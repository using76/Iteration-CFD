// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! A POSIX extended-regular-expression matcher, big enough for a case file.
//!
//! Written from:
//!   IEEE Std 1003.1 (POSIX.1-2017), Base Definitions ch. 9.4, "Extended
//!     Regular Expressions" - the grammar and the operator precedence
//!   S. C. Kleene, "Representation of events in nerve nets and finite
//!     automata", *Automata Studies*, Princeton (1956) - regular expressions
//!   K. Thompson, "Regular expression search algorithm",
//!     *Comm. ACM* 11 (1968) 419-422 - the recursive matcher this follows in
//!     spirit, with a continuation in place of a compiled program
//! No GPL-licensed source was consulted.
//!
//! # Why this exists at all
//!
//! An OpenFOAM dictionary key may be a quoted regular expression, and the
//! idiom is everywhere:
//!
//! ```text
//! boundaryField { ".*" { type zeroGradient; } }
//! solvers       { "(U|k|epsilon)" { solver PBiCGStab; } }
//! ```
//!
//! A reader that only does exact string lookup finds nothing in either case
//! and falls through to its defaults, silently. That is the failure this
//! module removes; see [`crate::io::dict::FoamDict::resolve`].
//!
//! # What is supported
//!
//! `.` `*` `+` `?` `|` `(...)` `[abc]` `[^a-z]` `{m}` `{m,}` `{m,n}`, the
//! anchors `^` and `$`, and a backslash to escape any of them. Character
//! classes accept ranges and a leading `^` for negation. That is the whole of
//! POSIX ERE except equivalence classes and collating symbols
//! (`[[:alpha:]]` and friends), which no case file uses, and back-references,
//! which ERE does not have.
//!
//! The match is always **anchored at both ends**, because a dictionary key is
//! a whole name: `inlet` must not be matched by the pattern `let`. `^` and
//! `$` are therefore accepted and ignored where they are redundant, which is
//! how a hand-written `"^wall.*$"` keeps working.
//!
//! # Cost
//!
//! Backtracking, exponential in the worst case on a pathological pattern such
//! as `(a*)*b`. That is acceptable here and nowhere else: the subject is a
//! patch name of a few characters, the pattern comes from the case author's
//! own file, and the alternative - Thompson's NFA simulation - buys nothing
//! at this size. If this is ever pointed at untrusted input, replace it.

/// One node of the parsed pattern.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// One literal character.
    Lit(char),
    /// `.` - any character. POSIX excludes the newline; a dictionary key
    /// cannot contain one, so this matches anything.
    Any,
    /// `[...]`: the set, and whether it was negated.
    Class { set: Vec<ClassItem>, negated: bool },
    /// A sequence, matched left to right.
    Seq(Vec<Node>),
    /// `a|b|c`, tried in order.
    Alt(Vec<Node>),
    /// `x{min,max}`; `max = None` is unbounded. `*` is `{0,}`, `+` is `{1,}`
    /// and `?` is `{0,1}`, so there is one repetition node rather than four.
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        /// Greedy in POSIX; kept explicit because `*?` appears in patterns
        /// copied from Perl-flavoured documentation and must not be a syntax
        /// error.
        greedy: bool,
    },
    /// `^` or `$`. Both are no-ops under a full-string match, but they must
    /// parse.
    Anchor,
    /// The empty alternative, as in `(a|)`.
    Empty,
}

#[derive(Debug, Clone, PartialEq)]
enum ClassItem {
    Ch(char),
    Range(char, char),
}

/// A compiled pattern.
#[derive(Debug, Clone)]
pub struct Regex {
    root: Node,
    src: String,
}

impl Regex {
    /// Parse `pattern`. The error names the offending pattern, because it
    /// came out of a case file and the user has to find it.
    pub fn new(pattern: &str) -> std::result::Result<Self, String> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser { s: &chars, i: 0 };
        let root = p.alternation()?;
        if p.i != chars.len() {
            return Err(format!(
                "regular expression \"{pattern}\": unbalanced ')' at character {}",
                p.i + 1
            ));
        }
        Ok(Self {
            root,
            src: pattern.to_string(),
        })
    }

    /// True when the pattern matches the WHOLE of `text`.
    pub fn is_match(&self, text: &str) -> bool {
        let t: Vec<char> = text.chars().collect();
        matches_at(&self.root, &t, 0, &mut |end| end == t.len())
    }

    pub fn as_str(&self) -> &str {
        &self.src
    }
}

/// True when `pattern` is worth compiling as a pattern at all.
///
/// A key with no ERE metacharacter in it can only ever match itself, so the
/// exact lookup that runs first has already dealt with it. Skipping those
/// keeps the pattern list short and, more usefully, keeps a plain patch name
/// out of the pattern arm entirely - so `inlet` never shadows `inlet2` by
/// accident of ordering.
pub fn looks_like_a_pattern(pattern: &str) -> bool {
    pattern.contains(|c| {
        matches!(
            c,
            '.' | '*' | '+' | '?' | '|' | '(' | '[' | '{' | '\\' | '^' | '$'
        )
    })
}

// ==========================================================================
//  Parsing
// ==========================================================================

struct Parser<'a> {
    s: &'a [char],
    i: usize,
}

type Parsed = std::result::Result<Node, String>;

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn alternation(&mut self) -> Parsed {
        let mut branches = vec![self.concatenation()?];
        while self.peek() == Some('|') {
            self.i += 1;
            branches.push(self.concatenation()?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().unwrap_or(Node::Empty)
        } else {
            Node::Alt(branches)
        })
    }

    fn concatenation(&mut self) -> Parsed {
        let mut items: Vec<Node> = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.postfix(atom)?);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap_or(Node::Empty),
            _ => Node::Seq(items),
        })
    }

    /// `*`, `+`, `?` and `{m,n}`, plus a trailing `?` that makes the
    /// repetition lazy.
    fn postfix(&mut self, atom: Node) -> Parsed {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.i += 1;
                (0, None)
            }
            Some('+') => {
                self.i += 1;
                (1, None)
            }
            Some('?') => {
                self.i += 1;
                (0, Some(1))
            }
            Some('{') if self.is_bound() => {
                self.i += 1;
                let min = self.number()?;
                let max = if self.peek() == Some(',') {
                    self.i += 1;
                    if self.peek() == Some('}') {
                        None
                    } else {
                        Some(self.number()?)
                    }
                } else {
                    Some(min)
                };
                if self.peek() != Some('}') {
                    return Err("regular expression: expected a closing brace".to_string());
                }
                self.i += 1;
                (min, max)
            }
            _ => return Ok(atom),
        };

        let mut greedy = true;
        if self.peek() == Some('?') {
            self.i += 1;
            greedy = false;
        }

        Ok(Node::Repeat {
            node: Box::new(atom),
            min,
            max,
            greedy,
        })
    }

    /// `{` only opens a bound when a digit follows; otherwise POSIX says it is
    /// a literal brace.
    fn is_bound(&self) -> bool {
        matches!(self.s.get(self.i + 1), Some(c) if c.is_ascii_digit())
    }

    fn number(&mut self) -> std::result::Result<u32, String> {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if start == self.i {
            return Err("regular expression: expected a repetition count".to_string());
        }
        self.s[start..self.i]
            .iter()
            .collect::<String>()
            .parse::<u32>()
            .map_err(|_| "regular expression: repetition count out of range".to_string())
    }

    fn atom(&mut self) -> Parsed {
        let Some(c) = self.peek() else {
            return Ok(Node::Empty);
        };
        self.i += 1;

        match c {
            '(' => {
                let inner = self.alternation()?;
                if self.peek() != Some(')') {
                    return Err("regular expression: missing ')'".to_string());
                }
                self.i += 1;
                Ok(inner)
            }
            '[' => self.class(),
            '.' => Ok(Node::Any),
            '^' | '$' => Ok(Node::Anchor),
            '\\' => match self.peek() {
                Some(e) => {
                    self.i += 1;
                    Ok(Node::Lit(e))
                }
                None => Err("regular expression: trailing backslash".to_string()),
            },
            // `*` and `+` here mean a repetition with nothing to repeat.
            '*' | '+' => Err(format!("regular expression: '{c}' has nothing to repeat")),
            other => Ok(Node::Lit(other)),
        }
    }

    /// `[abc]`, `[^a-z]`, `[]]`. POSIX gives `]` its literal meaning when it
    /// is the first character of the set, and `-` its literal meaning when it
    /// is first or last.
    fn class(&mut self) -> Parsed {
        let mut negated = false;
        if self.peek() == Some('^') {
            negated = true;
            self.i += 1;
        }

        let mut set: Vec<ClassItem> = Vec::new();
        let mut first = true;

        loop {
            let Some(c) = self.peek() else {
                return Err("regular expression: missing ']'".to_string());
            };
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;

            let lo = if c == '\\' {
                match self.peek() {
                    Some(e) => {
                        self.i += 1;
                        e
                    }
                    None => return Err("regular expression: trailing backslash".to_string()),
                }
            } else {
                c
            };

            // A `-` that is not followed by the closing bracket opens a range.
            if self.peek() == Some('-') && self.s.get(self.i + 1).copied() != Some(']') {
                self.i += 1;
                let Some(hi) = self.peek() else {
                    return Err("regular expression: missing ']'".to_string());
                };
                self.i += 1;
                set.push(ClassItem::Range(lo, hi));
            } else {
                set.push(ClassItem::Ch(lo));
            }
        }

        Ok(Node::Class { set, negated })
    }
}

// ==========================================================================
//  Matching
// ==========================================================================

fn class_matches(set: &[ClassItem], negated: bool, c: char) -> bool {
    let hit = set.iter().any(|item| match item {
        ClassItem::Ch(x) => *x == c,
        ClassItem::Range(a, b) => *a <= c && c <= *b,
    });
    hit != negated
}

/// Continuation-passing matcher: try to match `node` starting at `pos`, and
/// for each way it can succeed call `k` with the position after it. Returns
/// true as soon as `k` accepts, which is what makes the backtracking stop at
/// the first whole-string match rather than enumerating all of them.
fn matches_at(node: &Node, t: &[char], pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match node {
        Node::Empty | Node::Anchor => k(pos),

        Node::Lit(c) => t.get(pos) == Some(c) && k(pos + 1),

        Node::Any => pos < t.len() && k(pos + 1),

        Node::Class { set, negated } => match t.get(pos) {
            Some(c) if class_matches(set, *negated, *c) => k(pos + 1),
            _ => false,
        },

        Node::Alt(branches) => branches.iter().any(|b| matches_at(b, t, pos, k)),

        Node::Seq(items) => seq_at(items, t, pos, k),

        Node::Repeat {
            node,
            min,
            max,
            greedy,
        } => repeat_at(node, *min, *max, *greedy, t, pos, 0, k),
    }
}

fn seq_at(items: &[Node], t: &[char], pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match items.split_first() {
        None => k(pos),
        Some((head, rest)) => matches_at(head, t, pos, &mut |next| seq_at(rest, t, next, k)),
    }
}

#[allow(clippy::too_many_arguments)]
fn repeat_at(
    node: &Node,
    min: u32,
    max: Option<u32>,
    greedy: bool,
    t: &[char],
    pos: usize,
    done: u32,
    k: &mut dyn FnMut(usize) -> bool,
) -> bool {
    let may_stop = done >= min;
    let may_go = match max {
        Some(m) => done < m,
        None => true,
    };

    // Trying one more repetition first is what makes the operator greedy.
    // The zero-width guard (`next > pos`) is what stops `(a*)*` looping for
    // ever on an empty match.
    let more = |k: &mut dyn FnMut(usize) -> bool| -> bool {
        may_go
            && matches_at(node, t, pos, &mut |next| {
                next > pos && repeat_at(node, min, max, greedy, t, next, done + 1, k)
            })
    };

    if greedy {
        more(k) || (may_stop && k(pos))
    } else {
        (may_stop && k(pos)) || more(k)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p: &str, s: &str) -> bool {
        match Regex::new(p) {
            Ok(r) => r.is_match(s),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn the_two_idioms_that_matter() {
        // `".*" { type zeroGradient; }` in boundaryField.
        assert!(m(".*", "inlet"));
        assert!(m(".*", ""));
        // `"(U|k|epsilon)"` in fvSolution/solvers.
        assert!(m("(U|k|epsilon)", "epsilon"));
        assert!(m("(U|k|epsilon)", "U"));
        assert!(!m("(U|k|epsilon)", "omega"));
        assert!(!m("(U|k|epsilon)", "epsilonX"));
    }

    #[test]
    fn the_match_is_anchored_at_both_ends() {
        // A patch called `inlet` must not be found by the pattern `let`,
        // which is what an unanchored search would do.
        assert!(!m("let", "inlet"));
        assert!(!m("inle", "inlet"));
        assert!(m("inlet", "inlet"));
        // Redundant anchors are accepted and change nothing.
        assert!(m("^inlet$", "inlet"));
        assert!(m("^wall.*$", "wallLeft"));
    }

    #[test]
    fn repetition() {
        assert!(m("walls?", "wall"));
        assert!(m("walls?", "walls"));
        assert!(!m("walls?", "wallss"));
        assert!(m("a+", "aaa"));
        assert!(!m("a+", ""));
        assert!(m("a{2,3}", "aa"));
        assert!(m("a{2,3}", "aaa"));
        assert!(!m("a{2,3}", "a"));
        assert!(!m("a{2,3}", "aaaa"));
        assert!(m("a{2}", "aa"));
        assert!(m("a{2,}", "aaaaa"));
    }

    #[test]
    fn character_classes() {
        assert!(m("procBoundary[0-9]+", "procBoundary12"));
        assert!(!m("procBoundary[0-9]+", "procBoundary"));
        assert!(m("[^0-9]+", "wall"));
        assert!(!m("[^0-9]+", "wall1"));
        // `]` first is a literal.
        assert!(m("[]a]", "]"));
        // `-` last is a literal.
        assert!(m("[a-]", "-"));
    }

    #[test]
    fn escapes_and_literal_braces() {
        assert!(m(r"div\(phi,U\)", "div(phi,U)"));
        assert!(m(r"a\.b", "a.b"));
        assert!(!m(r"a\.b", "axb"));
        // `{` with no digit after it is a literal brace, per POSIX.
        assert!(m("a{b", "a{b"));
    }

    #[test]
    fn nested_groups_backtrack() {
        assert!(m("(ab|a)(b?)c", "abc"));
        assert!(m("(a*)*b", "aaab"));
        assert!(!m("(a*)*b", "aaa"));
        assert!(m("(inlet|outlet)[0-9]*", "outlet3"));
    }

    #[test]
    fn a_plain_name_is_not_a_pattern() {
        // The exact lookup has already handled these, and treating them as
        // patterns would let `inlet` shadow `inlet2` by ordering alone.
        assert!(!looks_like_a_pattern("inlet"));
        assert!(!looks_like_a_pattern("movingWall"));
        assert!(looks_like_a_pattern(".*"));
        assert!(looks_like_a_pattern("(U|k)"));
        assert!(looks_like_a_pattern("walls?"));
    }

    #[test]
    fn a_broken_pattern_is_an_error_not_a_panic() {
        assert!(Regex::new("(unclosed").is_err());
        assert!(Regex::new("[unclosed").is_err());
        assert!(Regex::new("closed)").is_err());
        assert!(Regex::new("*").is_err());
        assert!(Regex::new("trailing\\").is_err());
    }
}
