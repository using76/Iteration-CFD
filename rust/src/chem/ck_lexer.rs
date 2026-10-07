// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The lexical layer of the CHEMKIN mechanism reader: comments, block
//! keywords, THERMO cards and the `NAME /content/` items of an auxiliary
//! line. SPEC-LIT §129 is the contract.
//!
//! Written from:
//!   SPEC-LIT §129, §130.2 and §130.5
//!   R. J. Kee, F. M. Rupley & J. A. Miller, *Chemkin-II*, Sandia
//!     SAND89-8009 (1989), DOI 10.2172/5681118, Chapter IV and Tables I, II,
//!     IV and V, read as a format
//!   R. J. Kee, F. M. Rupley, E. Meeks & J. A. Miller, *CHEMKIN-III*, Sandia
//!     SAND96-8216 (1996), DOI 10.2172/481621, pp. 46-54 (non-integer
//!     coefficients, FORD, RORD, UNITS)
//!   NIST SP 811 (2008) Appendix B.8, for `1 cal = 4.184 J`, and the SI 2019
//!     exact values of `N_A`, `e` and `k_B`
//! No CHEMKIN, Cantera or other reader's source was opened. No GPL-licensed
//! source was consulted.
//!
//! Host only, and f64 everywhere: this layer holds no numbers at all.

/// The four blocks of a mechanism, in file order (SPEC-LIT §129.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Block {
    Elements,
    Species,
    Thermo,
    Reactions,
}

impl Block {
    /// The block's full keyword, for a message: a block is named with this
    /// text and never with its `Debug` form.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Block::Elements => "ELEMENTS",
            Block::Species => "SPECIES",
            Block::Thermo => "THERMO",
            Block::Reactions => "REACTIONS",
        }
    }

    /// The position in the required order ELEMENTS, SPECIES, THERMO, REACTIONS.
    pub(crate) fn rank(self) -> u8 {
        match self {
            Block::Elements => 0,
            Block::Species => 1,
            Block::Thermo => 2,
            Block::Reactions => 3,
        }
    }
}

/// The block a line starts, if any (SPEC-LIT §129.1): the first whitespace
/// token, upper-cased, begins with the short form `ELEM`, `SPEC`, `THER` or
/// `REAC` and is also a prefix of the full word. So `ELEM`, `ELEMENTS`,
/// `REAC` and `REACTION` are keywords, and `SPECX` is not.
pub(crate) fn block_keyword(line: &str) -> Option<Block> {
    let t = line.split_whitespace().next()?.to_ascii_uppercase();
    const TABLE: [(&str, &str, Block); 4] = [
        ("ELEM", "ELEMENTS", Block::Elements),
        ("SPEC", "SPECIES", Block::Species),
        ("THER", "THERMO", Block::Thermo),
        ("REAC", "REACTIONS", Block::Reactions),
    ];
    TABLE
        .into_iter()
        .find(|(short, full, _)| t.starts_with(short) && full.starts_with(t.as_str()))
        .map(|(.., b)| b)
}

/// The line with its `!` comment removed: the comment runs from the first
/// `!` to the end of the line. Tabs are whitespace for every caller.
pub(crate) fn code(line: &str) -> &str {
    match line.find('!') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// The text after the first whitespace token of a line: what a block keyword
/// line hands to its block's reader.
pub(crate) fn after_first_token(line: &str) -> &str {
    let t = line.trim_start();
    match t.find(char::is_whitespace) {
        Some(i) => &t[i..],
        None => "",
    }
}

/// A THERMO card (SPEC-LIT §129.3): a line of at least 80 bytes whose column
/// 80 carries `1`, `2`, `3` or `4`. A card is never a block end, so a species
/// named `REAC...` stays a record.
pub(crate) fn thermo_card(line: &str) -> bool {
    line.len() >= 80 && matches!(line.as_bytes()[79], b'1'..=b'4')
}

/// One item of an auxiliary line: a NAME with an optional `/content/`
/// (SPEC-LIT §129.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlashItem {
    /// The longest run of characters that are neither whitespace nor `/`.
    pub(crate) name: String,
    /// Everything between the opening `/` and the next `/`, or `None` when
    /// no `/` follows the NAME.
    pub(crate) content: Option<String>,
}

/// The items of one line (its comment already removed), read left to right:
/// skip whitespace; NAME is the longest run of characters that are neither
/// whitespace nor `/`; skip whitespace again; when the next character is
/// `/`, the content is everything up to the next `/`. An empty NAME, or a
/// `/` that never closes, is refused; the reason is returned and the caller
/// makes it an `Error::Parse` that names the line.
pub(crate) fn slash_items(line: &str) -> Result<Vec<SlashItem>, String> {
    let b = line.as_bytes();
    let mut i = 0usize;
    let mut items = Vec::new();
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() {
            return Ok(items);
        }
        let start = i;
        while i < b.len() && b[i] != b'/' && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        let name = &line[start..i];
        if name.is_empty() {
            return Err("a /.../ field opens with no name before its first slash".to_string());
        }
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let content = if i < b.len() && b[i] == b'/' {
            i += 1;
            let cs = i;
            while i < b.len() && b[i] != b'/' {
                i += 1;
            }
            if i >= b.len() {
                return Err(format!("the item {name} opens a /.../ field that never closes"));
            }
            let content = &line[cs..i];
            i += 1;
            Some(content.to_string())
        } else {
            None
        };
        items.push(SlashItem { name: name.to_string(), content });
    }
}
