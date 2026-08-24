// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The case contract: what happens when a case asks for something this solver
//! does not have.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §13.4, which states the rule for this section, for
//!     §11.7, and "everywhere else". The `-permissive` escape hatch is marked
//!     *DESIGN* there and is ours.
//! No GPL-licensed source was consulted.
//!
//! # The rule
//!
//! ```text
//! recognised and implemented   -> use it
//! recognised, not implemented  -> Error naming the setting and what is
//!                                 available
//! not recognised               -> Error naming the setting
//! ```
//!
//! Silent substitution produces a plausible wrong answer, which is worse than
//! no answer. A user who writes `Gauss vanLeer` and gets first-order upwind
//! has no way of finding out; a user who writes `nu banana` and gets
//! `nu = 1e-05` gets a Reynolds number off by whatever factor the typo cost,
//! and a converged, plotted, believed result.
//!
//! # The escape hatch
//!
//! *DESIGN.* One switch, `-permissive`, downgrades every rejection here to a
//! warning on stderr and falls back to a documented default. It exists so a
//! case migrated from elsewhere can be run at all. It prints **what it
//! substituted**, once per distinct setting, every run - a warning that scrolls
//! past a thousand times is a warning nobody reads, and one that never repeats
//! is one nobody sees.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::error::{Error, Result};

/// Whether unsupported settings are warnings rather than errors.
///
/// Process-wide because the alternative - threading a flag through every
/// reader, every model constructor and every kernel launcher - would put the
/// switch in fifty signatures to be read in one place. It is set once, from
/// the command line, before any case file is opened.
static PERMISSIVE: AtomicBool = AtomicBool::new(false);

/// Settings already warned about, so `-permissive` prints one line per
/// setting rather than one per patch, per field, per iteration.
static WARNED: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Turn `-permissive` on. Call once, from `main`, before reading the case.
pub fn set_permissive(on: bool) {
    PERMISSIVE.store(on, Ordering::Relaxed);
}

pub fn permissive() -> bool {
    PERMISSIVE.load(Ordering::Relaxed)
}

/// Forget which settings have been warned about. Tests only: the warn-once
/// set is process-wide and two tests in one binary would otherwise see each
/// other's state.
#[cfg(test)]
pub fn reset_warnings() {
    if let Ok(mut w) = WARNED.lock() {
        w.clear();
    }
}

/// The standard usage line for the switch, so every driver spells it the same.
pub const PERMISSIVE_USAGE: &str =
    "  -permissive     downgrade unsupported-setting errors to warnings and\n\
     \x20                 substitute a documented default";

/// Reject a setting this solver does not implement - or, under `-permissive`,
/// warn once and return `fallback`.
///
/// * `setting` names the dictionary entry, as the user wrote it:
///   `"divSchemes/div(phi,U)"`, `"boundaryField/inlet/type"`, `"nu"`.
/// * `value` is what the case said.
/// * `available` lists what this solver does have. Empty is allowed - some
///   settings have no menu, only a syntax - but where there is a menu it must
///   be printed, because "not supported" without it sends the user to the
///   source.
/// * `fallback_name` says what `-permissive` substitutes, in words. It is
///   printed, so it has to be the truth.
pub fn unsupported<T>(
    setting: &str,
    value: &str,
    available: &[&str],
    fallback_name: &str,
    fallback: T,
) -> Result<T> {
    let menu = if available.is_empty() {
        String::new()
    } else {
        format!("; available: {}", available.join(", "))
    };

    if !permissive() {
        return Err(Error::Config(format!(
            "{setting}: \"{value}\" is not supported by ofgpu{menu}\n  \
             (run with -permissive to substitute {fallback_name} and continue)"
        )));
    }

    warn_once(setting, &format!(
        "-permissive: {setting} \"{value}\" is not supported{menu}\n  \
         substituting {fallback_name}"
    ));

    Ok(fallback)
}

/// [`unsupported`] plus a NOTE printed on its own line.
///
/// For the case where "not supported" is the wrong summary on its own: the
/// feature exists, and what the user needs to know is *why it cannot be
/// reached from here*. Folding that sentence into `fallback_name` produces
/// nested parentheses and an unreadable line, so it gets its own.
pub fn unsupported_note<T>(
    setting: &str,
    value: &str,
    available: &[&str],
    note: &str,
    fallback_name: &str,
    fallback: T,
) -> Result<T> {
    let menu = if available.is_empty() {
        String::new()
    } else {
        format!("; available: {}", available.join(", "))
    };

    if !permissive() {
        return Err(Error::Config(format!(
            "{setting}: \"{value}\" is not supported by ofgpu{menu}
               note: {note}
               (run with -permissive to substitute {fallback_name} and continue)"
        )));
    }

    warn_once(setting, &format!(
        "-permissive: {setting} \"{value}\" is not supported{menu}
           note: {note}
           substituting {fallback_name}"
    ));

    Ok(fallback)
}

/// [`unsupported`] for an entry that could not be *parsed* at all, as opposed
/// to one naming something off the menu: `nu banana`, `deltaT later`.
///
/// Kept separate because the diagnostic is a different sentence - there is no
/// menu to print, the value is simply not a number - and because the reader
/// that hits it is usually several frames below anything that knows the file
/// name.
pub fn unreadable<T>(setting: &str, value: &str, expected: &str, fallback: T) -> Result<T> {
    if !permissive() {
        return Err(Error::Config(format!(
            "{setting}: \"{value}\" is not {expected}\n  \
             (run with -permissive to continue with the default)"
        )));
    }

    warn_once(
        setting,
        &format!(
            "-permissive: {setting} \"{value}\" is not {expected}; using the default"
        ),
    );

    Ok(fallback)
}

/// Print `msg` on stderr the first time this `setting` is seen.
///
/// The key is the setting rather than the whole message, so a `".*"` patch
/// entry that is wrong on four hundred patches produces one line.
pub fn warn_once(setting: &str, msg: &str) {
    let Ok(mut seen) = WARNED.lock() else {
        // A poisoned mutex means another thread panicked mid-warning. Print
        // anyway: losing the diagnostic is worse than printing it twice.
        eprintln!("[ofgpu] {msg}");
        return;
    };
    if seen.insert(setting.to_string()) {
        eprintln!("[ofgpu] {msg}");
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

/// Serialises every test, in ANY module, that touches the process-wide
/// permissive flag or the warn-once set.
///
/// The flag is global on purpose (a run is either permissive or it is not),
/// which makes tests that set it race each other under cargo's parallel
/// runner: a strict-mode assertion observes another module's
/// `set_permissive(true)` and fails only when the scheduler interleaves them
/// - the worst kind of flake. Take this guard FIRST in any test that calls
/// [`set_permissive`] or [`reset_warnings`]. Poisoning is ignored: a panic in
/// one such test must not cascade into every other one.
pub fn permissive_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: Mutex<()> = Mutex::new(());
    match GUARD.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All these take the crate-wide guard above - the flag is shared with
    // every other module's permissive tests, not just this file's.

    #[test]
    fn strict_is_the_default_and_names_the_setting_and_the_menu() {
        let _g = permissive_test_guard();
        set_permissive(false);

        let e = unsupported(
            "divSchemes/div(phi,U)",
            "Gauss vanLeer",
            &["Gauss linear", "Gauss upwind"],
            "upwind",
            7,
        )
        .unwrap_err();

        let s = e.to_string();
        // The three things a user needs to fix the case.
        assert!(s.contains("divSchemes/div(phi,U)"), "{s}");
        assert!(s.contains("Gauss vanLeer"), "{s}");
        assert!(s.contains("Gauss upwind"), "{s}");
        assert!(s.contains("-permissive"), "{s}");
    }

    #[test]
    fn permissive_substitutes_and_says_so() {
        let _g = permissive_test_guard();
        reset_warnings();
        set_permissive(true);

        let v = unsupported("nu", "banana", &[], "the built-in default", 1e-5f64);
        assert_eq!(v.ok(), Some(1e-5));

        let v = unreadable("deltaT", "later", "a number", 0.1f64);
        assert_eq!(v.ok(), Some(0.1));

        set_permissive(false);
    }

    #[test]
    fn unreadable_is_an_error_when_strict() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let e = unreadable("nu", "banana", "a number", 1e-5f64).unwrap_err();
        assert!(e.to_string().contains("banana"));
    }
}
