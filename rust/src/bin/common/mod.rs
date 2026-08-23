// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! Bits every ofgpu executable needs, and nothing a library user would.
//!
//! Not a directory cargo builds: `src/bin/<name>/` becomes a binary target
//! only when it contains `main.rs`, so this one is invisible to the target
//! auto-discovery and is pulled in with `#[path = "common/mod.rs"] mod common;`
//! by each binary that wants it.
//!
//! Everything here exists for one reason: **the C++ build and this port have
//! to be runnable side by side and diffed.** That makes the exact shape of a
//! printed number part of the interface, and `std::ostream`'s default is not
//! Rust's — `1e-05` versus `0.00001`, `1.500e-07` versus `1.5e-7`. The two
//! formatters below close that gap.

#![allow(dead_code)]

use ofgpu::{Gpu, Result, Scalar};

// ==========================================================================
//  Number formatting, C++ `std::ostream` style
// ==========================================================================

/// `std::ostream << double` with the default `precision(6)`, i.e. `%g`.
///
/// `1000`, `0.5`, `1e-05`, `1e+07`. Rust's `{}` writes the last two as
/// `0.00001` and `10000000`, which is a diff on every line that carries a
/// viscosity or a residual.
pub fn g(x: f64) -> String {
    g_prec(x, 6)
}

/// [`g`] with an explicit significant-digit count.
pub fn g_prec(x: f64, prec: i32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }

    // Decimal exponent AFTER rounding to `prec` significant digits: 9.999996e2
    // rounds to 1e3 and must be treated as exponent 3, not 2.
    let mut exp = x.abs().log10().floor() as i32;
    if format!("{:.*}", (prec - 1) as usize, x.abs() / 10f64.powi(exp)).starts_with("10") {
        exp += 1;
    }

    let trimmed = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };

    if exp < -4 || exp >= prec {
        let mantissa = trimmed(format!("{:.*}", (prec - 1) as usize, x / 10f64.powi(exp)));
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        trimmed(format!("{:.*}", (prec - 1 - exp).max(0) as usize, x))
    }
}

/// `std::scientific` with `setprecision(prec)`, i.e. `%.*e`.
///
/// The exponent is padded to two digits with an explicit sign, which is what
/// C and C++ do and what Rust's `{:e}` does not.
pub fn sci(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return g(x);
    }

    let s = format!("{:.*e}", prec, x);

    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp.trim_start_matches('+')),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

// ==========================================================================
//  Device banner
// ==========================================================================

/// `float` or `double`, whichever the `single` feature selected. Mirrors
/// `OFGPU_SCALAR_IS_FLOAT` in the C++ build.
pub fn precision_name() -> &'static str {
    if std::mem::size_of::<Scalar>() == 4 {
        "float"
    } else {
        "double"
    }
}

/// The header line every driver opens with:
/// `ofgpu k-epsilon | <device> sm_<cc> | <n> MiB | precision double`.
pub fn device_banner(gpu: &Gpu, tag: &str) -> Result<String> {
    let ctx = gpu.ctx();
    let (major, minor) = ctx.compute_capability()?;
    let total = ctx.total_mem()?;

    Ok(format!(
        "ofgpu {tag} | {} sm_{major}{minor} | {} MiB | precision {}",
        ctx.name()?,
        total >> 20,
        precision_name()
    ))
}

/// Resident device memory, as the benchmark reports it. `mem_get_info`
/// returns `(free, total)`; what a user cares about is the difference.
pub fn resident_mib(gpu: &Gpu) -> Result<(usize, usize)> {
    let (free, total) = gpu.mem_info()?;
    Ok(((total - free) >> 20, total >> 20))
}

// ==========================================================================
//  Command line
// ==========================================================================

/// The value following a flag, or a diagnostic naming the flag that is
/// missing one.
///
/// The C++ called `std::exit(1)` from inside its lambda; returning an error
/// lets the caller print it the same way it prints every other failure.
pub fn next_arg(args: &[String], i: &mut usize) -> Result<String> {
    let flag = args.get(*i).cloned().unwrap_or_default();
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| ofgpu::Error::Config(format!("missing value after {flag}")))
}

/// `std::atoi`: the leading integer, or zero. Deliberately not
/// `str::parse`, because the C++ accepts `50x` and a stricter reader here
/// would reject a command line the reference build runs.
pub fn atoi(s: &str) -> i64 {
    let t = s.trim_start();
    let (sign, digits) = match t.strip_prefix('-') {
        Some(d) => (-1i64, d),
        None => (1i64, t.strip_prefix('+').unwrap_or(t)),
    };

    let mut v: i64 = 0;
    for c in digits.chars() {
        match c.to_digit(10) {
            Some(d) => v = v.saturating_mul(10).saturating_add(i64::from(d)),
            None => break,
        }
    }

    sign * v
}

// ==========================================================================
//  Tests
// ==========================================================================

/// These run once per binary that includes the module, because each `[[bin]]`
/// is its own crate. That is the cost of sharing host code between binaries
/// without a third crate, and it is worth paying: a formatter that silently
/// drifts from `std::ostream` turns every side-by-side diff against the C++
/// build into noise, which is the one thing this module exists to prevent.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_matches_ostream_defaults() {
        assert_eq!(g(1000.0), "1000");
        assert_eq!(g(0.5), "0.5");
        assert_eq!(g(1e-5), "1e-05");
        assert_eq!(g(1.0), "1");
        assert_eq!(g(0.0), "0");
        assert_eq!(g(1e7), "1e+07");
        assert_eq!(g(-0.001), "-0.001");
        // Six significant digits, trailing zeros stripped.
        assert_eq!(g(6.355280e-05), "6.35528e-05");
        assert_eq!(g(0.09), "0.09");
    }

    #[test]
    fn sci_pads_the_exponent_to_two_digits() {
        // Rust's own `{:e}` writes `6.813e-1`; C and C++ write `6.813e-01`,
        // and these lines are diffed against the C++ build.
        assert_eq!(sci(0.6813, 3), "6.813e-01");
        assert_eq!(sci(0.0, 3), "0.000e+00");
        assert_eq!(sci(1.0, 3), "1.000e+00");
        assert_eq!(sci(7e-18, 0), "7e-18");
        assert_eq!(sci(1.5e7, 3), "1.500e+07");
        assert_eq!(sci(-2.5e-13, 3), "-2.500e-13");
    }

    #[test]
    fn atoi_takes_the_leading_integer_like_c() {
        assert_eq!(atoi("50"), 50);
        assert_eq!(atoi("-7"), -7);
        assert_eq!(atoi("  12abc"), 12);
        // A flag misread as a positional argument must become 0, which is what
        // the C++ benchmark's argument loop relies on.
        assert_eq!(atoi("-iters"), 0);
        assert_eq!(atoi(""), 0);
    }
}
