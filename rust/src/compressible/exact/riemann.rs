// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The exact solution of the Riemann problem of the 1-D Euler equations,
//! sampled at `(x, t)`: two constant states joined by a shock or a
//! rarefaction on each side and a contact between them. SPEC-LIT §127.2 is
//! the contract: equations (127.5) to (127.7) are solved here for the star
//! state, and (127.7a), (127.7b) with the paragraph after them are
//! evaluated here to sample the solution. The pressure function (127.5) is
//! monotone in `p`, so bisection on a doubled bracket finds `p*`; the wave
//! of a side is a shock exactly when `p* > p_K`.
//!
//! Written from:
//!   SPEC-LIT §127.2 - the equations, the vacuum condition, the sampling
//!     rules, and the oracle values the tests quote
//!   Ames Research Staff, *Equations, Tables, and Charts for Compressible
//!     Flow*, NACA Report 1135 (1953), NTRS 19930091059, a US-Government
//!     work - eq. 35 (the isentropic law), 84-85 (mass and momentum across
//!     a shock), 91 (the Hugoniot), 93 (the shock speed), read from the
//!     page images by the CMP-05 supervisor
//! Toro (2009) ch. 4 was not read. No compressible-flow code (OpenFOAM,
//! SU2, or any Riemann-solver library) was opened. No GPL-licensed source
//! was consulted.
//!
//! Precision (SPEC-LIT §127.6): `f64` everywhere, in both builds. An answer
//! key is a constant of a gate and never a `Scalar`, so it behaves
//! identically in the default build and in `--features single`. Every
//! refusal is an `Err(Error::Refused(..))` whose message starts
//! `compressible::exact::riemann::`.

use super::{bisect, check_finite, check_gamma, refuse};
use crate::error::Error;

/// A primitive state `(rho, u, p)` of the 1-D Euler equations: the input of
/// [`solve`], and what [`RiemannSolution::sample`] returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Primitive {
    /// The density.
    pub rho: f64,
    /// The x velocity.
    pub u: f64,
    /// The pressure.
    pub p: f64,
}

/// One wave of the pair that joins the left and right states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wave {
    /// A shock moving at `speed` in the lab frame, (127.7a).
    Shock {
        /// The shock speed `S_L` or `S_R`.
        speed: f64,
    },
    /// A rarefaction fan between `head` and `tail`, (127.7a): `head < tail`
    /// on the left, `tail < head` on the right.
    Rarefaction {
        /// The characteristic of the undisturbed state: `u_L - c_L` on the
        /// left, `u_R + c_R` on the right.
        head: f64,
        /// The characteristic of the star state.
        tail: f64,
    },
}

/// The full self-similar solution: [`solve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiemannSolution {
    /// The ratio of specific heats the solution was built with.
    pub gamma: f64,
    /// The left input state.
    pub left: Primitive,
    /// The right input state.
    pub right: Primitive,
    /// The star pressure, the root of (127.6).
    pub p_star: f64,
    /// The star velocity, (127.6).
    pub u_star: f64,
    /// The density between the left wave and the contact, (127.7).
    pub rho_star_left: f64,
    /// The density between the contact and the right wave, (127.7).
    pub rho_star_right: f64,
    /// The wave that carries the left state to the contact.
    pub left_wave: Wave,
    /// The wave that carries the right state to the contact.
    pub right_wave: Wave,
}

/// The sound speed `sqrt(gamma p / rho)`.
fn sound_speed(gamma: f64, rho: f64, p: f64) -> f64 {
    (gamma * p / rho).sqrt()
}

/// `f_K(p)` of (127.5): the velocity jump across the wave of side `K` at
/// pressure `p` - the shock branch of NACA (84)-(85) and (91) above `p_K`,
/// the isentropic rarefaction branch of NACA (35) at or below it.
fn f_k(gamma: f64, k: Primitive, p: f64) -> f64 {
    if p > k.p {
        let a = 2.0 / ((gamma + 1.0) * k.rho);
        let b = (gamma - 1.0) * k.p / (gamma + 1.0);
        (p - k.p) * (a / (p + b)).sqrt()
    } else {
        let c = sound_speed(gamma, k.rho, k.p);
        let e = (gamma - 1.0) / (2.0 * gamma);
        2.0 * c / (gamma - 1.0) * ((p / k.p).powf(e) - 1.0)
    }
}

/// The star pressure, the root of (127.6): `F(p) = f_L(p) + f_R(p) +
/// (u_R - u_L)`, which increases in `p`, by bisection on a bracket that
/// starts at `max(p_L, p_R)` and doubles until `F` turns non-negative. The
/// caller has checked gamma, finiteness, positivity and the vacuum
/// condition, so `F(0) < 0`.
fn star_pressure(gamma: f64, left: Primitive, right: Primitive) -> Result<f64, Error> {
    const WHO: &str = "riemann::solve";
    let f = |p: f64| f_k(gamma, left, p) + f_k(gamma, right, p) + (right.u - left.u);
    let mut hi = left.p.max(right.p);
    let mut doublings = 0;
    while f(hi) < 0.0 {
        hi *= 2.0;
        doublings += 1;
        if doublings > 200 {
            return Err(refuse(
                WHO,
                format!(
                    "the bracket for p* doubled 200 times from p = {} and F is still negative",
                    left.p.max(right.p)
                ),
            ));
        }
    }
    Ok(bisect(0.0, hi, |p| f(p) < 0.0))
}

/// The exact Riemann solution of the 1-D Euler equations, (127.5) to
/// (127.7): the two waves, the constant star state between them, and
/// nothing at the contact but a jump in density.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; an input not
/// finite; the `rho` or the `p` of either side `<= 0`; and a pair that
/// generates vacuum, `2 (c_L + c_R)/(gamma - 1) <= u_R - u_L`.
pub fn solve(gamma: f64, left: Primitive, right: Primitive) -> Result<RiemannSolution, Error> {
    const WHO: &str = "riemann::solve";
    check_gamma(WHO, gamma)?;
    check_finite(
        WHO,
        &[
            ("left.rho", left.rho),
            ("left.u", left.u),
            ("left.p", left.p),
            ("right.rho", right.rho),
            ("right.u", right.u),
            ("right.p", right.p),
        ],
    )?;
    for (side, rho) in [("left.rho", left.rho), ("right.rho", right.rho)] {
        if rho <= 0.0 {
            return Err(refuse(WHO, format!("{side} must be above 0, got {rho}")));
        }
    }
    for (side, p) in [("left.p", left.p), ("right.p", right.p)] {
        if p <= 0.0 {
            return Err(refuse(WHO, format!("{side} must be above 0, got {p}")));
        }
    }
    let c_l = sound_speed(gamma, left.rho, left.p);
    let c_r = sound_speed(gamma, right.rho, right.p);
    let gap = 2.0 * (c_l + c_r) / (gamma - 1.0);
    if gap <= right.u - left.u {
        return Err(refuse(
            WHO,
            format!(
                "the pair generates vacuum: 2 (c_L + c_R)/(gamma-1) = {gap} <= u_R - u_L = {}",
                right.u - left.u
            ),
        ));
    }
    let p_star = star_pressure(gamma, left, right)?;
    let f_l = f_k(gamma, left, p_star);
    let f_r = f_k(gamma, right, p_star);
    let u_star = 0.5 * (left.u + right.u) + 0.5 * (f_r - f_l);
    let q = (gamma - 1.0) / (gamma + 1.0);
    let e = (gamma - 1.0) / (2.0 * gamma);
    let density = |k: Primitive| {
        let r = p_star / k.p;
        if p_star > k.p {
            k.rho * (r + q) / (q * r + 1.0)
        } else {
            k.rho * r.powf(1.0 / gamma)
        }
    };
    let left_wave = if p_star > left.p {
        let qq = ((gamma + 1.0) / (2.0 * gamma) * (p_star / left.p) + e).sqrt();
        Wave::Shock {
            speed: left.u - c_l * qq,
        }
    } else {
        Wave::Rarefaction {
            head: left.u - c_l,
            tail: u_star - c_l * (p_star / left.p).powf(e),
        }
    };
    let right_wave = if p_star > right.p {
        let qq = ((gamma + 1.0) / (2.0 * gamma) * (p_star / right.p) + e).sqrt();
        Wave::Shock {
            speed: right.u + c_r * qq,
        }
    } else {
        Wave::Rarefaction {
            head: right.u + c_r,
            tail: u_star + c_r * (p_star / right.p).powf(e),
        }
    };
    Ok(RiemannSolution {
        gamma,
        left,
        right,
        p_star,
        u_star,
        rho_star_left: density(left),
        rho_star_right: density(right),
        left_wave,
        right_wave,
    })
}

/// The isentropic state of side `k` at sound speed `c` inside a fan,
/// (127.7b): `u` is given by the linear formula, and `rho` and `p` follow
/// isentropically from the undisturbed state, NACA (35).
fn fan_state(gamma: f64, k: Primitive, c: f64, u: f64) -> Primitive {
    let r = c / sound_speed(gamma, k.rho, k.p);
    Primitive {
        rho: k.rho * r.powf(2.0 / (gamma - 1.0)),
        u,
        p: k.p * r.powf(2.0 * gamma / (gamma - 1.0)),
    }
}

impl RiemannSolution {
    /// The primitive state at the reduced coordinate `xi = (x - x_0)/t`,
    /// the paragraph after (127.7b): `xi <= u*` reads the left side, else
    /// the right side. Unchecked - NaN in, NaN out.
    pub fn sample(&self, xi: f64) -> Primitive {
        let g = self.gamma;
        let star_left = Primitive {
            rho: self.rho_star_left,
            u: self.u_star,
            p: self.p_star,
        };
        let star_right = Primitive {
            rho: self.rho_star_right,
            u: self.u_star,
            p: self.p_star,
        };
        if xi <= self.u_star {
            match self.left_wave {
                Wave::Shock { speed } => {
                    if xi < speed {
                        self.left
                    } else {
                        star_left
                    }
                }
                Wave::Rarefaction { head, tail } => {
                    if xi <= head {
                        self.left
                    } else if xi >= tail {
                        star_left
                    } else {
                        let h = (g - 1.0) / 2.0;
                        let c_l = sound_speed(g, self.left.rho, self.left.p);
                        let u = 2.0 / (g + 1.0) * (c_l + h * self.left.u + xi);
                        let c = 2.0 / (g + 1.0) * (c_l + h * (self.left.u - xi));
                        fan_state(g, self.left, c, u)
                    }
                }
            }
        } else {
            match self.right_wave {
                Wave::Shock { speed } => {
                    if xi > speed {
                        self.right
                    } else {
                        star_right
                    }
                }
                Wave::Rarefaction { head, tail } => {
                    if xi >= head {
                        self.right
                    } else if xi <= tail {
                        star_right
                    } else {
                        let h = (g - 1.0) / 2.0;
                        let c_r = sound_speed(g, self.right.rho, self.right.p);
                        let u = 2.0 / (g + 1.0) * (-c_r + h * self.right.u + xi);
                        let c = 2.0 / (g + 1.0) * (c_r - h * (self.right.u - xi));
                        fan_state(g, self.right, c, u)
                    }
                }
            }
        }
    }

    /// The state at `(x, t)` for a diaphragm at `x_0`: [`sample`] at
    /// `xi = (x - x_0)/t`.
    ///
    /// Refuses (`Error::Refused`): `t <= 0` or an input not finite.
    pub fn sample_at(&self, x: f64, t: f64, x0: f64) -> Result<Primitive, Error> {
        const WHO: &str = "riemann::sample_at";
        check_finite(WHO, &[("x", x), ("t", t), ("x0", x0)])?;
        if t <= 0.0 {
            return Err(refuse(WHO, format!("t must be above 0, got {t}")));
        }
        Ok(self.sample((x - x0) / t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The relative difference `|got/want - 1|`.
    fn rel(got: f64, want: f64) -> f64 {
        (got / want - 1.0).abs()
    }

    fn close_rel(what: &str, got: f64, want: f64, tol: f64) {
        assert!(
            rel(got, want) <= tol,
            "{what}: got {got:e}, want {want:e}, rel {:e} > {tol:e}",
            rel(got, want)
        );
    }

    fn close_abs(what: &str, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got:e}, want {want:e}, abs {:e} > {tol:e}",
            (got - want).abs()
        );
    }

    /// The result is `Err(Error::Refused(msg))`, `msg` starts with
    /// `compressible::exact::` and contains every needle.
    fn refused<T: std::fmt::Debug>(res: Result<T, Error>, needles: &[&str]) {
        match res {
            Err(Error::Refused(msg)) => {
                assert!(
                    msg.starts_with("compressible::exact::"),
                    "refusal does not start with compressible::exact:: : {msg}"
                );
                for n in needles {
                    assert!(msg.contains(n), "refusal lacks {n:?}: {msg}");
                }
            }
            other => panic!("expected a refusal containing {needles:?}, got {other:?}"),
        }
    }

    /// The sod pair (1, 0, 1 | 0.125, 0, 0.1) at gamma 1.4.
    fn sod() -> RiemannSolution {
        solve(
            1.4,
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 1.0,
            },
            Primitive {
                rho: 0.125,
                u: 0.0,
                p: 0.1,
            },
        )
        .unwrap()
    }

    /// Checks a sampled state `(rho, u, p)` to rel 1e-12, abs 1e-15 where
    /// the wanted velocity is 0.
    fn sample_check(what: &str, got: Primitive, rho: f64, u: f64, p: f64) {
        close_rel(&format!("{what} rho"), got.rho, rho, 1e-12);
        if u == 0.0 {
            close_abs(&format!("{what} u"), got.u, 0.0, 1e-15);
        } else {
            close_rel(&format!("{what} u"), got.u, u, 1e-12);
        }
        close_rel(&format!("{what} p"), got.p, p, 1e-12);
    }

    #[test]
    fn sod_star_state_matches_the_printed_digits() {
        let s = sod();
        close_abs("p* 5 digits", s.p_star, 0.30313, 5e-6);
        close_abs("u* 5 digits", s.u_star, 0.92745, 5e-6);
        close_rel("p*", s.p_star, 0.30313017805064682, 1e-12);
        close_rel("u*", s.u_star, 0.92745262004894995, 1e-12);
        close_rel("rho*L", s.rho_star_left, 0.42631942817849519, 1e-12);
        close_rel("rho*R", s.rho_star_right, 0.26557371170530706, 1e-12);
        match s.left_wave {
            Wave::Rarefaction { head, tail } => {
                close_rel("left head", head, -1.1832159566199232, 1e-12);
                close_rel("left tail", tail, -0.07027281256118327, 1e-12);
            }
            other => panic!("left wave should be a rarefaction, got {other:?}"),
        }
        match s.right_wave {
            Wave::Shock { speed } => close_rel("right shock", speed, 1.7521557320301782, 1e-12),
            other => panic!("right wave should be a shock, got {other:?}"),
        }
    }

    #[test]
    fn sod_samples_at_t_0_2() {
        let s = sod();
        let at = |x: f64| s.sample_at(x, 0.2, 0.5).unwrap();
        sample_check("x 0.1", at(0.1), 1.0, 0.0, 1.0);
        sample_check(
            "x 0.3",
            at(0.3),
            0.87745253275527745,
            0.15267996384993601,
            0.8327470150499224,
        );
        sample_check("x 0.6", at(0.6), 0.42631942817849519, s.u_star, s.p_star);
        sample_check("x 0.75", at(0.75), 0.26557371170530706, s.u_star, s.p_star);
        sample_check("x 0.9", at(0.9), 0.125, 0.0, 0.1);
        let speed = match s.right_wave {
            Wave::Shock { speed } => speed,
            other => panic!("right wave should be a shock, got {other:?}"),
        };
        close_abs("shock at t 0.2", 0.5 + 0.2 * speed, 0.850431, 1e-6);
        close_abs("contact at t 0.2", 0.5 + 0.2 * s.u_star, 0.685491, 1e-6);
    }

    #[test]
    fn the_123_problem() {
        let s = solve(
            1.4,
            Primitive {
                rho: 1.0,
                u: -2.0,
                p: 0.4,
            },
            Primitive {
                rho: 1.0,
                u: 2.0,
                p: 0.4,
            },
        )
        .unwrap();
        close_rel("p*", s.p_star, 0.001893873420054763, 1e-12);
        close_abs("u*", s.u_star, 0.0, 1e-15);
        close_rel("rho*L", s.rho_star_left, 0.021852118206812831, 1e-12);
        close_rel("rho*R", s.rho_star_right, 0.021852118206812831, 1e-12);
        match s.left_wave {
            Wave::Rarefaction { head, tail } => {
                assert!(head < tail, "left fan runs the wrong way: {head} {tail}");
            }
            other => panic!("left wave should be a rarefaction, got {other:?}"),
        }
        match s.right_wave {
            Wave::Rarefaction { head, tail } => {
                assert!(tail < head, "right fan runs the wrong way: {tail} {head}");
            }
            other => panic!("right wave should be a rarefaction, got {other:?}"),
        }
        let at = |x: f64| s.sample_at(x, 0.15, 0.5).unwrap();
        sample_check(
            "x 0.2",
            at(0.2),
            0.40187757201646091,
            -1.3763904355376764,
            0.11163265889346136,
        );
        sample_check(
            "x 0.4",
            at(0.4),
            0.04439645533153016,
            -0.26527932442656532,
            0.0051091376899693629,
        );
        sample_check(
            "x 0.6",
            at(0.6),
            0.04439645533153016,
            0.26527932442656532,
            0.0051091376899693629,
        );
    }

    #[test]
    fn the_left_blast_wave() {
        let s = solve(
            1.4,
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 1000.0,
            },
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 0.01,
            },
        )
        .unwrap();
        close_rel("p*", s.p_star, 460.89378749138354, 1e-12);
        close_abs("p* printed", s.p_star, 460.894, 5e-4);
        close_rel("u*", s.u_star, 19.597451388723052, 1e-12);
        close_abs("u* printed", s.u_star, 19.5975, 5e-5);
        close_rel("rho*L", s.rho_star_left, 0.57506229847655549, 1e-12);
        close_rel("rho*R", s.rho_star_right, 5.999240704796234, 1e-12);
        let speed = match s.right_wave {
            Wave::Shock { speed } => speed,
            other => panic!("right wave should be a shock, got {other:?}"),
        };
        close_rel("right shock", speed, 23.517536966903236, 1e-12);
        let at = |x: f64| s.sample_at(x, 0.012, 0.5).unwrap();
        sample_check(
            "x 0.1",
            at(0.1),
            0.91230748781946552,
            3.4027004453384004,
            879.4228286278499,
        );
        sample_check(
            "x 0.3",
            at(0.3),
            0.61575337496781448,
            17.291589334227289,
            507.18864420297776,
        );
        sample_check("x 0.75", at(0.75), 5.999240704796234, s.u_star, s.p_star);
        sample_check("x 0.9", at(0.9), 1.0, 0.0, 0.01);
    }

    #[test]
    fn the_right_blast_wave() {
        let s = solve(
            1.4,
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 0.01,
            },
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 100.0,
            },
        )
        .unwrap();
        close_rel("p*", s.p_star, 46.095044248867966, 1e-12);
        close_rel("u*", s.u_star, -6.1963282497870359, 1e-12);
        close_rel("rho*L", s.rho_star_left, 5.9924168635152249, 1e-12);
        close_rel("rho*R", s.rho_star_right, 0.5751127897824123, 1e-12);
        let speed = match s.left_wave {
            Wave::Shock { speed } => speed,
            other => panic!("left wave should be a shock, got {other:?}"),
        };
        close_rel("left shock", speed, -7.4374762586943133, 1e-12);
        let at = |x: f64| s.sample_at(x, 0.035, 0.5).unwrap();
        sample_check("x 0.2", at(0.2), 1.0, 0.0, 0.01);
        sample_check("x 0.26", at(0.26), 5.9924168635152249, s.u_star, s.p_star);
        sample_check("x 0.4", at(0.4), 0.5751127897824123, s.u_star, s.p_star);
        sample_check(
            "x 0.9",
            at(0.9),
            0.97189682997458182,
            -0.33632344802316959,
            96.08779597816483,
        );
    }

    #[test]
    fn vacuum_and_bad_inputs_are_refused() {
        refused(
            solve(
                1.4,
                Primitive {
                    rho: 1.0,
                    u: -4.0,
                    p: 0.4,
                },
                Primitive {
                    rho: 1.0,
                    u: 4.0,
                    p: 0.4,
                },
            ),
            &["vacuum"],
        );
        refused(
            solve(
                1.4,
                Primitive {
                    rho: 0.0,
                    u: 0.0,
                    p: 1.0,
                },
                Primitive {
                    rho: 1.0,
                    u: 0.0,
                    p: 0.1,
                },
            ),
            &["rho"],
        );
        refused(
            solve(
                1.4,
                Primitive {
                    rho: 1.0,
                    u: 0.0,
                    p: -1.0,
                },
                Primitive {
                    rho: 1.0,
                    u: 0.0,
                    p: 0.1,
                },
            ),
            &["p"],
        );
        refused(
            solve(
                1.4,
                Primitive {
                    rho: f64::NAN,
                    u: 0.0,
                    p: 1.0,
                },
                Primitive {
                    rho: 1.0,
                    u: 0.0,
                    p: 0.1,
                },
            ),
            &["not finite"],
        );
        refused(
            solve(
                1.0,
                Primitive {
                    rho: 1.0,
                    u: 0.0,
                    p: 1.0,
                },
                Primitive {
                    rho: 1.0,
                    u: 0.0,
                    p: 0.1,
                },
            ),
            &["gamma"],
        );
        let s = sod();
        refused(s.sample_at(0.6, 0.0, 0.5), &["t"]);
        refused(s.sample_at(0.6, -1.0, 0.5), &["t"]);
    }
}
