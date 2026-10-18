// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The injector's size distribution (SPEC-LIT 147.1, VAL-71): the
//! Rosin-Rammler volume-fraction law an injector may draw its droplet
//! diameters from, the counter-based stratified draw that samples it, and
//! the per-event weight that keeps the discharged mass exact.
//!
//! The law is (147.1): `Q(d) = 1 - exp(-(d/X)^q)` is the fraction of the
//! total liquid VOLUME carried by drops of diameter less than `d`, with
//! (147.2) the mass median `MMD = X (ln 2)^(1/q)` and (147.3) the inverse
//! `d(u) = X (-ln(1-u))^(1/q)`. The draw is (147.4): parcel `i` of `n` in an
//! event lands in the `i`-th of `n` equal strata of `u`, at an offset
//! `r = (floor(h/2^11) + 1/2) 2^-53` keyed on the parcel's (66.9) identity
//! through SplitMix64's finaliser - a counter-based draw in the sense of
//! Salmon, Moraes, Dror & Shaw (SC '11), so every quantity is a pure
//! function of `(injector, event, index, seed)` and the same seed gives the
//! same bits. The weight is (147.5): parcel `i` carries
//! `n_p,i = c / d_i^3` droplets with `c = mdot dt stride / (n rho_l pi/6)`,
//! so the event's mass is `mdot dt stride` to round-off, exactly as (66.13)
//! makes it for one diameter.
//!
//! Everything here is the f64 host twin of the device's draw:
//! `cuda/parcels.cu` evaluates (147.3)-(147.5) in the SAME operation order
//! in `double` in BOTH builds and rounds `d` and `n_p` once each to
//! [`Scalar`]; `tests/parcel_sizes.rs` measures the two against each other.
//!
//! Provenance: LITERATURE - Rosin & Rammler 1933 (SPEC-LIT 147.1). No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::Scalar;

/// SPEC-LIT 147.1: how an injector's droplet diameters are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SizeDistribution {
    /// Every droplet has `Injector::diameter` - SPEC-LIT 66.8, unchanged.
    #[default]
    Monodisperse,
    /// (147.1)-(147.5). `Injector::diameter` is not read.
    RosinRammler(RosinRammler),
}

/// SPEC-LIT (147.1)'s two parameters plus the draw's key. `x` is the
/// characteristic diameter `X` in m (Q(X) = 1 - 1/e) and `q` the spread;
/// both are widened to f64 in every formula. `seed` keys (147.4)'s offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RosinRammler {
    /// Characteristic diameter X, m.
    pub x: Scalar,
    /// Spread.
    pub q: Scalar,
    /// The draw's key.
    pub seed: u64,
}

/// (147.4)'s multiplier of the seed: the odd constant SplitMix64 seeds with,
/// already the golden-ratio wrap of (66.9)'s family.
pub const SEED_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's finalising mix (Steele, Lea & Flood 2014; Vigna's reference
/// implementation, public domain) - (66.9)'s three xor-shifts and two odd
/// multiplications applied to `z` alone. A bijection on `u64`, so a distinct
/// key gives a distinct hash exactly.
#[must_use]
pub fn mix64(z: u64) -> u64 {
    let mut z = z;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z
}

/// (147.4)'s offset `r` inside parcel `uid`'s stratum:
/// `h = mix64(uid XOR seed*SEED_MIX)`, `r = (floor(h/2^11) + 1/2) * 2^-53`.
/// Strictly inside (0, 1) for every key: the top 53 bits of a bijection's
/// output plus one half of the last of them. `ofpMix64` in `cuda/parcels.cu`
/// is this function bit for bit.
#[must_use]
pub fn stratum_offset(uid: u64, seed: u64) -> f64 {
    let h = mix64(uid ^ seed.wrapping_mul(SEED_MIX));
    // 2^-53 exactly.
    ((h >> 11) as f64 + 0.5) * 1.1102230246251565404e-16
}

/// (147.4): the uniform parcel `index` of `n` draws - the `index`-th of `n`
/// equal strata of `u`, at `r` inside it. The sample's distribution function
/// is then within 1/n of (147.1) for ANY `r`, which is why one uniform per
/// parcel is enough.
#[must_use]
pub fn stratified_u(index: u32, n: u32, uid: u64, seed: u64) -> f64 {
    (index as f64 + stratum_offset(uid, seed)) / n as f64
}

/// (147.5)'s per-event coefficient
/// `c = mdot dt stride / (n rho_l pi/6)`, in f64 from the widened
/// `Scalar`s. The device divides it by the STORED `d_i^3` and rounds once;
/// the host needs it again to refuse an `n_p` that would not fit.
#[must_use]
pub fn event_mass_coefficient(
    mass_flow: Scalar,
    dt: Scalar,
    stride: i32,
    n: u32,
    rho_liquid: Scalar,
) -> f64 {
    mass_flow as f64 * dt as f64 * stride as f64
        / (n as f64 * rho_liquid as f64 * std::f64::consts::FRAC_PI_6)
}

impl RosinRammler {
    /// (147.1): `1 - exp(-(d/X)^q)` - the volume fraction in drops thinner
    /// than `d` - evaluated as `-expm1` so `Q` stays exact as `d` grows past
    /// `X`. `0` for `d <= 0`.
    #[must_use]
    pub fn cdf(&self, d: f64) -> f64 {
        if d <= 0.0 {
            return 0.0;
        }
        -((-(d / self.x as f64).powf(self.q as f64)).exp_m1())
    }

    /// (147.3): the inverse of (147.1), `X (-ln(1-u))^(1/q)`.
    #[must_use]
    pub fn quantile(&self, u: f64) -> f64 {
        self.x as f64 * (-(-u).ln_1p()).powf(1.0 / self.q as f64)
    }

    /// (147.2): the mass median diameter, `Q(MMD) = 1/2`.
    #[must_use]
    pub fn mass_median(&self) -> f64 {
        self.x as f64 * std::f64::consts::LN_2.powf(1.0 / self.q as f64)
    }

    /// The two draws (147.4) can actually produce, `n` strata: the smallest
    /// from the `u` just above 0, the largest from the `u` just below 1.
    /// `validate` refuses an injector either extreme cannot survive in
    /// [`Scalar`].
    #[must_use]
    pub fn draw_range(&self, n: u32) -> (f64, f64) {
        let x = self.x as f64;
        let q = self.q as f64;
        let smallest = x * (-(-(f64::powi(2.0, -54) / n as f64)).ln_1p()).powf(1.0 / q);
        let largest = x * ((n as f64).ln() + 54.0 * std::f64::consts::LN_2).powf(1.0 / q);
        (smallest, largest)
    }

    /// SPEC-LIT 147.1's refusals, by name. Called from [`Parcels::new`] with
    /// `j` the injector's index, `n` its parcels per event and `c` the
    /// (147.5) coefficient [`event_mass_coefficient`] computed for it - the
    /// last because a smallest draw so fine that `c / d^3` will not fit in
    /// [`Scalar`] is an `n_p` overflow, and that belongs where a human can
    /// be told (S13.4), not inside a kernel.
    pub fn validate(&self, j: usize, n: u32, c: f64) -> Result<()> {
        let x = self.x as f64;
        let q = self.q as f64;
        if !x.is_finite() || x <= 0.0 {
            return Err(Error::Config(format!(
                "parcels: injector {j} Rosin-Rammler characteristic diameter X is {}; \
                 it must be finite and positive",
                self.x
            )));
        }
        if !q.is_finite() || q <= 0.0 {
            return Err(Error::Config(format!(
                "parcels: injector {j} Rosin-Rammler spread q is {}; \
                 it must be finite and positive",
                self.q
            )));
        }
        let (smallest, largest) = self.draw_range(n);
        if !largest.is_finite() || largest > Scalar::MAX as f64 {
            return Err(Error::Config(format!(
                "parcels: injector {j} Rosin-Rammler largest draw is {largest} m with \
                 X {} q {} over {n} parcels; it overflows Scalar (max {})",
                self.x,
                self.q,
                Scalar::MAX
            )));
        }
        if smallest as Scalar == 0.0 {
            return Err(Error::Config(format!(
                "parcels: injector {j} Rosin-Rammler smallest draw {smallest} m with \
                 X {} q {} over {n} parcels rounds to 0 in Scalar",
                self.x, self.q
            )));
        }
        let n_p_max = c / (smallest * smallest * smallest);
        if !n_p_max.is_finite() || n_p_max > Scalar::MAX as f64 {
            return Err(Error::Config(format!(
                "parcels: injector {j} Rosin-Rammler smallest draw {smallest} m needs \
                 n_p {n_p_max}, which overflows Scalar (max {}) at this massFlow; \
                 lower parcelsPerEvent or raise X",
                Scalar::MAX
            )));
        }
        Ok(())
    }
}
