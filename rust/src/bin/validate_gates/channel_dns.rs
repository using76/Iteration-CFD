// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 110-A's recorded runs: the plane channel at three `Re_tau` against
//! Moser, Kim & Mansour (1999) (SPEC-LIT §110.2, §110.5; docs/17 VAL-51).**
//!
//! `cases/channelDns{180,395,590}.jsonc` ran on `ofgpu-lowmach` exactly as
//! §110.5's recipe says: 40 000 iterations, `-check 5000`, on scratch copies
//! with ONLY `cells` edited to `N_y = 32 / 64 / 128`, and `ofgpu-sample column
//! <copy> 40000 y 0.045 0.02 -mean` took each column. The verdict is
//! `validate.rs`'s `check_channel_dns_on`, unchanged in its bands; this file
//! holds the data it replays, so a run changes DATA, not code (§110.1).
//!
//! What each record carries, and where it comes from:
//! - `u_b_plus`: the sampler's volume-weighted column mean of `U_x` over the
//!   case's own `u_tau = sqrt(g_x h)` (§110.2's force balance), per level;
//! - `tau_w_measured_kin`: the driver's printed KINEMATIC wall sink
//!   `sum nu_eff |dU_par| deltaCoeffs |Sf|` (§32.5.2) over its printed total
//!   wall area, on the finest level - §110.2's `u_input`;
//! - `residual_u`: the finest run's last printed `|U| res`;
//! - `profile_fine`: `(y, U_x)` of the finest column's cells below the
//!   centreline `y = h`, wall to centre, as the sampler printed them.
//!
//! The single build's three finest runs are recorded beside them for the
//! class-R row (docs/17 section 5.1 rule 4: each lands in the f64 build's bands). A run
//! that did not finish is a `None` in `SINGLE`, and `SINGLE_NOT_RECORDED`
//! says why, in the driver's own `run ended:` words (§110.1: a run that
//! diverged is recorded as exactly that).
//!
//! The ignored test `record_channel_dns` reads a runs directory (env
//! `VAL51_RUNS`: `logs/<stem>.log`, `logs/<stem>.col`, `cases/<stem>.jsonc`)
//! and prints the literals below.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §110.2 (the gate, its bands and functionals), §110.5
//!     (the recipe and the record), §32.5.2 (the measured wall shear), §94
//!     (the mesh study) and §110.1 (the replay shape)
//!   Moser, Kim & Mansour, Phys. Fluids 11 (1999) 943, DOI 10.1063/1.869966 -
//!     only through the fetched key files, read at run time; no number of
//!     theirs is in this file
//! No GPL-licensed source was consulted.

use ofgpu::Scalar;

use crate::ChannelDnsRecord;

/// The commit the f64 and single binaries of every record were built at.
pub(crate) const BUILT_AT: &str = "67af46d";

/// One single-build finest run (class R, docs/17 section 5.1 rule 4): the same case and
/// recipe as the f64 finest level, built with `--features single`.
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct ChannelDnsSingle {
    pub(crate) n_y: usize,
    pub(crate) u_b_plus: Scalar,
    pub(crate) tau_w_measured_kin: Scalar,
    pub(crate) iterations: usize,
    pub(crate) residual_u: Scalar,
    pub(crate) profile_fine: &'static [(Scalar, Scalar)],
}

// RECORDS-BEGIN (printed by `record_channel_dns`)
pub(crate) static RECORDS: [ChannelDnsRecord; 3] = [
    ChannelDnsRecord {
        re_tau: 178.12,
        body_force: 0.89231,
        n_y: [32, 64, 128],
        u_b_plus: [1.529190091e1, 1.627768072e1, 1.671379228e1],
        tau_w_measured_kin: 1.784625000e-2,
        iterations: 40000,
        residual_u: 2.99689e-2,
        profile_fine: &[
            (4.05167e-6, 4.82046e-3),
            (1.25105e-5, 1.48802e-2),
            (2.17115e-5, 2.58178e-2),
            (3.17198e-5, 3.77093e-2),
            (4.26061e-5, 5.06374e-2),
            (5.44476e-5, 6.46918e-2),
            (6.73281e-5, 7.99697e-2),
            (8.13387e-5, 9.65768e-2),
            (9.65785e-5, 1.14627e-1),
            (1.13155e-4, 1.34246e-1),
            (1.31187e-4, 1.55566e-1),
            (1.508e-4, 1.78735e-1),
            (1.72134e-4, 2.03908e-1),
            (1.95341e-4, 2.31258e-1),
            (2.20583e-4, 2.60966e-1),
            (2.48039e-4, 2.93231e-1),
            (2.77905e-4, 3.28267e-1),
            (3.10391e-4, 3.663e-1),
            (3.45728e-4, 4.07573e-1),
            (3.84165e-4, 4.5234e-1),
            (4.25974e-4, 5.00866e-1),
            (4.71451e-4, 5.53415e-1),
            (5.20919e-4, 6.10237e-1),
            (5.74727e-4, 6.71538e-1),
            (6.33255e-4, 7.37417e-1),
            (6.96919e-4, 8.07767e-1),
            (7.66169e-4, 8.82125e-1),
            (8.41494e-4, 9.59531e-1),
            (9.23428e-4, 1.03853e0),
            (1.01255e-3, 1.11745e0),
            (1.10949e-3, 1.19478e0),
            (1.21494e-3, 1.26942e0),
            (1.32964e-3, 1.34076e0),
            (1.45441e-3, 1.40854e0),
            (1.59012e-3, 1.47278e0),
            (1.73773e-3, 1.53362e0),
            (1.8983e-3, 1.59131e0),
            (2.07296e-3, 1.64615e0),
            (2.26294e-3, 1.69843e0),
            (2.46959e-3, 1.74845e0),
            (2.69437e-3, 1.79651e0),
            (2.93887e-3, 1.84286e0),
            (3.20482e-3, 1.88774e0),
            (3.49411e-3, 1.93135e0),
            (3.80878e-3, 1.9739e0),
            (4.15106e-3, 2.01554e0),
            (4.52336e-3, 2.05641e0),
            (4.92834e-3, 2.09664e0),
            (5.36884e-3, 2.13633e0),
            (5.848e-3, 2.17556e0),
            (6.36919e-3, 2.21438e0),
            (6.93612e-3, 2.25285e0),
            (7.55278e-3, 2.29098e0),
            (8.22355e-3, 2.32875e0),
            (8.95317e-3, 2.36612e0),
            (9.74681e-3, 2.40299e0),
            (1.06101e-2, 2.43918e0),
            (1.15491e-2, 2.47441e0),
            (1.25705e-2, 2.50828e0),
            (1.36815e-2, 2.54012e0),
            (1.489e-2, 2.56899e0),
            (1.62045e-2, 2.59354e0),
            (1.76344e-2, 2.61204e0),
            (1.91897e-2, 2.62246e0),
        ],
    },
    ChannelDnsRecord {
        re_tau: 392.24,
        body_force: 4.32709,
        n_y: [32, 64, 128],
        u_b_plus: [1.699123697e1, 1.806184045e1, 1.857747735e1],
        tau_w_measured_kin: 8.654171875e-2,
        iterations: 40000,
        residual_u: 5.72422e-2,
        profile_fine: &[
            (4.05167e-6, 2.33759e-2),
            (1.25105e-5, 7.21589e-2),
            (2.17115e-5, 1.25198e-1),
            (3.17198e-5, 1.82864e-1),
            (4.26061e-5, 2.45554e-1),
            (5.44476e-5, 3.13704e-1),
            (6.73281e-5, 3.87783e-1),
            (8.13387e-5, 4.68299e-1),
            (9.65785e-5, 5.55798e-1),
            (1.13155e-4, 6.5087e-1),
            (1.31187e-4, 7.54141e-1),
            (1.508e-4, 8.66275e-1),
            (1.72134e-4, 9.87958e-1),
            (1.95341e-4, 1.11987e0),
            (2.20583e-4, 1.26265e0),
            (2.48039e-4, 1.41673e0),
            (2.77905e-4, 1.58217e0),
            (3.10391e-4, 1.75817e0),
            (3.45728e-4, 1.94254e0),
            (3.84165e-4, 2.13145e0),
            (4.25974e-4, 2.31998e0),
            (4.71451e-4, 2.50359e0),
            (5.20919e-4, 2.67909e0),
            (5.74727e-4, 2.84483e0),
            (6.33255e-4, 3.00033e0),
            (6.96919e-4, 3.14578e0),
            (7.66169e-4, 3.28182e0),
            (8.41494e-4, 3.40925e0),
            (9.23428e-4, 3.52898e0),
            (1.01255e-3, 3.64193e0),
            (1.10949e-3, 3.74894e0),
            (1.21494e-3, 3.85082e0),
            (1.32964e-3, 3.94829e0),
            (1.45441e-3, 4.04199e0),
            (1.59012e-3, 4.13249e0),
            (1.73773e-3, 4.22027e0),
            (1.8983e-3, 4.30577e0),
            (2.07296e-3, 4.38935e0),
            (2.26294e-3, 4.47134e0),
            (2.46959e-3, 4.55201e0),
            (2.69437e-3, 4.63161e0),
            (2.93887e-3, 4.71034e0),
            (3.20482e-3, 4.78838e0),
            (3.49411e-3, 4.86589e0),
            (3.80878e-3, 4.94301e0),
            (4.15106e-3, 5.01983e0),
            (4.52336e-3, 5.09646e0),
            (4.92834e-3, 5.17296e0),
            (5.36884e-3, 5.24938e0),
            (5.848e-3, 5.32575e0),
            (6.36919e-3, 5.40205e0),
            (6.93612e-3, 5.47824e0),
            (7.55278e-3, 5.55425e0),
            (8.22355e-3, 5.62992e0),
            (8.95317e-3, 5.70503e0),
            (9.74681e-3, 5.77922e0),
            (1.06101e-2, 5.85202e0),
            (1.15491e-2, 5.92268e0),
            (1.25705e-2, 5.99018e0),
            (1.36815e-2, 6.05306e0),
            (1.489e-2, 6.10933e0),
            (1.62045e-2, 6.15646e0),
            (1.76344e-2, 6.19142e0),
            (1.91897e-2, 6.2109e0),
        ],
    },
    ChannelDnsRecord {
        re_tau: 587.19,
        body_force: 9.69728,
        n_y: [32, 64, 128],
        u_b_plus: [1.778370220e1, 1.891703200e1, 1.949438054e1],
        tau_w_measured_kin: 1.939453125e-1,
        iterations: 40000,
        residual_u: 5.48234e-2,
        profile_fine: &[
            (4.05167e-6, 5.23868e-2),
            (1.25105e-5, 1.61712e-1),
            (2.17115e-5, 2.80576e-1),
            (3.17198e-5, 4.09804e-1),
            (4.26061e-5, 5.50287e-1),
            (5.44476e-5, 7.02989e-1),
            (6.73281e-5, 8.68944e-1),
            (8.13387e-5, 1.04925e0),
            (9.65785e-5, 1.24507e0),
            (1.13155e-4, 1.45758e0),
            (1.31187e-4, 1.6879e0),
            (1.508e-4, 1.93695e0),
            (1.72134e-4, 2.20505e0),
            (1.95341e-4, 2.4912e0),
            (2.20583e-4, 2.79187e0),
            (2.48039e-4, 3.10009e0),
            (2.77905e-4, 3.40652e0),
            (3.10391e-4, 3.70243e0),
            (3.45728e-4, 3.98219e0),
            (3.84165e-4, 4.24332e0),
            (4.25974e-4, 4.48551e0),
            (4.71451e-4, 4.7097e0),
            (5.20919e-4, 4.9174e0),
            (5.74727e-4, 5.11035e0),
            (6.33255e-4, 5.29034e0),
            (6.96919e-4, 5.45909e0),
            (7.66169e-4, 5.61816e0),
            (8.41494e-4, 5.76895e0),
            (9.23428e-4, 5.91272e0),
            (1.01255e-3, 6.05051e0),
            (1.10949e-3, 6.18327e0),
            (1.21494e-3, 6.31178e0),
            (1.32964e-3, 6.43673e0),
            (1.45441e-3, 6.55869e0),
            (1.59012e-3, 6.67816e0),
            (1.73773e-3, 6.79557e0),
            (1.8983e-3, 6.91129e0),
            (2.07296e-3, 7.02563e0),
            (2.26294e-3, 7.13889e0),
            (2.46959e-3, 7.25129e0),
            (2.69437e-3, 7.36305e0),
            (2.93887e-3, 7.47435e0),
            (3.20482e-3, 7.58534e0),
            (3.49411e-3, 7.69616e0),
            (3.80878e-3, 7.80693e0),
            (4.15106e-3, 7.91773e0),
            (4.52336e-3, 8.02864e0),
            (4.92834e-3, 8.13971e0),
            (5.36884e-3, 8.25095e0),
            (5.848e-3, 8.36236e0),
            (6.36919e-3, 8.47389e0),
            (6.93612e-3, 8.58543e0),
            (7.55278e-3, 8.69683e0),
            (8.22355e-3, 8.80783e0),
            (8.95317e-3, 8.91804e0),
            (9.74681e-3, 9.02693e0),
            (1.06101e-2, 9.13372e0),
            (1.15491e-2, 9.23728e0),
            (1.25705e-2, 9.33606e0),
            (1.36815e-2, 9.42787e0),
            (1.489e-2, 9.5098e0),
            (1.62045e-2, 9.57821e0),
            (1.76344e-2, 9.62882e0),
            (1.91897e-2, 9.65696e0),
        ],
    },
];
pub(crate) static SINGLE: [Option<ChannelDnsSingle>; 3] = [
    None,
    Some(ChannelDnsSingle {
        n_y: 128,
        u_b_plus: 1.857747735e1,
        tau_w_measured_kin: 8.656109375e-2,
        iterations: 40000,
        residual_u: 4.24727e-2,
        profile_fine: &[
            (4.05168e-6, 2.33758e-2),
            (1.25105e-5, 7.21587e-2),
            (2.17115e-5, 1.25198e-1),
            (3.17198e-5, 1.82863e-1),
            (4.26062e-5, 2.45554e-1),
            (5.44477e-5, 3.13704e-1),
            (6.73282e-5, 3.87783e-1),
            (8.13388e-5, 4.68298e-1),
            (9.65787e-5, 5.55797e-1),
            (1.13156e-4, 6.50868e-1),
            (1.31187e-4, 7.54139e-1),
            (1.508e-4, 8.66273e-1),
            (1.72135e-4, 9.87955e-1),
            (1.95341e-4, 1.11987e0),
            (2.20583e-4, 1.26265e0),
            (2.4804e-4, 1.41673e0),
            (2.77906e-4, 1.58217e0),
            (3.10392e-4, 1.75816e0),
            (3.45728e-4, 1.94254e0),
            (3.84165e-4, 2.13144e0),
            (4.25975e-4, 2.31998e0),
            (4.71452e-4, 2.50358e0),
            (5.2092e-4, 2.67908e0),
            (5.74727e-4, 2.84483e0),
            (6.33256e-4, 3.00032e0),
            (6.9692e-4, 3.14578e0),
            (7.6617e-4, 3.28181e0),
            (8.41495e-4, 3.40925e0),
            (9.2343e-4, 3.52898e0),
            (1.01255e-3, 3.64192e0),
            (1.1095e-3, 3.74894e0),
            (1.21494e-3, 3.85082e0),
            (1.32964e-3, 3.94829e0),
            (1.45441e-3, 4.04199e0),
            (1.59012e-3, 4.13249e0),
            (1.73773e-3, 4.22027e0),
            (1.8983e-3, 4.30577e0),
            (2.07296e-3, 4.38935e0),
            (2.26294e-3, 4.47134e0),
            (2.46959e-3, 4.55201e0),
            (2.69437e-3, 4.63161e0),
            (2.93887e-3, 4.71034e0),
            (3.20482e-3, 4.78838e0),
            (3.49411e-3, 4.8659e0),
            (3.80878e-3, 4.94301e0),
            (4.15106e-3, 5.01983e0),
            (4.52337e-3, 5.09646e0),
            (4.92834e-3, 5.17296e0),
            (5.36885e-3, 5.24938e0),
            (5.848e-3, 5.32574e0),
            (6.3692e-3, 5.40204e0),
            (6.93612e-3, 5.47824e0),
            (7.55278e-3, 5.55425e0),
            (8.22355e-3, 5.62991e0),
            (8.95317e-3, 5.70502e0),
            (9.74681e-3, 5.77921e0),
            (1.06101e-2, 5.85201e0),
            (1.15491e-2, 5.92267e0),
            (1.25705e-2, 5.99017e0),
            (1.36815e-2, 6.05305e0),
            (1.489e-2, 6.10932e0),
            (1.62045e-2, 6.15644e0),
            (1.76344e-2, 6.19141e0),
            (1.91897e-2, 6.21088e0),
        ],
    }),
    Some(ChannelDnsSingle {
        n_y: 128,
        u_b_plus: 1.949424430e1,
        tau_w_measured_kin: 1.939859375e-1,
        iterations: 40000,
        residual_u: 6.19563e-2,
        profile_fine: &[
            (4.05168e-6, 5.23858e-2),
            (1.25105e-5, 1.61709e-1),
            (2.17115e-5, 2.80571e-1),
            (3.17198e-5, 4.09796e-1),
            (4.26062e-5, 5.50277e-1),
            (5.44477e-5, 7.02976e-1),
            (6.73282e-5, 8.68927e-1),
            (8.13388e-5, 1.04923e0),
            (9.65787e-5, 1.24505e0),
            (1.13156e-4, 1.45755e0),
            (1.31187e-4, 1.68787e0),
            (1.508e-4, 1.93692e0),
            (1.72135e-4, 2.20502e0),
            (1.95341e-4, 2.49116e0),
            (2.20583e-4, 2.79182e0),
            (2.4804e-4, 3.10005e0),
            (2.77906e-4, 3.40647e0),
            (3.10392e-4, 3.70238e0),
            (3.45728e-4, 3.98214e0),
            (3.84165e-4, 4.24326e0),
            (4.25975e-4, 4.48546e0),
            (4.71452e-4, 4.70965e0),
            (5.2092e-4, 4.91735e0),
            (5.74727e-4, 5.1103e0),
            (6.33256e-4, 5.29029e0),
            (6.9692e-4, 5.45903e0),
            (7.6617e-4, 5.6181e0),
            (8.41495e-4, 5.7689e0),
            (9.2343e-4, 5.91266e0),
            (1.01255e-3, 6.05045e0),
            (1.1095e-3, 6.18321e0),
            (1.21494e-3, 6.31172e0),
            (1.32964e-3, 6.43667e0),
            (1.45441e-3, 6.55862e0),
            (1.59012e-3, 6.67809e0),
            (1.73773e-3, 6.7955e0),
            (1.8983e-3, 6.91122e0),
            (2.07296e-3, 7.02557e0),
            (2.26294e-3, 7.13882e0),
            (2.46959e-3, 7.25122e0),
            (2.69437e-3, 7.36298e0),
            (2.93887e-3, 7.47427e0),
            (3.20482e-3, 7.58526e0),
            (3.49411e-3, 7.69609e0),
            (3.80878e-3, 7.80685e0),
            (4.15106e-3, 7.91765e0),
            (4.52337e-3, 8.02856e0),
            (4.92834e-3, 8.13963e0),
            (5.36885e-3, 8.25087e0),
            (5.848e-3, 8.36228e0),
            (6.3692e-3, 8.47381e0),
            (6.93612e-3, 8.58535e0),
            (7.55278e-3, 8.69675e0),
            (8.22355e-3, 8.80775e0),
            (8.95317e-3, 8.91796e0),
            (9.74681e-3, 9.02685e0),
            (1.06101e-2, 9.13363e0),
            (1.15491e-2, 9.23719e0),
            (1.25705e-2, 9.33597e0),
            (1.36815e-2, 9.42778e0),
            (1.489e-2, 9.50972e0),
            (1.62045e-2, 9.57813e0),
            (1.76344e-2, 9.62875e0),
            (1.91897e-2, 9.65689e0),
        ],
    }),
];
pub(crate) const SINGLE_NOT_RECORDED: [&str; 3] = [
    "the run did not finish: 'run ended: diverged | diverged at outer iteration 32503: a field went non-finite (NaN/Inf) | exit code 2'",
    "",
    "",
];
// RECORDS-END

#[cfg(test)]
mod tests {
    use super::*;

    /// §110.2's three body forces, the force balance `g_x = (Re_tau nu/h)^2/h`
    /// at the key headers' `Re_tau`, to the five digits the cases carry.
    const G_X: [f64; 3] = [0.89231, 4.32709, 9.69728];
    const RE_TAU: [f64; 3] = [178.12, 392.24, 587.19];

    #[test]
    fn the_records_carry_the_cases_inputs() {
        for (i, r) in RECORDS.iter().enumerate() {
            assert_eq!(f64::from(r.re_tau), f64::from(RE_TAU[i] as Scalar));
            assert_eq!(f64::from(r.body_force), f64::from(G_X[i] as Scalar));
            let g = (RE_TAU[i] * 1.5e-5 / 0.02_f64).powi(2) / 0.02;
            assert!((g / G_X[i] - 1.0).abs() < 1e-5, "110.2's balance: {g} against {}", G_X[i]);
            assert_eq!(r.n_y, [32, 64, 128]);
            assert_eq!(r.iterations, 40_000);
            assert_eq!(r.profile_fine.len(), 64, "the finest column's lower half");
            assert!(r.profile_fine.windows(2).all(|t| t[1].0 > t[0].0), "wall to centre");
            match &SINGLE[i] {
                Some(s) => assert_eq!((s.n_y, s.iterations, s.profile_fine.len()), (128, 40_000, 64)),
                None => assert!(SINGLE_NOT_RECORDED[i].contains("diverged"), "a missing record says why"),
            }
        }
    }

    /// Reads one run's two outputs: `(column rows (y, U_x), column mean U_x)`
    /// from the sampler and `(last |U| res, kinematic wall sink, wall area,
    /// iterations)` from the driver's log.
    fn read_run(dir: &std::path::Path, stem: &str) -> (Vec<(f64, f64)>, f64, f64, f64, f64, usize) {
        let col = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.col"))).expect("col");
        let log = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.log"))).expect("log");
        let mut rows = Vec::new();
        let mut mean = None;
        for line in col.lines().skip(1) {
            let t = line.trim();
            if let Some(m) = t.strip_prefix("volume-weighted mean:") {
                mean = m.split_whitespace().next().map(|v| v.parse::<f64>().expect("mean"));
                continue;
            }
            let f: Vec<f64> = t.split_whitespace().map(|v| v.parse().expect("row")).collect();
            rows.push((f[0], f[1]));
        }
        let res = log
            .lines()
            .filter(|l| l.starts_with("iter "))
            .last()
            .and_then(|l| l.split("|U| res").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .map(|v| v.parse::<f64>().expect("res"))
            .expect("an iter line");
        let after = |key: &str| -> f64 {
            let l = log.lines().find(|l| l.contains(key)).unwrap_or_else(|| panic!("{key}"));
            let rest = &l[l.find(key).expect("key") + key.len()..];
            rest.split_whitespace().next().expect("value").parse().expect("number")
        };
        let sink = after("wall sink sum nu_eff |dU_par| deltaCoeffs |Sf| =");
        let area = after("total wall area:");
        let ended = log.lines().find(|l| l.starts_with("run ended:")).expect("run ended");
        let iters = ended
            .split('|')
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .map(|v| v.parse().expect("iterations"))
            .expect("iterations");
        (rows, mean.expect("a -mean line"), res, sink, area, iters)
    }

    fn lower_half(rows: &[(f64, f64)]) -> Vec<(f64, f64)> {
        rows.iter().copied().filter(|&(y, _)| y < 0.02).collect()
    }

    fn print_profile(rows: &[(f64, f64)]) {
        println!("        profile_fine: &[");
        for (y, u) in rows {
            println!("            ({y:e}, {u:e}),");
        }
        println!("        ],");
    }

    /// The recorder (§110.1's split): prints `RECORDS` and `SINGLE` from the
    /// runs directory `VAL51_RUNS`. Not part of any verification.
    #[test]
    #[ignore = "reads a recorded runs directory (env VAL51_RUNS); prints the record literals"]
    fn record_channel_dns() {
        let dir = std::path::PathBuf::from(std::env::var("VAL51_RUNS").expect("VAL51_RUNS"));
        let legs = ["180", "395", "590"];
        println!("pub(crate) static RECORDS: [ChannelDnsRecord; 3] = [");
        for (i, leg) in legs.iter().enumerate() {
            let u_tau = (G_X[i] * 0.02).sqrt();
            let mut ub = [0.0; 3];
            let mut fine = None;
            for (j, n) in [32, 64, 128].iter().enumerate() {
                let stem = format!("channelDns{leg}_n{n}");
                let case = std::fs::read_to_string(dir.join("cases").join(format!("{stem}.jsonc"))).expect("case");
                assert!(case.contains(&format!("\"cells\":  [8, {n}, 1]")), "{stem}: cells");
                assert!(case.contains(&format!("\"bodyForce\": [{}, 0, 0]", G_X[i])), "{stem}: g_x");
                let run = read_run(&dir, &stem);
                assert_eq!(run.0.len(), *n, "{stem}: the column has N_y cells");
                ub[j] = run.1 / u_tau;
                fine = Some(run);
            }
            let (rows, _, res, sink, area, iters) = fine.expect("finest");
            println!("    ChannelDnsRecord {{");
            println!("        re_tau: {},", RE_TAU[i]);
            println!("        body_force: {},", G_X[i]);
            println!("        n_y: [32, 64, 128],");
            println!("        u_b_plus: [{:.9e}, {:.9e}, {:.9e}],", ub[0], ub[1], ub[2]);
            println!("        tau_w_measured_kin: {:.9e},", sink / area);
            println!("        iterations: {iters},");
            println!("        residual_u: {res:e},");
            print_profile(&lower_half(&rows));
            println!("    }},");
        }
        println!("];");
        let mut why = Vec::new();
        println!("pub(crate) static SINGLE: [Option<ChannelDnsSingle>; 3] = [");
        for (i, leg) in legs.iter().enumerate() {
            let u_tau = (G_X[i] * 0.02).sqrt();
            let stem = format!("channelDns{leg}_n128s");
            let log = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.log"))).expect("log");
            let ended = log.lines().find(|l| l.starts_with("run ended:")).expect("run ended").to_string();
            if !ended.starts_with("run ended: budget") {
                println!("    None,");
                why.push(format!("the run did not finish: '{}'", ended.trim()));
                continue;
            }
            why.push(String::new());
            let (rows, mean, res, sink, area, iters) = read_run(&dir, &stem);
            assert_eq!(rows.len(), 128, "{stem}: the column has N_y cells");
            println!("    Some(ChannelDnsSingle {{");
            println!("        n_y: 128,");
            println!("        u_b_plus: {:.9e},", mean / u_tau);
            println!("        tau_w_measured_kin: {:.9e},", sink / area);
            println!("        iterations: {iters},");
            println!("        residual_u: {res:e},");
            print_profile(&lower_half(&rows));
            println!("    }}),");
        }
        println!("];");
        println!("pub(crate) const SINGLE_NOT_RECORDED: [&str; 3] = [");
        for w in &why {
            println!("    {w:?},");
        }
        println!("];");
    }
}
