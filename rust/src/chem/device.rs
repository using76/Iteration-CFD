// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The device kernels of SPEC-LIT §132.9 (`cuda/chem.cu`): one thread per
//! cell, the molar production rates (130.1) from `(T, p, Y)` through
//! (128.9), and the constant-pressure reactor's right-hand side
//! (132.7)-(132.9) with §132.3's one-sided finite-difference Jacobian.
//! Both compute in double in every build, reproducing the f64 host twins -
//! [`crate::chem::kinetics::production_rates_tpy`], and
//! [`crate::chem::reactor::ConstPressureReactor::rhs`] with
//! [`crate::chem::rosenbrock::fd_jacobian`] - choice for choice and in the
//! hosts' operation order. A mechanism is flattened once to the two device
//! arrays of §132.9 D2 ([`flatten`]); a refused cell writes NaN to all its
//! outputs and a status code ([`status`]).
//!
//! §132.10 (CHR-07b) adds the chemistry step itself: [`integrate`] runs the
//! host twin [`crate::chem::reactor::ConstPressureReactor::advance`] - the
//! Rosenbrock of §132.7 over one flow step dt, chemistry in double in both
//! builds - one thread per cell, and [`integrate_fields`] is its ofscalar
//! crossing (§132.6).
//!
//! No GPL-licensed source was consulted.

use crate::chem::mechanism::{Blending, Collider, Mechanism};
use crate::chem::reactor::ReactorOptions;
use crate::chem::transport::{Lewis, TransportSet};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::{Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};

/// The per-cell refusal codes of §132.9 D3 and §132.10, in the host twins'
/// order of checks. A kernel cannot return an `Error::Config`, so a refused
/// cell writes NaN to all its outputs and its code here; it cannot be used
/// silently.
pub mod status {
    pub const OK: i32 = 0;
    pub const BAD_PRESSURE: i32 = 1;
    pub const BAD_TEMPERATURE: i32 = 2;
    pub const BAD_MASS_FRACTION: i32 = 3;
    pub const NO_SPECIES: i32 = 4;
    pub const OUTSIDE_THERMO_RANGE: i32 = 5;
    /// A singular `I - gamma h J` (a pivot exactly 0 or not finite).
    pub const SINGULAR_MATRIX: i32 = 6;
    /// More step attempts than `max_steps`.
    pub const MAX_STEPS: i32 = 7;
    /// A `Y_k` below `-atol_y` at advance's observer.
    pub const BELOW_NEGATIVE_ATOL: i32 = 8;
    /// `h_in` not finite or not above 0 (R5's `h0`).
    pub const BAD_STEP: i32 = 9;
    /// A present species' T outside its TM-4513 eq. (10) transport range
    /// (SPEC-LIT 133.4's transport kernels).
    pub const OUTSIDE_TRANSPORT_RANGE: i32 = 10;
}

/// The flat mechanism of §132.9 D2: the whole mechanism as two arrays, one
/// of doubles and one of ints, in the layout `cuda/chem.cu`'s header
/// states. The two field lists must stay identical; this is the Rust one.
///
/// `mi` header, 11 ints: `[0]` K, `[1]` I (reactions), `[2]` n_tb (dense
/// efficiency rows), `[3]` base_r, `[4]` base_o, `[5]` base_s, `[6]`
/// base_e (offsets into mf), `[7]` base_need_g, `[8]` base_rx, `[9]`
/// base_ord_sp, `[10]` base_sto_sp (offsets into mi).
///
/// `mf`: the species block at 0, 18 doubles per species k at 18k - W
/// (kg/mol), t_low, t_mid, t_high, lower a1..a7, upper a1..a7; then at 18K
/// `R_GAS` and `p_ref`, taken from the host so that no constant is spelled
/// twice; then at base_r, 14 doubles per reaction - A, beta, theta (the
/// reaction rate), revA, revBeta, revTheta (the REV rate, or 0), lowA,
/// lowBeta, lowTheta (the fall-off LOW rate, or 0), q0..q4 (Troe: a, T***,
/// T*, T**, 0 - T** 0 when absent; SRI: a, b, c, d, e; otherwise five 0s);
/// then at base_o one order value per order entry; then at base_s one nu
/// ([`crate::chem::Rational::to_f64`]) per stoichiometry entry; then at
/// base_e n_tb rows of K alphas each (alpha = the listed efficiency, else
/// 1.0).
///
/// `mi` at base_need_g: K ints, 1 when species k appears in the reactants
/// or the products of a reaction that is reversible without REV (the only
/// time G/RT is read), else 0. At base_rx, 13 ints per reaction: kind
/// (2 fall-off, else 1 third body, else 0), rev_kind (0 irreversible,
/// 2 explicit REV, else 1 from ln K_c), tb_row (the dense row of the
/// reaction's third body or `(+M)` collider, else -1), collider_species
/// (the `(+SPECIES)` index, else -1), blending (-1 none, 0 Lindemann,
/// 1 Troe without T**, 2 Troe with T**, 3 SRI), fo_begin, fo_end (the
/// reaction's range of the forward-order list), ro_begin, ro_end (the
/// reverse-order list), re_begin, re_end (the reactant list), pr_begin,
/// pr_end (the product list), in stored order. At base_ord_sp: the species
/// index of each order entry, the same index space as base_o's values; at
/// base_sto_sp: the species index of each stoichiometry entry, the same
/// index space as base_s's values.
pub fn flatten(mech: &Mechanism) -> (Vec<f64>, Vec<i32>) {
    let sp = mech.thermo().species();
    let k = sp.len();
    let rxs = mech.reactions();
    let i = rxs.len();

    // One dense efficiency row per plain `+M` and per `(+M)` fall-off
    // collider, assigned in mechanism order; a `(+SPECIES)` collider takes
    // its species' concentration alone and has no row.
    let mut n_tb = 0usize;
    let mut tb_row = vec![-1i32; i];
    for (r, row) in rxs.iter().zip(&mut tb_row) {
        let mixture = r.third_body.is_some()
            || r.falloff
                .as_ref()
                .is_some_and(|f| matches!(f.collider, Collider::Mixture(_)));
        if mixture {
            *row = n_tb as i32;
            n_tb += 1;
        }
    }

    // The per-reaction records, and the order and stoichiometry lists:
    // forward then reverse, reactants then products, in mechanism order.
    let mut rx_i = Vec::with_capacity(13 * i);
    let mut rx_f = Vec::with_capacity(i);
    let mut order_val: Vec<f64> = Vec::new();
    let mut order_sp: Vec<i32> = Vec::new();
    let mut sto_val: Vec<f64> = Vec::new();
    let mut sto_sp: Vec<i32> = Vec::new();
    for (r, &row) in rxs.iter().zip(&tb_row) {
        let kind = if r.falloff.is_some() {
            2
        } else if r.third_body.is_some() {
            1
        } else {
            0
        };
        let rev_kind = if !r.reversible {
            0
        } else if r.reverse.is_some() {
            2
        } else {
            1
        };
        let col_sp = match r.falloff.as_ref().map(|f| &f.collider) {
            Some(Collider::Species(j)) => *j as i32,
            _ => -1,
        };
        let blending = r
            .falloff
            .as_ref()
            .map_or(-1, |f| match f.blending {
                Blending::Lindemann => 0,
                Blending::Troe { t2: None, .. } => 1,
                Blending::Troe { t2: Some(_), .. } => 2,
                Blending::Sri { .. } => 3,
            });
        let fo_begin = order_sp.len() as i32;
        for (s, o) in r.forward_orders() {
            order_val.push(o);
            order_sp.push(s as i32);
        }
        let fo_end = order_sp.len() as i32;
        let ro_begin = order_sp.len() as i32;
        for (s, o) in r.reverse_orders() {
            order_val.push(o);
            order_sp.push(s as i32);
        }
        let ro_end = order_sp.len() as i32;
        let re_begin = sto_sp.len() as i32;
        for (s, nu) in &r.reactants {
            sto_val.push(nu.to_f64());
            sto_sp.push(*s as i32);
        }
        let re_end = sto_sp.len() as i32;
        let pr_begin = sto_sp.len() as i32;
        for (s, nu) in &r.products {
            sto_val.push(nu.to_f64());
            sto_sp.push(*s as i32);
        }
        let pr_end = sto_sp.len() as i32;
        rx_i.extend_from_slice(&[
            kind, rev_kind, row, col_sp, blending, fo_begin, fo_end, ro_begin, ro_end,
            re_begin, re_end, pr_begin, pr_end,
        ]);
        let (rev_a, rev_b, rev_t) = r
            .reverse
            .as_ref()
            .map_or((0.0, 0.0, 0.0), |v| (v.a, v.beta, v.theta));
        let (low_a, low_b, low_t) = r
            .falloff
            .as_ref()
            .map_or((0.0, 0.0, 0.0), |f| (f.low.a, f.low.beta, f.low.theta));
        let (q0, q1, q2, q3, q4) = match r.falloff.as_ref().map(|f| &f.blending) {
            Some(Blending::Troe { a, t3, t1, t2 }) => (*a, *t3, *t1, t2.unwrap_or(0.0), 0.0),
            Some(Blending::Sri { a, b, c, d, e }) => (*a, *b, *c, *d, *e),
            _ => (0.0, 0.0, 0.0, 0.0, 0.0),
        };
        rx_f.push([
            r.rate.a, r.rate.beta, r.rate.theta, rev_a, rev_b, rev_t, low_a, low_b,
            low_t, q0, q1, q2, q3, q4,
        ]);
    }

    // need_g: the species a reversible reaction without REV touches; only
    // theirs is a G/RT the kernels evaluate, and only theirs a range the T
    // must sit inside (§130.7).
    let mut need_g = vec![0i32; k];
    for r in rxs.iter().filter(|r| r.reversible && r.reverse.is_none()) {
        for (s, _) in r.reactants.iter().chain(&r.products) {
            need_g[*s] = 1;
        }
    }

    let base_r = 18 * k + 2;
    let base_o = base_r + 14 * i;
    let base_s = base_o + order_val.len();
    let base_e = base_s + sto_val.len();

    let mut mf: Vec<f64> = Vec::with_capacity(base_e + n_tb * k);
    for s in sp {
        mf.extend_from_slice(&[s.molar_mass, s.t_low, s.t_mid, s.t_high]);
        mf.extend_from_slice(&s.lower);
        mf.extend_from_slice(&s.upper);
    }
    mf.push(crate::chem::R_GAS);
    mf.push(mech.thermo().p_ref());
    for f in &rx_f {
        mf.extend_from_slice(f);
    }
    mf.extend_from_slice(&order_val);
    mf.extend_from_slice(&sto_val);
    for (r, &row) in rxs.iter().zip(&tb_row) {
        if row < 0 {
            continue;
        }
        let eff: &[(usize, f64)] = match r.third_body.as_ref() {
            Some(tb) => &tb.efficiencies,
            None => match &r.falloff.as_ref().expect("a dense row has a collider").collider {
                Collider::Mixture(tb) => &tb.efficiencies,
                Collider::Species(_) => continue,
            },
        };
        let mut alphas = vec![1.0f64; k];
        for (j, a) in eff {
            alphas[*j] = *a;
        }
        mf.extend_from_slice(&alphas);
    }

    let mut mi = Vec::with_capacity(11 + k + rx_i.len() + order_sp.len() + sto_sp.len());
    mi.extend_from_slice(&[
        k as i32,
        i as i32,
        n_tb as i32,
        base_r as i32,
        base_o as i32,
        base_s as i32,
        base_e as i32,
        11,
        11 + k as i32,
        (11 + k + rx_i.len()) as i32,
        (11 + k + rx_i.len() + order_sp.len()) as i32,
    ]);
    mi.extend_from_slice(&need_g);
    mi.extend_from_slice(&rx_i);
    mi.extend_from_slice(&order_sp);
    mi.extend_from_slice(&sto_sp);

    (mf, mi)
}

/// The kernels of `cuda/chem.cu`, loaded once per device.
pub struct ChemKernels {
    production: CudaFunction,
    jacobian: CudaFunction,
    integrate: CudaFunction,
    pack: CudaFunction,
    unpack: CudaFunction,
    transport: CudaFunction,
    prandtl: CudaFunction,
    lewis: CudaFunction,
}

impl ChemKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::CHEM)?;
        Ok(Self {
            production: k.func("chemProductionRates")?,
            jacobian: k.func("chemReactorJacobian")?,
            integrate: k.func("chemIntegrate")?,
            pack: k.func("chemPackState")?,
            unpack: k.func("chemUnpackState")?,
            transport: k.func("chemMixtureTransport")?,
            prandtl: k.func("chemPrandtlConductivity")?,
            lewis: k.func("chemLewisDiffusivity")?,
        })
    }
}

/// The two flat arrays of [`flatten`] on the device, uploaded once per
/// mechanism; the kernels read the layout out of them, so a launch carries
/// no pointer arithmetic of its own.
pub struct DeviceMechanism {
    mf: DevBuf<f64>,
    mi: DevBuf<i32>,
    n_species: usize,
    n_reactions: usize,
}

impl DeviceMechanism {
    pub fn upload(gpu: &Gpu, mech: &Mechanism) -> Result<Self> {
        let (mf, mi) = flatten(mech);
        Ok(Self {
            mf: gpu.upload(&mf)?,
            mi: gpu.upload(&mi)?,
            n_species: mech.species().len(),
            n_reactions: mech.reactions().len(),
        })
    }

    /// The species count, K.
    pub fn n_species(&self) -> usize {
        self.n_species
    }

    /// The reaction count, I.
    pub fn n_reactions(&self) -> usize {
        self.n_reactions
    }
}

/// The f64 scratch of §132.9 the two kernels need, SoA with stride
/// [`ChemWork::n_cells`]: `conc`, `g`, `omega` and `yp` carry K values per
/// cell, `ypert` and `fpert` K+1. Everything computes in double in both
/// builds, so the scratch is f64 in both; a CUDA thread cannot size a
/// local array at run time, which is why the per-cell vectors live here.
pub struct ChemWork {
    conc: DevBuf<f64>,
    g: DevBuf<f64>,
    omega: DevBuf<f64>,
    yp: DevBuf<f64>,
    ypert: DevBuf<f64>,
    fpert: DevBuf<f64>,
    n_cells: usize,
}

impl ChemWork {
    pub fn new(gpu: &Gpu, dm: &DeviceMechanism, n_cells: usize) -> Result<Self> {
        let k = dm.n_species();
        Ok(Self {
            conc: gpu.zeros(k * n_cells)?,
            g: gpu.zeros(k * n_cells)?,
            omega: gpu.zeros(k * n_cells)?,
            yp: gpu.zeros(k * n_cells)?,
            ypert: gpu.zeros((k + 1) * n_cells)?,
            fpert: gpu.zeros((k + 1) * n_cells)?,
            n_cells,
        })
    }

    /// The cell count the scratch was sized for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

/// A launch whose buffers do not hold the layout the call asks for is a
/// refusal naming the function, the buffer and both lengths.
fn check_layout(f: &str, buf: &str, have: usize, need: usize) -> Result<()> {
    if have < need {
        return Err(Error::Config(format!(
            "{f}: the buffer {buf} holds {have} element(s), the layout of this \
             call needs {need}"
        )));
    }
    Ok(())
}

fn check_cells(f: &str, n: usize, cells: usize) -> Result<()> {
    if n > cells {
        return Err(Error::Config(format!(
            "{f}: the call asks for {n} cell(s), the work buffers were sized for \
             {cells}"
        )));
    }
    Ok(())
}
/// (130.1) for every cell, from `(T, p, Y)` through (128.9): the device
/// [`crate::chem::kinetics::production_rates_tpy`] (§132.9). Layouts:
/// `t[c]`, `p[c]`, `y[k*n+c]`, `omega[k*n+c]`, `status[c]`. A refused cell
/// writes NaN to every omega_k and its [`status`] code; `n == 0` launches
/// nothing. No sync, no allocation, so the call captures.
///
/// The omega output is ofscalar: the kernel accumulates in double and
/// rounds once on the way out (§132.9).
#[allow(clippy::too_many_arguments)]
pub fn production_rates(
    gpu: &Gpu,
    k: &ChemKernels,
    dm: &DeviceMechanism,
    work: &mut ChemWork,
    t: &DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    omega: &mut DevBuf<Scalar>,
    status: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    let f = "chemProductionRates";
    let ks = dm.n_species();
    check_cells(f, n, work.n_cells())?;
    check_layout(f, "t", t.len(), n)?;
    check_layout(f, "p", p.len(), n)?;
    check_layout(f, "y", y.len(), ks * n)?;
    check_layout(f, "omega", omega.len(), ks * n)?;
    check_layout(f, "status", status.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.production)
            .arg(t)
            .arg(p)
            .arg(y)
            .arg(&mut *omega)
            .arg(&mut *status)
            .arg(&mut work.conc)
            .arg(&mut work.g)
            .arg(&mut work.omega)
            .arg(&dm.mf)
            .arg(&dm.mi)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}
/// The constant-pressure reactor's right-hand side (132.7)-(132.9) at every
/// cell, and §132.3's one-sided finite-difference Jacobian column by
/// column, `delta_j = 2^-26 max(|y_j|, atol_j)`: the device
/// [`crate::chem::reactor::ConstPressureReactor::rhs`] with
/// [`crate::chem::rosenbrock::fd_jacobian`] (§132.9). Layouts:
/// `state[i*n+c]` with i = K the temperature, `f0` like state, `atol[i]`
/// K+1 values shared by all cells, `jac[(i*(K+1)+j)*n+c] =
/// d f_i / d y_j`. The status is the first refusal met at `y` or at any
/// perturbed column; a refused cell writes NaN to its f0 and every J
/// entry. `n == 0` launches nothing; the call captures.
#[allow(clippy::too_many_arguments)]
pub fn reactor_jacobian(
    gpu: &Gpu,
    k: &ChemKernels,
    dm: &DeviceMechanism,
    work: &mut ChemWork,
    state: &DevBuf<f64>,
    p: &DevBuf<f64>,
    atol: &DevBuf<f64>,
    f0: &mut DevBuf<f64>,
    jac: &mut DevBuf<f64>,
    status: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    let f = "chemReactorJacobian";
    let ks = dm.n_species();
    check_cells(f, n, work.n_cells())?;
    check_layout(f, "state", state.len(), (ks + 1) * n)?;
    check_layout(f, "p", p.len(), n)?;
    check_layout(f, "atol", atol.len(), ks + 1)?;
    check_layout(f, "f0", f0.len(), (ks + 1) * n)?;
    check_layout(f, "jac", jac.len(), (ks + 1) * (ks + 1) * n)?;
    check_layout(f, "status", status.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.jacobian)
            .arg(state)
            .arg(p)
            .arg(atol)
            .arg(&mut *f0)
            .arg(&mut *jac)
            .arg(&mut *status)
            .arg(&mut work.conc)
            .arg(&mut work.g)
            .arg(&mut work.omega)
            .arg(&mut work.yp)
            .arg(&mut work.ypert)
            .arg(&mut work.fpert)
            .arg(&dm.mf)
            .arg(&dm.mi)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}
/// The length of [`DeviceReactor`]'s one f64 control array, in the layout
/// its doc comment states and `cuda/chem.cu`'s header repeats.
pub const CTRL_LEN: usize = 48;

/// The controls of one device chemistry step (SPEC-LIT §132.10): the
/// coefficient set and tolerances of [`ReactorOptions`] as one f64 array
/// the `chemIntegrate` kernel reads. `h0` and `ignition_rise` are not
/// uploaded - the step is per cell and R11 is host-only - so only these
/// fields are validated here, each refusal naming its value:
///
/// * `rtol` not finite or below 0;
/// * `atol_y` or `atol_t` not finite or not above 0;
/// * `h_max` NaN, not above 0, or -inf (+inf, the default, is the
///   unbounded cap);
/// * `max_steps` 0 or above `i32::MAX`.
///
/// The control array's layout, in `cuda/chem.cu`'s header with the same
/// field list: `[0]` gamma, `[1..17)` alpha 4x4 row-major, `[17..33)`
/// gamma_ij 4x4 row-major, `[33..37)` b, `[37..41)` b_hat, `[41]` stages,
/// `[42]` embedded_order, `[43]` rtol, `[44]` atol_y, `[45]` atol_t,
/// `[46]` h_max (+inf allowed), `[47]` max_steps. The coefficients come
/// from `opts.method.coefficients()`, so no constant is spelled twice.
#[derive(Debug)]
pub struct DeviceReactor {
    ctrl: DevBuf<f64>,
    opts: ReactorOptions,
}

impl DeviceReactor {
    pub fn new(gpu: &Gpu, opts: &ReactorOptions) -> Result<Self> {
        const TAG: &str = "(SPEC-LIT §132.10)";
        if !opts.rtol.is_finite() || opts.rtol < 0.0 {
            return Err(Error::Config(format!(
                "rtol = {:e} must be finite and at least 0 {TAG}",
                opts.rtol
            )));
        }
        if !opts.atol_y.is_finite() || opts.atol_y <= 0.0 {
            return Err(Error::Config(format!(
                "atol_y = {:e} must be finite and above 0 {TAG}",
                opts.atol_y
            )));
        }
        if !opts.atol_t.is_finite() || opts.atol_t <= 0.0 {
            return Err(Error::Config(format!(
                "atol_t = {:e} must be finite and above 0 {TAG}",
                opts.atol_t
            )));
        }
        if opts.h_max.is_nan() || opts.h_max <= 0.0 {
            return Err(Error::Config(format!(
                "h_max = {:e} must be finite and above 0, or +inf {TAG}",
                opts.h_max
            )));
        }
        if opts.max_steps == 0 || opts.max_steps > i32::MAX as usize {
            return Err(Error::Config(format!(
                "max_steps = {} must be at least 1 and at most {} {TAG}",
                opts.max_steps,
                i32::MAX
            )));
        }
        let c = opts.method.coefficients();
        let mut ctrl = vec![0.0f64; CTRL_LEN];
        ctrl[0] = c.gamma;
        for i in 0..4 {
            for j in 0..4 {
                ctrl[1 + 4 * i + j] = c.alpha[i][j];
                ctrl[17 + 4 * i + j] = c.gamma_ij[i][j];
            }
        }
        ctrl[33..37].copy_from_slice(&c.b);
        ctrl[37..41].copy_from_slice(&c.b_hat);
        ctrl[41] = c.stages as f64;
        ctrl[42] = c.embedded_order as f64;
        ctrl[43] = opts.rtol;
        ctrl[44] = opts.atol_y;
        ctrl[45] = opts.atol_t;
        ctrl[46] = opts.h_max;
        ctrl[47] = opts.max_steps as f64;
        Ok(Self {
            ctrl: gpu.upload(&ctrl)?,
            opts: opts.clone(),
        })
    }

    /// The options the reactor was built from.
    pub fn options(&self) -> &ReactorOptions {
        &self.opts
    }
}

/// The f64 scratch of one device chemistry step (SPEC-LIT §132.10), SoA
/// with stride [`IntegrateWork::n_cells`], `m = K + 1`. `vecs` holds ten
/// vectors of `m` rows per cell - f0, y_new, est, arg, rhs, v, k1, k2, k3,
/// k4 - element `(i, cell c)` of a vector at `[i * n + c]`; `jac` and `lu`
/// hold the row-major `m x m` matrices, entry `(i, j)` at
/// `[(i * m + j) * n + c]`; `piv` the LU row swaps. `state` and `p` are the
/// f64 columns [`integrate_fields`]' pack widens the ofscalar fields into.
/// Everything computes in double in every build (§132.6).
pub struct IntegrateWork {
    vecs: DevBuf<f64>,
    jac: DevBuf<f64>,
    lu: DevBuf<f64>,
    piv: DevBuf<i32>,
    state: DevBuf<f64>,
    p: DevBuf<f64>,
    n_cells: usize,
}

impl IntegrateWork {
    pub fn new(gpu: &Gpu, dm: &DeviceMechanism, n_cells: usize) -> Result<Self> {
        let m = dm.n_species() + 1;
        let n = n_cells;
        Ok(Self {
            vecs: gpu.zeros(10 * m * n)?,
            jac: gpu.zeros(m * m * n)?,
            lu: gpu.zeros(m * m * n)?,
            piv: gpu.zeros(m * n)?,
            state: gpu.zeros(m * n)?,
            p: gpu.zeros(n)?,
            n_cells,
        })
    }

    /// The cell count the scratch was sized for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

/// The device chemistry step of SPEC-LIT §132.10: one thread per cell runs
/// the host twin [`crate::chem::reactor::ConstPressureReactor::advance`]
/// over one flow step `dt` - `integrate_observed` with its `ros_step`,
/// `lu_factor`, `lu_solve`, `fd_jacobian` and `error_norm`, and advance's
/// observer (R10's clip / refusal) - in the host's operation order, with
/// chemistry in double in every build (§132.6). Layouts: `state[i*n+c]` =
/// `(Y_1..Y_K, T)` in place, `p[c]`, `h[c]` in = the first trial step
/// (advance's `h0`), out = the step proposed after the last accepted step
/// (`RosStats::h_next`), `counts[0*n+c]` accepted, `[1*n+c]` rejected,
/// `[2*n+c]` clipped. A refused cell writes NaN to its whole state column
/// and to `h[c]`; its counts hold what stood at the refusal. Rust-side
/// refusals (`Error::Config`, "(SPEC-LIT §132.10)"): `dt` not finite or
/// below 0; `n` above `work.n_cells()` or `iw.n_cells()`; a buffer shorter
/// than the layout. `dt == 0` or `n == 0` launches nothing (buffers
/// untouched). No sync, no allocation (capturable).
#[allow(clippy::too_many_arguments)]
pub fn integrate(
    gpu: &Gpu,
    k: &ChemKernels,
    dm: &DeviceMechanism,
    work: &mut ChemWork,
    iw: &mut IntegrateWork,
    dr: &DeviceReactor,
    state: &mut DevBuf<f64>,
    p: &DevBuf<f64>,
    h: &mut DevBuf<f64>,
    dt: f64,
    status: &mut DevBuf<i32>,
    counts: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    let f = "chemIntegrate";
    if !dt.is_finite() || dt < 0.0 {
        return Err(Error::Config(format!(
            "{f}: dt = {dt} s must be finite and at least 0 (SPEC-LIT §132.10)"
        )));
    }
    let ks = dm.n_species();
    check_cells(f, n, work.n_cells())?;
    check_cells(f, n, iw.n_cells())?;
    check_layout(f, "state", state.len(), (ks + 1) * n)?;
    check_layout(f, "p", p.len(), n)?;
    check_layout(f, "h", h.len(), n)?;
    check_layout(f, "status", status.len(), n)?;
    check_layout(f, "counts", counts.len(), 3 * n)?;
    check_layout(f, "ctrl", dr.ctrl.len(), CTRL_LEN)?;
    if n == 0 || dt == 0.0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.integrate)
            .arg(&mut *state)
            .arg(p)
            .arg(&mut *h)
            .arg(&mut *status)
            .arg(&mut *counts)
            .arg(&dr.ctrl)
            .arg(&mut iw.vecs)
            .arg(&mut iw.jac)
            .arg(&mut iw.lu)
            .arg(&mut iw.piv)
            .arg(&mut work.conc)
            .arg(&mut work.g)
            .arg(&mut work.omega)
            .arg(&mut work.yp)
            .arg(&mut work.ypert)
            .arg(&mut work.fpert)
            .arg(&dm.mf)
            .arg(&dm.mi)
            .arg(&dt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// [`integrate`]'s ofscalar crossing (SPEC-LIT §132.6, §132.10): kernel
/// `chemPackState` widens `t[c]`, `y[k*n+c]`, `p[c]` exactly into
/// [`IntegrateWork`]'s f64 state and pressure, [`integrate`]'s launch runs
/// on them, and kernel `chemUnpackState` rounds the integrated state once
/// into `t` and `y`. In the f64 build that is bitwise [`integrate`] (the
/// widenings are identities); in the f32 build the flow state crosses as
/// Scalar and the chemistry still computes in double (§132.6). The same
/// refusals and the same `dt == 0` rule as [`integrate`].
#[allow(clippy::too_many_arguments)]
pub fn integrate_fields(
    gpu: &Gpu,
    k: &ChemKernels,
    dm: &DeviceMechanism,
    work: &mut ChemWork,
    iw: &mut IntegrateWork,
    dr: &DeviceReactor,
    t: &mut DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    y: &mut DevBuf<Scalar>,
    h: &mut DevBuf<f64>,
    dt: f64,
    status: &mut DevBuf<i32>,
    counts: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    let f = "chemPackState";
    if !dt.is_finite() || dt < 0.0 {
        return Err(Error::Config(format!(
            "{f}: dt = {dt} s must be finite and at least 0 (SPEC-LIT §132.10)"
        )));
    }
    let ks = dm.n_species();
    check_cells(f, n, work.n_cells())?;
    check_cells(f, n, iw.n_cells())?;
    check_layout(f, "t", t.len(), n)?;
    check_layout(f, "p", p.len(), n)?;
    check_layout(f, "y", y.len(), ks * n)?;
    check_layout(f, "h", h.len(), n)?;
    check_layout(f, "status", status.len(), n)?;
    check_layout(f, "counts", counts.len(), 3 * n)?;
    check_layout(f, "ctrl", dr.ctrl.len(), CTRL_LEN)?;
    if n == 0 || dt == 0.0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.pack)
            .arg(&mut *t)
            .arg(p)
            .arg(&mut *y)
            .arg(&mut iw.state)
            .arg(&mut iw.p)
            .arg(&dm.mi)
            .arg(&nl)
            .launch(cfg_for(n))?;
        gpu.stream()
            .launch_builder(&k.integrate)
            .arg(&mut iw.state)
            .arg(&iw.p)
            .arg(&mut *h)
            .arg(&mut *status)
            .arg(&mut *counts)
            .arg(&dr.ctrl)
            .arg(&mut iw.vecs)
            .arg(&mut iw.jac)
            .arg(&mut iw.lu)
            .arg(&mut iw.piv)
            .arg(&mut work.conc)
            .arg(&mut work.g)
            .arg(&mut work.omega)
            .arg(&mut work.yp)
            .arg(&mut work.ypert)
            .arg(&mut work.fpert)
            .arg(&dm.mf)
            .arg(&dm.mi)
            .arg(&dt)
            .arg(&nl)
            .launch(cfg_for(n))?;
        gpu.stream()
            .launch_builder(&k.unpack)
            .arg(&mut iw.state)
            .arg(&mut *t)
            .arg(&mut *y)
            .arg(&dm.mi)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}
/// The flat transport layout of SPEC-LIT §133.4's device twins, one f64
/// array: 40 doubles per species k at `40k` - `[0]` W (kg/mol), `[1]` n_v
/// (1..3, as a double), `[2..20)` three viscosity fits of 6 doubles
/// `(t_lo, t_hi, a, b, c, d)` (the unused slots 0.0), `[20]` n_c, `[21..39)`
/// three conductivity fits of 6 doubles (the unused slots 0.0), `[39]` 0.0.
/// The twin of this layout is the transport header comment in
/// `cuda/chem.cu`; the two field lists must stay identical. The species
/// count is not in the array - the launches carry it as an argument, as
/// `ChemKernels::new` loaded it.
pub fn flatten_transport(ts: &TransportSet) -> Vec<f64> {
    let mut v = Vec::with_capacity(40 * ts.species().len());
    for (k, sp) in ts.species().iter().enumerate() {
        v.push(ts.molar_masses()[k]);
        v.push(sp.viscosity.len() as f64);
        for i in 0..3 {
            match sp.viscosity.get(i) {
                Some(f) => v.extend_from_slice(&[f.t_lo, f.t_hi, f.a, f.b, f.c, f.d]),
                None => v.extend_from_slice(&[0.0; 6]),
            }
        }
        v.push(sp.conductivity.len() as f64);
        for i in 0..3 {
            match sp.conductivity.get(i) {
                Some(f) => v.extend_from_slice(&[f.t_lo, f.t_hi, f.a, f.b, f.c, f.d]),
                None => v.extend_from_slice(&[0.0; 6]),
            }
        }
        v.push(0.0);
    }
    v
}

/// [`flatten_transport`]'s array on the device, uploaded once per
/// [`TransportSet`].
pub struct DeviceTransport {
    tf: DevBuf<f64>,
    n_species: usize,
}

impl DeviceTransport {
    pub fn upload(gpu: &Gpu, ts: &TransportSet) -> Result<Self> {
        let tf = flatten_transport(ts);
        Ok(Self {
            tf: gpu.upload(&tf)?,
            n_species: ts.species().len(),
        })
    }

    /// The species count, K.
    pub fn n_species(&self) -> usize {
        self.n_species
    }
}

/// The f64 scratch of §133.4's mixture kernel, SoA with stride
/// [`TransportWork::n_cells`]: the mole fractions and the two
/// pure-species property columns, K values per cell each. Everything
/// computes in double in both builds, so the scratch is f64 in both.
pub struct TransportWork {
    x: DevBuf<f64>,
    mu: DevBuf<f64>,
    lam: DevBuf<f64>,
    n_cells: usize,
}

impl TransportWork {
    pub fn new(gpu: &Gpu, dt: &DeviceTransport, n_cells: usize) -> Result<Self> {
        let k = dt.n_species();
        Ok(Self {
            x: gpu.zeros(k * n_cells)?,
            mu: gpu.zeros(k * n_cells)?,
            lam: gpu.zeros(k * n_cells)?,
            n_cells,
        })
    }

    /// The cell count the scratch was sized for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

/// The stated constants of the Lewis-number route on the device: the
/// per-species Lewis numbers and the Prandtl number the launches carry as
/// an argument.
pub struct DeviceLewis {
    le: DevBuf<f64>,
    pr: f64,
    n_species: usize,
}

impl DeviceLewis {
    pub fn upload(gpu: &Gpu, l: &Lewis) -> Result<Self> {
        Ok(Self {
            le: gpu.upload(&l.le)?,
            pr: l.pr,
            n_species: l.le.len(),
        })
    }

    /// The species count, K.
    pub fn n_species(&self) -> usize {
        self.n_species
    }
}

/// The device [`crate::chem::transport::TransportSet::mixture_ty`] (SPEC-LIT
/// §133.4): the Wilke mixture viscosity (133.8) with (133.10) and the
/// mixture conductivity (133.9) with (133.10) and (133.11), from the
/// TM-4513 eq. (10) fits, one thread per cell, in the host's operation
/// order and choice for choice. Layouts: `t[c]`, `y[k*n+c]`, `mu[c]`,
/// `lambda[c]`, `status[c]`. Per-cell checks in the host's order: T not
/// finite or not above 0 -> 2 (BAD_TEMPERATURE); any y negative or not
/// finite -> 3; the sum of Y/W NaN or not above 0 -> 4; a present
/// species' T outside its viscosity or conductivity range -> 10
/// (OUTSIDE_TRANSPORT_RANGE). A refused cell writes NaN to its mu and
/// lambda. `n == 0` launches nothing; no sync, no allocation (capturable).
#[allow(clippy::too_many_arguments)]
pub fn transport_mixture(
    gpu: &Gpu,
    k: &ChemKernels,
    dt: &DeviceTransport,
    work: &mut TransportWork,
    t: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    mu: &mut DevBuf<Scalar>,
    lambda: &mut DevBuf<Scalar>,
    status: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    let f = "chemMixtureTransport";
    let ks = dt.n_species();
    check_cells(f, n, work.n_cells())?;
    check_layout(f, "t", t.len(), n)?;
    check_layout(f, "y", y.len(), ks * n)?;
    check_layout(f, "mu", mu.len(), n)?;
    check_layout(f, "lambda", lambda.len(), n)?;
    check_layout(f, "status", status.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let kl = ks as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.transport)
            .arg(t)
            .arg(y)
            .arg(&mut *mu)
            .arg(&mut *lambda)
            .arg(&mut *status)
            .arg(&mut work.x)
            .arg(&mut work.mu)
            .arg(&mut work.lam)
            .arg(&dt.tf)
            .arg(&kl)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The Lewis route's conductivity (SPEC-LIT §133.4): `lambda[c] =
/// (mu cp)/Pr`, the device [`crate::chem::transport::prandtl_conductivity`]
/// in double in every build, rounded once on the way out. Layouts: `mu[c]`,
/// `cp[c]`, `lambda[c]`. `n == 0` launches nothing; no sync, no allocation
/// (capturable).
pub fn transport_prandtl(
    gpu: &Gpu,
    k: &ChemKernels,
    l: &DeviceLewis,
    mu: &DevBuf<Scalar>,
    cp: &DevBuf<Scalar>,
    lambda: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    let f = "chemPrandtlConductivity";
    check_layout(f, "mu", mu.len(), n)?;
    check_layout(f, "cp", cp.len(), n)?;
    check_layout(f, "lambda", lambda.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.prandtl)
            .arg(mu)
            .arg(cp)
            .arg(&mut *lambda)
            .arg(&l.pr)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The Lewis route's mixture diffusivity (SPEC-LIT §133.4):
/// `d[k*n+c] = lambda/((rho cp) le_k)`, the device
/// [`crate::chem::transport::lewis_diffusivity`] in double in every build,
/// rounded once on the way out. Layouts: `lambda[c]`, `rho[c]`, `cp[c]`,
/// `d[k*n+c]`. `n == 0` launches nothing; no sync, no allocation
/// (capturable).
#[allow(clippy::too_many_arguments)]
pub fn transport_lewis(
    gpu: &Gpu,
    k: &ChemKernels,
    l: &DeviceLewis,
    lambda: &DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    cp: &DevBuf<Scalar>,
    d: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    let f = "chemLewisDiffusivity";
    let ks = l.n_species();
    check_layout(f, "lambda", lambda.len(), n)?;
    check_layout(f, "rho", rho.len(), n)?;
    check_layout(f, "cp", cp.len(), n)?;
    check_layout(f, "d", d.len(), ks * n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let kl = ks as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.lewis)
            .arg(lambda)
            .arg(rho)
            .arg(cp)
            .arg(&mut *d)
            .arg(&l.le)
            .arg(&kl)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::capture_replays_bitwise;
    use crate::chem::transport::read_table_iv_csv;
    use crate::chem::{P_ATM, ReactorOptions, SpeciesThermo, ThermoSet};
    use crate::field_ops::FieldKernels;

    /// The ten TM-4513 transport records of reference/mcbride-gordon-reno1993/
    /// table_IV.csv (SPEC-LIT §133.4).
    const TABLE_IV: &str =
        include_str!("../../../reference/mcbride-gordon-reno1993/table_IV.csv");

    /// The process's device, if the box has one: every GPU test returns
    /// early rather than failing.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// The ten TM-4513 records of reference/mcbride-gordon-reno1993/
    /// table_II.csv (SPEC-LIT §128.6, §130.7).
    const TABLE_II: &str =
        include_str!("../../../reference/mcbride-gordon-reno1993/table_II.csv");

    /// The table's species codes: name and elements (§128.6's code list).
    #[allow(clippy::type_complexity)]
    const CODES: &[(&str, &str, &[(&str, u32)])] = &[
        ("1", "N2", &[("N", 2)]),
        ("2", "O2", &[("O", 2)]),
        ("5", "CH4", &[("C", 1), ("H", 4)]),
        ("6", "CO2", &[("C", 1), ("O", 2)]),
        ("7", "H2O", &[("H", 2), ("O", 1)]),
        ("8", "CO", &[("C", 1), ("O", 1)]),
        ("9", "H2", &[("H", 2)]),
        ("10", "OH", &[("O", 1), ("H", 1)]),
        ("11", "O", &[("O", 1)]),
        ("12", "H", &[("H", 1)]),
    ];

    /// The records as a set at `P_ATM`, as tests/chem_kinetics.rs's
    /// `tm_set`: bounds 200/1000/6000 K for all ten species.
    fn tm_set() -> ThermoSet {
        let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
        for line in TABLE_II.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
                continue;
            }
            let c: Vec<&str> = line.split(',').collect();
            rows.push((
                c[0].to_string(),
                c[1].parse().unwrap(),
                c[3].parse().unwrap(),
                [
                    c[4].parse().unwrap(),
                    c[5].parse().unwrap(),
                    c[6].parse().unwrap(),
                    c[7].parse().unwrap(),
                    c[8].parse().unwrap(),
                    c[9].parse().unwrap(),
                    c[10].parse().unwrap(),
                ],
            ));
        }
        let mut v = Vec::new();
        for (code, name, els) in CODES {
            let pick = |t_lo: f64| {
                rows.iter()
                    .find(|(c, t, _, _)| c == code && *t == t_lo)
                    .map(|r| (r.2, r.3))
                    .unwrap()
            };
            let (mol_weight, upper) = pick(1000.0);
            let (_, lower) = pick(200.0);
            v.push(SpeciesThermo {
                name: name.to_string(),
                elements: els.iter().map(|(s, n)| (s.to_string(), *n)).collect(),
                molar_mass: mol_weight / 1000.0,
                t_low: 200.0,
                t_mid: 1000.0,
                t_high: 6000.0,
                upper,
                lower,
                line: 0,
            });
        }
        ThermoSet::from_species(v, P_ATM, "TM-4513").unwrap()
    }

    /// A mechanism of the given reaction lines over the ten TM-4513
    /// species; the concentration order is H2 O2 H2O OH O H CO CO2 CH4 N2.
    fn mech(rx: &[&str]) -> Mechanism {
        let mut text: Vec<&str> = vec![
            "ELEMENTS H O C N END",
            "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END",
            "REACTIONS",
        ];
        text.extend_from_slice(rx);
        text.push("END");
        Mechanism::read_chemkin(&text.join("\n"), "kin-fixture", Some(&tm_set()))
            .unwrap_or_else(|e| panic!("the fixture does not read: {e}"))
    }

    /// The four reactions of the capture gate, out of §130.7's fixture: an
    /// elementary reversible one, a plain `+M` with efficiencies, a Troe
    /// fall-off, and an irreversible global with FORD.
    fn fixture() -> Mechanism {
        mech(&[
            "H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0",
            "H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0",
            "  H2O/6.0/ H2/2.0/",
            "CO + O (+M) <=> CO2 (+M)          2.0E+10   0.0   2500.0",
            "  LOW / 5.0E+14 0.0 3000.0 /",
            "  TROE / 0.6 100.0 1500.0 5000.0 /",
            "CH4 + 2 O2 => CO2 + 2 H2O         1.0E+12   0.0  30000.0",
            "  FORD /CH4 0.2/ FORD /O2 1.3/",
        ])
    }

    /// SPEC-LIT §132.9's and §132.10's capture row: the chemistry kernels -
    /// the rates, the Jacobian and the chemistry step's ofscalar crossing -
    /// capture into a CUDA graph and replay bitwise, the rates feeding the
    /// next iteration's T.
    #[test]
    fn chem_device_replays_under_capture() {
        let Some(g) = gpu() else { return };
        let m = fixture();
        let ks = m.species().len();
        let n = 64;
        // T_c = 900 + 10c, p = 101325, Y_k = (1 + (c + 3k) % 7) normalised
        // to sum 1.
        let mix = |c: usize| -> Vec<f64> {
            let s: f64 = (0..ks).map(|k| 1.0 + ((c + 3 * k) % 7) as f64).sum();
            (0..ks)
                .map(|k| (1.0 + ((c + 3 * k) % 7) as f64) / s)
                .collect()
        };
        let t0: Vec<Scalar> = (0..n).map(|c| (900.0 + 10.0 * c as f64) as Scalar).collect();
        let p0: Vec<Scalar> = vec![101325.0; n];
        let mut y0 = vec![0.0 as Scalar; ks * n];
        for c in 0..n {
            for (k, v) in mix(c).iter().enumerate() {
                y0[k * n + c] = *v as Scalar;
            }
        }
        // The Jacobian's fixed f64 state (Y_c, T_c), and its tolerances.
        let mut state0 = vec![0.0f64; (ks + 1) * n];
        for c in 0..n {
            for (k, v) in mix(c).iter().enumerate() {
                state0[k * n + c] = *v;
            }
            state0[ks * n + c] = 900.0 + 10.0 * c as f64;
        }
        let pj0 = vec![101325.0f64; n];
        let mut atol0 = vec![1.0e-14f64; ks];
        atol0.push(1.0e-8);

        // CHR-07b's chemistry step, on its own ofscalar fields: the same
        // mix(c) at T_c = 1200 + 10c, p = 101325, the first trial step
        // h = 1e-10, one flow step dt = 1e-6.
        let ti0: Vec<Scalar> = (0..n).map(|c| (1200.0 + 10.0 * c as f64) as Scalar).collect();
        let pi0: Vec<Scalar> = vec![101325.0; n];
        let mut yi0 = vec![0.0 as Scalar; ks * n];
        for c in 0..n {
            for (k, v) in mix(c).iter().enumerate() {
                yi0[k * n + c] = *v as Scalar;
            }
        }
        let hi0 = vec![1.0e-10f64; n];

        let report = capture_replays_bitwise(
            &g,
            "the chemistry device kernels (SPEC-LIT 132.9)",
            || {
                let ck = ChemKernels::new(&g)?;
                let fk = FieldKernels::new(&g)?;
                let dm = DeviceMechanism::upload(&g, &m)?;
                let work = ChemWork::new(&g, &dm, n)?;
                let iw = IntegrateWork::new(&g, &dm, n)?;
                let dr = DeviceReactor::new(&g, &ReactorOptions::default())?;
                // CHR-08's transport on the same ten species: the
                // table_IV records as a TransportSet, the Lewis route at
                // Pr 0.7 with Le 1 everywhere, and the constant rho and
                // cp fields its two extra kernels read.
                let tset = TransportSet::for_thermo(
                    &tm_set(),
                    &read_table_iv_csv(TABLE_IV, "table_IV.csv")?,
                )?;
                let dtr = DeviceTransport::upload(&g, &tset)?;
                let tw = TransportWork::new(&g, &dtr, n)?;
                let dlewis = DeviceLewis::upload(&g, &Lewis::new(0.7, vec![1.0; ks])?)?;
                let rho_t = g.upload(&vec![1.0 as Scalar; n])?;
                let cp_t = g.upload(&vec![1000.0 as Scalar; n])?;
                let mu_t = g.zeros::<Scalar>(n)?;
                let lam_t = g.zeros::<Scalar>(n)?;
                let lam_p = g.zeros::<Scalar>(n)?;
                let d_t = g.zeros::<Scalar>(ks * n)?;
                let st_t = g.zeros::<i32>(n)?;
                Ok((
                    ck, fk, dm, work,
                    g.upload(&t0)?,
                    g.upload(&p0)?,
                    g.upload(&y0)?,
                    g.zeros::<Scalar>(ks * n)?,
                    g.zeros::<i32>(n)?,
                    g.upload(&state0)?,
                    g.upload(&pj0)?,
                    g.upload(&atol0)?,
                    g.zeros::<f64>((ks + 1) * n)?,
                    g.zeros::<f64>((ks + 1) * (ks + 1) * n)?,
                    g.zeros::<i32>(n)?,
                    dr,
                    iw,
                    g.upload(&ti0)?,
                    g.upload(&pi0)?,
                    g.upload(&yi0)?,
                    g.upload(&hi0)?,
                    g.zeros::<i32>(n)?,
                    g.zeros::<i32>(3 * n)?,
                    dtr,
                    tw,
                    dlewis,
                    rho_t,
                    cp_t,
                    mu_t,
                    lam_t,
                    lam_p,
                    d_t,
                    st_t,
                ))
            },
            |(ck, fk, dm, work, t, p, y, omega, st_p, state, pj, atol, f0, jac, st_j,
              dr, iw, t2, p2, y2, h2, st2, cnt2,
              dtr, tw, dlewis, rho_t, cp_t, mu_t, lam_t, lam_p, d_t, st_t)| {
                production_rates(&g, ck, dm, work, t, p, y, omega, st_p, n)?;
                // The rates feed the next iteration's T, so a replay that
                // skipped the chemistry cannot pass.
                crate::field_ops::scale_field(&g, fk, omega, 1.0e-12, n)?;
                crate::field_ops::add_field(&g, fk, t, omega, n)?;
                reactor_jacobian(&g, ck, dm, work, state, pj, atol, f0, jac, st_j, n)?;
                integrate_fields(&g, ck, dm, work, iw, dr, t2, p2, y2, h2, 1.0e-6, st2, cnt2, n)?;
                // CHR-08's transport on the step's ofscalar fields, after
                // the chemistry: the Wilke mixture, then the Lewis route
                // from the device mu and lambda.
                transport_mixture(&g, ck, dtr, tw, t2, y2, mu_t, lam_t, st_t, n)?;
                transport_prandtl(&g, ck, dlewis, mu_t, cp_t, lam_p, n)?;
                transport_lewis(&g, ck, dlewis, lam_t, rho_t, cp_t, d_t, n)
            },
            |(_ck, _fk, _dm, _work, t, _p, _y, omega, st_p, _state, _pj, _atol, f0, jac, st_j,
              _dr, _iw, t2, _p2, y2, h2, st2, cnt2,
              _dtr, _tw, _dlewis, _rho_t, _cp_t, mu_t, lam_t, lam_p, d_t, st_t)| {
                Ok(vec![
                    ("t", g.download(t)?),
                    ("omega", g.download(omega)?),
                    ("production status", g.download(st_p)?.iter().map(|&v| v as Scalar).collect()),
                    ("jacobian status", g.download(st_j)?.iter().map(|&v| v as Scalar).collect()),
                    ("f0", g.download(f0)?.iter().map(|&v| v as Scalar).collect()),
                    ("jac", g.download(jac)?.iter().map(|&v| v as Scalar).collect()),
                    ("integrate t", g.download(t2)?),
                    ("integrate y", g.download(y2)?),
                    ("integrate h", g.download(h2)?.iter().map(|&v| v as Scalar).collect()),
                    ("integrate status", g.download(st2)?.iter().map(|&v| v as Scalar).collect()),
                    ("integrate counts", g.download(cnt2)?.iter().map(|&v| v as Scalar).collect()),
                    ("transport mu", g.download(mu_t)?),
                    ("transport lambda", g.download(lam_t)?),
                    ("prandtl lambda", g.download(lam_p)?),
                    ("lewis d", g.download(d_t)?),
                    ("transport status", g.download(st_t)?.iter().map(|&v| v as Scalar).collect()),
                ])
            },
        )
        .expect("SPEC-LIT 132.9: the chemistry kernels must capture and replay bitwise");
        println!("  chemistry device kernels: {report}");
    }
}
