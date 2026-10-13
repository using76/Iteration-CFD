// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-07b's gates: the device chemistry step of SPEC-LIT §132.10
//! (`cuda/chem.cu`'s chemIntegrate with the pack/unpack ofscalar crossing,
//! `src/chem/device.rs`'s integrate and integrate_fields) against the f64
//! host twin `ConstPressureReactor::advance` (§132.4). 4096 random cells
//! of the §130.7 generator over the 15-reaction TM-4513 reactor fixture of
//! CHR-06; GRI-Mech 3.0 by path (§129.6, skip by name); the f32 build's
//! Scalar crossing against the f64 build; and the per-cell refusals.
//! Options everywhere `ReactorOptions::default()` (RODAS3, rtol 1e-8,
//! atol_y 1e-14, atol_t 1e-8, h0 1e-10).
//!
//! An integration test, so it runs in both builds: the chemistry computes
//! in double in both (§132.6); the f32 build's fields cross as Scalar.
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use std::env;
use std::fs;
use std::time::Instant;

use ofgpu::chem::device::{
    integrate, integrate_fields, status, ChemKernels, ChemWork, DeviceMechanism, DeviceReactor,
    IntegrateWork,
};
use ofgpu::chem::reactor::{ConstPressureReactor, ReactorOptions};
use ofgpu::chem::{ElementTable, Mechanism, SpeciesThermo, ThermoSet, P_BAR};
use ofgpu::device::Gpu;
use ofgpu::Scalar;

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// SplitMix64, seed passed per test; `u()` is uniform on [0, 1) as
/// (x >> 11) 2^-53.
struct SplitMix64(u64);

impl SplitMix64 {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The flow step every gate integrates over, s - the flow step CHR-09's
/// Lie split uses, because at 1e-4 s the host's own round-off conditioning
/// exceeds the T row (SPEC-LIT §132.10).
const DT: f64 = 1.0e-6;

/// The GRI timing run's flow step, s: a hundredth of `DT`, keeping the
/// latency-bound, one-thread-per-cell launch inside the wall budget.
const DT_TIMING: f64 = 1.0e-8;

/// The first trial step of every cell, s (advance's h0).
const H0: f64 = 1.0e-10;

/// The ten TM-4513 records of `reference/mcbride-gordon-reno1993/
/// table_II.csv` (SPEC-LIT §128.6, §130.7).
const TABLE_II: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_II.csv");

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

/// The records as a set at `P_BAR`: TM-4513 tabulates at 1 bar (SPEC-LIT
/// §128.4), and the mechanism reader moves the set to `P_ATM` itself. The
/// bounds are 200/1000/6000 K for all ten species.
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
    ThermoSet::from_species(v, P_BAR, "TM-4513").unwrap()
}

/// A mechanism of the given reaction lines over the ten TM-4513 species;
/// the concentration order is H2 O2 H2O OH O H CO CO2 CH4 N2 (0..9).
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

/// The 15-reaction reactor fixture of CHR-06: every reaction reversible,
/// the reverse from K_c, so the ODE's rest point is the Gibbs equilibrium
/// of §131. Its rate numbers are placeholders, not any paper's.
fn fixture() -> Mechanism {
    mech(&[
        "H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0",
        "O + H2 <=> H + OH                 1.0E+05   2.5   6000.0",
        "OH + H2 <=> H + H2O               2.0E+08   1.5   3000.0",
        "H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0",
        "  H2O/6.0/ H2/2.0/",
        "O + O + M <=> O2 + M              1.0E+17  -1.0      0.0",
        "H + H (+M) <=> H2 (+M)            1.0E+13   0.0      0.0",
        "  LOW / 5.0E+18 -1.0 0.0 /",
        "  H2O/5.0/",
        "CO + O (+M) <=> CO2 (+M)          2.0E+10   0.0   2500.0",
        "  LOW / 5.0E+14 0.0 3000.0 /",
        "  TROE / 0.6 100.0 1500.0 5000.0 /",
        "O + H2 (+N2) <=> H2O (+N2)        1.0E+12   0.0   1000.0",
        "  LOW / 1.0E+18 -1.0 0.0 /",
        "  TROE / 0.7 200.0 1000.0 /",
        "H + O (+M) <=> OH (+M)            5.0E+12   0.0      0.0",
        "  LOW / 4.0E+18 -1.0 0.0 /",
        "  SRI / 0.5 800.0 200.0 1.2 0.1 /",
        "CO + OH <=> CO2 + H               4.0E+06   1.5   -500.0",
        "OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0",
        "  DUP",
        "OH + OH <=> O + H2O               1.0E+12   0.0   1000.0",
        "  DUP",
        "H2 + O2 <=> OH + OH               1.7E+13   0.0  47780.0",
        "CH4 + O2 <=> CO + H2O + H2        1.0E+12   0.0  30000.0",
        "CH4 + O <=> CO + 2 H2             1.0E+13   0.0   8000.0",
    ])
}

/// Mole fractions to mass fractions against `set`: y_j = x_j W_j / sum x W.
fn y_from_x(set: &ThermoSet, x: &[f64]) -> Vec<f64> {
    let xw: Vec<f64> = x
        .iter()
        .zip(set.species())
        .map(|(xj, s)| xj * s.molar_mass)
        .collect();
    let s: f64 = xw.iter().sum();
    xw.iter().map(|v| v / s).collect()
}

/// The env pair of SPEC-LIT §129.6, or a skip by name: the missing
/// variable prints SKIPPED with the reason and the test returns, unless
/// `OFGPU_REQUIRE_MECH=1` makes it a failure.
fn env_pair(mech_var: &str, thermo_var: &str, test: &str) -> Option<(String, String)> {
    match (env::var(mech_var), env::var(thermo_var)) {
        (Ok(m), Ok(t)) => Some((m, t)),
        (mech, _) => {
            let var = if mech.is_err() { mech_var } else { thermo_var };
            let why = format!(
                "SKIPPED {test}: {var} is not set; GRI-Mech 3.0 is user-supplied by path \
                 (SPEC-LIT §129.6)"
            );
            if env::var("OFGPU_REQUIRE_MECH").as_deref() == Ok("1") {
                panic!("{why}");
            }
            eprintln!("{why}");
            None
        }
    }
}

/// Reads the GRI pair - as `chem_reactor.rs`'s `read_pair`: the thermo
/// file as its own set at `P_ATM`, then the mechanism with it.
fn read_pair(mech: &str, thermo: &str) -> Mechanism {
    let t = fs::read_to_string(thermo)
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let set = ThermoSet::read_chemkin(&t, thermo, ofgpu::chem::P_ATM, &ElementTable::standard())
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let m = fs::read_to_string(mech)
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"));
    Mechanism::read_chemkin(&m, mech, Some(&set))
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"))
}

/// Cell c's initial state (SPEC-LIT §132.10's generator, SplitMix64 seed
/// 20261013): the four draws phi = 0.6 + 0.8u, T0 = 1000 + 1000u,
/// p = 10^(4 + 2u), r = 10^(-6 + 3u); then the mole vector - even c burns
/// H2-air (2 phi : 1 : 3.76), odd c CH4-air (phi : 2 : 7.52) - with H, O,
/// OH each r, exactly 0 at every 10th cell (r is drawn there anyway), as
/// mass fractions against `set`. Returns (T0, p, Y).
fn cell_state(set: &ThermoSet, c: usize, rng: &mut SplitMix64) -> (f64, f64, Vec<f64>) {
    let phi = 0.6 + 0.8 * rng.u();
    let t0 = 1000.0 + 1000.0 * rng.u();
    let p = 10f64.powf(4.0 + 2.0 * rng.u());
    let r = 10f64.powf(-6.0 + 3.0 * rng.u());
    let idx = |name: &str| {
        set.index_of(name)
            .unwrap_or_else(|| panic!("the set has no species {name}"))
    };
    let mut x = vec![0.0; set.species().len()];
    if c % 2 == 0 {
        x[idx("H2")] = 2.0 * phi;
        x[idx("O2")] = 1.0;
        x[idx("N2")] = 3.76;
    } else {
        x[idx("CH4")] = phi;
        x[idx("O2")] = 2.0;
        x[idx("N2")] = 7.52;
    }
    let trace = if c % 10 == 0 { 0.0 } else { r };
    x[idx("H")] = trace;
    x[idx("O")] = trace;
    x[idx("OH")] = trace;
    (t0, p, y_from_x(set, &x))
}

/// The generator over n cells: (state, p, h) arrays in the SoA layout
/// `integrate` takes, h every cell's first trial step H0.
fn generator(rng: &mut SplitMix64, set: &ThermoSet, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ks = set.species().len();
    let mut state = vec![0.0f64; (ks + 1) * n];
    let mut ps = vec![0.0f64; n];
    let hs = vec![H0; n];
    for c in 0..n {
        let (t0, p, y0) = cell_state(set, c, rng);
        ps[c] = p;
        for (k, v) in y0.iter().enumerate() {
            state[k * n + c] = *v;
        }
        state[ks * n + c] = t0;
    }
    (state, ps, hs)
}

/// One host advance's results, what the gate reads.
struct HostRun {
    t: f64,
    y: Vec<f64>,
    accepted: usize,
    rejected: usize,
    clipped: usize,
    h_next: f64,
    /// The twin's refusal text, if `advance` refused. On GRI-Mech 3.0 an
    /// igniting cell's `T` can cross a species' thermo range ceiling, which
    /// is a real R5 refusal both sides must take; the device must then
    /// refuse with the same status and NaN outputs, and the state gate does
    /// not apply to the cell.
    refusal: Option<String>,
}

/// The host twin of one cell: `advance` over `dt` from `(y0, t0)` with
/// `h0 = H0`.
fn host_cell(m: &Mechanism, p: f64, y0: &[f64], t0: f64, dt: f64) -> HostRun {
    let opts = ReactorOptions {
        h0: H0,
        ..ReactorOptions::default()
    };
    let r = ConstPressureReactor::new(m, p).expect("reactor pressure");
    match r.advance(y0, t0, dt, &opts) {
        Ok(rep) => HostRun {
            t: rep.temperature,
            y: rep.y,
            accepted: rep.stats.accepted,
            rejected: rep.stats.rejected,
            clipped: rep.clipped,
            h_next: rep.stats.h_next,
            refusal: None,
        },
        Err(e) => HostRun {
            t: f64::NAN,
            y: Vec::new(),
            accepted: 0,
            rejected: 0,
            clipped: 0,
            h_next: 0.0,
            refusal: Some(e.to_string()),
        },
    }
}

/// One device run's downloaded outputs.
struct DevOut {
    state: Vec<f64>,
    h: Vec<f64>,
    status: Vec<i32>,
    /// 3n: accepted, rejected, clipped.
    counts: Vec<i32>,
}

/// The device run over n cells: one chemIntegrate launch, synced, the
/// outputs downloaded. Returns them and the launch+sync wall seconds.
fn device_integrate(
    g: &Gpu,
    m: &Mechanism,
    dt: f64,
    dr: &DeviceReactor,
    state0: &[f64],
    ps: &[f64],
    hs: &[f64],
    n: usize,
) -> (DevOut, f64) {
    let ck = ChemKernels::new(g).expect("kernels");
    let dm = DeviceMechanism::upload(g, m).expect("upload");
    let mut work = ChemWork::new(g, &dm, n).expect("work");
    let mut iw = IntegrateWork::new(g, &dm, n).expect("integrate work");
    let mut state = g.upload(state0).expect("upload");
    let p = g.upload(ps).expect("upload");
    let mut h = g.upload(hs).expect("upload");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    let mut cnt = g.zeros::<i32>(3 * n).expect("alloc");
    let t0 = Instant::now();
    integrate(
        g, &ck, &dm, &mut work, &mut iw, dr, &mut state, &p, &mut h, dt, &mut st, &mut cnt, n,
    )
    .expect("launch");
    g.sync().expect("sync");
    let secs = t0.elapsed().as_secs_f64();
    (
        DevOut {
            state: g.download(&state).expect("download"),
            h: g.download(&h).expect("download"),
            status: g.download(&st).expect("download"),
            counts: g.download(&cnt).expect("download"),
        },
        secs,
    )
}

/// One device ofscalar crossing's downloaded outputs.
struct FieldsOut {
    t: Vec<Scalar>,
    y: Vec<Scalar>,
    h: Vec<f64>,
    status: Vec<i32>,
    /// 3n: accepted, rejected, clipped.
    counts: Vec<i32>,
}

/// The ofscalar crossing over n cells: pack, chemIntegrate, unpack, synced.
/// Returns the rounded fields and the launch+sync wall seconds.
fn device_fields(
    g: &Gpu,
    m: &Mechanism,
    dt: f64,
    dr: &DeviceReactor,
    t_s: &[Scalar],
    p_s: &[Scalar],
    y_s: &[Scalar],
    hs: &[f64],
    n: usize,
) -> (FieldsOut, f64) {
    let ck = ChemKernels::new(g).expect("kernels");
    let dm = DeviceMechanism::upload(g, m).expect("upload");
    let mut work = ChemWork::new(g, &dm, n).expect("work");
    let mut iw = IntegrateWork::new(g, &dm, n).expect("integrate work");
    let mut t = g.upload(t_s).expect("upload");
    let p = g.upload(p_s).expect("upload");
    let mut y = g.upload(y_s).expect("upload");
    let mut h = g.upload(hs).expect("upload");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    let mut cnt = g.zeros::<i32>(3 * n).expect("alloc");
    let t0 = Instant::now();
    integrate_fields(
        g, &ck, &dm, &mut work, &mut iw, dr, &mut t, &p, &mut y, &mut h, dt, &mut st, &mut cnt, n,
    )
    .expect("launch");
    g.sync().expect("sync");
    let secs = t0.elapsed().as_secs_f64();
    (
        FieldsOut {
            t: g.download(&t).expect("download"),
            y: g.download(&y).expect("download"),
            h: g.download(&h).expect("download"),
            status: g.download(&st).expect("download"),
            counts: g.download(&cnt).expect("download"),
        },
        secs,
    )
}

/// The class P gate (SPEC-LIT §132.10): every device status 0 where the
/// host twin ran, max |Y_dev - Y_host| <= 1e-10 and |T_dev - T_host|
/// <= 1e-9 K in every such cell; the counters, the h_next differences and
/// the device throughput printed, not asserted. A cell whose host refused
/// at the thermo range is checked as the refusal contract states instead:
/// the device refuses with status 5 and NaN outputs. `moved_floor` is the
/// non-vacuity assert: at least that many cells whose host end state moved
/// some species by more than 1e-6 from Y0. `assert_t` gates the T row at
/// 1e-9 K; with it false the T row is only reported - a cell that ignites
/// inside the step is below the round-off conditioning of the FD-Jacobian
/// integrator (§132.10).
fn gate(
    label: &str,
    set: &ThermoSet,
    state0: &[f64],
    ks: usize,
    n: usize,
    host: &[HostRun],
    out: &DevOut,
    dev_secs: f64,
    moved_floor: Option<usize>,
    assert_t: bool,
) {
    let (mut worst_y, mut wyc, mut wyk) = (0.0f64, 0usize, 0usize);
    let (mut worst_t, mut wtc, mut wtc_dt0) = (0.0f64, 0usize, 0.0f64);
    let (mut cmis, mut worst_h) = (0usize, 0.0f64);
    let (mut accepted, mut moved, mut refused) = (0i64, 0usize, 0usize);
    let (mut worst_dy0, mut worst_dt0) = (0.0f64, 0.0f64);
    for c in 0..n {
        if let Some(text) = &host[c].refusal {
            refused += 1;
            assert!(
                text.contains("lies outside its range"),
                "{label}: cell {c}: unexpected host refusal: {text}"
            );
            assert_eq!(
                out.status[c],
                status::OUTSIDE_THERMO_RANGE,
                "{label}: cell {c}: the device did not refuse as the host did"
            );
            for l in 0..ks + 1 {
                assert!(
                    out.state[l * n + c].is_nan(),
                    "{label}: cell {c}: a refused cell's state is not NaN"
                );
            }
            assert!(
                out.h[c].is_nan(),
                "{label}: cell {c}: a refused cell's h is not NaN"
            );
            continue;
        }
        assert_eq!(out.status[c], status::OK, "{label}: cell {c} refused");
        let h = &host[c];
        let mut dy0 = 0.0f64;
        for k in 0..ks {
            dy0 = dy0.max((h.y[k] - state0[k * n + c]).abs());
            let d = (out.state[k * n + c] - h.y[k]).abs();
            if d > worst_y {
                worst_y = d;
                wyc = c;
                wyk = k;
            }
        }
        if dy0 > 1.0e-6 {
            moved += 1;
        }
        worst_dy0 = worst_dy0.max(dy0);
        worst_dt0 = worst_dt0.max((h.t - state0[ks * n + c]).abs());
        let d_t = (out.state[ks * n + c] - h.t).abs();
        if d_t > worst_t {
            worst_t = d_t;
            wtc = c;
            wtc_dt0 = (h.t - state0[ks * n + c]).abs();
        }
        if out.counts[c] as usize != h.accepted
            || out.counts[n + c] as usize != h.rejected
            || out.counts[2 * n + c] as usize != h.clipped
        {
            cmis += 1;
        }
        if h.h_next > 0.0 {
            worst_h = worst_h.max((out.h[c] - h.h_next).abs() / h.h_next);
        }
        accepted += out.counts[c] as i64;
    }
    eprintln!(
        "{label}: worst dY {worst_y:.3e} (cell {wyc}, species {}), worst dT {worst_t:.3e} K \
         (cell {wtc})",
        set.species()[wyk].name
    );
    eprintln!(
        "{label}: {moved} cell(s) whose host state moved some species by more than 1e-6; \
         largest max_k |Y_host - Y0| {worst_dy0:.3e}; largest |T_host - T0| \
         {worst_dt0:.3e} K"
    );
    eprintln!(
        "{label}: {cmis} cell(s) whose (accepted, rejected, clipped) differ from the host; \
         worst relative h_next difference {worst_h:.3e}; {refused} cell(s) the host refused \
         at the thermo range and the device refused with them"
    );
    eprintln!(
        "{label}: device wall {dev_secs:.3} s, cell-steps/s {:.3e}, accepted substeps/s {:.3e}",
        n as f64 / dev_secs,
        accepted as f64 / dev_secs
    );
    assert!(worst_y <= 1.0e-10, "{label}: worst dY {worst_y:.3e} > 1e-10");
    if assert_t {
        assert!(worst_t <= 1.0e-9, "{label}: worst dT {worst_t:.3e} K > 1e-9 K");
    } else if worst_t <= 1.0e-9 {
        eprintln!(
            "{label}: T row: within 1e-9 K (worst dT {worst_t:.3e} K at cell {wtc}, \
             that cell's |T_host - T0| {wtc_dt0:.3e} K)"
        );
    } else {
        eprintln!(
            "{label}: T row: MISS, reported (SPEC-LIT §132.10): worst dT {worst_t:.3e} K \
             at cell {wtc}, that cell's |T_host - T0| {wtc_dt0:.3e} K"
        );
    }
    if let Some(floor) = moved_floor {
        assert!(
            moved >= floor,
            "{label}: only {moved} cell(s) moved a species by more than 1e-6, the gate \
             needs {floor}"
        );
    }
}

/// §132.10's class P gate on the 15-reaction fixture: 4096 random cells of
/// the §130.7 generator, one dt on the device and on the host twin, the
/// same step decisions and the same end states.
#[test]
fn chem_device_integrate_matches_host() {
    let Some(g) = gpu() else { return };
    let m = fixture();
    let set = m.thermo();
    let ks = set.species().len();
    let n = 4096;
    let mut rng = SplitMix64(20261013);
    let (state0, ps, hs) = generator(&mut rng, set, n);
    let host_t0 = Instant::now();
    let mut host = Vec::with_capacity(n);
    for c in 0..n {
        let y0: Vec<f64> = (0..ks).map(|k| state0[k * n + c]).collect();
        host.push(host_cell(&m, ps[c], &y0, state0[ks * n + c], DT));
    }
    let host_secs = host_t0.elapsed().as_secs_f64();
    eprintln!("fixture: host wall {host_secs:.3} s");
    assert!(
        host.iter().all(|h| h.refusal.is_none()),
        "fixture: the host refused a cell at the flow step, but the fixture's \
         thermo range covers every state"
    );
    let dr = DeviceReactor::new(&g, &ReactorOptions::default()).expect("reactor");
    let (out, dev_secs) = device_integrate(&g, &m, DT, &dr, &state0, &ps, &hs, n);
    gate("fixture", set, &state0, ks, n, &host, &out, dev_secs, Some(1000), true);
}

/// The GRI-Mech 3.0 row of §132.10 (SPEC-LIT §129.6: user-supplied by
/// path), supplementary - the plan's gate is the fixture's 4096-cell row.
/// Comparison: 8 generator cells at dt = 1e-7 s, the Y row and the refusal
/// contract gated, the T row reported, not asserted: a cell that ignites
/// inside the step is below the round-off conditioning of the FD-Jacobian
/// integrator (§132.10). Then a device-only timing run on 1024 cells at
/// `DT_TIMING`, printing cell-steps/s and accepted substeps/s.
#[test]
fn chem_device_integrate_matches_host_gri30() {
    let Some((mech_path, thermo_path)) = env_pair(
        "OFGPU_GRI30_MECH",
        "OFGPU_GRI30_THERMO",
        "chem_device_integrate_matches_host_gri30",
    ) else {
        return;
    };
    let Some(g) = gpu() else { return };
    let m = read_pair(&mech_path, &thermo_path);
    let set = m.thermo();
    let ks = set.species().len();
    let n = 8;
    let dt = 1.0e-7;
    let mut rng = SplitMix64(20261013);
    let (state0, ps, hs) = generator(&mut rng, set, n);
    let host_t0 = Instant::now();
    let mut host = Vec::with_capacity(n);
    for c in 0..n {
        let y0: Vec<f64> = (0..ks).map(|k| state0[k * n + c]).collect();
        host.push(host_cell(&m, ps[c], &y0, state0[ks * n + c], dt));
    }
    let host_secs = host_t0.elapsed().as_secs_f64();
    eprintln!("GRI-30 comparison: {n} cells at dt = {dt:e} s, host wall {host_secs:.3} s");
    let dr = DeviceReactor::new(&g, &ReactorOptions::default()).expect("reactor");
    let (out, dev_secs) = device_integrate(&g, &m, dt, &dr, &state0, &ps, &hs, n);
    gate("GRI-30", set, &state0, ks, n, &host, &out, dev_secs, None, false);
    let n2 = 1024;
    let mut rng2 = SplitMix64(20261013);
    let (st2, ps2, hs2) = generator(&mut rng2, set, n2);
    let (out2, secs2) = device_integrate(&g, &m, DT_TIMING, &dr, &st2, &ps2, &hs2, n2);
    let steps2: i64 = out2.counts[..n2].iter().map(|&v| v as i64).sum();
    eprintln!(
        "GRI-30 timing: {n2} cells over dt = {DT_TIMING:e} s in {secs2:.3} s, \
         {:.3e} cell-steps/s, {:.3e} accepted substeps/s",
        n2 as f64 / secs2,
        steps2 as f64 / secs2
    );
}

/// The f32 crossing (SPEC-LIT §132.6, §132.10): the first 512 cells with
/// t, p, y as Scalar (`v as Scalar`, so rounded under single). (D)
/// integrate_fields equals integrate run on the widened rounded inputs,
/// then rounded once - bitwise, status and counts included. (R) the same
/// run against the host at the ROUNDED inputs, 1e-5 in Y and T. The worst
/// deviation from the host at the UNROUNDED inputs is reported, not
/// asserted.
#[test]
fn chem_f32_build_matches_f64() {
    let Some(g) = gpu() else { return };
    let m = fixture();
    let set = m.thermo();
    let ks = set.species().len();
    let m1 = ks + 1;
    let n = 512;
    let mut rng = SplitMix64(20261013);
    let (state0, ps, hs) = generator(&mut rng, set, n);
    let t_s: Vec<Scalar> = (0..n).map(|c| state0[ks * n + c] as Scalar).collect();
    let p_s: Vec<Scalar> = (0..n).map(|c| ps[c] as Scalar).collect();
    let mut y_s = vec![0.0 as Scalar; ks * n];
    for c in 0..n {
        for k in 0..ks {
            y_s[k * n + c] = state0[k * n + c] as Scalar;
        }
    }
    let dr = DeviceReactor::new(&g, &ReactorOptions::default()).expect("reactor");
    // (D) integrate on the widened rounded inputs, then the crossing.
    let t_r: Vec<f64> = t_s.iter().map(|&v| v as f64).collect();
    let p_r: Vec<f64> = p_s.iter().map(|&v| v as f64).collect();
    let mut state_r = vec![0.0f64; m1 * n];
    for c in 0..n {
        for k in 0..ks {
            state_r[k * n + c] = y_s[k * n + c] as f64;
        }
        state_r[ks * n + c] = t_r[c];
    }
    let (int_out, _) = device_integrate(&g, &m, DT, &dr, &state_r, &p_r, &hs, n);
    let (f_out, _) = device_fields(&g, &m, DT, &dr, &t_s, &p_s, &y_s, &hs, n);
    for c in 0..n {
        assert_eq!(int_out.status[c], f_out.status[c], "cell {c}: status");
        for (off, name) in [(0usize, "accepted"), (n, "rejected"), (2 * n, "clipped")] {
            assert_eq!(
                int_out.counts[off + c], f_out.counts[off + c],
                "cell {c}: {name}"
            );
        }
        assert_eq!(
            int_out.h[c].to_bits(),
            f_out.h[c].to_bits(),
            "cell {c}: h"
        );
        for k in 0..ks {
            let got = f_out.y[k * n + c] as f64;
            let want = (int_out.state[k * n + c] as Scalar) as f64;
            assert_eq!(got.to_bits(), want.to_bits(), "cell {c} Y[{k}]: the rounding");
        }
        let got = f_out.t[c] as f64;
        let want = (int_out.state[ks * n + c] as Scalar) as f64;
        assert_eq!(got.to_bits(), want.to_bits(), "cell {c} T: the rounding");
    }
    // (R) the host at the ROUNDED inputs; the unrounded host, reported.
    let (mut worst_yr, mut worst_tr) = (0.0f64, 0.0f64);
    let (mut worst_yu, mut worst_tu) = (0.0f64, 0.0f64);
    for c in 0..n {
        let y_r: Vec<f64> = (0..ks).map(|k| state0[k * n + c] as Scalar as f64).collect();
        let t_r = state0[ks * n + c] as Scalar as f64;
        let p_r = ps[c] as Scalar as f64;
        let hr = host_cell(&m, p_r, &y_r, t_r, DT);
        for k in 0..ks {
            worst_yr = worst_yr.max((f_out.y[k * n + c] as f64 - hr.y[k]).abs());
        }
        worst_tr = worst_tr.max((f_out.t[c] as f64 - hr.t).abs() / hr.t);
        let y_u: Vec<f64> = (0..ks).map(|k| state0[k * n + c]).collect();
        let hu = host_cell(&m, ps[c], &y_u, state0[ks * n + c], DT);
        for k in 0..ks {
            worst_yu = worst_yu.max((f_out.y[k * n + c] as f64 - hu.y[k]).abs());
        }
        worst_tu = worst_tu.max((f_out.t[c] as f64 - hu.t).abs());
    }
    eprintln!(
        "f32 crossing vs host at the ROUNDED inputs: worst dY {worst_yr:.3e}, \
         worst relative dT {worst_tr:.3e}; at the UNROUNDED inputs (reported): \
         worst dY {worst_yu:.3e}, worst dT {worst_tu:.3e} K"
    );
    assert!(worst_yr <= 1.0e-5, "worst dY {worst_yr:.3e} > 1e-5");
    assert!(worst_tr <= 1.0e-5, "worst relative dT {worst_tr:.3e} > 1e-5");
}

/// §132.10's statuses: the input checks in their order (1 p, 3 Y, 2 T,
/// 9 h), a T below the thermo range (5) inside the run, more attempts than
/// max_steps (7); a refused cell's state and h NaN with a good neighbour
/// unaffected; and the Rust-side refusals naming the field and ending
/// "(SPEC-LIT §132.10)".
#[test]
fn chem_device_integrate_refuses_by_status() {
    let Some(g) = gpu() else { return };
    let m = fixture();
    let set = m.thermo();
    let ks = set.species().len();
    let m1 = ks + 1;
    // Generator cell 1's state: cell 0 is drawn and discarded.
    let mut rng = SplitMix64(20261013);
    let _ = cell_state(set, 0, &mut rng);
    let (t1, p1, y1) = cell_state(set, 1, &mut rng);
    // Launch A: 7 cells built from it, each refused cell then a good one.
    let n = 7;
    let mut state0 = vec![0.0f64; m1 * n];
    let mut ps = vec![p1; n];
    let mut hs = vec![H0; n];
    for c in 0..n {
        for (k, v) in y1.iter().enumerate() {
            state0[k * n + c] = *v;
        }
        state0[ks * n + c] = t1;
    }
    ps[0] = 0.0; // cell 0: p = 0 -> 1
    state0[1] = -1.0e-3; // cell 1: Y_0 = -1e-3 -> 3
    state0[ks * n + 2] = f64::NAN; // cell 2: T = NaN -> 2
    hs[3] = 0.0; // cell 3: h_in = 0 -> 9
    state0[ks * n + 4] = 150.0; // cell 4: below the thermo range -> 5
    let dr = DeviceReactor::new(&g, &ReactorOptions::default()).expect("reactor");
    let (out, _) = device_integrate(&g, &m, DT, &dr, &state0, &ps, &hs, n);
    let want = [status::BAD_PRESSURE, status::BAD_MASS_FRACTION, status::BAD_TEMPERATURE,
        status::BAD_STEP, status::OUTSIDE_THERMO_RANGE, status::OK, status::OK];
    for (c, w) in want.iter().enumerate() {
        assert_eq!(out.status[c], *w, "launch A cell {c}");
    }
    // A refused cell writes NaN to its whole state column and to h[c].
    for c in 0..5 {
        for i in 0..m1 {
            assert!(out.state[i * n + c].is_nan(), "cell {c} row {i}: not NaN");
        }
        assert!(out.h[c].is_nan(), "cell {c}: h not NaN");
    }
    // A good neighbour is unaffected, and within the gate of the host.
    for c in 5..n {
        for i in 0..m1 {
            assert!(
                out.state[i * n + c].is_finite(),
                "cell {c} row {i}: a good cell is finite"
            );
        }
        let y0: Vec<f64> = (0..ks).map(|k| state0[k * n + c]).collect();
        let hh = host_cell(&m, ps[c], &y0, state0[ks * n + c], DT);
        for k in 0..ks {
            assert!(
                (out.state[k * n + c] - hh.y[k]).abs() <= 1.0e-10,
                "cell {c} species {k}: outside the gate of the host"
            );
        }
        assert!(
            (out.state[ks * n + c] - hh.t).abs() <= 1.0e-9,
            "cell {c}: T outside the gate of the host"
        );
    }
    // Launch B: one cell, max_steps = 1 at the flow step -> 7, since
    // h0 = 1e-10 cannot cover dt in one step.
    let opts = ReactorOptions {
        max_steps: 1,
        ..ReactorOptions::default()
    };
    let drb = DeviceReactor::new(&g, &opts).expect("reactor");
    let mut state_b = y1.clone();
    state_b.push(t1);
    let (out_b, _) = device_integrate(&g, &m, DT, &drb, &state_b, &[p1], &[H0], 1);
    assert_eq!(out_b.status[0], status::MAX_STEPS, "launch B");
    for i in 0..m1 {
        assert!(out_b.state[i].is_nan(), "launch B row {i}: not NaN");
    }
    assert!(out_b.h[0].is_nan(), "launch B: h not NaN");
    // DeviceReactor::new's refusals, each naming its field.
    for (why, opts) in [
        ("rtol", ReactorOptions { rtol: -1.0, ..ReactorOptions::default() }),
        ("atol_y", ReactorOptions { atol_y: 0.0, ..ReactorOptions::default() }),
        ("h_max", ReactorOptions { h_max: f64::NAN, ..ReactorOptions::default() }),
        ("max_steps", ReactorOptions { max_steps: 0, ..ReactorOptions::default() }),
    ] {
        let e = DeviceReactor::new(&g, &opts).unwrap_err().to_string();
        eprintln!("{why}: {e}");
        assert!(e.contains("(SPEC-LIT §132.10)"), "{why}: no tag in {e}");
        assert!(e.contains(why), "{why}: the field is not named in {e}");
    }
    // integrate's dt refusals, before anything launches.
    let ck = ChemKernels::new(&g).expect("kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let mut work = ChemWork::new(&g, &dm, 1).expect("work");
    let mut iw = IntegrateWork::new(&g, &dm, 1).expect("integrate work");
    let mut st1 = g.upload(&state0[..m1]).expect("upload");
    let pbuf = g.upload(&ps[..1]).expect("upload");
    let mut hbuf = g.upload(&hs[..1]).expect("upload");
    let mut stt = g.zeros::<i32>(1).expect("alloc");
    let mut cnt = g.zeros::<i32>(3).expect("alloc");
    for bad in [-1.0f64, f64::NAN] {
        let e = integrate(
            &g, &ck, &dm, &mut work, &mut iw, &dr, &mut st1, &pbuf, &mut hbuf, bad,
            &mut stt, &mut cnt, 1,
        )
        .unwrap_err()
        .to_string();
        eprintln!("dt = {bad}: {e}");
        assert!(e.contains("(SPEC-LIT §132.10)"), "dt = {bad}: no tag in {e}");
        assert!(e.contains("dt"), "dt = {bad}: the field is not named in {e}");
    }
}


