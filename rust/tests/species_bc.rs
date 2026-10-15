// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-29's gates (SPEC-LIT §120.15): the species boundary conditions
//! `semiPermeableBaffleMassFraction` and `specieTransfer` as §4 triple
//! rewrites on the solved members of a species set.
//!
//! The rig is a 16 x 1 x 1 duct of unit length whose internal face between
//! cells 7 and 8 is removed and re-added as two coincident wall faces,
//! `baffleA` (+x out of cell 7) and `baffleB` (-x out of cell 8); y and z are
//! empty. The reference is the film law of Bird, Stewart & Lightfoot,
//! *Transport Phenomena*, 2nd ed., section 22.1 - a flux proportional to a
//! concentration difference through a mass-transfer coefficient - with the
//! difference taken between the two cells adjacent to the baffle (120.15a),
//! and the series closed form (120.15d) of the 1-D slab, which is exact in
//! the finite-volume discretisation because the profile in each half is
//! linear.
//!
//! An integration test, so it runs in both builds. Tolerances are class R
//! (docs/17 §5.1 rule 4, D-F32-1): the f64 number in the f64 build,
//! `max(t, min(t 2^29, 1e-5))` under `single`; every comparison is made in
//! f64. A machine without a card passes vacuously.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_setup::setup_scalar_field;
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField};
use ofgpu::mesh::PatchInfo;
use ofgpu::species::{Species, SpeciesCoeffs};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, HostMesh,
    Label, PatchKind, Result, Scalar,
};

/// The process's device, if the box has one.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// Class R (D-F32-1): `t` in the f64 build, `max(t, min(t 2^29, 1e-5))`
/// under `single`.
const fn tol_r(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped {
            t
        } else {
            capped
        }
    } else {
        t
    }
}

#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

const N: usize = 16;
/// The cut: the face between cells `CUT` and `CUT + 1` becomes the baffle.
const CUT: usize = 7;
/// The molecular diffusivity of every species, m2/s (SpeciesCoeffs' default).
const D: f64 = 2.0e-5;
const NU: f64 = 1.5e-5;

/// The duct of the module doc.
fn baffle_duct() -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize| GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(1.0, N),
        y: axis(0.1, 1),
        z: axis(0.1, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "empty", "empty", "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    };
    let mut raw = blockgen::raw_mesh(&spec)?;
    let n_if = raw.neighbour.len();
    let cut = (0..n_if)
        .find(|&f| raw.owner[f] == CUT as Label && raw.neighbour[f] == (CUT + 1) as Label)
        .expect("the face between cells 7 and 8");

    let face = raw.faces[cut].clone();
    let mut faces = Vec::new();
    let mut owner = Vec::new();
    let mut neighbour = Vec::new();
    for f in 0..n_if {
        if f != cut {
            faces.push(raw.faces[f].clone());
            owner.push(raw.owner[f]);
            neighbour.push(raw.neighbour[f]);
        }
    }
    for f in n_if..raw.faces.len() {
        faces.push(raw.faces[f].clone());
        owner.push(raw.owner[f]);
    }
    let n_bf = raw.faces.len() - n_if;
    faces.push(face.clone());
    owner.push(CUT as Label);
    let mut rev = face;
    rev.reverse();
    faces.push(rev);
    owner.push((CUT + 1) as Label);

    for (k, name) in ["baffleA", "baffleB"].into_iter().enumerate() {
        raw.patches.push(PatchInfo {
            name: name.into(),
            type_name: "wall".into(),
            kind: PatchKind::Wall,
            start: n_bf + k,
            size: 1,
            nbr_patch: None,
        });
    }
    raw.faces = faces;
    raw.owner = owner;
    raw.neighbour = neighbour;
    ofgpu::io::polymesh::build_host_mesh(&raw)
}

fn face_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
        .start
}

fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

fn fixed(v: f64) -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![v as Scalar];
    s
}

/// A `semiPermeableBaffleMassFraction` entry; `c = None` leaves it out.
fn baffle(c: Option<&str>, nbr: Option<&str>) -> PatchFieldSpec {
    let mut s = ps("semiPermeableBaffleMassFraction");
    if let Some(c) = c {
        s.extra.insert("c".into(), c.into());
    }
    s.neighbour_patch = nbr.map(String::from);
    s
}

fn transfer(c: &str, y_inf: Option<&str>) -> PatchFieldSpec {
    let mut s = ps("specieTransfer");
    s.extra.insert("c".into(), c.into());
    if let Some(y) = y_inf {
        s.extra.insert("Yinf".into(), y.into());
    }
    s
}

/// A raw species field over the duct's eight patches.
fn raw_field(
    name: &str,
    x: [PatchFieldSpec; 2],
    b: [PatchFieldSpec; 2],
    internal: Vec<Scalar>,
) -> RawScalarField {
    let mut raw = RawScalarField {
        name: name.into(),
        ..Default::default()
    };
    let [x0, x1] = x;
    let [b0, b1] = b;
    for (p, s) in [
        ("xMin", x0),
        ("xMax", x1),
        ("yMin", ps("empty")),
        ("yMax", ps("empty")),
        ("zMin", ps("empty")),
        ("zMax", ps("empty")),
        ("baffleA", b0),
        ("baffleB", b1),
    ] {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = internal;
    raw
}

/// Transient controls: Euler at `dt`, no relaxation, tight solves.
fn controls(dt: f64) -> TurbulenceControls {
    let mut c = TurbulenceControls {
        k_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        ddt: DdtScheme::Euler,
        sn_grad: ofgpu::fv::SnGradScheme::Uncorrected,
        ..TurbulenceControls::default()
    };
    c.k_solver.report_residuals = false;
    c.k_solver.tolerance = 1e-15;
    c.k_solver.rel_tol = 0.0;
    c.k_solver.max_iter = 400;
    c
}

fn names() -> Vec<String> {
    ["A", "B", "N2"].map(String::from).to_vec()
}

fn index_of(sp: &Species<'_>, name: &str) -> usize {
    sp.names().iter().position(|n| n == name).expect("solved")
}

fn cells(g: &Gpu, f: &GpuScalarField) -> Vec<f64> {
    g.download(&f.f).expect("download").into_iter().map(w64).collect()
}

// ==========================================================================
//  species_semipermeable_baffle_flux
// ==========================================================================

/// **SPEC-LIT §120.15, row R.** Species A crosses the baffle with
/// `c = 1e-4 m/s` between `Y = 1` at xMin and `Y = 0` at xMax; species B is
/// held back (`c = 0`) and starts non-uniform. Marched to the steady state,
/// A's baffle flux `c (Y_7 - Y_8)` (120.15a) equals every interior face's
/// diffusive flux and the series closed form (120.15d)
/// `J = 1 / (15 h / D + 1 / c)` to 1e-12 relative (tol_r under single);
/// B's baffle faces carry `fr = 0` and `g = 0` bitwise - a flux of exactly
/// zero - and B relaxes to a uniform value on each side. (B's mass per side
/// is held in `species_transfer_budget_closes`, at a time step whose
/// matrix keeps the conserved mode well conditioned; at `dt = 1e7` it is
/// the linear solver's residual that moves it.)
#[test]
fn species_semipermeable_baffle_flux() {
    let Some(g) = gpu() else { return };
    let hm = baffle_duct().expect("the baffle duct");
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let dt = 1.0e7;
    let coeffs = vec![SpeciesCoeffs::default(); 3];
    assert_eq!(coeffs[0].d, D as Scalar, "the test's D is the default diffusivity");
    let mut sp =
        Species::new(&g, &hm, &m, &names(), &coeffs, "N2", NU as Scalar, controls(dt)).expect("set");

    let c_a = 1.0e-4;
    let raw_a = raw_field(
        "A",
        [fixed(1.0), fixed(0.0)],
        [baffle(Some("1e-4"), Some("baffleB")), baffle(Some("1e-4"), Some("baffleA"))],
        vec![0.5],
    );
    let b0: Vec<Scalar> = (0..N).map(|i| (1e-3 * (1.0 + 0.5 * (i as f64 * 0.7).sin())) as Scalar).collect();
    let raw_b = raw_field(
        "B",
        [ps("zeroGradient"), ps("zeroGradient")],
        [baffle(Some("0"), Some("baffleB")), baffle(Some("0"), Some("baffleA"))],
        b0,
    );
    let (ia, ib) = (index_of(&sp, "A"), index_of(&sp, "B"));
    sp.setup_member(&g, &hm, ia, &raw_a).expect("A's boundaries");
    sp.setup_member(&g, &hm, ib, &raw_b).expect("B's boundaries");
    assert_eq!(sp.boundary_rules(ia).len(), 2, "A carries one rule per baffle side");
    assert_eq!(sp.boundary_rules(ib).len(), 2, "B carries one rule per baffle side");
    sp.initialise(&g).expect("initialise");

    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    let flow = FlowState::new(&u, &phi, NU as Scalar);

    for _ in 0..150 {
        sp.correct(&g, &flow, &nut).expect("correct");
    }

    let ya = cells(&g, sp.get(ia).expect("A").field());
    let h = 1.0 / N as f64;
    let j_closed = 1.0 / (15.0 * h / D + 1.0 / c_a);
    let j_baffle = c_a * (ya[CUT] - ya[CUT + 1]);
    let mut worst = (j_baffle - j_closed).abs() / j_closed;
    for i in 0..N - 1 {
        if i == CUT {
            continue;
        }
        let j = D * (ya[i] - ya[i + 1]) / h;
        worst = worst.max((j - j_closed).abs() / j_closed);
    }
    let j_in = D * (1.0 - ya[0]) / (0.5 * h);
    let j_out = D * ya[N - 1] / (0.5 * h);
    worst = worst.max((j_in - j_closed).abs() / j_closed);
    worst = worst.max((j_out - j_closed).abs() / j_closed);
    println!(
        "  [120.15] semi-permeable baffle: J = {j_closed:.6e}, c (Y_P - Y_N) = {j_baffle:.6e}, \
         worst relative difference over every face {worst:.3e} (tol {:.0e})",
        tol_r(1e-12)
    );
    assert!(worst <= tol_r(1e-12), "the baffle flux misses (120.15d) by {worst:e}");
    assert!(j_baffle > 1e-6, "the baffle carries a flux: {j_baffle:e}");

    // B: the impermeable species - fr and g exactly zero on both baffle faces.
    let fb = sp.get(ib).expect("B").field();
    let fr = g.download(&fb.fr).expect("fr");
    let rg = g.download(&fb.ref_grad).expect("ref_grad");
    let kinds = g.download(&fb.bc_kind).expect("kinds");
    for name in ["baffleA", "baffleB"] {
        let f = face_of(&hm, name);
        assert_eq!(kinds[f], BcKind::Mixed as Label, "{name}: the rule's face is a mixed face");
        assert_eq!(fr[f].to_bits(), (0.0 as Scalar).to_bits(), "{name}: B's fr must be +0");
        assert_eq!(rg[f].to_bits(), (0.0 as Scalar).to_bits(), "{name}: B's g must be +0");
    }
    let yb = cells(&g, fb);
    let spread = yb[..=CUT].iter().fold(0.0f64, |a, v| a.max((v - yb[0]).abs()));
    assert!(spread < 1e-3 * yb[0], "B relaxed to uniform on its side ({spread:e})");
}

// ==========================================================================
//  species_bc_refused_without_species
// ==========================================================================

fn refusal(r: Result<()>) -> String {
    match r {
        Ok(()) => panic!("must be refused"),
        Err(e) => e.to_string(),
    }
}

/// **SPEC-LIT §120.15, row D.** Outside a species set both names are
/// refused BY NAME - by the generic reader and by a plain scalar field's
/// set-up - and the species set refuses an entry it cannot lower, naming
/// the condition and the patch.
#[test]
fn species_bc_refused_without_species() {
    for name in ["semiPermeableBaffleMassFraction", "specieTransfer"] {
        let msg = BcKind::from_name(name, "T", "baffleA").expect_err("refused").to_string();
        assert!(msg.contains(name), "{msg}");
        assert!(msg.contains("SPEC-LIT 120.15"), "{msg}");
        assert!(msg.contains("species"), "{msg}");
    }

    let Some(g) = gpu() else { return };
    let hm = baffle_duct().expect("the baffle duct");
    let m = GpuMesh::upload(&g, &hm).expect("mesh");

    // A case without species: the temperature field with a species condition.
    let mut t = GpuScalarField::zeros(&g, &m, "T").expect("T");
    let raw_t = raw_field(
        "T",
        [fixed(300.0), transfer("1e-4", None)],
        [ps("zeroGradient"), ps("zeroGradient")],
        vec![300.0],
    );
    let msg = refusal(setup_scalar_field(&g, &mut t, &raw_t, &hm));
    assert!(msg.contains("specieTransfer") && msg.contains("120.15"), "{msg}");

    let fresh = || {
        Species::new(&g, &hm, &m, &names(), &vec![SpeciesCoeffs::default(); 3], "N2", NU as Scalar, controls(1.0))
            .expect("set")
    };
    let ends = || [fixed(1.0), fixed(0.0)];
    let cases: Vec<(&str, RawScalarField, &str)> = vec![
        ("no c", raw_field("A", ends(), [baffle(None, Some("baffleB")), baffle(Some("1e-4"), Some("baffleA"))], vec![0.5]), "`c`"),
        ("c < 0", raw_field("A", ends(), [baffle(Some("-1e-4"), Some("baffleB")), baffle(Some("-1e-4"), Some("baffleA"))], vec![0.5]), "c ="),
        ("no partner", raw_field("A", ends(), [baffle(Some("1e-4"), None), baffle(Some("1e-4"), Some("baffleA"))], vec![0.5]), "neighbourPatch"),
        ("unknown partner", raw_field("A", ends(), [baffle(Some("1e-4"), Some("nowhere")), baffle(Some("1e-4"), Some("baffleA"))], vec![0.5]), "nowhere"),
        ("another c", raw_field("A", ends(), [baffle(Some("1e-4"), Some("baffleB")), baffle(Some("2e-4"), Some("baffleA"))], vec![0.5]), "same `c`"),
        ("partner not a baffle", raw_field("A", ends(), [baffle(Some("1e-4"), Some("baffleB")), ps("zeroGradient")], vec![0.5]), "baffleB"),
        ("Yinf outside [0, 1]", raw_field("A", [fixed(1.0), transfer("1e-4", Some("1.5"))], [ps("zeroGradient"), ps("zeroGradient")], vec![0.5]), "Yinf"),
    ];
    for (what, raw, key) in cases {
        let mut sp = fresh();
        let i = index_of(&sp, "A");
        let msg = refusal(sp.setup_member(&g, &hm, i, &raw));
        println!("  [120.15] refused ({what}): {}", msg.lines().next().unwrap_or(""));
        assert!(msg.contains(key), "{what}: the refusal must say {key}: {msg}");
        assert!(msg.contains("120.15"), "{what}: {msg}");
    }

    // An empty patch cannot carry a transfer.
    let mut raw = raw_field("A", ends(), [ps("zeroGradient"), ps("zeroGradient")], vec![0.5]);
    raw.boundary.insert("yMin".into(), transfer("1e-4", None));
    let mut sp = fresh();
    let i = index_of(&sp, "A");
    let msg = refusal(sp.setup_member(&g, &hm, i, &raw));
    assert!(msg.contains("yMin") && msg.contains("specieTransfer"), "{msg}");

    // D = 0 leaves the half-cell with no conductance to carry c through.
    let mut coeffs = vec![SpeciesCoeffs::default(); 3];
    coeffs[0].d = 0.0;
    let mut sp = Species::new(&g, &hm, &m, &names(), &coeffs, "N2", NU as Scalar, controls(1.0)).expect("set");
    let i = index_of(&sp, "A");
    let raw = raw_field("A", [fixed(1.0), transfer("1e-4", None)], [ps("zeroGradient"), ps("zeroGradient")], vec![0.5]);
    let msg = refusal(sp.setup_member(&g, &hm, i, &raw));
    assert!(msg.contains("D = 0"), "{msg}");

    // The same raw without a species condition is set up exactly as
    // setup_scalar_field sets it up: no rule.
    let mut sp = fresh();
    let i = index_of(&sp, "A");
    let plain = raw_field("A", ends(), [ps("zeroGradient"), ps("zeroGradient")], vec![0.5]);
    sp.setup_member(&g, &hm, i, &plain).expect("a plain member");
    assert!(sp.boundary_rules(i).is_empty());
}

// ==========================================================================
//  species_transfer_budget_closes
// ==========================================================================

/// **SPEC-LIT §120.15 and (86.5), row R.** On the mass-weighted equation,
/// with a density varying in space, the baffle (`c = 1e-4`) and a
/// `specieTransfer` sink at xMax (`c = 5e-5`, `Y_inf = 0`) move each
/// species' mass by exactly what the rules let through: per step,
/// `|Δ Σ rho Y V - dt Σ_b rho_m c |S_b| (Y_ref - Y_P) - Δ clip| <= 1e-12`
/// of the mass (tol_r), `rho_m` the mean of a baffle's two face densities
/// and a transfer face's own, with the transfer above 1e-6 of it. A third solved
/// species C carries both conditions with `c = 0`: its mass on each side
/// of the baffle stays what it was, per step, to 1e-12 of it (tol_r).
#[test]
fn species_transfer_budget_closes() {
    let Some(g) = gpu() else { return };
    let hm = baffle_duct().expect("the baffle duct");
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let dt = 1.0;
    let four: Vec<String> = ["A", "B", "C", "N2"].map(String::from).to_vec();
    let mut sp = Species::new(&g, &hm, &m, &four, &vec![SpeciesCoeffs::default(); 4], "N2", NU as Scalar, controls(dt))
        .expect("set");
    sp.use_mass_weighting(&g).expect("mass weighting");

    let ya0: Vec<Scalar> = (0..N).map(|i| if i <= CUT { 0.6 } else { 0.2 } as Scalar).collect();
    let yb0: Vec<Scalar> = (0..N).map(|i| (0.05 + 0.01 * i as f64) as Scalar).collect();
    let yc0: Vec<Scalar> = (0..N).map(|i| (0.02 + 0.03 * (i as f64 * 0.9).sin().abs()) as Scalar).collect();
    let make = |nm: &str, c: &str, y: Vec<Scalar>| {
        raw_field(
            nm,
            [ps("zeroGradient"), transfer(if c == "0" { "0" } else { "5e-5" }, Some("0"))],
            [baffle(Some(c), Some("baffleB")), baffle(Some(c), Some("baffleA"))],
            y,
        )
    };
    for (nm, c, y) in [("A", "1e-4", ya0), ("B", "1e-4", yb0), ("C", "0", yc0)] {
        let i = index_of(&sp, nm);
        sp.setup_member(&g, &hm, i, &make(nm, c, y)).expect("boundaries");
        assert_eq!(sp.boundary_rules(i).len(), 3);
    }
    sp.initialise(&g).expect("initialise");

    // rho varying in space, fixed in time, its boundary value the owner's.
    let rho_c: Vec<Scalar> = (0..n).map(|i| (0.4 + 0.1 * (i as f64 * 0.37).sin()) as Scalar).collect();
    let mut rho = GpuScalarField::zeros(&g, &m, "rho").expect("rho");
    g.write(&mut rho.bc_kind, &vec![BcKind::Calculated as Label; hm.n_boundary_faces]).expect("kinds");
    for b in [&mut rho.f, &mut rho.f0, &mut rho.f00] {
        g.write(b, &rho_c).expect("rho");
    }
    let rho_b: Vec<Scalar> = hm.b_face_cells.iter().map(|&c| rho_c[c as usize]).collect();
    g.write(&mut rho.bf, &rho_b).expect("rho bf");
    let rho_h: Vec<f64> = rho_c.iter().map(|v| w64(*v)).collect();
    let vol: Vec<f64> = hm.v.iter().map(|v| w64(*v)).collect();
    let total: f64 = (0..n).map(|c| rho_h[c] * vol[c]).sum();

    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    let flow = FlowState::new(&u, &phi, NU as Scalar);

    let solved: Vec<String> = sp.names().to_vec();
    let mass = |sp: &Species<'_>, nm: &str| -> f64 {
        let y = cells(&g, sp.by_name(nm).expect("solved").field());
        (0..n).map(|c| rho_h[c] * y[c] * vol[c]).sum()
    };
    let ledger = |sp: &Species<'_>, nm: &str| -> f64 {
        g.download(sp.clip_ledger(nm).expect("ledger")).expect("d").iter().map(|v| w64(*v)).sum()
    };
    let side_mass = |sp: &Species<'_>, nm: &str| -> (f64, f64) {
        let y = cells(&g, sp.by_name(nm).expect("solved").field());
        let l: f64 = (0..=CUT).map(|c| rho_h[c] * y[c] * vol[c]).sum();
        let r: f64 = (CUT + 1..N).map(|c| rho_h[c] * y[c] * vol[c]).sum();
        (l, r)
    };
    let (mut worst, mut largest, mut held) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..3 {
        let before: Vec<(f64, f64)> = solved.iter().map(|nm| (mass(&sp, nm), ledger(&sp, nm))).collect();
        let c_before = side_mass(&sp, "C");
        sp.correct_with_density(&g, &flow, &nut, Some(&rho)).expect("correct");
        let c_after = side_mass(&sp, "C");
        held = held
            .max((c_after.0 - c_before.0).abs() / c_before.0)
            .max((c_after.1 - c_before.1).abs() / c_before.1);
        for (i, nm) in solved.iter().enumerate() {
            let st = sp.get(i).expect("solved");
            let y = cells(&g, st.field());
            let rf = g.download(&st.field().ref_value).expect("ref");
            let rho_f = g.download(&st.face_density().expect("rho_f").bf).expect("rho_f");
            let mut inflow = 0.0;
            for r in sp.boundary_rules(i) {
                for j in 0..r.size {
                    let f = r.start + j;
                    let cell = hm.b_face_cells[f] as usize;
                    // (120.15a): a baffle face carries the two sides' mean
                    // face density, a transfer face its own.
                    let rho_m = match r.nbr_start {
                        Some(s) => 0.5 * (w64(rho_f[f]) + w64(rho_f[s + j])),
                        None => w64(rho_f[f]),
                    };
                    inflow += rho_m * w64(r.c) * w64(hm.b_mag_sf[f]) * (w64(rf[f]) - y[cell]);
                }
            }
            let (m0, l0) = before[i];
            let defect = mass(&sp, nm) - m0 - dt * inflow - (ledger(&sp, nm) - l0);
            let rel = defect.abs() / total;
            assert!(rel.is_finite(), "{nm}: the budget is not finite");
            worst = worst.max(rel);
            largest = largest.max((dt * inflow).abs() / total);
        }
    }
    println!(
        "  [120.15] species budget with the transfer rules: worst defect {worst:.3e} of the mass, \
         largest transfer {largest:.3e} (tol {:.0e}); the held-back species' side masses \
         moved by {held:.3e}",
        tol_r(1e-12)
    );
    assert!(held <= tol_r(1e-12), "C crossed the baffle or the wall: {held:e}");
    assert!(largest > 1e-6, "the rules moved only {largest:e} of the mass");
    assert!(worst <= tol_r(1e-12), "(86.5) with the transfer rules: defect {worst:e}");
}
