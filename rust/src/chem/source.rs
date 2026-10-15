// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The split chemistry step of SPEC-LIT §132.5, as built by CHR-09
//! (§132.11): for one flow step `dt` every cell's constant-pressure reactor
//! (132.7)-(132.9) advances the transported state `(Y°, T°)` to `(Y*, T*)`
//! on the device ([`crate::chem::device::integrate_fields`], chemistry in
//! double in both builds, §132.6), and each solved species of the
//! [`Species`] set is handed the source (132.10)
//!
//! ```text
//! S_Y,k = rho (Y*_k - Y°_k) / dt
//! ```
//!
//! through `correct_with_density`'s `su` - `(Y*_k - Y°_k)/dt` on §19's
//! constant-density equation, `rho (Y*_k - Y°_k)/dt` on §86's mass-weighted
//! one, each the right-hand side of the equation it enters. That is Lie
//! splitting, chemistry then transport, the plan's §2.1 order. `T*` is kept
//! for the energy equation's `Q_chem` (CHR-10b); this module does not touch
//! `T`.
//!
//! Every mechanism species is a member of the set (solved, or the inert
//! remainder) and every member is a mechanism species: the reactor needs the
//! whole composition, and a name on one side only is refused by name.
//!
//! Written from SPEC-LIT §132.4, §132.5 and §86.
//! No GPL-licensed source was consulted.

use crate::chem::device::{
    integrate_fields, status, ChemKernels, ChemWork, DeviceMechanism, DeviceReactor,
    IntegrateWork,
};
use crate::chem::mechanism::Mechanism;
use crate::chem::reactor::ReactorOptions;
use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{self, FieldKernels};
use crate::species::Species;
use crate::Scalar;

const TAG: &str = "(SPEC-LIT §132.11)";

/// The refusal for a [`Species`] set that is not the one the step was
/// built on: its solved species `i` is missing.
fn other_set(i: usize) -> Error {
    Error::Config(format!(
        "chemistry: the species set has no solved species {i}; the split step must be \
         stepped with the set it was built on {TAG}"
    ))
}

/// Where mechanism species `k` lives in the [`Species`] set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// The `i`-th solved species.
    Solved(usize),
    /// The inert remainder `1 - Σ`.
    Inert,
}

/// The split chemistry step over one [`Species`] set (SPEC-LIT §132.11).
///
/// Holds the device mechanism, the reactor's control array, the f64 scratch
/// of [`integrate_fields`] and the packed state, all allocated once in
/// [`ChemSource::new`], so [`ChemSource::step`] allocates nothing and reads
/// nothing back (§81).
pub struct ChemSource {
    ck: ChemKernels,
    fk: FieldKernels,
    dm: DeviceMechanism,
    dr: DeviceReactor,
    work: ChemWork,
    iw: IntegrateWork,
    /// Mechanism order: `slots[k]` is where species `k` is in the set.
    slots: Vec<Slot>,
    /// `[K * n]`, `y[k * n + c]`: `Y°` packed, then `Y*` in place.
    y: DevBuf<Scalar>,
    /// `[K * n]`: `Y°`, kept for (132.10).
    y0: DevBuf<Scalar>,
    /// `[n]`: one species' `Y*`, unpacked for the field kernels.
    tmp: DevBuf<Scalar>,
    /// `[n]`: `T°` copied in, `T*` after the step.
    t: DevBuf<Scalar>,
    /// `[n]`: each cell's step memory - in, the first trial step; out, the
    /// step the controller proposes next (`RosStats::h_next`). Seeded with
    /// `opts.h0` and carried from flow step to flow step.
    h: DevBuf<f64>,
    status: DevBuf<i32>,
    counts: DevBuf<i32>,
    n: usize,
}

impl ChemSource {
    /// Build the step for `species` on `n_cells` cells and give the set its
    /// reaction sources ([`Species::use_reaction_sources`]).
    ///
    /// Refused by name: a mechanism species that is not in the set, a member
    /// of the set that is not in the mechanism, a field shorter than
    /// `n_cells`, and every refusal of [`DeviceReactor::new`].
    pub fn new(
        gpu: &Gpu,
        mech: &Mechanism,
        species: &mut Species<'_>,
        opts: &ReactorOptions,
        n_cells: usize,
    ) -> Result<Self> {
        let mut slots = Vec::with_capacity(mech.species().len());
        for name in mech.species() {
            let slot = if name == species.inert_name() {
                Slot::Inert
            } else if let Some(i) = species.names().iter().position(|s| s == name) {
                Slot::Solved(i)
            } else {
                return Err(Error::Config(format!(
                    "chemistry: the mechanism's species \"{name}\" is not one of the \
                     transported species {:?} or the inert \"{}\"; the reactor needs \
                     the whole composition {TAG}",
                    species.names(),
                    species.inert_name()
                )));
            };
            slots.push(slot);
        }
        let in_mech = |s: &str| mech.species().iter().any(|m| m == s);
        for name in species.names().iter().map(String::as_str).chain([species.inert_name()]) {
            if !in_mech(name) {
                return Err(Error::Config(format!(
                    "chemistry: the transported species \"{name}\" is not in the \
                     mechanism, whose species are {:?} {TAG}",
                    mech.species()
                )));
            }
        }
        for (k, slot) in slots.iter().enumerate() {
            let len = match slot {
                Slot::Solved(i) => species.get(*i).map_or(0, |st| st.field().f.len()),
                Slot::Inert => species.inert().f.len(),
            };
            if len < n_cells {
                return Err(Error::Config(format!(
                    "chemistry: species \"{}\" holds {len} cells, the step was built \
                     for {n_cells} {TAG}",
                    mech.species()[k]
                )));
            }
        }

        let dm = DeviceMechanism::upload(gpu, mech)?;
        let ks = dm.n_species();
        let dr = DeviceReactor::new(gpu, opts)?;
        species.use_reaction_sources(gpu)?;
        let nc = n_cells.max(1);
        Ok(Self {
            ck: ChemKernels::new(gpu)?,
            fk: FieldKernels::new(gpu)?,
            work: ChemWork::new(gpu, &dm, n_cells)?,
            iw: IntegrateWork::new(gpu, &dm, n_cells)?,
            dm,
            dr,
            slots,
            y: gpu.zeros(ks * nc)?,
            y0: gpu.zeros(ks * nc)?,
            tmp: gpu.zeros(nc)?,
            t: gpu.zeros(nc)?,
            h: gpu.upload(&vec![opts.h0; nc])?,
            status: gpu.zeros(nc)?,
            counts: gpu.zeros(3 * nc)?,
            n: n_cells,
        })
    }

    /// One split chemistry step over `dt` (SPEC-LIT §132.5): integrate every
    /// cell from the set's current `Y°` and `t` at the pressure `p`, then
    /// write each solved species' source (132.10). `rho` is `Some` exactly
    /// when the set is mass-weighted (§86), as `correct_with_density` asks.
    ///
    /// Refused by name: `dt` not finite or not above 0, a density on the
    /// constant-density set or none on the mass-weighted one, `t` or `p`
    /// shorter than the cell count. A cell the reactor refuses (a status
    /// other than [`status::OK`], see [`ChemSource::refused_cells`]) carries
    /// NaN into its sources. No allocation, no read-back (capturable).
    pub fn step(
        &mut self,
        gpu: &Gpu,
        species: &mut Species<'_>,
        t: &DevBuf<Scalar>,
        p: &DevBuf<Scalar>,
        rho: Option<&GpuScalarField>,
        dt: f64,
    ) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::Config(format!(
                "chemistry: the split step's dt = {dt} s must be finite and above 0 {TAG}"
            )));
        }
        match (species.is_mass_weighted(), rho) {
            (true, None) => {
                return Err(Error::Config(format!(
                    "chemistry: the species set is integrated in rho*Y (SPEC-LIT §86) and \
                     the split step was given no density for rho (Y* - Y°)/dt {TAG}"
                )))
            }
            (false, Some(_)) => {
                return Err(Error::Config(format!(
                    "chemistry: the species set is the constant-density equation and the \
                     split step was given a density it would not read {TAG}"
                )))
            }
            _ => {}
        }
        if !species.has_reaction_sources() {
            return Err(Error::Config(format!(
                "chemistry: the species set carries no reaction sources; build the \
                 step with ChemSource::new on this set {TAG}"
            )));
        }
        let n = self.n;
        for (what, len) in [("T", t.len()), ("p", p.len())] {
            if len < n {
                return Err(Error::Config(format!(
                    "chemistry: {what} holds {len} cells, the step was built for {n} {TAG}"
                )));
            }
        }
        if n == 0 {
            return Ok(());
        }

        // Y° in mechanism order, and T° - the reactor's state.
        for (k, slot) in self.slots.iter().enumerate() {
            let src = match slot {
                Slot::Solved(i) => &species.get(*i).ok_or_else(|| other_set(*i))?.field().f,
                Slot::Inert => &species.inert().f,
            };
            let mut dst = self.y.slice_mut(k * n..(k + 1) * n);
            gpu.stream().memcpy_dtod(&src.slice(0..n), &mut dst)?;
        }
        gpu.stream().memcpy_dtod(&self.y, &mut self.y0)?;
        gpu.stream().memcpy_dtod(&t.slice(0..n), &mut self.t.slice_mut(0..n))?;

        integrate_fields(
            gpu,
            &self.ck,
            &self.dm,
            &mut self.work,
            &mut self.iw,
            &self.dr,
            &mut self.t,
            p,
            &mut self.y,
            &mut self.h,
            dt,
            &mut self.status,
            &mut self.counts,
            n,
        )?;

        // (132.10): su_k = (Y*_k - Y°_k) / dt, times rho on §86's equation.
        let inv_dt = (1.0 / dt) as Scalar;
        for (k, slot) in self.slots.iter().enumerate() {
            let Slot::Solved(i) = *slot else { continue };
            let su = species.reaction_source_mut(i).ok_or_else(|| other_set(i))?;
            gpu.stream()
                .memcpy_dtod(&self.y0.slice(k * n..(k + 1) * n), &mut su.slice_mut(0..n))?;
            field_ops::scale_field(gpu, &self.fk, su, -1.0, n)?;
            gpu.stream()
                .memcpy_dtod(&self.y.slice(k * n..(k + 1) * n), &mut self.tmp.slice_mut(0..n))?;
            field_ops::add_field(gpu, &self.fk, su, &self.tmp, n)?;
            field_ops::scale_field(gpu, &self.fk, su, inv_dt, n)?;
            if let Some(rho) = rho {
                field_ops::multiply_field(gpu, &self.fk, su, &rho.f, n)?;
            }
        }
        Ok(())
    }

    /// `T*` after the last [`ChemSource::step`], per cell: the temperature
    /// the reactor reached, for (132.10)'s `Q_chem` (CHR-10b).
    pub fn temperature(&self) -> &DevBuf<Scalar> {
        &self.t
    }

    /// The cells the last step's reactor refused (a status other than
    /// [`status::OK`]), as `(cell, status)`. One read-back, so a diagnostic
    /// and not a loop step.
    pub fn refused_cells(&self, gpu: &Gpu) -> Result<Vec<(usize, i32)>> {
        let s = gpu.download(&self.status)?;
        Ok(s.iter()
            .take(self.n)
            .enumerate()
            .filter(|(_, v)| **v != status::OK)
            .map(|(c, v)| (c, *v))
            .collect())
    }

    /// The cell count the step was built for.
    pub fn n_cells(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::capture_replays_bitwise;
    use crate::chem::{SpeciesThermo, ThermoSet, P_ATM};
    use crate::field::{GpuSurfaceScalarField, GpuVectorField};
    use crate::mesh::{GpuMesh, HostMesh};
    use crate::species::SpeciesCoeffs;
    use crate::turbulence::{FlowState, TurbulenceControls};
    use crate::Vec3;

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

    /// The records as a set at `P_ATM`, as `chem::device`'s tests build it:
    /// bounds 200/1000/6000 K for all ten species.
    fn tm_set() -> ThermoSet {
        let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
        for line in TABLE_II.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
                continue;
            }
            let c: Vec<&str> = line.split(',').collect();
            let mut a = [0.0f64; 7];
            for (j, v) in a.iter_mut().enumerate() {
                *v = c[4 + j].parse().unwrap();
            }
            rows.push((c[0].to_string(), c[1].parse().unwrap(), c[3].parse().unwrap(), a));
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

    /// Two of §130.7's PLACEHOLDER-rate fixture reactions over the ten
    /// TM-4513 species: an elementary reversible one and a `+M` one.
    fn fixture() -> Mechanism {
        let text = [
            "ELEMENTS H O C N END",
            "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END",
            "REACTIONS",
            "H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0",
            "H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0",
            "  H2O/6.0/ H2/2.0/",
            "END",
        ];
        Mechanism::read_chemkin(&text.join("\n"), "kin-fixture", Some(&tm_set()))
            .unwrap_or_else(|e| panic!("the fixture does not read: {e}"))
    }

    fn boxed(n: usize) -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([n, n, n], Vec3::new(0.01, 0.01, 0.01));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// Everything one iteration of the gate touches.
    struct Rig<'m> {
        sp: Species<'m>,
        cs: ChemSource,
        t: DevBuf<Scalar>,
        p: DevBuf<Scalar>,
    }

    /// **SPEC-LIT §81.7, the gate `src/chem/source.rs` is registered
    /// under (§132.11).** One split chemistry step followed by the species
    /// correction it feeds captures into a CUDA graph and replays bit for
    /// bit against the per-launch path. `T*` is copied back into `T` each
    /// iteration, so the chemistry's state moves from replay to replay and
    /// a graph that skipped it cannot pass.
    #[test]
    fn chem_source_replays_under_capture() {
        let Some(g) = gpu() else { return };
        let hm = boxed(3);
        let m = GpuMesh::upload(&g, &hm).expect("mesh");
        let n = hm.n_cells;
        let mech = fixture();
        let names: Vec<String> = mech.species().to_vec();
        let coeffs = vec![SpeciesCoeffs::default(); names.len()];
        let mut ctrl = TurbulenceControls {
            steady: false,
            delta_t: 1.0e-6,
            k_relax: 1.0,
            ddt: crate::timescheme::DdtScheme::Euler,
            ..Default::default()
        };
        ctrl.k_solver.fixed_iters = true;
        ctrl.k_solver.max_iter = 4;
        ctrl.k_solver.report_residuals = false;

        let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
        let internal: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|f| Vec3::new(0.05, 0.0, 0.0).dot(hm.sf[f]))
            .collect();
        g.write(&mut phi.f, &internal).expect("phi");
        let nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
        let flow = FlowState::new(&u, &phi, 1.5e-5);

        // H2-air at phi 1 (2 : 1 : 3.76 by moles) with H, O and OH traces,
        // as mass fractions; T_c = 1200 + 10c, p = 1 atm.
        let x_of = |s: &str| match s {
            "H2" => 2.0,
            "O2" => 1.0,
            "N2" => 3.76,
            "H" | "O" | "OH" => 1.0e-3,
            _ => 0.0,
        };
        let set = tm_set();
        let xw: Vec<f64> = set
            .species()
            .iter()
            .map(|s| x_of(&s.name) * s.molar_mass)
            .collect();
        let sum: f64 = xw.iter().sum();
        let y_of = |name: &str| {
            let j = set.index_of(name).expect("species");
            (xw[j] / sum) as Scalar
        };
        let t0: Vec<Scalar> = (0..n).map(|c| (1200.0 + 10.0 * c as f64) as Scalar).collect();

        let build = || -> Result<Rig<'_>> {
            let mut sp = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, ctrl)?;
            for j in 0..sp.n_solved() {
                let name = sp.names()[j].clone();
                let v = vec![y_of(&name); n];
                g.write(&mut sp.get_mut(j).expect("solved").field_mut().f, &v)?;
            }
            sp.initialise(&g)?;
            let cs = ChemSource::new(&g, &mech, &mut sp, &ReactorOptions::default(), n)?;
            Ok(Rig { sp, cs, t: g.upload(&t0)?, p: g.upload(&vec![P_ATM as Scalar; n])? })
        };
        let report = capture_replays_bitwise(
            &g,
            "the split chemistry step (SPEC-LIT 132.11)",
            build,
            |r: &mut Rig| {
                r.cs.step(&g, &mut r.sp, &r.t, &r.p, None, 1.0e-6)?;
                g.stream().memcpy_dtod(r.cs.temperature(), &mut r.t)?;
                r.sp.correct(&g, &flow, &nut).map(|_| ())
            },
            |r: &Rig| {
                Ok(vec![
                    ("H2", g.download(&r.sp.by_name("H2").expect("H2").field().f)?),
                    ("OH", g.download(&r.sp.by_name("OH").expect("OH").field().f)?),
                    ("H2O", g.download(&r.sp.by_name("H2O").expect("H2O").field().f)?),
                    ("su OH", g.download(r.sp.reaction_source("OH").expect("su"))?),
                    ("inert", g.download(&r.sp.inert().f)?),
                    ("T*", g.download(&r.t)?),
                ])
            },
        )
        .expect("SPEC-LIT §81.7: the split chemistry step must capture and replay bitwise");
        println!("  chem source: {report}");
    }
}
