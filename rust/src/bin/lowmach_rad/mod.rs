// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Radiation in `ofgpu-lowmach`'s outer iteration - SPEC-LIT 137.8 (CHR-23).
//!
//! [`crate::io::case_rad`] lowers the case's `radiation` block (SPEC-LIT
//! 137.2) - the JSONC route and the `constant/radiationProperties` route
//! through one lowering - into a [`ofgpu::io::case_rad::LoweredRadiation`].
//! This module runs it: CHR-18's P1 (SPEC-LIT 135.7) or CHR-20/21's fvDOM
//! (SPEC-LIT 136.7-136.8) solves when the case's schedule says so, registers
//! (137.2)'s linearised source at the current `T` every iteration, and
//! balances the radiating walls' `T` conditions, (137.5), so the energy
//! equation sees the radiation the walls receive. After the loop one more
//! solve runs at the final `T` and (137.3)'s two sides are formed on the
//! host, (137.6).
//!
//! The constraint's `Q` is (137.4)'s `q + sp*T` once a sink is live - that
//! half lives in [`ofgpu::energy::EnergySources::constraint_q`], not here.
//!
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};
use ofgpu::device::{cfg_for, DevBuf};
use ofgpu::energy::EnergySources;
use ofgpu::field::{BcKind, GpuScalarField};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::case_rad::LoweredRadiation;
use ofgpu::io::fields::{RawScalarField, RawVectorField};
use ofgpu::radiation::dom::{DomControls, DomMedium, DomSolve};
use ofgpu::radiation::dom_quad::Quadrature;
use ofgpu::radiation::p1::{P1, P1Medium};
use ofgpu::radiation::{RadiationModel, SIGMA_SB};
use ofgpu::mesh::PatchKind;
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, KernelSet, Label, Result, Scalar, Vec3};

use crate::common::CaseNumerics;

/// How a run's radiation ended, for `main`'s summary and the tests: the
/// solves the schedule made, (137.6)'s two sides and scale in watts, and the
/// last solve's `G` and wall flux, as downloaded.
#[derive(Debug)]
pub struct RadiationEnd {
    /// The solves the `every` schedule made inside the loop; the finish's
    /// own final solve is not counted (SPEC-LIT 137.8).
    pub solves: usize,
    /// (137.6)'s `L = sum_b (q_r.n)_b |S_b|`, watts.
    pub walls_w: f64,
    /// (137.6)'s `R = sum_P kappa (4 sigma T_P^4 - G_P) V_P`, watts.
    pub volume_w: f64,
    /// (137.6)'s scale `S = sum_b |(q_r.n)_b| |S_b| + sum_P kappa
    /// (4 sigma T_P^4 + G_P) V_P`, watts.
    pub scale_w: f64,
    /// The final solve's incident radiation, `[n_cells]` W/m2.
    pub g: Vec<Scalar>,
    /// The final solve's wall flux `(q_r.n)_b`, `[n_bf]` W/m2, positive out
    /// of the medium.
    pub wall_flux: Vec<Scalar>,
}

/// The one radiation solve a run carries: CHR-18's P1 or CHR-20/21's fvDOM.
enum Solver {
    P1(P1),
    Dom(DomSolve),
}

/// Radiation as `ofgpu-lowmach` runs it (SPEC-LIT 137.8). Built once, before
/// the loop; `step`/`register`/`balance_walls` run in that order every outer
/// iteration, after the thermostat and before `update_target_divergence`.
pub struct CaseRadiation {
    model: RadiationModel,
    /// The medium's constant `kappa`, 1/m - `wsgg`/`wsggGrey` are refused at
    /// the driver (SPEC-LIT 137.8: CHR-22's three open items), so one number
    /// is the whole absorption the (137.6) volume terms need.
    kappa: Scalar,
    solver: Solver,
    /// Solve every `every`-th outer iteration, counted from 0 (SPEC-LIT
    /// 137.8's rule 1).
    every: usize,
    calls: usize,
    solves: usize,
    /// The radiating wall faces whose `ref_value` step 3 rewrites, (137.5),
    /// as boundary-face indices.
    balanced: Vec<usize>,
    /// The stated flux `q_stated,b` of each balanced face, captured at
    /// [`Self::new`]: 0 on a face rewritten from `zeroGradient`, the case's
    /// own `q` on one that already was `fixedFluxTemperature`.
    q_stated: Vec<Scalar>,
    /// The `radSourcePair` scratch, fvDOM only: P1's
    /// `register_energy_source` launches the same kernel itself.
    q_src: DevBuf<Scalar>,
    sp_src: DevBuf<Scalar>,
    source_pair: CudaFunction,
    n: usize,
}

impl CaseRadiation {
    /// Lower the case's radiation into a running solve. `t` is `T`, already
    /// set up: every radiating wall's faces - the mesh's `wall` patches,
    /// which (SPEC-LIT 137.8's patch table) all carry a Marshak or
    /// grey-diffuse condition - must state a `T` condition radiation can
    /// ride, and the ones that are `zeroGradient` (adiabatic) are rewritten
    /// to `fixedFluxTemperature` with `q_stated = 0` (`fr = 0`, `ref_value
    /// = 0`, `ref_grad = 0`), per (137.5)'s set-up. A face that already is
    /// `fixedFluxTemperature` keeps its own stated `q` as `q_stated`; a
    /// `fixedValue` wall is not balanced (its temperature is stated); any
    /// other condition is refused by name.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        rad: &LoweredRadiation,
        num: &CaseNumerics,
        t: &mut GpuScalarField,
    ) -> Result<Self> {
        // SPEC-LIT 137.8: `wsgg`/`wsggGrey` at the driver, naming CHR-22's
        // three open items - the lowering alone cannot refuse them, because
        // the library's weighted grey-gas rows are the CHR-22 machinery
        // around them.
        let kappa = match &rad.config.absorption {
            ofgpu::radiation::props::Absorption::Constant { kappa } => *kappa as Scalar,
            ofgpu::radiation::props::Absorption::Wsgg { .. }
            | ofgpu::radiation::props::Absorption::WsggGrey { .. } => {
                return Err(Error::Config(
                    "radiation: wsgg/wsggGrey is refused in ofgpu-lowmach - CHR-22's \
                     per-cell a_i(T, X) and kappa_i(T, X), the wall weight a_i(T_w) \
                     and the transparent window are named and not built, so a \
                     weighted medium has no driver to run it (SPEC-LIT 137.8)"
                        .to_string(),
                ));
            }
        };
        let model = rad.config.model;
        let solver = match model {
            RadiationModel::P1 => {
                let medium = P1Medium::from_config(&rad.config)?;
                // SPEC-LIT 137.8: P1's `G` solve answers to the case's own
                // `G` rule when it gave one, else PCG on a DIC.
                let fallback = SolverControls {
                    solver: LinearSolverKind::PCG,
                    precon: Preconditioner::Dic,
                    ..SolverControls::default()
                };
                let ctrl = num.solver("G", fallback)?;
                Solver::P1(P1::new(gpu, hm, m, &medium, &rad.p1_walls(hm)?, ctrl)?)
            }
            RadiationModel::FvDom => {
                let medium = DomMedium::from_config(&rad.config)?;
                // The default quadrature is S4 (SPEC-LIT 137.2's table).
                let quad = Quadrature::from_name(rad.quadrature.as_deref().unwrap_or("S4"))?;
                let mut ctrl = DomControls::default();
                ctrl.linear = num.solver("Ii", ctrl.linear)?;
                // The sweep tolerance and cap are the case's when it gave
                // them, else the library's own defaults.
                if let Some(v) = rad.scatter_tol {
                    ctrl.tol = v;
                }
                if let Some(v) = rad.max_sweeps {
                    ctrl.max_sweeps = v;
                }
                Solver::Dom(DomSolve::new(
                    gpu,
                    hm,
                    m,
                    &medium,
                    &quad,
                    &rad.dom_patches(hm)?,
                    ctrl,
                )?)
            }
            RadiationModel::ViewFactor => {
                return Err(Error::Config(
                    "radiation: viewFactor is CHR-24's enclosure model, not built \
                     in ofgpu-lowmach (SPEC-LIT 137.8)"
                        .to_string(),
                ));
            }
        };
        let n = m.n_cells;
        let k = KernelSet::new(gpu, ofgpu::kernels::RADIATION)?;
        let source_pair = k.func("radSourcePair")?;
        let mut rad_case = Self {
            model,
            kappa,
            solver,
            every: rad.every,
            calls: 0,
            solves: 0,
            balanced: Vec::new(),
            q_stated: Vec::new(),
            q_src: gpu.zeros(n.max(1))?,
            sp_src: gpu.zeros(n.max(1))?,
            source_pair,
            n,
        };
        rad_case.claim_walls(gpu, hm, t)?;
        Ok(rad_case)
    }

    /// (137.5)'s set-up: walk the mesh's `wall` patches, read their faces'
    /// `T` condition, rewrite every `zeroGradient` face to
    /// `fixedFluxTemperature` with `q_stated = 0`, adopt every
    /// `fixedFluxTemperature` face's own `q`, leave `fixedValue` alone, and
    /// refuse every other condition by patch name.
    fn claim_walls(&mut self, gpu: &Gpu, hm: &HostMesh, t: &mut GpuScalarField) -> Result<()> {
        let mut kinds = gpu.download(&t.bc_kind)?;
        let mut fr = gpu.download(&t.fr)?;
        let mut ref_value = gpu.download(&t.ref_value)?;
        let mut ref_grad = gpu.download(&t.ref_grad)?;
        let zg = BcKind::ZeroGradient as Label;
        let ffq = BcKind::FixedFluxTemperature as Label;
        let fv = BcKind::FixedValue as Label;
        for p in &hm.patches {
            if p.kind != PatchKind::Wall || p.size == 0 {
                continue;
            }
            for bf in p.start..p.start + p.size {
                if kinds[bf] == zg {
                    // (137.5)'s set-up triple: q_stated = 0, the same three
                    // numbers field_setup writes for a stated flux.
                    kinds[bf] = ffq;
                    fr[bf] = 0.0;
                    ref_value[bf] = 0.0;
                    ref_grad[bf] = 0.0;
                    self.balanced.push(bf);
                    self.q_stated.push(0.0);
                } else if kinds[bf] == ffq {
                    self.balanced.push(bf);
                    self.q_stated.push(ref_value[bf]);
                } else if kinds[bf] != fv {
                    // A fixedValue wall keeps its own temperature (its
                    // conductive flux is whatever the solve gives); any
                    // other condition cannot carry a stated flux.
                    return Err(Error::Config(format!(
                        "radiation: wall patch {} carries a T condition radiation \
                         cannot ride, on face {bf} - a radiating wall's T must be \
                         fixedValue, zeroGradient or fixedFluxTemperature (SPEC-LIT \
                         137.8)",
                        p.name
                    )));
                }
            }
        }
        gpu.write(&mut t.bc_kind, &kinds)?;
        gpu.write(&mut t.fr, &fr)?;
        gpu.write(&mut t.ref_value, &ref_value)?;
        gpu.write(&mut t.ref_grad, &ref_grad)?;
        Ok(())
    }

    /// Rule 1 (SPEC-LIT 137.8): solve when due. Outer iterations are counted
    /// from 0, and iteration `i` solves when `i mod every = 0`; between
    /// solves `G` is held. `true` when this call solved.
    pub fn step(&mut self, gpu: &Gpu, m: &GpuMesh, t: &GpuScalarField) -> Result<bool> {
        let due = self.calls % self.every == 0;
        if due {
            match &mut self.solver {
                Solver::P1(p) => {
                    p.solve(gpu, m, t)?;
                }
                Solver::Dom(d) => {
                    d.solve(gpu, m, t)?;
                }
            }
            self.solves += 1;
        }
        self.calls += 1;
        Ok(due)
    }

    /// Rule 2: register (137.2) at `T* =` the current `T`, every iteration,
    /// from the held `G`. P1 through its own `register_energy_source`; fvDOM
    /// through the same `radSourcePair` kernel with `sigma` unweighted
    /// (SPEC-LIT 137.8). The explicit half goes to `q`, the implicit sink to
    /// `sp` - which is what makes [`EnergySources::constraint_q`] read
    /// (137.4)'s `q + sp*T` from here on.
    pub fn register(&mut self, gpu: &Gpu, t: &GpuScalarField, s: &mut EnergySources) -> Result<()> {
        match &mut self.solver {
            Solver::P1(p) => p.register_energy_source(gpu, t, s),
            Solver::Dom(d) => {
                let kappa = self.kappa;
                let sigma = SIGMA_SB;
                let nl = self.n as Label;
                unsafe {
                    gpu.stream()
                        .launch_builder(&self.source_pair)
                        .arg(&mut self.q_src)
                        .arg(&mut self.sp_src)
                        .arg(d.g())
                        .arg(&t.f)
                        .arg(&kappa)
                        .arg(&sigma)
                        .arg(&nl)
                        .launch(cfg_for(self.n))?;
                }
                s.register_explicit(gpu, &self.q_src)?;
                s.register_implicit_sink(gpu, &self.sp_src)
            }
        }
    }

    /// Rule 3 (137.5): rewrite `ref_value` on the balanced faces only - one
    /// download and one upload of `T`'s `ref_value`, host arithmetic -
    /// `q_stated + (q_r.n)`. `Energy`'s own `update_fixed_flux` turns it
    /// into `ref_grad`.
    pub fn balance_walls(&self, gpu: &Gpu, t: &mut GpuScalarField) -> Result<()> {
        if self.balanced.is_empty() {
            return Ok(());
        }
        let flux = self.wall_flux(gpu)?;
        let mut rv = gpu.download(&t.ref_value)?;
        for (i, &bf) in self.balanced.iter().enumerate() {
            rv[bf] = self.q_stated[i] + flux[bf];
        }
        gpu.write(&mut t.ref_value, &rv)
    }

    /// The balanced faces OR'd into a `set_fixed_flux_walls` face list, so
    /// `Energy::update_fixed_flux` converts them along with the case's own
    /// `fixedFluxTemperature` faces (SPEC-LIT 137.8).
    pub fn fixed_flux_faces(&self, base: &mut [bool]) {
        for &bf in &self.balanced {
            base[bf] = true;
        }
    }

    /// The solves the schedule made so far.
    pub fn solves(&self) -> usize {
        self.solves
    }

    fn wall_flux(&self, gpu: &Gpu) -> Result<Vec<Scalar>> {
        match &self.solver {
            Solver::P1(p) => gpu.download(p.wall_flux()),
            Solver::Dom(d) => gpu.download(d.wall_flux()),
        }
    }

    /// The report and the outputs (SPEC-LIT 137.8): after the loop, one more
    /// radiation solve at the final `T` - not counted in `every` - and then
    /// (137.6)'s two sides and scale on the host. Each term is computed in
    /// f64 from the downloaded values and rounded once to [`Scalar`]; each
    /// sum is `exactsum::host_exact_sum`. The faces are every boundary face
    /// of every patch, since the wall flux is 0 where nothing radiates.
    pub fn finish(&mut self, gpu: &Gpu, m: &GpuMesh, hm: &HostMesh, t: &GpuScalarField) -> Result<RadiationEnd> {
        match &mut self.solver {
            Solver::P1(p) => {
                p.solve(gpu, m, t)?;
            }
            Solver::Dom(d) => {
                d.solve(gpu, m, t)?;
            }
        }
        let flux = self.wall_flux(gpu)?;
        let g_host: Vec<Scalar> = match &self.solver {
            Solver::P1(p) => gpu.download(&p.g().f)?,
            Solver::Dom(d) => gpu.download(d.g())?,
        };
        let tv = gpu.download(&t.f)?;
        let mut walls: Vec<Scalar> = Vec::with_capacity(hm.n_boundary_faces);
        let mut scale_faces: Vec<Scalar> = Vec::with_capacity(hm.n_boundary_faces);
        for (bf, &q) in flux.iter().enumerate() {
            let area = f64::from(hm.b_sf[bf].mag());
            walls.push((f64::from(q) * area) as Scalar);
            scale_faces.push((f64::from(q).abs() * area) as Scalar);
        }
        let mut volume: Vec<Scalar> = Vec::with_capacity(hm.n_cells);
        let mut scale_cells: Vec<Scalar> = Vec::with_capacity(hm.n_cells);
        let (kappa, sigma) = (f64::from(self.kappa), f64::from(SIGMA_SB));
        for c in 0..hm.n_cells {
            let emission = 4.0 * sigma * f64::from(tv[c]) * f64::from(tv[c]) * f64::from(tv[c]) * f64::from(tv[c]);
            let v = f64::from(hm.v[c]);
            volume.push((kappa * (emission - f64::from(g_host[c])) * v) as Scalar);
            scale_cells.push((kappa * (emission + f64::from(g_host[c])) * v) as Scalar);
        }
        let mut scale = scale_faces;
        scale.extend(scale_cells);
        let walls_w = f64::from(ofgpu::exactsum::host_exact_sum(&walls));
        let volume_w = f64::from(ofgpu::exactsum::host_exact_sum(&volume));
        let scale_w = f64::from(ofgpu::exactsum::host_exact_sum(&scale));
        let gap = walls_w - volume_w;
        println!(
            "radiation (137.3): walls {} W | volume {} W | gap {} W | scale {} W | solves {}",
            crate::common::sci(walls_w, 6),
            crate::common::sci(volume_w, 6),
            crate::common::sci(gap, 6),
            crate::common::sci(scale_w, 6),
            self.solves
        );
        Ok(RadiationEnd {
            solves: self.solves,
            walls_w,
            volume_w,
            scale_w,
            g: g_host,
            wall_flux: flux,
        })
    }

    /// The two radiation fields every write time writes (SPEC-LIT 137.8):
    /// `G` (W/m2; boundary values P1's `G` faces, the owner cell's under
    /// fvDOM) and `qr` (W/m2; the cell `q_r`, boundary values
    /// `(q_r.n)_b n_b`), both `calculated`.
    pub fn output(&self, gpu: &Gpu, hm: &HostMesh) -> Result<(RawScalarField, RawVectorField)> {
        const DIMS: &str = "[1 0 -3 0 0 0 0]";
        let flux = self.wall_flux(gpu)?;
        let g_int: Vec<Scalar> = match &self.solver {
            Solver::P1(p) => gpu.download(&p.g().f)?,
            Solver::Dom(d) => gpu.download(d.g())?,
        };
        let g_bf: Vec<Scalar> = match &self.solver {
            // P1 solves for `G` and carries its own boundary faces.
            Solver::P1(p) => gpu.download(&p.g().bf)?,
            // fvDOM holds the cells' `G`; a boundary face shows its owner.
            Solver::Dom(_) => flux
                .iter()
                .zip(hm.owner.iter())
                .map(|(_, &c)| g_int[c as usize])
                .collect(),
        };
        let qr_int: Vec<Vec3> = match &self.solver {
            Solver::P1(p) => gpu.download(p.qr())?,
            Solver::Dom(d) => gpu.download(d.q())?,
        };
        let mut seed = RawScalarField {
            name: "G".to_string(),
            dimensions: DIMS.to_string(),
            internal: g_int,
            ..RawScalarField::default()
        };
        let mut seed_v = RawVectorField {
            name: "qr".to_string(),
            dimensions: DIMS.to_string(),
            internal: qr_int,
            ..RawVectorField::default()
        };
        for p in &hm.patches {
            let mut spec = ofgpu::io::fields::PatchFieldSpec::default();
            spec.type_name = "calculated".to_string();
            spec.value = vec![Scalar::default(); p.size];
            spec.value_v = vec![Vec3::default(); p.size];
            for i in 0..p.size {
                let bf = p.start + i;
                spec.value[i] = g_bf[bf];
                // (q_r.n)_b n_b, with n_b the OUTWARD unit normal.
                let n = hm.b_sf[bf];
                let area = f64::from(n.mag());
                let (x, y, z) = (f64::from(n.x) / area, f64::from(n.y) / area, f64::from(n.z) / area);
                spec.value_v[i] = Vec3::new(
                    (f64::from(flux[bf]) * x) as Scalar,
                    (f64::from(flux[bf]) * y) as Scalar,
                    (f64::from(flux[bf]) * z) as Scalar,
                );
            }
            seed.boundary.insert(p.name.clone(), spec.clone());
            seed_v.boundary.insert(p.name.clone(), spec);
        }
        Ok((seed, seed_v))
    }

    /// The radiative column of the wall heat-flux report: per wall patch,
    /// `sum_b (q_r.n)_b |S_b|` in W (SPEC-LIT 137.8), positive out of the
    /// medium. Patches with no faces are skipped.
    pub fn wall_report(&self, gpu: &Gpu, hm: &HostMesh) -> Result<Vec<(String, f64)>> {
        let flux = self.wall_flux(gpu)?;
        let mut out = Vec::new();
        for p in &hm.patches {
            if p.kind != PatchKind::Wall || p.size == 0 {
                continue;
            }
            let mut w = 0.0f64;
            for bf in p.start..p.start + p.size {
                w += f64::from(flux[bf]) * f64::from(hm.b_sf[bf].mag());
            }
            out.push((p.name.clone(), w));
        }
        Ok(out)
    }

    /// One line for the start-up banner.
    pub fn describe(&self) -> String {
        let model = match self.model {
            RadiationModel::P1 => "P1",
            _ => "fvDOM",
        };
        format!(
            "radiation: {} medium, kappa {} 1/m, solve every {} ({} wall faces balanced)",
            model,
            crate::common::g(f64::from(self.kappa)),
            self.every,
            self.balanced.len()
        )
    }
}

/// SPEC-LIT 137.8's refusals of what radiation cannot ride with:
/// a `motion` block or `-rigidBody` (the radiation geometry is built once),
/// and `every > 1` with `-restartFrom` (the solve phase is not in the
/// checkpoint).
pub fn refuse_combinations(
    rad: Option<&LoweredRadiation>,
    motion: bool,
    rigid_body: bool,
    restart: bool,
) -> Result<()> {
    let Some(rad) = rad else { return Ok(()) };
    if motion {
        return Err(Error::Config(
            "radiation: a case with a `motion` block is refused - the radiation \
             geometry is built once, and a moving mesh would need it rebuilt \
             (SPEC-LIT 137.8)"
                .to_string(),
        ));
    }
    if rigid_body {
        return Err(Error::Config(
            "radiation: -rigidBody is refused - the radiation geometry is built \
             once, and a moving mesh would need it rebuilt (SPEC-LIT 137.8)"
                .to_string(),
        ));
    }
    if restart && rad.every > 1 {
        return Err(Error::Config(format!(
            "radiation: `every` {} with -restartFrom is refused - the solve phase \
             is not in the checkpoint, so a resumed run would hold a `G` it never \
             solved for (SPEC-LIT 137.8)",
            rad.every
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
