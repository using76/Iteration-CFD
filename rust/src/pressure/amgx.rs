// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! NVIDIA AMGX as a pressure backend, behind the `amgx` Cargo feature.
//!
//! # Why it is a feature, and why the feature is off
//!
//! AMGX is not installed anywhere and does not ship with the CUDA toolkit; it
//! has to be built from source. NVIDIA's own notes put Windows support at
//! "very limited", and the newest toolkit they list as verified is CUDA 12.2
//! against this machine's 13.3. A build that may not succeed cannot be a hard
//! dependency of a crate whose test suite has to stay green, so:
//!
//! * the feature is **off by default**;
//! * with it off the crate compiles and every test passes, and this file still
//!   provides an [`AmgxBackend`] so the selector can list it;
//! * with it off the backend reports itself *unavailable*, with the reason,
//!   rather than silently not being a candidate. "AMGX was never tried" and
//!   "AMGX is not compiled in" are different facts and the decision table says
//!   which one it is.
//!
//! If the build fails, that is a reported outcome, not a broken repository.
//!
//! As it happens the build did succeed here - AMGX 2.5.0 against CUDA 13.x -
//! and the backend runs and agrees with PBiCGStab. It stays behind the feature
//! anyway: what is verified is one build on one machine, and a crate that
//! cannot be compiled without a from-source dependency is a crate most people
//! cannot compile.
//!
//! # Building it
//!
//! ```text
//! cmake -S third_party/AMGX -B third_party/AMGX/build -G Ninja \
//!       -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
//! cmake --build third_party/AMGX/build
//!
//! set AMGX_DIR=...\third_party\AMGX
//! cargo test --release --features amgx        # amgxsh.dll must be on PATH
//! ```
//!
//! # A cost worth knowing about
//!
//! `AMGX_solver_setup` is called on every solve, because the AMG hierarchy
//! depends on the coefficient values and `rAUf` changes every outer iteration.
//! On the plume-sized matrix that setup is most of AMGX's 31 ms. AMGX can be
//! told to keep the coarsening and only rebuild the operators
//! (`structure_reuse_levels`), which is the first thing to try if AMGX ever
//! needs to win a measurement.
//!
//! # What it is fed
//!
//! The CSR that [`crate::ldu::CsrPattern`] already builds. That permutation
//! exists precisely so an external solver can consume the matrix without the
//! host ever seeing it: [`crate::ldu_ops::csr_fill`] gathers the LDU
//! coefficients into `val` on the device, and AMGX's `AMGX_matrix_upload_all`
//! copies with `cudaMemcpyDefault`, which under unified addressing accepts a
//! device pointer. So the matrix goes device to device.
//!
//! # What it cannot represent
//!
//! The CSR carries `diag`, `upper` and `lower` only. A cyclic patch's
//! `boundary_coeffs` never enters it - `Amul` applies that term separately
//! against the cell across the interface - so a mesh with a coupled patch
//! would hand AMGX a *different matrix* from the one PBiCGStab solves.
//! [`AmgxBackend::setup`] refuses in that case rather than producing an answer
//! that would then be silently one boundary term short.
//!
//! Provenance: ORIGINAL - the AMGX backend, behind the `amgx` feature. AMGX
//! itself is NVIDIA's and BSD-3-Clause; it is linked, not vendored, and is
//! recorded in `../NOTICE`. Nothing of its source is reproduced here.
//! `PROVENANCE.md` carries the row. No GPL-licensed source was consulted.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::Scalar;

use super::{PressureBackend, SystemProbe};

/// Below this many cells an algebraic multigrid hierarchy cannot pay for its
/// own setup whatever the machine, so the backend declines rather than wasting
/// a trial solve on it.
///
/// This is a FLOOR, not a crossover. Above it the selector still decides by
/// measurement; a hardcoded crossover is exactly what
/// [`super::choose_pressure_backend`] exists to avoid.
pub const MIN_CELLS: usize = 20_000;

/// The solver configuration handed to AMGX.
///
/// Conjugate gradients preconditioned by one classical-AMG V cycle - AMGX's
/// own `PCG_CLASSICAL_V_JACOBI.json`, spelled out inline so it is visible here
/// rather than in a file nobody reads and so the crate needs no data files at
/// run time.
///
/// Two settings matter for the selector and are set with it in mind:
///
/// * `convergence=RELATIVE_INI` with `tolerance=1e-12`. The default `1e-6`
///   would leave an answer about `1e-6` from the reference, which the
///   agreement gate would then - correctly - throw out. A backend has to be
///   asked for an answer good enough to be verified.
/// * `store_res_history=1`, without which
///   `AMGX_solver_get_iteration_residual` throws and AMGX prints a stack trace
///   to stderr on every solve.
pub const DEFAULT_CONFIG: &str = concat!(
    "config_version=2,",
    "solver(main)=PCG,",
    "main:max_iters=1000,",
    "main:tolerance=1e-12,",
    "main:norm=L2,",
    "main:convergence=RELATIVE_INI,",
    "main:monitor_residual=1,",
    "main:store_res_history=1,",
    "main:print_solve_stats=0,",
    "main:obtain_timings=0,",
    "main:preconditioner(amg)=AMG,",
    "amg:algorithm=CLASSICAL,",
    "amg:interpolator=D2,",
    "amg:max_iters=1,",
    "amg:cycle=V,",
    "amg:presweeps=1,",
    "amg:postsweeps=1,",
    "amg:max_levels=50,",
    "amg:smoother(jac)=BLOCK_JACOBI,",
    "amg:monitor_residual=0,",
    "amg:print_solve_stats=0,",
    "amg:print_grid_stats=0",
);

/// NVIDIA AMGX.
pub struct AmgxBackend {
    config: String,
    #[cfg(feature = "amgx")]
    inner: Option<imp::Session>,
    #[cfg(not(feature = "amgx"))]
    _unused: (),
}

impl Default for AmgxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AmgxBackend {
    pub fn new() -> Self {
        Self::with_config(DEFAULT_CONFIG)
    }

    pub fn with_config(config: &str) -> Self {
        Self {
            config: config.to_string(),
            #[cfg(feature = "amgx")]
            inner: None,
            #[cfg(not(feature = "amgx"))]
            _unused: (),
        }
    }

    pub fn config(&self) -> &str {
        &self.config
    }

    /// `true` when the crate was built with the `amgx` feature.
    pub const fn compiled_in() -> bool {
        cfg!(feature = "amgx")
    }

    /// The structural half of applicability, shared by both builds so the two
    /// cannot drift apart: AMG needs a symmetric matrix and enough cells for
    /// its setup to be worth doing.
    fn suits(probe: &SystemProbe) -> bool {
        probe.symmetric && probe.n_cells >= MIN_CELLS
    }
}

// ==========================================================================
//  Feature OFF - present, listed, unavailable
// ==========================================================================

#[cfg(not(feature = "amgx"))]
impl PressureBackend for AmgxBackend {
    fn name(&self) -> &'static str {
        "AMGX"
    }

    fn applicable(&self, _probe: &SystemProbe) -> bool {
        false
    }

    fn why_not(&self, probe: &SystemProbe) -> String {
        if !Self::suits(probe) {
            if !probe.symmetric {
                return "feature 'amgx' not enabled (and the matrix is not symmetric)".into();
            }
            return format!(
                "feature 'amgx' not enabled (and {} cells is below the {MIN_CELLS} \
                 an AMG setup needs to pay for itself)",
                probe.n_cells
            );
        }
        "feature 'amgx' not enabled".into()
    }

    fn setup(
        &mut self,
        _gpu: &Gpu,
        _hm: &HostMesh,
        _m: &GpuMesh,
        _probe: &SystemProbe,
    ) -> Result<()> {
        Err(Error::Config(
            "AMGX backend: the crate was built without the 'amgx' feature".into(),
        ))
    }

    fn solve(
        &mut self,
        _gpu: &Gpu,
        _p: &mut DevBuf<Scalar>,
        _a: &GpuLduMatrix,
        _m: &GpuMesh,
    ) -> Result<SolverPerformance> {
        Err(Error::Config(
            "AMGX backend: the crate was built without the 'amgx' feature".into(),
        ))
    }
}

// ==========================================================================
//  Feature ON
// ==========================================================================

#[cfg(feature = "amgx")]
impl PressureBackend for AmgxBackend {
    fn name(&self) -> &'static str {
        "AMGX"
    }

    fn applicable(&self, probe: &SystemProbe) -> bool {
        Self::suits(probe)
    }

    fn why_not(&self, probe: &SystemProbe) -> String {
        if !probe.symmetric {
            return "the matrix is not symmetric, and this configuration is an AMG \
                    hierarchy built for one"
                .into();
        }
        format!(
            "{} cells is below the {MIN_CELLS} an AMG setup needs to pay for itself",
            probe.n_cells
        )
    }

    fn setup(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        _probe: &SystemProbe,
    ) -> Result<()> {
        if hm.b_nbr_cell.iter().any(|c| *c >= 0) {
            return Err(Error::Config(
                "AMGX backend: this mesh has a coupled (cyclic) patch, whose \
                 boundaryCoeffs live outside the CSR - AMGX would be solving a \
                 different matrix from the reference"
                    .into(),
            ));
        }
        self.inner = Some(imp::Session::new(gpu, hm, m, &self.config)?);
        Ok(())
    }

    fn solve(
        &mut self,
        gpu: &Gpu,
        p: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<SolverPerformance> {
        let s = self
            .inner
            .as_mut()
            .ok_or_else(|| Error::Config("AMGX backend: setup() was not called".into()))?;
        s.solve(gpu, p, a, m)
    }
}

// ==========================================================================
//  The FFI, only compiled with the feature
// ==========================================================================

#[cfg(feature = "amgx")]
mod imp {
    //! Hand-written bindings to the handful of `amgx_c.h` entry points a
    //! single-GPU, single-rank solve needs.
    //!
    //! Written out rather than bindgen-ed because it is fourteen functions
    //! against a C API that NVIDIA calls "C-API stable", and adding a
    //! build-time dependency on libclang to a crate whose whole build story is
    //! "cargo build" would cost more than it saves.

    use std::ffi::{c_char, c_int, c_void, CString};

    use cudarc::driver::{DevicePtr, DevicePtrMut};

    use crate::device::{DevBuf, Gpu};
    use crate::error::{Error, Result};
    use crate::ldu::{CsrPattern, GpuCsrMatrix, GpuLduMatrix};
    use crate::ldu_ops::{self, LduKernels};
    use crate::mesh::{GpuMesh, HostMesh};
    use crate::solver::SolverPerformance;
    use crate::Scalar;

    pub type AmgxRc = c_int;
    pub type Handle = *mut c_void;

    const AMGX_RC_OK: AmgxRc = 0;

    /// `AMGX_mode_dDDI` / `AMGX_mode_dFFI` from `amgx_config.h`: device memory,
    /// with the matrix and the vectors in the precision this crate solves in.
    #[cfg(not(feature = "single"))]
    const MODE: c_int = 8193;
    #[cfg(feature = "single")]
    const MODE: c_int = 8465;

    extern "C" {
        fn AMGX_initialize() -> AmgxRc;
        fn AMGX_finalize() -> AmgxRc;
        fn AMGX_get_error_string(err: AmgxRc, buf: *mut c_char, buf_len: c_int) -> AmgxRc;

        fn AMGX_config_create(cfg: *mut Handle, options: *const c_char) -> AmgxRc;
        fn AMGX_config_destroy(cfg: Handle) -> AmgxRc;

        fn AMGX_resources_create_simple(rsc: *mut Handle, cfg: Handle) -> AmgxRc;
        fn AMGX_resources_destroy(rsc: Handle) -> AmgxRc;

        fn AMGX_matrix_create(mtx: *mut Handle, rsc: Handle, mode: c_int) -> AmgxRc;
        fn AMGX_matrix_destroy(mtx: Handle) -> AmgxRc;
        fn AMGX_matrix_upload_all(
            mtx: Handle,
            n: c_int,
            nnz: c_int,
            block_dimx: c_int,
            block_dimy: c_int,
            row_ptrs: *const c_int,
            col_indices: *const c_int,
            data: *const c_void,
            diag_data: *const c_void,
        ) -> AmgxRc;
        fn AMGX_matrix_replace_coefficients(
            mtx: Handle,
            n: c_int,
            nnz: c_int,
            data: *const c_void,
            diag_data: *const c_void,
        ) -> AmgxRc;

        fn AMGX_vector_create(vec: *mut Handle, rsc: Handle, mode: c_int) -> AmgxRc;
        fn AMGX_vector_destroy(vec: Handle) -> AmgxRc;
        fn AMGX_vector_upload(
            vec: Handle,
            n: c_int,
            block_dim: c_int,
            data: *const c_void,
        ) -> AmgxRc;
        fn AMGX_vector_download(vec: Handle, data: *mut c_void) -> AmgxRc;

        fn AMGX_solver_create(slv: *mut Handle, rsc: Handle, mode: c_int, cfg: Handle) -> AmgxRc;
        fn AMGX_solver_destroy(slv: Handle) -> AmgxRc;
        fn AMGX_solver_setup(slv: Handle, mtx: Handle) -> AmgxRc;
        fn AMGX_solver_solve(slv: Handle, rhs: Handle, sol: Handle) -> AmgxRc;
        fn AMGX_solver_get_iterations_number(slv: Handle, n: *mut c_int) -> AmgxRc;
        fn AMGX_solver_get_iteration_residual(
            slv: Handle,
            it: c_int,
            idx: c_int,
            res: *mut f64,
        ) -> AmgxRc;
        fn AMGX_solver_get_status(slv: Handle, st: *mut c_int) -> AmgxRc;
    }

    /// Turn an `AMGX_RC` into a `Result`, with AMGX's own message attached.
    fn check(what: &str, rc: AmgxRc) -> Result<()> {
        if rc == AMGX_RC_OK {
            return Ok(());
        }
        let mut buf = [0i8; 512];
        // SAFETY: AMGX writes at most `buf.len()` bytes into the buffer and
        // NUL-terminates. Ignoring its return code is deliberate - we are
        // already reporting a failure and a failure to describe it must not
        // replace the original.
        let msg = unsafe {
            let _ = AMGX_get_error_string(rc, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int);
            let bytes: Vec<u8> = buf
                .iter()
                .take_while(|b| **b != 0)
                .map(|b| *b as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        Err(Error::Config(format!("AMGX: {what} failed ({rc}): {msg}")))
    }

    /// Everything AMGX owns for one matrix, freed in reverse order on drop.
    pub struct Session {
        cfg: Handle,
        rsc: Handle,
        mtx: Handle,
        rhs: Handle,
        sol: Handle,
        slv: Handle,

        csr: GpuCsrMatrix,
        lduk: LduKernels,
        uploaded: bool,
    }

    impl Session {
        pub fn new(gpu: &Gpu, hm: &HostMesh, _m: &GpuMesh, config: &str) -> Result<Self> {
            let pattern = CsrPattern::build(hm);
            let csr = pattern.upload(gpu)?;
            let lduk = LduKernels::new(gpu)?;

            let options = CString::new(config).map_err(|_| {
                Error::Config("AMGX: the configuration string contains a NUL".into())
            })?;

            let mut cfg: Handle = std::ptr::null_mut();
            let mut rsc: Handle = std::ptr::null_mut();
            let mut mtx: Handle = std::ptr::null_mut();
            let mut rhs: Handle = std::ptr::null_mut();
            let mut sol: Handle = std::ptr::null_mut();
            let mut slv: Handle = std::ptr::null_mut();

            // SAFETY: every pointer below is either a stack slot AMGX writes a
            // handle into or a handle AMGX itself produced. Nothing here
            // dereferences device memory on the host.
            unsafe {
                check("initialize", AMGX_initialize())?;
                check("config_create", AMGX_config_create(&mut cfg, options.as_ptr()))?;
                check(
                    "resources_create_simple",
                    AMGX_resources_create_simple(&mut rsc, cfg),
                )?;
                check("matrix_create", AMGX_matrix_create(&mut mtx, rsc, MODE))?;
                check("vector_create(rhs)", AMGX_vector_create(&mut rhs, rsc, MODE))?;
                check("vector_create(sol)", AMGX_vector_create(&mut sol, rsc, MODE))?;
                check("solver_create", AMGX_solver_create(&mut slv, rsc, MODE, cfg))?;
            }

            Ok(Self { cfg, rsc, mtx, rhs, sol, slv, csr, lduk, uploaded: false })
        }

        pub fn solve(
            &mut self,
            gpu: &Gpu,
            p: &mut DevBuf<Scalar>,
            a: &GpuLduMatrix,
            _m: &GpuMesh,
        ) -> Result<SolverPerformance> {
            // Device-to-device: the LDU coefficients are gathered into the CSR
            // value array by a kernel, and AMGX's uploads go through
            // cudaMemcpyDefault, so the matrix never reaches the host.
            ldu_ops::csr_fill(gpu, &self.lduk, &mut self.csr, a)?;
            gpu.sync()?;

            let n = self.csr.n_rows as c_int;
            let nnz = self.csr.nnz as c_int;
            let stream = gpu.stream();

            let (row_ptr, _g0) = self.csr.row_ptr.device_ptr(stream);
            let (col_ind, _g1) = self.csr.col_ind.device_ptr(stream);
            let (val, _g2) = self.csr.val.device_ptr(stream);
            let (source, _g3) = a.source.device_ptr(stream);
            let (psi, _g4) = p.device_ptr_mut(stream);

            // SAFETY: the five pointers are live device allocations of exactly
            // the lengths passed alongside them, and AMGX only reads/writes
            // within those lengths.
            unsafe {
                if !self.uploaded {
                    check(
                        "matrix_upload_all",
                        AMGX_matrix_upload_all(
                            self.mtx,
                            n,
                            nnz,
                            1,
                            1,
                            row_ptr as *const c_int,
                            col_ind as *const c_int,
                            val as *const c_void,
                            std::ptr::null(),
                        ),
                    )?;
                    self.uploaded = true;
                } else {
                    check(
                        "matrix_replace_coefficients",
                        AMGX_matrix_replace_coefficients(
                            self.mtx,
                            n,
                            nnz,
                            val as *const c_void,
                            std::ptr::null(),
                        ),
                    )?;
                }

                check(
                    "vector_upload(rhs)",
                    AMGX_vector_upload(self.rhs, n, 1, source as *const c_void),
                )?;
                check(
                    "vector_upload(sol)",
                    AMGX_vector_upload(self.sol, n, 1, psi as *const c_void),
                )?;

                check("solver_setup", AMGX_solver_setup(self.slv, self.mtx))?;
                check("solver_solve", AMGX_solver_solve(self.slv, self.rhs, self.sol))?;
                check("vector_download", AMGX_vector_download(self.sol, psi as *mut c_void))?;
            }

            let mut iters: c_int = 0;
            let mut status: c_int = 0;
            let mut res0 = 0.0f64;
            let mut res1 = 0.0f64;
            unsafe {
                let _ = AMGX_solver_get_iterations_number(self.slv, &mut iters);
                let _ = AMGX_solver_get_status(self.slv, &mut status);
                if iters > 0 {
                    let _ = AMGX_solver_get_iteration_residual(self.slv, 0, 0, &mut res0);
                    let _ = AMGX_solver_get_iteration_residual(self.slv, iters - 1, 0, &mut res1);
                }
            }

            Ok(SolverPerformance {
                initial_residual: res0 as Scalar,
                final_residual: res1 as Scalar,
                n_iterations: iters.max(0) as usize,
                // AMGX_SOLVE_SUCCESS == 0
                converged: status == 0,
            })
        }
    }

    impl Drop for Session {
        fn drop(&mut self) {
            // SAFETY: each handle was produced by the matching create call and
            // is destroyed exactly once, in reverse order of creation. Errors
            // are ignored because a destructor has nowhere to report them and
            // leaking would be worse.
            unsafe {
                let _ = AMGX_solver_destroy(self.slv);
                let _ = AMGX_vector_destroy(self.sol);
                let _ = AMGX_vector_destroy(self.rhs);
                let _ = AMGX_matrix_destroy(self.mtx);
                let _ = AMGX_resources_destroy(self.rsc);
                let _ = AMGX_config_destroy(self.cfg);
                let _ = AMGX_finalize();
            }
        }
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// With the feature off the backend is still a candidate; what it is not
    /// is applicable. The distinction is the whole point of the file.
    #[test]
    fn without_the_feature_it_is_listed_but_unavailable() {
        let b = AmgxBackend::new();
        let probe = SystemProbe { n_cells: 1_000_000, symmetric: true, ..Default::default() };

        assert_eq!(b.name(), "AMGX");
        assert_eq!(AmgxBackend::compiled_in(), cfg!(feature = "amgx"));

        if !AmgxBackend::compiled_in() {
            assert!(!b.applicable(&probe));
            assert!(
                b.why_not(&probe).contains("feature 'amgx' not enabled"),
                "{}",
                b.why_not(&probe)
            );
        }
    }

    #[test]
    fn a_small_mesh_is_below_the_amg_floor() {
        let b = AmgxBackend::new();
        let probe = SystemProbe { n_cells: 100, symmetric: true, ..Default::default() };
        assert!(!b.applicable(&probe));
        assert!(!AmgxBackend::suits(&probe));
    }
}
