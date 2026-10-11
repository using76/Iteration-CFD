// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
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
//! Nothing is linked at build time any more: the library is opened at RUN
//! time with `libloading` (SPEC-LIT §141.20), so `--features amgx` compiles
//! on a machine that has no AMGX at all, and absence there is reported -
//! "AMGX library not loaded: tried ..." - not a link error. The recipe that
//! builds the DLL is `../tools/amgx/build.cmd`. At run time the search is:
//!
//! ```text
//! AMGX_DLL set:            that file, alone
//! else AMGX_DIR set at D:  D/<file>, D/bin/, D/lib/, D/build/, D/build/Release/
//! else:                    the bare name, so the system loader searches PATH
//! ```
//!
//! `OFGPU_REQUIRE_AMGX=1` makes absence an error; without it absence prints
//! "AMGX not loaded, skipped: ..." and the run goes on without AMGX.
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

// ==========================================================================
//  Finding the library (compiled in BOTH feature states)
// ==========================================================================

/// The shared-library file name the backend opens (SPEC-LIT §141.20): the
/// shared build, which is the only AMGX build that exports the C API.
#[cfg(windows)]
pub const LIB_NAME: &str = "amgxsh.dll";
#[cfg(not(windows))]
pub const LIB_NAME: &str = "libamgxsh.so";

/// The run-time search of SPEC-LIT §141.20, on fixed inputs so it can be
/// tested without touching the environment. An explicit setting is used
/// ALONE, so a typo never loads a different copy of the library from
/// somewhere else; the empty string counts as unset.
pub fn candidates_from(
    dll: Option<&std::ffi::OsStr>,
    dir: Option<&std::ffi::OsStr>,
) -> Vec<std::path::PathBuf> {
    if let Some(dll) = dll.filter(|v| !v.is_empty()) {
        return vec![std::path::PathBuf::from(dll)];
    }
    if let Some(dir) = dir.filter(|v| !v.is_empty()) {
        let d = std::path::PathBuf::from(dir);
        return vec![
            d.join(LIB_NAME),
            d.join("bin").join(LIB_NAME),
            d.join("lib").join(LIB_NAME),
            d.join("build").join(LIB_NAME),
            d.join("build").join("Release").join(LIB_NAME),
        ];
    }
    vec![std::path::PathBuf::from(LIB_NAME)]
}

/// [`candidates_from`] on this process's `AMGX_DLL` and `AMGX_DIR`.
pub fn candidates() -> Vec<std::path::PathBuf> {
    candidates_from(
        std::env::var_os("AMGX_DLL").as_deref(),
        std::env::var_os("AMGX_DIR").as_deref(),
    )
}

/// `true` iff the value is exactly `1` - SPEC-LIT §141.20's "the value `1`,
/// and only `1`, makes absence a failure".
pub fn require_from(v: Option<&std::ffi::OsStr>) -> bool {
    v == Some(std::ffi::OsStr::new("1"))
}

/// [`require_from`] on this process's `OFGPU_REQUIRE_AMGX`.
pub fn required() -> bool {
    require_from(std::env::var_os("OFGPU_REQUIRE_AMGX").as_deref())
}

/// What absence means: required is an error naming the switch, not required
/// is a printed skip and `Ok(false)` - the convention of a machine without
/// a card.
pub fn absence_verdict(required: bool, reason: &str) -> Result<bool> {
    if required {
        return Err(Error::Config(format!("OFGPU_REQUIRE_AMGX=1 but {reason}")));
    }
    eprintln!("AMGX not loaded, skipped: {reason}");
    Ok(false)
}

/// `Ok(true)` when the AMGX library is loaded in this process; otherwise
/// the [`absence_verdict`] for the reason it is not. A device test that
/// needs AMGX starts with `require_loaded()?` and gets a real failure under
/// `OFGPU_REQUIRE_AMGX=1`, a vacuous pass without it.
pub fn require_loaded() -> Result<bool> {
    #[cfg(feature = "amgx")]
    match imp::api() {
        Ok(_) => Ok(true),
        Err(reason) => absence_verdict(required(), &reason),
    }
    #[cfg(not(feature = "amgx"))]
    {
        absence_verdict(required(), "feature 'amgx' not enabled")
    }
}

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
        // `suits` first, so a small mesh never even attempts the load.
        Self::suits(probe) && library().is_ok()
    }

    fn why_not(&self, probe: &SystemProbe) -> String {
        if let Err(reason) = library() {
            return reason;
        }
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
        // SPEC-LIT §48.2 removed the guard that used to stand here. A coupled
        // patch's `boundaryCoeffs` now has a column in the CSR and
        // `lduCsrFill` writes it, so the exported matrix IS the operator
        // `amul` applies - which it was not before, and which is why every
        // periodic mesh was refused. `CsrPattern::build` still refuses the one
        // case it cannot represent (two cells joined by both an internal face
        // and a coupled boundary face), and that refusal arrives from
        // `Session::new` below.
        library().map_err(Error::Config)?;
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
//  Opening the library, only compiled with the feature
// ==========================================================================

/// The file the process's one open came from - SPEC-LIT §141.20's search run
/// over [`candidates`] and cached. `Err` carries the reason the search
/// failed, which is the backend's "unavailable" reason and `setup`'s error.
#[cfg(feature = "amgx")]
pub fn library() -> std::result::Result<std::path::PathBuf, String> {
    imp::api().map(|a| a.path.clone())
}

/// Open each candidate in turn and return the one that opened - the search
/// WITHOUT keeping anything: the library is dropped before returning and
/// the process-wide cache is never touched. A device test probes with this
/// before committing to a machine, and the error is the same string the
/// real load would have reported.
#[cfg(feature = "amgx")]
pub fn probe_paths(
    paths: &[std::path::PathBuf],
) -> std::result::Result<std::path::PathBuf, String> {
    let api = imp::open_resolved(paths)?;
    let path = api.path.clone();
    drop(api); // nothing persists: the next query reopens from scratch
    Ok(path)
}

// ==========================================================================
//  The FFI, only compiled with the feature
// ==========================================================================

#[cfg(feature = "amgx")]
mod imp {
    //! Hand-written bindings to the handful of `amgx_c.h` entry points a
    //! single-GPU, single-rank solve needs, resolved from a RUN-time-loaded
    //! library (SPEC-LIT §141.20) instead of a build-time link.
    //!
    //! Written out rather than bindgen-ed because it is fourteen functions
    //! against a C API that NVIDIA calls "C-API stable", and adding a
    //! build-time dependency on libclang to a crate whose whole build story is
    //! "cargo build" would cost more than it saves.

    use std::ffi::{c_char, c_int, c_void, CString};
    use std::path::{Path, PathBuf};

    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use libloading::{Library, Symbol};

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

    /// The resolved C API: one raw entry point per `amgx_c.h` function this
    /// backend binds, plus the `Library` that keeps the code they point into
    /// mapped. Resolved ONCE at open (SPEC-LIT §141.20); a missing symbol
    /// refuses the open, naming the entry point and the file.
    #[allow(non_snake_case)]
    pub struct Api {
        /// The file that opened - what `library()` reports.
        pub path: PathBuf,
        /// Outlives every pointer below: the `API` static that holds this
        /// struct is never dropped, so the code stays mapped for the whole
        /// process. Dropping the last `Library` would unload it.
        _lib: Library,

        pub AMGX_initialize: unsafe extern "C" fn() -> AmgxRc,
        pub AMGX_finalize: unsafe extern "C" fn() -> AmgxRc,
        pub AMGX_get_error_string:
            unsafe extern "C" fn(AmgxRc, *mut c_char, c_int) -> AmgxRc,
        pub AMGX_config_create:
            unsafe extern "C" fn(*mut Handle, *const c_char) -> AmgxRc,
        pub AMGX_config_destroy: unsafe extern "C" fn(Handle) -> AmgxRc,
        pub AMGX_resources_create_simple:
            unsafe extern "C" fn(*mut Handle, Handle) -> AmgxRc,
        pub AMGX_resources_destroy: unsafe extern "C" fn(Handle) -> AmgxRc,
        pub AMGX_matrix_create:
            unsafe extern "C" fn(*mut Handle, Handle, c_int) -> AmgxRc,
        pub AMGX_matrix_destroy: unsafe extern "C" fn(Handle) -> AmgxRc,
        pub AMGX_matrix_upload_all: unsafe extern "C" fn(
            Handle,
            c_int,
            c_int,
            c_int,
            c_int,
            *const c_int,
            *const c_int,
            *const c_void,
            *const c_void,
        ) -> AmgxRc,
        pub AMGX_matrix_replace_coefficients:
            unsafe extern "C" fn(Handle, c_int, c_int, *const c_void, *const c_void) -> AmgxRc,
        pub AMGX_vector_create:
            unsafe extern "C" fn(*mut Handle, Handle, c_int) -> AmgxRc,
        pub AMGX_vector_destroy: unsafe extern "C" fn(Handle) -> AmgxRc,
        pub AMGX_vector_upload:
            unsafe extern "C" fn(Handle, c_int, c_int, *const c_void) -> AmgxRc,
        pub AMGX_vector_download:
            unsafe extern "C" fn(Handle, *mut c_void) -> AmgxRc,
        pub AMGX_solver_create:
            unsafe extern "C" fn(*mut Handle, Handle, c_int, Handle) -> AmgxRc,
        pub AMGX_solver_destroy: unsafe extern "C" fn(Handle) -> AmgxRc,
        pub AMGX_solver_setup: unsafe extern "C" fn(Handle, Handle) -> AmgxRc,
        pub AMGX_solver_solve:
            unsafe extern "C" fn(Handle, Handle, Handle) -> AmgxRc,
        pub AMGX_solver_get_iterations_number:
            unsafe extern "C" fn(Handle, *mut c_int) -> AmgxRc,
        pub AMGX_solver_get_iteration_residual:
            unsafe extern "C" fn(Handle, c_int, c_int, *mut f64) -> AmgxRc,
        pub AMGX_solver_get_status:
            unsafe extern "C" fn(Handle, *mut c_int) -> AmgxRc,
    }

    /// One entry point by name, typed by the caller's let binding. The raw
    /// fn pointer is copied out of the `Symbol`, so nothing borrows the
    /// library afterwards - the `API` static keeps it mapped instead.
    fn resolve_symbol<T: Copy + 'static>(
        lib: &Library,
        path: &Path,
        name: &str,
    ) -> std::result::Result<T, String> {
        let nul = format!("{name}\0");
        // SAFETY: `nul` is NUL-terminated as `get` requires, and the pointer
        // stays valid because the library is never unloaded (see `Api::_lib`).
        let sym: Symbol<'_, T> = unsafe { lib.get(nul.as_bytes()) }.map_err(|_| {
            format!("AMGX library {} lacks symbol {name}", path.display())
        })?;
        Ok(*sym)
    }

    impl Api {
        /// Resolve every entry point in an already-opened library. The first
        /// missing symbol refuses the whole open, naming the entry point and
        /// the file (SPEC-LIT §141.20).
        fn open_with(path: PathBuf, lib: Library) -> std::result::Result<Self, String> {
            let p = &path;
            Ok(Self {
                AMGX_initialize: resolve_symbol(&lib, p, "AMGX_initialize")?,
                AMGX_finalize: resolve_symbol(&lib, p, "AMGX_finalize")?,
                AMGX_get_error_string: resolve_symbol(&lib, p, "AMGX_get_error_string")?,
                AMGX_config_create: resolve_symbol(&lib, p, "AMGX_config_create")?,
                AMGX_config_destroy: resolve_symbol(&lib, p, "AMGX_config_destroy")?,
                AMGX_resources_create_simple: resolve_symbol(
                    &lib,
                    p,
                    "AMGX_resources_create_simple",
                )?,
                AMGX_resources_destroy: resolve_symbol(&lib, p, "AMGX_resources_destroy")?,
                AMGX_matrix_create: resolve_symbol(&lib, p, "AMGX_matrix_create")?,
                AMGX_matrix_destroy: resolve_symbol(&lib, p, "AMGX_matrix_destroy")?,
                AMGX_matrix_upload_all: resolve_symbol(&lib, p, "AMGX_matrix_upload_all")?,
                AMGX_matrix_replace_coefficients: resolve_symbol(
                    &lib,
                    p,
                    "AMGX_matrix_replace_coefficients",
                )?,
                AMGX_vector_create: resolve_symbol(&lib, p, "AMGX_vector_create")?,
                AMGX_vector_destroy: resolve_symbol(&lib, p, "AMGX_vector_destroy")?,
                AMGX_vector_upload: resolve_symbol(&lib, p, "AMGX_vector_upload")?,
                AMGX_vector_download: resolve_symbol(&lib, p, "AMGX_vector_download")?,
                AMGX_solver_create: resolve_symbol(&lib, p, "AMGX_solver_create")?,
                AMGX_solver_destroy: resolve_symbol(&lib, p, "AMGX_solver_destroy")?,
                AMGX_solver_setup: resolve_symbol(&lib, p, "AMGX_solver_setup")?,
                AMGX_solver_solve: resolve_symbol(&lib, p, "AMGX_solver_solve")?,
                AMGX_solver_get_iterations_number: resolve_symbol(
                    &lib,
                    p,
                    "AMGX_solver_get_iterations_number",
                )?,
                AMGX_solver_get_iteration_residual: resolve_symbol(
                    &lib,
                    p,
                    "AMGX_solver_get_iteration_residual",
                )?,
                AMGX_solver_get_status: resolve_symbol(&lib, p, "AMGX_solver_get_status")?,
                path,
                _lib: lib,
            })
        }
    }

    /// Turn an `AMGX_RC` into a `Result`, with AMGX's own message attached.
    fn check(what: &str, rc: AmgxRc) -> Result<()> {
        if rc == AMGX_RC_OK {
            return Ok(());
        }
        let api = api().map_err(Error::Config)?;
        let mut buf = [0i8; 512];
        // SAFETY: AMGX writes at most `buf.len()` bytes into the buffer and
        // NUL-terminates. Ignoring its return code is deliberate - we are
        // already reporting a failure and a failure to describe it must not
        // replace the original.
        let msg = unsafe {
            let _ = (api.AMGX_get_error_string)(
                rc,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            );
            let bytes: Vec<u8> = buf
                .iter()
                .take_while(|b| **b != 0)
                .map(|b| *b as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        Err(Error::Config(format!("AMGX: {what} failed ({rc}): {msg}")))
    }

    /// The process's one open, resolved - SPEC-LIT §141.20's "opened once per
    /// process, on the first query, and stays open". A refusal is cached too:
    /// re-running a search that cannot succeed on every solve would only
    /// re-print the same reason.
    static API: std::sync::OnceLock<std::result::Result<Api, String>> =
        std::sync::OnceLock::new();

    /// The cached [`Api`], opened lazily over [`super::candidates`] on the
    /// first query. `Err` carries the exact reason string the backend
    /// reports everywhere.
    pub fn api() -> std::result::Result<&'static Api, String> {
        API.get_or_init(|| open_resolved(&super::candidates()))
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// The SPEC-LIT §141.20 load-error string: every candidate tried, the
    /// two variables that would fix it, and the loader's last error.
    fn load_record(paths: &[PathBuf]) -> std::result::Result<(PathBuf, Library), String> {
        let mut last: Option<String> = None;
        for p in paths {
            // SAFETY: loading a library runs its initialisation routines -
            // for AMGX, nothing beyond CUDA's on first use - and the user
            // pointed this search at it.
            match unsafe { Library::new(p) } {
                Ok(lib) => return Ok((p.clone(), lib)),
                Err(e) => last = Some(e.to_string()),
            }
        }
        let tried: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        Err(format!(
            "AMGX library not loaded: tried {}; set AMGX_DLL to the library \
             file or AMGX_DIR to its directory (last error: {})",
            tried.join(", "),
            last.as_deref().unwrap_or("nothing was tried"),
        ))
    }

    /// Open the first candidate that opens and resolve every entry point.
    /// This is the whole load path; the cache and the probe both run it.
    pub fn open_resolved(paths: &[PathBuf]) -> std::result::Result<Api, String> {
        let (path, lib) = load_record(paths)?;
        Api::open_with(path, lib)
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
            let pattern = CsrPattern::build(hm)?;
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
            let api = api().map_err(Error::Config)?;
            unsafe {
                check("initialize", (api.AMGX_initialize)())?;
                check("config_create", (api.AMGX_config_create)(&mut cfg, options.as_ptr()))?;
                check(
                    "resources_create_simple",
                    (api.AMGX_resources_create_simple)(&mut rsc, cfg),
                )?;
                check("matrix_create", (api.AMGX_matrix_create)(&mut mtx, rsc, MODE))?;
                check("vector_create(rhs)", (api.AMGX_vector_create)(&mut rhs, rsc, MODE))?;
                check("vector_create(sol)", (api.AMGX_vector_create)(&mut sol, rsc, MODE))?;
                check(
                    "solver_create",
                    (api.AMGX_solver_create)(&mut slv, rsc, MODE, cfg),
                )?;
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
            let api = api().map_err(Error::Config)?;
            unsafe {
                if !self.uploaded {
                    check(
                        "matrix_upload_all",
                        (api.AMGX_matrix_upload_all)(
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
                        (api.AMGX_matrix_replace_coefficients)(
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
                    (api.AMGX_vector_upload)(self.rhs, n, 1, source as *const c_void),
                )?;
                check(
                    "vector_upload(sol)",
                    (api.AMGX_vector_upload)(self.sol, n, 1, psi as *const c_void),
                )?;

                check("solver_setup", (api.AMGX_solver_setup)(self.slv, self.mtx))?;
                check("solver_solve", (api.AMGX_solver_solve)(self.slv, self.rhs, self.sol))?;
                check(
                    "vector_download",
                    (api.AMGX_vector_download)(self.sol, psi as *mut c_void),
                )?;
            }

            // Everything above - the download included - AMGX issued on the
            // CUDA runtime's legacy default stream, while this crate runs on
            // a non-blocking stream; a device-to-device copy does not
            // host-synchronise, so `gpu.sync()` (the crate's stream only)
            // would not order the caller's next read of `psi` after it.
            // Synchronise the CONTEXT, which waits for AMGX's stream too.
            gpu.ctx().synchronize()?;

            let mut iters: c_int = 0;
            let mut status: c_int = 0;
            let mut res0 = 0.0f64;
            let mut res1 = 0.0f64;
            unsafe {
                let _ = (api.AMGX_solver_get_iterations_number)(self.slv, &mut iters);
                let _ = (api.AMGX_solver_get_status)(self.slv, &mut status);
                if iters > 0 {
                    let _ = (api.AMGX_solver_get_iteration_residual)(self.slv, 0, 0, &mut res0);
                    let _ =
                        (api.AMGX_solver_get_iteration_residual)(self.slv, iters - 1, 0, &mut res1);
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
            // The library is process-wide and never unloaded (the `API`
            // static holds it), so the pointers below are valid for as long
            // as any Session can exist.
            let Ok(api) = api() else { return };
            // SAFETY: each handle was produced by the matching create call and
            // is destroyed exactly once, in reverse order of creation. Errors
            // are ignored because a destructor has nowhere to report them and
            // leaking would be worse.
            unsafe {
                let _ = (api.AMGX_solver_destroy)(self.slv);
                let _ = (api.AMGX_vector_destroy)(self.sol);
                let _ = (api.AMGX_vector_destroy)(self.rhs);
                let _ = (api.AMGX_matrix_destroy)(self.mtx);
                let _ = (api.AMGX_resources_destroy)(self.rsc);
                let _ = (api.AMGX_config_destroy)(self.cfg);
                let _ = (api.AMGX_finalize)();
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

    use std::path::PathBuf;

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

    /// The search of SPEC-LIT §141.20 on fixed inputs: an explicit
    /// `AMGX_DLL` is used alone, `AMGX_DIR` expands to the five places a
    /// build puts the DLL, the empty string is unset, and what is left is
    /// the bare name the system loader resolves along PATH.
    #[test]
    fn the_search_order_is_amgx_dll_then_amgx_dir_then_path() {
        use std::ffi::OsStr;
        use std::path::Path;

        assert_eq!(
            candidates_from(Some(OsStr::new("X/a.dll")), Some(OsStr::new("D"))),
            vec![Path::new("X/a.dll").to_path_buf()]
        );
        let d = Path::new("D");
        assert_eq!(
            candidates_from(None, Some(OsStr::new("D"))),
            vec![
                d.join(LIB_NAME),
                d.join("bin").join(LIB_NAME),
                d.join("lib").join(LIB_NAME),
                d.join("build").join(LIB_NAME),
                d.join("build").join("Release").join(LIB_NAME),
            ]
        );
        assert_eq!(
            candidates_from(Some(OsStr::new("")), None),
            vec![PathBuf::from(LIB_NAME)]
        );
        assert_eq!(
            candidates_from(None, Some(OsStr::new(""))),
            vec![PathBuf::from(LIB_NAME)]
        );
    }

    /// `1` requires; `0`, `true`, the empty string and unset do not.
    #[test]
    fn only_the_value_1_requires_amgx() {
        use std::ffi::OsStr;

        assert!(require_from(Some(OsStr::new("1"))));
        assert!(!require_from(Some(OsStr::new("0"))));
        assert!(!require_from(Some(OsStr::new("true"))));
        assert!(!require_from(Some(OsStr::new(""))));
        assert!(!require_from(None));
    }

    /// Absent and required is an error naming the switch and the reason;
    /// absent and not required is a printed skip and `Ok(false)`.
    #[test]
    fn require_turns_absence_into_an_error() {
        let err = absence_verdict(true, "R").expect_err("required absence is an error");
        let msg = err.to_string();
        assert!(msg.contains("OFGPU_REQUIRE_AMGX=1"), "{msg}");
        assert!(msg.contains("R"), "{msg}");
        assert_eq!(absence_verdict(false, "R").ok(), Some(false));
    }

    /// A candidate that does not exist gives the "AMGX library not loaded"
    /// reason, names the path it tried, and names the variable that would
    /// have fixed it - without touching the process-wide cache.
    #[cfg(feature = "amgx")]
    #[test]
    fn a_missing_library_is_unavailable_with_the_paths_it_tried() {
        let err = probe_paths(&[PathBuf::from("Z:/no/such/dir/amgxsh.dll")])
            .expect_err("nothing exists at that path");
        assert!(err.contains("AMGX library not loaded"), "{err}");
        assert!(err.contains("Z:/no/such/dir/amgxsh.dll"), "{err}");
        assert!(err.contains("AMGX_DLL"), "{err}");
    }
}
