// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The GPU handle: context, stream, modules, memory.
//!
//! Everything the rest of the crate does to the device goes through here, so
//! that if `cudarc`'s pre-1.0 API shifts, one file absorbs it.
//!
//! Ownership rules this enforces, which the C++ version could only document:
//!
//! * a `DevBuf<T>` frees its memory when it drops, and cannot be freed twice;
//! * a buffer cannot outlive the context that allocated it (`Arc`);
//! * a `Graph` is `!Send`/`!Sync`, so the "graph objects are not internally
//!   synchronised" rule in the CUDA docs becomes a compile error rather than
//!   a race.
//!
//! Provenance: ORIGINAL - the cudarc wrapper (context, dedicated non-blocking
//! stream, `DevBuf`, `KernelSet`, CUDA-graph capture). No external source: this
//! is ownership plumbing over the CUDA driver API, with no CFD analogue
//! anywhere. `PROVENANCE.md`, *GPU plumbing and tooling - original*. No
//! GPL-licensed source was consulted.

use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use crate::error::{Error, Result};

/// An owned device buffer. Named rather than aliased so the rest of the crate
/// never spells `CudaSlice` directly.
pub type DevBuf<T> = CudaSlice<T>;

/// Threads per block. 256 is what the C++ version used and what the kernels
/// were tuned against; changing it changes nothing about correctness because
/// every kernel is written as a flat 1-D grid with an out-of-range early exit.
pub const BLOCK: u32 = 256;

/// Launch configuration for `n` independent items.
///
/// Returns a zero-block config for `n == 0`; callers must skip the launch,
/// because a grid dimension of zero is an invalid configuration, not a no-op.
#[inline]
pub fn cfg_for(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;

        // Event tracking OFF, and it must be off BEFORE the first allocation:
        // cudarc only skips it for buffers created afterwards. With it on,
        // every `.arg(&buf)` can drop a `SyncOnDrop` that synchronises the
        // stream, which is illegal mid-capture and pointless here anyway -
        // this crate uses exactly one stream, so there is no cross-stream
        // hazard for cudarc to guard against.
        //
        // SAFETY: the contract is "the user manages stream synchronisation".
        // There is one stream and it is created immediately below; no buffer
        // is ever touched from another.
        unsafe { ctx.disable_event_tracking() };

        // A dedicated non-blocking stream, NOT `default_stream()`.
        //
        // `default_stream()` hands back the legacy NULL stream, and
        // `cuStreamBeginCapture` refuses it outright with
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED - so using it would quietly
        // cost us CUDA graphs entirely. A non-blocking stream also stops the
        // legacy stream's implicit synchronisation with every other stream in
        // the process, which matters the moment this crate is embedded in
        // something larger.
        let stream = ctx.new_stream()?;

        Ok(Self { ctx, stream })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.ctx.name()?)
    }

    /// Load one of the CUBIN blobs from [`crate::kernels`].
    pub fn load(&self, cubin: &[u8]) -> Result<Arc<CudaModule>> {
        Ok(self.ctx.load_module(Ptx::from_binary(cubin.to_vec()))?)
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    /// Total and free device memory, in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (free, total) = cudarc::driver::result::mem_get_info()?;
        Ok((free, total))
    }

    // ---- memory -----------------------------------------------------------

    pub fn zeros<T>(&self, n: usize) -> Result<DevBuf<T>>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        Ok(self.stream.alloc_zeros::<T>(n)?)
    }

    pub fn upload<T: DeviceRepr>(&self, src: &[T]) -> Result<DevBuf<T>> {
        Ok(self.stream.clone_htod(src)?)
    }

    pub fn download<T: DeviceRepr>(&self, src: &DevBuf<T>) -> Result<Vec<T>> {
        Ok(self.stream.clone_dtoh(src)?)
    }

    pub fn fill_zero<T>(&self, dst: &mut DevBuf<T>) -> Result<()>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        Ok(self.stream.memset_zeros(dst)?)
    }

    /// Overwrite an existing buffer from the host. Setup only - nothing in the
    /// time loop is allowed to call this.
    pub fn write<T: DeviceRepr>(&self, dst: &mut DevBuf<T>, src: &[T]) -> Result<()> {
        if src.len() != dst.len() {
            return Err(Error::Config(format!(
                "write: host slice has {} elements, device buffer has {}",
                src.len(),
                dst.len()
            )));
        }
        Ok(self.stream.memcpy_htod(src, dst)?)
    }

    // ---- CUDA graphs ------------------------------------------------------

    /// Capture everything `body` launches into a replayable graph.
    ///
    /// The captured region must contain **no host round-trip** - no
    /// synchronisation, no read-back, no allocation. That is exactly why the
    /// solver has a fixed-iteration mode: an adaptive convergence test reads a
    /// flag back to the host every few iterations, and a graph cannot capture
    /// a decision the host makes.
    ///
    /// Returns `None` if the capture produced no work.
    pub fn capture<F>(&self, body: F) -> Result<Option<Graph>>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<()>,
    {
        // Relaxed rather than Global: Global would fail the capture if any
        // other thread in the process touched a legacy default stream, which
        // is not something this crate can promise about its embedder.
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;

        // If the body fails mid-capture the stream is left in capturing mode,
        // so end it before propagating.
        if let Err(e) = body(&self.stream) {
            let _ = self
                .stream
                .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            return Err(e);
        }

        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;

        Ok(graph.map(Graph::new))
    }
}

/// A captured, instantiated graph.
///
/// `CudaGraph` is `!Send` and `!Sync` in cudarc, and that propagates here, so
/// the compiler refuses to let one cross a thread boundary.
pub struct Graph {
    inner: CudaGraph,
    uploaded: bool,
}

impl Graph {
    fn new(inner: CudaGraph) -> Self {
        Self { inner, uploaded: false }
    }

    /// Push the graph to the device ahead of the first launch, so the first
    /// replay is not slower than the rest.
    pub fn upload(&mut self) -> Result<()> {
        self.inner.upload()?;
        self.uploaded = true;
        Ok(())
    }

    pub fn launch(&self) -> Result<()> {
        Ok(self.inner.launch()?)
    }

    pub fn is_uploaded(&self) -> bool {
        self.uploaded
    }
}

/// Resolve kernel entry points once and hold them, so the time loop never
/// does a string lookup.
pub struct KernelSet {
    module: Arc<CudaModule>,
}

impl KernelSet {
    pub fn new(gpu: &Gpu, cubin: &[u8]) -> Result<Self> {
        Ok(Self { module: gpu.load(cubin)? })
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        self.module.load_function(name).map_err(|_| {
            Error::Config(format!(
                "kernel '{name}' is not in the loaded module - \
                 is it declared extern \"C\"?"
            ))
        })
    }
}
