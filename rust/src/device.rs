// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

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
//! stream, `DevBuf`, `KernelSet`, CUDA-graph capture, the per-process pool
//! reading of SPEC-LIT 111.2). No external source: this is ownership plumbing
//! over the CUDA driver API, with no CFD analogue anywhere. `PROVENANCE.md`,
//! *GPU plumbing and tooling - original*. No GPL-licensed source was
//! consulted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

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
    /// True between `begin_capture` and `end_capture`. See
    /// [`Gpu::refuse_during_capture`] and `SPEC-LIT` 81.3.
    capturing: AtomicBool,
}

/// This process's own device memory, read from the stream-ordered memory
/// pool every [`Gpu::zeros`] and [`Gpu::upload`] allocates from.
///
/// Unlike [`Gpu::mem_info`], which reports the whole device and, under the
/// Windows display driver, counts another process's memory only some of the
/// time, these figures belong to this process alone. SPEC-LIT 111.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolUsage {
    /// Bytes handed out to live buffers (`CU_MEMPOOL_ATTR_USED_MEM_CURRENT`).
    pub used: u64,
    /// Bytes the pool holds from the driver, used plus cached
    /// (`CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT`).
    pub reserved: u64,
    /// The highest `used` so far (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`).
    pub used_high: u64,
    /// Cached bytes the pool keeps at a synchronise
    /// (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`); 0 is the driver default.
    pub release_threshold: u64,
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

        Ok(Self { ctx, stream, capturing: AtomicBool::new(false) })
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
        self.refuse_during_capture("load")?;
        Ok(self.ctx.load_module(Ptx::from_binary(cubin.to_vec()))?)
    }

    pub fn sync(&self) -> Result<()> {
        self.refuse_during_capture("sync")?;
        Ok(self.stream.synchronize()?)
    }

    /// Total and free device memory, in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.refuse_during_capture("mem_info")?;
        let (free, total) = cudarc::driver::result::mem_get_info()?;
        Ok((free, total))
    }

    /// This process's own pool figures, SPEC-LIT 111.2. A host query, so it
    /// is refused during a capture like [`Gpu::mem_info`] (§81.3).
    pub fn pool_usage(&self) -> Result<PoolUsage> {
        use cudarc::driver::sys::CUmemPool_attribute as A;
        self.refuse_during_capture("pool_usage")?;
        self.ctx.bind_to_thread()?;
        // SAFETY: `cu_device()` is the device this context was created on,
        // so `get_mem_pool` receives a device the driver returned; each
        // attribute read below is one of the four `cuuint64_t` attributes and
        // writes exactly the eight bytes of `v`.
        unsafe {
            let pool = cudarc::driver::result::device::get_mem_pool(self.ctx.cu_device())?;
            let read = |a: A| -> Result<u64> {
                let mut v: u64 = 0;
                cudarc::driver::result::mem_pool::get_attribute(
                    pool,
                    a,
                    (&mut v as *mut u64).cast(),
                )?;
                Ok(v)
            };
            Ok(PoolUsage {
                used: read(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
                reserved: read(A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
                used_high: read(A::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
                release_threshold: read(A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)?,
            })
        }
    }

    // ---- the capture guard -------------------------------------------
    //
    // SPEC-LIT 81.3. A CUDA graph records a sequence of *device* work. The
    // moment the host is asked a question mid-capture - "how much memory is
    // free", "what does this buffer hold", "give me a fresh allocation" -
    // there is nothing to record, and the driver answers with
    // `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`: a number, naming nothing.
    //
    // That is the failure this crate has to be able to read. A future module
    // that slips a `download` into its per-iteration path breaks capture for
    // every module downstream of it, and the driver's error says only that
    // some operation somewhere was unsupported. So the crate asks first, and
    // names the operation and the rule it broke.

    /// True while [`Gpu::capture`] is recording this stream.
    #[inline]
    pub fn is_capturing(&self) -> bool {
        self.capturing.load(Ordering::Relaxed)
    }

    /// Refuse `op` if a capture is in progress, naming it.
    ///
    /// `SPEC-LIT` 13.4: refused by name, with the alternative. Every caller is
    /// a host round-trip, and the alternative is always the same one - keep
    /// the value on the device.
    #[inline]
    fn refuse_during_capture(&self, op: &str) -> Result<()> {
        if self.is_capturing() {
            return Err(Error::Config(format!(
                "Gpu::{op} was called inside a CUDA-graph capture. A \
                 graph records device work; {op} is a host round-trip, so \
                 there is nothing to record, and the capture would otherwise \
                 fail with an unattributed \
                 CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED. Keep the value on the \
                 device - solver.rs keeps its residuals there for exactly \
                 this reason - or move the call outside the captured region. \
                 SPEC-LIT 81.3"
            )));
        }
        Ok(())
    }

    // ---- memory -----------------------------------------------------------

    #[track_caller]
    pub fn zeros<T>(&self, n: usize) -> Result<DevBuf<T>>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        self.refuse_during_capture("zeros")?;
        let at = std::panic::Location::caller();
        let buf = self.stream.alloc_zeros::<T>(n)?;
        if mem_trace_on() {
            self.trace_allocation(at, n * std::mem::size_of::<T>())?;
        }
        Ok(buf)
    }

    #[track_caller]
    pub fn upload<T: DeviceRepr>(&self, src: &[T]) -> Result<DevBuf<T>> {
        self.refuse_during_capture("upload")?;
        let at = std::panic::Location::caller();
        let buf = self.stream.clone_htod(src)?;
        if mem_trace_on() {
            self.trace_allocation(at, std::mem::size_of_val(src))?;
        }
        Ok(buf)
    }

    /// One `mem-trace` line on stderr, SPEC-LIT 111.2. The stream is
    /// synchronised first: without it the card's free figure lags the queued
    /// work and the bytes are charged to a later allocation.
    fn trace_allocation(&self, at: &std::panic::Location<'_>, bytes: usize) -> Result<()> {
        self.stream.synchronize()?;
        let (free, _) = cudarc::driver::result::mem_get_info()?;
        let pool = self.pool_usage()?;
        eprintln!(
            "mem-trace {}:{} | requested {bytes} B | pool used {} B | card free {free} B",
            at.file(),
            at.line(),
            pool.used
        );
        Ok(())
    }

    pub fn download<T: DeviceRepr>(&self, src: &DevBuf<T>) -> Result<Vec<T>> {
        self.refuse_during_capture("download")?;
        Ok(self.stream.clone_dtoh(src)?)
    }

    /// Zero a buffer in place.
    ///
    /// Deliberately **not** guarded against capture: `cuMemsetD8Async` becomes
    /// a memset node in the graph and replays correctly, so zeroing an
    /// accumulator at the top of an iteration is legal and several modules do
    /// it. It is the one device-side write on this type that is (`SPEC-LIT`
    /// 81.3, row *memset*).
    pub fn fill_zero<T>(&self, dst: &mut DevBuf<T>) -> Result<()>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        Ok(self.stream.memset_zeros(dst)?)
    }

    /// Overwrite an existing buffer from the host. Setup only - nothing in the
    /// time loop is allowed to call this, and inside a capture it is refused
    /// by name (`SPEC-LIT` 81.3).
    pub fn write<T: DeviceRepr>(&self, dst: &mut DevBuf<T>, src: &[T]) -> Result<()> {
        if src.len() != dst.len() {
            return Err(Error::Config(format!(
                "write: host slice has {} elements, device buffer has {}",
                src.len(),
                dst.len()
            )));
        }
        self.refuse_during_capture("write")?;
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
        // One stream, so one capture. A nested `capture` would silently fold
        // the inner region into the outer graph and hand back `None`, which
        // reads as "the body launched nothing" - the most misleading answer
        // available. Refuse it by name instead.
        if self.is_capturing() {
            return Err(Error::Config(
                "Gpu::capture was called while a capture is already \
                 recording this stream. There is one stream, so captures do \
                 not nest: the inner region would be folded into the outer \
                 graph and this call would return None, which reads as \
                 \"the body launched nothing\" - SPEC-LIT 81.3"
                    .to_string(),
            ));
        }

        // Relaxed rather than Global: Global would fail the capture if any
        // other thread in the process touched a legacy default stream, which
        // is not something this crate can promise about its embedder.
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        self.capturing.store(true, Ordering::Relaxed);

        // If the body fails mid-capture the stream is left in capturing mode,
        // so end it before propagating. The flag is cleared FIRST, because
        // `end_capture` is itself one of the calls the flag would refuse if
        // it were routed through the guard, and because a body that failed
        // must not leave the guard armed for the next caller.
        if let Err(e) = body(&self.stream) {
            self.capturing.store(false, Ordering::Relaxed);
            let _ = self
                .stream
                .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            return Err(e);
        }

        self.capturing.store(false, Ordering::Relaxed);
        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;

        Ok(graph.map(Graph::new))
    }
}

/// `OFGPU_MEM_TRACE` in the environment, read once. SPEC-LIT 111.2.
fn mem_trace_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OFGPU_MEM_TRACE").is_some())
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

    /// What the captured graph is actually made of, by node type.
    ///
    /// This is not decoration. `cudarc` allocates with `cuMemAllocAsync`,
    /// which is **stream-ordered and therefore capturable**: a module that
    /// allocates inside its per-iteration path does not fail the capture, it
    /// records a `MEM_ALLOC` node and succeeds. The graph then does a device
    /// allocation on every replay - which is most of what the graph existed to
    /// remove - and with
    /// `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` it frees it again on
    /// the next launch, so it does not even leak. Nothing complains.
    ///
    /// `alloc == 0 && free == 0 && host == 0` is the statement that the
    /// captured region is pure device work, and it is the only way to see it
    /// from outside. `SPEC-LIT` 81.4.
    pub fn shape(&self) -> Result<GraphShape> {
        use cudarc::driver::sys as cu;

        let g = self.inner.cu_graph();
        let mut n: usize = 0;
        // SAFETY: `g` is owned by `self.inner` and outlives this call. A null
        // node pointer with a null count is how the driver is asked for the
        // count alone.
        unsafe { cu::cuGraphGetNodes(g, std::ptr::null_mut(), &mut n) }.result()?;

        let mut nodes: Vec<cu::CUgraphNode> = vec![std::ptr::null_mut(); n];
        if n > 0 {
            // SAFETY: `nodes` has room for exactly the `n` the driver just
            // reported, and `n` is passed back unchanged.
            unsafe { cu::cuGraphGetNodes(g, nodes.as_mut_ptr(), &mut n) }.result()?;
        }

        let mut sh = GraphShape { total: n, ..Default::default() };
        for node in nodes.into_iter().take(n) {
            let mut t = cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            // SAFETY: `node` came from `cuGraphGetNodes` on a live graph.
            unsafe { cu::cuGraphNodeGetType(node, &mut t) }.result()?;
            match t {
                cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => sh.kernel += 1,
                cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => sh.memset += 1,
                cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => sh.memcpy += 1,
                cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => sh.host += 1,
                cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => sh.alloc += 1,
                cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => sh.free += 1,
                _ => sh.other += 1,
            }
        }
        Ok(sh)
    }
}

/// The node-type census of a captured graph. See [`Graph::shape`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphShape {
    /// Every node, whatever its kind.
    pub total: usize,
    /// Kernel launches - the work the graph exists to replay.
    pub kernel: usize,
    /// `cuMemsetD*Async`. Legal and expected: zeroing an accumulator.
    pub memset: usize,
    /// Device-to-device copies. Legal.
    pub memcpy: usize,
    /// A host callback. **Never legal in an iteration**: it serialises the
    /// replay on the CPU, which is what the graph was removing.
    pub host: usize,
    /// A stream-ordered allocation. **Never legal in an iteration.**
    pub alloc: usize,
    /// A stream-ordered free. **Never legal in an iteration.**
    pub free: usize,
    /// Events, child graphs, semaphores: none of this crate's doing.
    pub other: usize,
}

impl GraphShape {
    /// True when the graph is nothing but device work: no host callback, no
    /// allocation, no free.
    pub fn is_pure_device_work(&self) -> bool {
        self.host == 0 && self.alloc == 0 && self.free == 0
    }

    /// The impurities, named, for an assertion message. Empty when pure.
    pub fn impurities(&self) -> Vec<String> {
        let mut v = Vec::new();
        if self.host > 0 {
            v.push(format!("{} host-callback node(s)", self.host));
        }
        if self.alloc > 0 {
            v.push(format!(
                "{} stream-ordered allocation node(s) - something in the                  iteration calls a device allocator, so every replay allocates",
                self.alloc
            ));
        }
        if self.free > 0 {
            v.push(format!("{} stream-ordered free node(s)", self.free));
        }
        v
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool reading counts this process's own live buffers, and the
    /// driver defaults SPEC-LIT 111.1 measured hold. Upper bounds are
    /// deliberately absent: other lib tests share the pool in parallel.
    #[test]
    fn pool_usage_counts_this_process_own_buffers() {
        let Ok(gpu) = Gpu::new(0) else { return; };
        let held = gpu.zeros::<f64>(8 << 20).unwrap(); // 64 MiB, held live.
        gpu.sync().unwrap();
        let pool = gpu.pool_usage().unwrap();
        assert!(
            pool.used >= 64 << 20,
            "the pool's used bytes are {} after one live 64 MiB allocation",
            pool.used
        );
        assert!(
            pool.reserved >= pool.used,
            "reserved {} is below used {} - the pool cannot hand out bytes it does not hold",
            pool.reserved,
            pool.used
        );
        assert!(
            pool.used_high >= pool.used,
            "used_high {} is below used {} - a high-water mark cannot sit under the current level",
            pool.used_high,
            pool.used
        );
        assert!(
            pool.release_threshold == 0,
            "the pool's release threshold is {} bytes, not the 0 SPEC-LIT 111.1 measured: the pool now keeps freed memory at a synchronise",
            pool.release_threshold
        );
        drop(held);
    }
}
