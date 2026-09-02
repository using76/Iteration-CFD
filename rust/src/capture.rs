// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The CUDA-graph capture gate, and the registry that makes it hard to skip.
//!
//! `SPEC-LIT` 81. CUDA-graph capture is one of this crate's published results:
//! one outer iteration is replayed from a single submission instead of some
//! hundreds of driver calls. The claim attached to it is not "it is faster" -
//! it is **"it is faster and the answer does not move a bit"**. A capture that
//! quietly changes the answer is worse than no capture at all, because the run
//! still finishes and still prints numbers.
//!
//! Two things can take that guarantee away, and neither announces itself:
//!
//! * a module puts a **host round-trip** in its per-iteration path - a
//!   `download`, a `sync`, an allocation. `Gpu`'s guard (`SPEC-LIT` 81.3)
//!   turns the first two into an error naming the call. The third does not
//!   even fail: `cudarc` allocates with `cuMemAllocAsync`, which is
//!   stream-ordered and therefore *capturable*, so the graph records a
//!   `MEM_ALLOC` node, succeeds, and allocates on every replay. Only
//!   [`crate::device::GraphShape`] can see that;
//! * a module keeps **per-iteration state on the host** - a counter, a clock,
//!   a coefficient it recomputes in Rust. Capture runs that host code once;
//!   replay runs the kernels many times. The capture succeeds, the graph is
//!   pure device work, and the answer is still wrong. Only a bitwise
//!   comparison against the per-launch path sees *that*.
//!
//! So the gate is both: capture, look at what was captured, replay it, and
//! require the result to be bit-for-bit what the per-launch path produced.
//! [`capture_replays_bitwise`] is that protocol, written once.
//!
//! # Why there is a registry
//!
//! Before this section the guarantee was guarded in five places - `solver.rs`,
//! `models/k_epsilon.rs` and the three parcel modules - while forty-one
//! sections of new physics had been added around them. Nothing said which
//! modules were covered, so nothing could say which were not. A list of
//! covered modules maintained by hand would have had exactly the defect
//! `SPEC-LIT` 69 names: maintained beside a tree that already knows the
//! answer.
//!
//! [`REGISTRY`] is therefore checked against the tree, not trusted. The
//! population is derived from disk - every `.rs` under `src/` outside
//! `src/bin/` that either launches a kernel or exposes a per-iteration entry
//! point - and a file in that population with no row **fails the tests**. A
//! new module is gated, excused by name, or it stops the build. It cannot be
//! quietly absent.
//!
//! Provenance: ORIGINAL - a test harness and a source-derived registry over
//! this crate's own modules. There is no external source for it; the CUDA
//! Graph semantics it relies on are the CUDA Driver API's, cited in
//! `PROVENANCE.md` under *GPU plumbing and tooling - original*. No
//! GPL-licensed source was consulted.

use crate::device::GraphShape;
use crate::error::{Error, Result};
use crate::{Gpu, Scalar};

// ==========================================================================
//  81.5  The protocol
// ==========================================================================

/// Iterations run before anything is measured, so that what the graph records
/// is an ordinary steady iteration and not the first one - the first launch of
/// a kernel also loads its module.
const WARM: usize = 2;

/// How many times the captured graph is replayed, and how many extra
/// per-launch iterations the reference runs. More than one, because a graph
/// that is right once and wrong on the second replay is the interesting
/// failure: it means the captured region depended on something that existed
/// only at capture time.
const REPLAYS: usize = 3;

/// What one gate proved.
#[derive(Debug, Clone)]
pub(crate) struct Replay {
    /// The node census of the captured graph.
    pub shape: GraphShape,
    /// Buffers compared, and elements across all of them.
    pub buffers: usize,
    pub elements: usize,
}

impl std::fmt::Display for Replay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} nodes ({} kernel, {} memset, {} memcpy); {} replays bitwise \
             over {} buffer(s) / {} value(s)",
            self.shape.total,
            self.shape.kernel,
            self.shape.memset,
            self.shape.memcpy,
            REPLAYS,
            self.buffers,
            self.elements
        )
    }
}

/// The device buffers one iteration writes, named.
pub(crate) type Snapshot = Vec<(&'static str, Vec<Scalar>)>;

/// **The gate.** Capture one iteration of `what`, replay it, and require the
/// result to be bit-for-bit what the per-launch path produced.
///
/// * `build` constructs the module in a known state. It is called **twice**,
///   and the two instances must be identical - anything random or
///   clock-dependent in it makes the comparison meaningless, so there is
///   nothing random in any caller;
/// * `iterate` runs exactly one outer iteration;
/// * `state` reads back every device buffer the iteration writes. Whatever it
///   omits is not gated, so it should omit nothing.
///
/// Four things are asserted, and they fail differently on purpose:
///
/// 1. the capture produced a graph at all - an empty graph means the region
///    launched nothing, which passes every other check vacuously;
/// 2. the graph is pure device work: no host callback, no allocation node
///    (`SPEC-LIT` 81.4);
/// 3. **nothing escaped the captured stream** - the state after `capture`
///    returns equals the state before it was called. Stream capture records
///    without running, so device state that moves anyway is work that never
///    went onto the captured stream: a launch on another stream, or a
///    synchronous copy. Such work runs once, at capture, and never again on
///    replay, and the graph is silently missing it;
/// 4. `REPLAYS` replays equal `REPLAYS` per-launch iterations, bitwise.
pub(crate) fn capture_replays_bitwise<M>(
    gpu: &Gpu,
    what: &str,
    build: impl Fn() -> Result<M>,
    iterate: impl Fn(&mut M) -> Result<()>,
    state: impl Fn(&M) -> Result<Snapshot>,
) -> Result<Replay> {
    // ---- the reference: everything launched one kernel at a time ----------
    let mut eager = build()?;
    for _ in 0..WARM + REPLAYS {
        iterate(&mut eager)?;
    }
    gpu.sync()?;
    let want = state(&eager)?;
    drop(eager);

    // ---- the same run, with the last REPLAYS iterations replayed ----------
    let mut graphed = build()?;
    for _ in 0..WARM {
        iterate(&mut graphed)?;
    }
    gpu.sync()?;
    let before = state(&graphed)?;

    let captured = gpu.capture(|_| iterate(&mut graphed)).map_err(|e| {
        Error::Config(format!(
            "{what}: the iteration could not be captured into a CUDA graph, \
             so the graph path is not available to it at all - {e}"
        ))
    })?;
    let Some(mut graph) = captured else {
        return Err(Error::Config(format!(
            "{what}: capture produced an EMPTY graph. The iteration launched \
             nothing, so every check below would pass without meaning \
             anything - SPEC-LIT 81.5"
        )));
    };

    // (2) what did we actually capture?
    let shape = graph.shape()?;
    if !shape.is_pure_device_work() {
        return Err(Error::Config(format!(
            "{what}: the captured graph is not pure device work - {}. The \
             graph replays that on every launch, which is most of what \
             capture existed to remove - SPEC-LIT 81.4",
            shape.impurities().join("; ")
        )));
    }
    if shape.kernel == 0 {
        return Err(Error::Config(format!(
            "{what}: the captured graph has {} node(s) and not one kernel \
             launch - SPEC-LIT 81.4",
            shape.total
        )));
    }

    // (3) capture records; nothing may have escaped the stream.
    let after = state(&graphed)?;
    if let Some(m) = first_difference(&before, &after) {
        return Err(Error::Config(format!(
            "{what}: state MOVED while the graph was being captured, at {m}. \
             Stream capture records device work without running it, so \
             state that moves anyway came from work that never went onto \
             the captured stream - a launch on another stream, or a \
             synchronous copy. It ran once, at capture, and will never \
             run again on replay, so the graph is silently missing it - \
             SPEC-LIT 81.5 assertion 3"
        )));
    }

    graph.upload()?;
    for _ in 0..REPLAYS {
        graph.launch()?;
    }
    gpu.sync()?;
    let got = state(&graphed)?;

    // (4) the whole point.
    if let Some(m) = first_difference(&want, &got) {
        return Err(Error::Config(format!(
            "{what}: {REPLAYS} graph replays did NOT reproduce {REPLAYS} \
             per-launch iterations, at {m}. Capture that changes the answer \
             is worse than no capture - SPEC-LIT 81.5 assertion 4"
        )));
    }

    let elements = want.iter().map(|(_, v)| v.len()).sum();
    Ok(Replay { shape, buffers: want.len(), elements })
}

/// The first place two snapshots differ, described well enough to debug from.
fn first_difference(a: &Snapshot, b: &Snapshot) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("buffer count {} vs {}", a.len(), b.len()));
    }
    for ((na, va), (nb, vb)) in a.iter().zip(b.iter()) {
        if na != nb {
            return Some(format!("buffer name '{na}' vs '{nb}'"));
        }
        if va.len() != vb.len() {
            return Some(format!("'{na}': length {} vs {}", va.len(), vb.len()));
        }
        let differing = va
            .iter()
            .zip(vb.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        if differing > 0 {
            let (i, (x, y)) = va
                .iter()
                .zip(vb.iter())
                .enumerate()
                .find(|(_, (x, y))| x.to_bits() != y.to_bits())
                .expect("a differing element was just counted");
            return Some(format!(
                "'{na}' element {i} of {n}: {x:e} ({xb:#x}) vs {y:e} ({yb:#x}); \
                 {differing} of {n} values differ",
                n = va.len(),
                xb = x.to_bits(),
                yb = y.to_bits()
            ));
        }
    }
    None
}

/// One named device buffer, downloaded. Shorthand for the `state` closures.
pub(crate) fn buf(
    gpu: &Gpu,
    name: &'static str,
    d: &crate::DevBuf<Scalar>,
) -> Result<(&'static str, Vec<Scalar>)> {
    Ok((name, gpu.download(d)?))
}

/// One named scalar field, downloaded.
pub(crate) fn field(
    gpu: &Gpu,
    name: &'static str,
    f: &crate::GpuScalarField,
) -> Result<(&'static str, Vec<Scalar>)> {
    Ok((name, gpu.download(&f.f)?))
}

pub(crate) mod registry;

#[cfg(test)]
mod gates;
