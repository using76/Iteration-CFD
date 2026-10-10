// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! [`ThreadComm`]: N ranks as N threads of one process (SPEC-LIT §138.1.3).
//!
//! One `Mutex` and one `Condvar` guard a shared state of two mailbox maps and
//! an abort slot:
//!
//! * point-to-point messages, keyed by the envelope `(src, dst, tag)`, each
//!   key holding a FIFO queue, so a message is matched by what it says and
//!   never by when it arrived;
//! * collective messages, keyed by `(src, dst, sequence)`, a separate channel
//!   no tag can reach. Every collective call takes the next per-rank
//!   sequence number and sends every other rank a header (its kind, its root
//!   and its block length) with its bytes. A rank checks the header of every
//!   peer against its own, so a collective that differs between ranks fails
//!   on every rank, by name, instead of mixing bytes;
//! * the abort slot, set when a rank's closure panics inside
//!   [`ThreadComm::run`]. From then on every blocked or later call returns
//!   `Err` naming that rank, without waiting for the timeout.
//!
//! Sends copy their bytes into the mailbox before the call waits on
//! anything, so no call ever waits for a receiver. A wait for a message ends
//! at the world's timeout ([`DEFAULT_TIMEOUT`] unless chosen), naming the
//! waiting rank, what it waited for and the timeout in milliseconds. A
//! mismatch or a bad length does not abort the world: each rank that sees it
//! fails on its own, and a rank whose call was fine returns `Ok`.
//!
//! Provenance: ORIGINAL - the design is ours (docs/17 PAR-01); the semantics
//! are written from the MPI-4.1 report (MPI Forum, 2023), sections 4.2.3,
//! 4.5, 7.1, 7.3, 7.4, 7.7, 7.14 and 10.1.2.1, read as a specification. No
//! MPI implementation's source was opened. No GPL-licensed source was
//! consulted.

use std::any::Any;
use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::{Comm, DEFAULT_TIMEOUT, MAX_TAG};
use crate::error::{Error, Result};

/// Which collective a call is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Barrier,
    Allgather,
    Allgatherv,
    Bcast,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Barrier => "barrier",
            Kind::Allgather => "allgather",
            Kind::Allgatherv => "allgatherv",
            Kind::Bcast => "bcast",
        }
    }
}

/// What a rank says about one collective call: the part that every rank must
/// agree on. `root` and `len` are 0 for a kind that has no such parameter.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Header {
    kind: Kind,
    root: usize,
    len: usize,
}

/// One collective message: the sender's header and its bytes.
struct CollMsg {
    header: Header,
    data: Vec<u8>,
}

/// Everything the ranks share, behind one lock.
struct State {
    /// Point-to-point mailboxes, `(src, dst, tag)` -> FIFO of messages. An
    /// emptied queue is removed, so the map holds only what is in flight.
    p2p: HashMap<(usize, usize, u32), VecDeque<Vec<u8>>>,
    /// Collective mailboxes, `(src, dst, sequence)` -> the one message.
    coll: HashMap<(usize, usize, u64), CollMsg>,
    /// Set once, by the first panicking rank: the message every later call
    /// returns.
    abort: Option<String>,
}

struct Shared {
    size: usize,
    timeout: Duration,
    state: Mutex<State>,
    ready: Condvar,
}

impl Shared {
    /// The lock. A poisoned lock is still usable: nothing here runs caller
    /// code while holding it, so the state is never left half-written.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Spend the world: every blocked and later call returns `why`. The first
    /// reason wins.
    fn abort(&self, why: String) {
        let mut g = self.lock();
        if g.abort.is_none() {
            g.abort = Some(why);
        }
        drop(g);
        self.ready.notify_all();
    }

    /// Block until `take` yields a value, the world aborts, or the timeout
    /// passes. The guard comes back with the value so one critical section
    /// can wait for several messages in turn.
    ///
    /// The order is fixed: abort first, then the message, then the clock, so
    /// a message that arrives as the timeout expires is still delivered.
    fn wait_for<'a, T>(
        &'a self,
        mut g: MutexGuard<'a, State>,
        rank: usize,
        waiting_for: impl Fn() -> String,
        mut take: impl FnMut(&mut State) -> Option<T>,
    ) -> Result<(MutexGuard<'a, State>, T)> {
        let deadline = Instant::now().checked_add(self.timeout);
        loop {
            if let Some(why) = &g.abort {
                return Err(Error::Config(why.clone()));
            }
            if let Some(v) = take(&mut g) {
                return Ok((g, v));
            }
            let remaining = match deadline {
                Some(d) => {
                    let r = d.saturating_duration_since(Instant::now());
                    if r.is_zero() {
                        return Err(Error::Config(format!(
                            "comm: rank {rank}: {} - nothing arrived within the timeout of {} ms",
                            waiting_for(),
                            self.timeout.as_millis()
                        )));
                    }
                    r
                }
                // A timeout too long to add to the clock never expires.
                None => self.timeout,
            };
            g = self
                .ready
                .wait_timeout(g, remaining)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

/// One rank's handle on a thread world. `Send`, so it can move to its
/// thread, and not `Sync`: a rank is one thread's business.
pub struct ThreadComm {
    rank: usize,
    shared: Arc<Shared>,
    /// The next collective sequence number of this rank.
    seq: Cell<u64>,
}

impl std::fmt::Debug for ThreadComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadComm")
            .field("rank", &self.rank)
            .field("size", &self.shared.size)
            .finish()
    }
}

fn config<T>(msg: String) -> Result<T> {
    Err(Error::Config(msg))
}

/// The text of a panic payload, when it is a string.
fn panic_text(payload: &(dyn Any + Send)) -> Option<String> {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
}

impl ThreadComm {
    /// A world of `n` ranks with the [`DEFAULT_TIMEOUT`]. Entry r of the
    /// result is rank r. Refuses `n == 0`.
    pub fn world(n: usize) -> Result<Vec<ThreadComm>> {
        Self::world_with_timeout(n, DEFAULT_TIMEOUT)
    }

    /// A world of `n` ranks whose waits end after `timeout`. Refuses
    /// `n == 0` and a zero `timeout`.
    pub fn world_with_timeout(n: usize, timeout: Duration) -> Result<Vec<ThreadComm>> {
        if n == 0 {
            return config("comm: a world needs at least one rank, got 0".to_string());
        }
        if timeout.is_zero() {
            return config("comm: the timeout must be above zero, got 0 ms".to_string());
        }
        let shared = Arc::new(Shared {
            size: n,
            timeout,
            state: Mutex::new(State {
                p2p: HashMap::new(),
                coll: HashMap::new(),
                abort: None,
            }),
            ready: Condvar::new(),
        });
        Ok((0..n)
            .map(|rank| ThreadComm {
                rank,
                shared: Arc::clone(&shared),
                seq: Cell::new(0),
            })
            .collect())
    }

    /// Run `f` once per rank on `n` scoped threads, with the
    /// [`DEFAULT_TIMEOUT`], and return the `n` results in rank order. See
    /// [`ThreadComm::run_with_timeout`].
    pub fn run<R, F>(n: usize, f: F) -> Result<Vec<R>>
    where
        R: Send,
        F: Fn(&ThreadComm) -> R + Sync,
    {
        Self::run_with_timeout(n, DEFAULT_TIMEOUT, f)
    }

    /// Run `f(&handle_r)` on one scoped thread per rank and return the `n`
    /// results in rank order.
    ///
    /// A rank whose closure panics aborts the world: its panic is caught,
    /// the world is marked aborted naming that rank, and every waiter is
    /// woken, so every blocked or later call on any rank returns `Err`
    /// without waiting for the timeout. `run` then returns `Err` naming the
    /// lowest panicking rank, with the panic message when the payload is a
    /// `&str` or a `String`.
    pub fn run_with_timeout<R, F>(n: usize, timeout: Duration, f: F) -> Result<Vec<R>>
    where
        R: Send,
        F: Fn(&ThreadComm) -> R + Sync,
    {
        let comms = Self::world_with_timeout(n, timeout)?;
        let shared = Arc::clone(&comms[0].shared);
        let f = &f;
        // `Ok(result)`, or `Err((rank, panic message))`, in rank order.
        let mut spawn_failure: Option<Error> = None;
        let outcomes: Vec<std::result::Result<R, (usize, Option<String>)>> =
            std::thread::scope(|s| {
                let mut joins = Vec::with_capacity(n);
                for comm in comms {
                    let rank = comm.rank;
                    let spawned = std::thread::Builder::new()
                        .name(format!("comm-rank-{rank}"))
                        .spawn_scoped(s, move || {
                            match catch_unwind(AssertUnwindSafe(|| f(&comm))) {
                                Ok(r) => Ok(r),
                                Err(payload) => {
                                    let text = panic_text(payload.as_ref());
                                    let detail = text.as_deref().map(|t| format!(": {t}"));
                                    comm.shared.abort(format!(
                                        "comm: world aborted: rank {rank} panicked{}",
                                        detail.as_deref().unwrap_or("")
                                    ));
                                    Err((rank, text))
                                }
                            }
                        });
                    match spawned {
                        Ok(h) => joins.push((rank, h)),
                        Err(e) => {
                            let why =
                                format!("comm: the thread of rank {rank} could not start: {e}");
                            // The ranks already running would wait for the
                            // one that never came.
                            shared.abort(why.clone());
                            spawn_failure = Some(Error::Config(why));
                            break;
                        }
                    }
                }
                joins
                    .into_iter()
                    .map(|(rank, h)| match h.join() {
                        Ok(outcome) => outcome,
                        // A panic outside the caught closure, in this module.
                        Err(payload) => Err((rank, panic_text(payload.as_ref()))),
                    })
                    .collect()
            });
        if let Some(e) = spawn_failure {
            return Err(e);
        }
        let mut results = Vec::with_capacity(n);
        for outcome in outcomes {
            match outcome {
                Ok(r) => results.push(r),
                Err((rank, text)) => {
                    return config(match text {
                        Some(t) => format!("comm: rank {rank} panicked: {t}"),
                        None => format!("comm: rank {rank} panicked"),
                    });
                }
            }
        }
        Ok(results)
    }

    /// A refusal from this rank: `comm: rank r: <msg>`.
    fn fail<T>(&self, msg: String) -> Result<T> {
        config(format!("comm: rank {}: {msg}", self.rank))
    }

    /// R9, the peer half: every peer is in `0..size` and none is listed
    /// twice. `list` says which list it is, for the message.
    fn check_peers(&self, list: &str, peers: impl Iterator<Item = usize>) -> Result<()> {
        let size = self.shared.size;
        let mut seen = vec![false; size];
        for p in peers {
            if p >= size {
                return self.fail(format!(
                    "exchange {list} peer {p} is out of range for size {size}"
                ));
            }
            if seen[p] {
                return self.fail(format!("exchange lists peer {p} twice in its {list} list"));
            }
            seen[p] = true;
        }
        Ok(())
    }

    /// One collective call. Takes the next sequence number, sends `payload`
    /// with `header` to every other rank, then reads every other rank's
    /// message in rank order and checks its header against this one's.
    /// Returns the peers' bytes, indexed by rank; this rank's own entry is
    /// empty, for the caller to fill from its own argument.
    fn collective(&self, header: Header, payload: &[u8]) -> Result<Vec<Vec<u8>>> {
        let n = self.shared.size;
        let me = self.rank;
        let seq = self.seq.get();
        self.seq.set(seq + 1);

        let mut g = self.shared.lock();
        if let Some(why) = &g.abort {
            return Err(Error::Config(why.clone()));
        }
        for p in (0..n).filter(|&p| p != me) {
            g.coll.insert(
                (me, p, seq),
                CollMsg {
                    header,
                    data: payload.to_vec(),
                },
            );
        }
        if n > 1 {
            self.shared.ready.notify_all();
        }

        let mut out = vec![Vec::new(); n];
        for p in (0..n).filter(|&p| p != me) {
            let (g2, msg) = self.shared.wait_for(
                g,
                me,
                || {
                    format!(
                        "collective #{seq} ({}) is waiting for rank {p}",
                        header.kind.name()
                    )
                },
                |st| st.coll.remove(&(p, me, seq)),
            )?;
            g = g2;
            if msg.header != header {
                return Err(self.mismatch(seq, header, p, msg.header));
            }
            out[p] = msg.data;
        }
        Ok(out)
    }

    /// R6: this rank's call and `peer`'s call at the same sequence number
    /// differ. Names the first thing that differs: the kind, then the root,
    /// then the length.
    fn mismatch(&self, seq: u64, mine: Header, peer: usize, theirs: Header) -> Error {
        let me = self.rank;
        let msg = if mine.kind != theirs.kind {
            format!(
                "collective #{seq} differs between ranks: rank {me} called {} but rank {peer} \
                 called {}",
                mine.kind.name(),
                theirs.kind.name()
            )
        } else if mine.root != theirs.root {
            format!(
                "collective #{seq} ({}): rank {me} passes root {} but rank {peer} passes root {}",
                mine.kind.name(),
                mine.root,
                theirs.root
            )
        } else {
            let what = if mine.kind == Kind::Bcast {
                "a buffer"
            } else {
                "a block"
            };
            format!(
                "collective #{seq} ({}): rank {me} passes {what} of {} bytes but rank {peer} \
                 passes {}",
                mine.kind.name(),
                mine.len,
                theirs.len
            )
        };
        Error::Config(format!("comm: rank {me}: {msg}"))
    }
}

impl Comm for ThreadComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.shared.size
    }

    fn transport(&self) -> &'static str {
        "threads"
    }

    fn barrier(&self) -> Result<()> {
        self.collective(
            Header {
                kind: Kind::Barrier,
                root: 0,
                len: 0,
            },
            &[],
        )
        .map(|_| ())
    }

    fn exchange(
        &self,
        tag: u32,
        sends: &[(usize, &[u8])],
        recvs: &mut [(usize, &mut [u8])],
    ) -> Result<()> {
        // R9: everything a caller can get wrong alone, before anything moves.
        if tag > MAX_TAG {
            return self.fail(format!(
                "exchange tag {tag} is above the largest tag {MAX_TAG}"
            ));
        }
        self.check_peers("send", sends.iter().map(|(p, _)| *p))?;
        self.check_peers("receive", recvs.iter().map(|(p, _)| *p))?;

        let me = self.rank;
        let mut g = self.shared.lock();
        if let Some(why) = &g.abort {
            return Err(Error::Config(why.clone()));
        }
        // R4: the bytes are copied out before this call waits on anything.
        for (dst, bytes) in sends {
            g.p2p
                .entry((me, *dst, tag))
                .or_default()
                .push_back(bytes.to_vec());
        }
        if !sends.is_empty() {
            self.shared.ready.notify_all();
        }

        let mut got: Vec<Vec<u8>> = Vec::with_capacity(recvs.len());
        for (src, buf) in recvs.iter() {
            let (src, want) = (*src, buf.len());
            let (g2, msg) = self.shared.wait_for(
                g,
                me,
                || format!("exchange is waiting for the message from rank {src} under tag {tag}"),
                |st| {
                    let key = (src, me, tag);
                    let q = st.p2p.get_mut(&key)?;
                    let m = q.pop_front();
                    if q.is_empty() {
                        st.p2p.remove(&key);
                    }
                    m
                },
            )?;
            g = g2;
            // R5: exact lengths.
            if msg.len() != want {
                return self.fail(format!(
                    "the message from rank {src} under tag {tag} has {} bytes but the receive \
                     buffer holds {want}",
                    msg.len()
                ));
            }
            got.push(msg);
        }
        drop(g);
        for ((_, buf), msg) in recvs.iter_mut().zip(got) {
            buf.copy_from_slice(&msg);
        }
        Ok(())
    }

    fn allgather(&self, mine: &[u8], all: &mut [u8]) -> Result<()> {
        let n = self.shared.size;
        let len = mine.len();
        if all.len() != len.saturating_mul(n) {
            return self.fail(format!(
                "allgather receive buffer holds {} bytes but the block of {len} bytes times \
                 size {n} is {}",
                all.len(),
                len.saturating_mul(n)
            ));
        }
        let blocks = self.collective(
            Header {
                kind: Kind::Allgather,
                root: 0,
                len,
            },
            mine,
        )?;
        for (q, block) in blocks.iter().enumerate() {
            let dst = &mut all[q * len..(q + 1) * len];
            dst.copy_from_slice(if q == self.rank {
                mine
            } else {
                block.as_slice()
            });
        }
        Ok(())
    }

    fn allgatherv(&self, mine: &[u8]) -> Result<Vec<Vec<u8>>> {
        let mut blocks = self.collective(
            Header {
                kind: Kind::Allgatherv,
                root: 0,
                len: 0,
            },
            mine,
        )?;
        blocks[self.rank] = mine.to_vec();
        Ok(blocks)
    }

    fn bcast(&self, root: usize, buf: &mut [u8]) -> Result<()> {
        let n = self.shared.size;
        if root >= n {
            return self.fail(format!("bcast root {root} is out of range for size {n}"));
        }
        let is_root = root == self.rank;
        let header = Header {
            kind: Kind::Bcast,
            root,
            len: buf.len(),
        };
        // Only the root sends bytes; the others send their header alone, so
        // that every rank can still check every other rank's call.
        let parts = self.collective(header, if is_root { &*buf } else { &[] })?;
        if !is_root {
            buf.copy_from_slice(&parts[root]);
        }
        Ok(())
    }
}
