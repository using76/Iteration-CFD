// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! One error type for the whole crate.
//!
//! `cudarc::driver::DriverError` deliberately does not implement
//! `std::error::Error`, so it cannot be `?`-ed into a boxed error. Rather than
//! sprinkle `.map_err()` everywhere, everything funnels through this enum -
//! which is also where the domain errors (a malformed polyMesh, a field whose
//! size does not match the mesh) belong.
//!
//! Provenance: ORIGINAL - the crate's error type. No external source.
//! `PROVENANCE.md`, *GPU plumbing and tooling - original*. No GPL-licensed
//! source was consulted.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("CUDA driver error: {0:?}")]
    Driver(cudarc::driver::DriverError),

    #[error("{path}: {msg}")]
    Parse { path: String, msg: String },

    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{0}")]
    Mesh(String),

    #[error("field '{field}': {msg}")]
    Field { field: String, msg: String },

    #[error("{0}")]
    Config(String),
}

// DriverError is a plain enum without an Error impl, so From has to be manual.
impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error::Driver(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Attach a path to an `io::Error`, which on its own never says which file it
/// was - the single most annoying thing about io errors.
pub trait IoContext<T> {
    fn path(self, p: impl AsRef<std::path::Path>) -> Result<T>;
}

impl<T> IoContext<T> for std::result::Result<T, std::io::Error> {
    fn path(self, p: impl AsRef<std::path::Path>) -> Result<T> {
        self.map_err(|source| Error::Io {
            path: p.as_ref().display().to_string(),
            source,
        })
    }
}

/// Shorthand for a parse failure that names the file.
pub fn parse_err<T>(path: impl AsRef<std::path::Path>, msg: impl Into<String>) -> Result<T> {
    Err(Error::Parse {
        path: path.as_ref().display().to_string(),
        msg: msg.into(),
    })
}
