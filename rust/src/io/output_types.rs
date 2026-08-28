// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The one input type every result writer consumes.
//!
//! Writers (vtu, nvdb, vdb, usda, the OpenFOAM writer behind the seam) take
//! a `HostMesh` reference plus a flat list of named fields. Defining the type
//! here, once, is what lets the writers be built in parallel and the output
//! seam unify them without touching their internals.
//!
//! Provenance: ORIGINAL - the one input type every result writer consumes.
//! Designed here so the writers can be built independently behind one seam; it
//! describes no external format of its own (each writer names its own).
//! `PROVENANCE.md`, *New I/O formats and machinery*. No GPL-licensed source was
//! consulted.

use crate::{Scalar, Vec3};

/// Borrowed cell-centred data for one field.
pub enum FieldValues<'a> {
    Scalar(&'a [Scalar]),
    Vector(&'a [Vec3]),
}

/// One named output field.
pub struct OutputField<'a> {
    pub name: &'a str,
    pub values: FieldValues<'a>,
}

impl<'a> OutputField<'a> {
    pub fn scalar(name: &'a str, v: &'a [Scalar]) -> Self {
        Self { name, values: FieldValues::Scalar(v) }
    }
    pub fn vector(name: &'a str, v: &'a [Vec3]) -> Self {
        Self { name, values: FieldValues::Vector(v) }
    }
    pub fn len(&self) -> usize {
        match self.values { FieldValues::Scalar(s) => s.len(), FieldValues::Vector(v) => v.len() }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}
