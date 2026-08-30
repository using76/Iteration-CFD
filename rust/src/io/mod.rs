// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Reading and writing OpenFOAM ASCII case files.
//!
//! ofgpu links against no part of OpenFOAM: it reads `constant/polyMesh` and
//! the `0/` directory itself so a case can be run on a machine that has never
//! had a FOAM installation. Everything needed for that lives here.
//!
//! * [`tokenizer`] — the `ISstream`-compatible lexer, and the source-text
//!   helpers ([`slurp`], [`check_ascii_format`]) every reader below starts with
//! * [`dict`] — any dictionary, flattened to `"a/b/c"` keys
//! * [`polymesh`] — `constant/polyMesh` and the `HostMesh` built from it
//! * [`fields`] — `volScalarField` / `volVectorField`, in and out
//! * [`regex`] — the POSIX ERE matcher a quoted dictionary key needs
//! * [`contract`] — what happens when a case asks for something this solver
//!   does not have (SPEC-LIT §13.4)
//! * [`schemes`] — `system/fvSchemes`, read one equation at a time
//! * [`case`] — pulling a whole case directory together
//!
//! Binary-format files are refused up front, before tokenising, with the
//! `foamFormatConvert` incantation that fixes them.
//!
//! Provenance: FORMAT, and carried across from this project's own earlier C++
//! I/O layer when the crate moved to Rust. That C++ was written from the case
//! format as it appears in DATA FILES - not from any CFD code's source - and
//! the format itself, not another program, is the specification here. A file
//! format is not a work of authorship and interoperability is the whole
//! purpose. See `PROVENANCE.md`, *Case format interoperability - format*, and
//! its note on what "carried over" means. No GPL-licensed source was
//! consulted.

pub mod case;
pub mod contract;
pub mod dict;
pub mod fields;
pub mod polymesh;
pub mod regex;
pub mod schemes;
pub mod tokenizer;

pub use contract::{permissive, set_permissive, unreadable, unsupported};
pub use dict::FoamDict;
pub use schemes::{DivEntry, FvSchemes};
pub use regex::Regex;
pub use tokenizer::{check_ascii_format, slurp, Tok, Tokenizer};

// polymesh, fields and case are owned by another module author; re-exported
// wholesale so this file does not have to guess at their type names, and does
// not have to be edited every time one is added.
pub use case::*;
pub use fields::*;
pub use polymesh::*;
pub mod vtu;
pub mod usda;
pub mod msh;
pub mod nvdb;
pub mod vdb;
pub mod case_json;
pub mod case_cht;
pub mod case_dc;
pub mod output_types;
pub mod writer;
pub mod output_plan;

pub use writer::{
    cartesian_info, CartesianInfo, FoamField, FoamPayload, FoamWriter, NvdbWriter, ResultWriter,
    UsdaWriter, VdbWriter, VtuWriter, WriteCtx,
};
pub use output_types::{FieldValues, OutputField};
pub use output_plan::{
    build_writers, parse_output_formats, refuse_output_named_twice, ExactSpec, FieldSelection,
    OutputFormat, OutputPipeline, OutputPlan, RestartSpec, VisSpec, OUTPUT_FORMAT_NAMES,
};
