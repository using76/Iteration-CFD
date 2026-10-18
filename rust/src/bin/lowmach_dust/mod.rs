// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-lowmach`'s dust step - SPEC-LIT §157.4 (docs/17 DST-06 part 3).
//!
//! A case whose `combustion.model` is `FSC` or `TFC` with a `dust` block
//! (§154, §157.4) lowers to a `LoweredDust` and runs here:
//! [`refuse_settings`] refuses, by name and before a field is set up,
//! everything this driver cannot honour - a steady run, a ddt other than
//! Euler (the heat release rides the c~ solve's one old level), a
//! checkpoint (the .mcr carries no c~), a motion block, `-heaterPower`.
//! [`c_controls`] reads the c~ equation's own `solvers`/`relaxation`/
//! `gradSchemes`/`divSchemes` entries exactly as `species_controls` reads
//! the species'; [`dust_turbulence_of`] lifts the frozen k/omega/nu_t the
//! closures of §154.4 read off the coupled model; [`DustStep`] carries one
//! outer iteration's borrow of the [`DustSource`] into the driver's unit of
//! work; [`describe`] is the banner line. The step itself is
//! `ofgpu::dust::source::DustSource::before_flow` in §157.4's order.
//! `ofgpu-par` and `ofgpu-decompose` never reach a dust case at all: they
//! load through `common::load_case`, whose part-1 seam
//! (`LoweredCase::refuse_dust`) refuses one by name.
//!
//! Written from SPEC-LIT §157.4, §155.3, §155.6, §156 and §133.10. The
//! specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use ofgpu::dust::source::{DustSource, DustTurbulence};
use ofgpu::field::GpuScalarField;
use ofgpu::io::DivEntry;
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::TurbulenceControls;
use ofgpu::{Error, Result, Scalar};

use crate::common::{g, sci, CaseNumerics};

/// The run settings §157.4 refuses with a dust case, by name, before a
/// field is set up.
pub struct DustShape {
    pub transient: bool,
    pub ddt: DdtScheme,
    pub restart: bool,
    pub motion: bool,
    pub heater: bool,
}

fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} (SPEC-LIT §157.4)"))
}

/// §157.4's driver refusals for a run of `shape`: each names the setting
/// that asked for it, before anything is built.
pub fn refuse_settings(shape: &DustShape) -> Result<()> {
    if !shape.transient {
        return Err(refused(
            "run",
            "a steady run: the c~ equation (155.3) and the closures' flame-development \
             clock integrate physical time; run the dust model transient with -endTime and \
             -deltaT",
        ));
    }
    if shape.ddt != DdtScheme::Euler {
        return Err(refused(
            "numerics.ddt",
            &format!(
                "{:?} with a dust case: the c~ equation keeps one old level and the heat \
                 release (152.12) rides that solve's one old level; only Euler is run",
                shape.ddt
            ),
        ));
    }
    if shape.restart {
        return Err(refused(
            "restart",
            "a checkpoint (-restartFrom, -restartWrite or an output restart series) with a \
             dust case: the .mcr format carries no c~, so a resumed run would restart the \
             flame at 0",
        ));
    }
    if shape.motion {
        return Err(refused(
            "motion",
            "a moving mesh with a dust case: the c~ equation is not run on the ALE flux",
        ));
    }
    if shape.heater {
        return Err(refused(
            "-heaterPower",
            "the dust model's heat release is (152.12)'s own registration on the same source \
             registry; a second, uniform W would double-count the burn",
        ));
    }
    Ok(())
}

/// The c~ equation's controls by its own entry names (§13.4.1), as
/// `species_controls` reads the species': `solvers` "c" (fallback the
/// turbulence set's own), `relaxation` "c" (1 where the case names none, a
/// dust run being transient), `grad(c)`, `snGradSchemes default`, and the
/// convection `div(phi,c)` where the case names one, otherwise §155.3's
/// bounded Gauss upwind - never the case's `default` entry.
/// `steady = false` and `delta_t = dt`: (155.3) has a time derivative, and
/// `ProgressEquation::new` refuses a steady set by name.
pub fn c_controls(
    num: &CaseNumerics<'_>,
    base: &TurbulenceControls,
    dt: Scalar,
) -> Result<(TurbulenceControls, DivEntry)> {
    let conv = match num.div_named("div(phi,c)")? {
        Some(a) => a,
        None => DivEntry { scheme: ofgpu::fv::DivScheme::Upwind, bounded: true },
    };
    let mut c = *base;
    c.k_solver = num.solver("c", base.k_solver)?;
    c.k_relax = num.relax("c", 1.0)?;
    c.grad_scheme = num.grad("grad(c)")?;
    c.sn_grad = num.sn_grad("default")?;
    c.div_scheme = conv.scheme;
    c.bounded_convection = conv.bounded;
    c.steady = false;
    c.delta_t = dt;
    Ok((c, conv))
}

/// One outer iteration's dust arguments: the source itself, the frozen
/// turbulence its closures read (`None` for the laminar step, whose k and
/// omega are the zero ones of decision D5), and the step's end time - the
/// clock `t_fd` runs from `DustSetup::t_fd_origin`.
pub struct DustStep<'a, 'm> {
    pub src: &'a mut DustSource<'m>,
    pub turb: Option<DustTurbulence<'a>>,
    pub t_new: f64,
}

/// The frozen turbulence one step's closures read, lifted off a coupled
/// model's own fields: `"k"` and `"omega"` of `output_fields` and the model's
/// `nut`. `None` for a laminar model, which owns neither - its closures run
/// on the zero buffers [`DustSource`] holds for exactly that.
pub fn dust_turbulence_of<'a>(
    fields: &[(&'static str, &'a GpuScalarField)],
    nut: &'a GpuScalarField,
) -> Option<DustTurbulence<'a>> {
    let k = fields.iter().find(|(n, _)| *n == "k").map(|(_, f)| &f.f)?;
    let omega = fields.iter().find(|(n, _)| *n == "omega").map(|(_, f)| &f.f)?;
    Some(DustTurbulence { k, omega, nu_t: &nut.f })
}

/// One banner line for the start-up log: the model, the thermo mode and the
/// state of (152.9)-(152.10), (153.2) and (155.9) the step runs with. A
/// Gate DST-VES case appends its prescribed `U_t` law and the density the
/// propagation reads (SPEC-LIT §157.6).
pub fn describe(ds: &DustSource<'_>) -> String {
    let s = ds.setup();
    let mut line = format!(
        "dust (SPEC-LIT §157.4): model {:?}, thermo {:?}, T_u {} K, T_b {} K, sigma {}, \
         S_L {} m/s, Psi {} (dimensionless), t_r {} s, max flame Courant {}",
        s.fsc.model,
        s.gas.mode,
        g(s.t_u),
        g(s.t_b),
        g(s.sigma),
        g(s.s_l),
        sci(s.psi, 4),
        sci(s.t_r, 4),
        g(s.max_flame_courant),
    );
    if let Some(law) = s.prescribed_ut {
        line.push_str(&format!(
            ", prescribed U_t S0 {} alpha {} beta {}, unburned density {:?}",
            g(law.s0),
            g(law.alpha),
            g(law.beta),
            s.unburned_density,
        ));
    }
    line
}

#[cfg(test)]
mod tests;
