// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Vertical slice: nvcc -> PTX -> cudarc -> launch -> read back, on the exact
//! gather pattern the solver depends on. Verifies against a CPU computation.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use ofgpu::{Label, Result, Scalar};

fn main() -> Result<()> {
    // ---- a tiny 1-D mesh: 5 cells in a line, 4 internal faces -------------
    let n_cells: Label = 5;
    let n_faces: usize = 4;

    let owner: Vec<Label> = vec![0, 1, 2, 3];
    let neighbour: Vec<Label> = vec![1, 2, 3, 4];

    // cell -> face CSR (each interior cell touches two faces)
    let cf_offset: Vec<Label> = vec![0, 1, 3, 5, 7, 8];
    let cf_face: Vec<Label> = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let cf_own: Vec<Label> = vec![1, 0, 1, 0, 1, 0, 1, 0];

    let diag: Vec<Scalar> = vec![4.0, 4.5, 5.0, 5.5, 6.0];
    let upper: Vec<Scalar> = vec![-1.0, -1.25, -1.5, -1.75];
    let lower: Vec<Scalar> = vec![-0.5, -0.75, -0.9, -1.1];
    let psi: Vec<Scalar> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // ---- CPU reference, deliberately a SCATTER, not a gather --------------
    // Deliberately the opposite loop structure to the kernel's gather, so
    // agreement means something.
    let mut expect = vec![0.0 as Scalar; n_cells as usize];
    for c in 0..n_cells as usize {
        expect[c] = diag[c] * psi[c];
    }
    for f in 0..n_faces {
        let l = owner[f] as usize;
        let u = neighbour[f] as usize;
        expect[u] += lower[f] * psi[l];
        expect[l] += upper[f] * psi[u];
    }

    // ---- device ----------------------------------------------------------
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    println!("device : {}", ctx.name()?);
    println!("scalar : {}", if size_of::<Scalar>() == 8 { "f64" } else { "f32" });

    let module = ctx.load_module(Ptx::from_binary(ofgpu::kernels::PROBE.to_vec()))?;
    let f = module.load_function("probeAmul")?;

    let d_psi = stream.clone_htod(&psi)?;
    let d_diag = stream.clone_htod(&diag)?;
    let d_upper = stream.clone_htod(&upper)?;
    let d_lower = stream.clone_htod(&lower)?;
    let d_owner = stream.clone_htod(&owner)?;
    let d_neighbour = stream.clone_htod(&neighbour)?;
    let d_cf_offset = stream.clone_htod(&cf_offset)?;
    let d_cf_face = stream.clone_htod(&cf_face)?;
    let d_cf_own = stream.clone_htod(&cf_own)?;
    let mut d_out = stream.alloc_zeros::<Scalar>(n_cells as usize)?;

    let cfg = LaunchConfig::for_num_elems(n_cells as u32);

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut d_out)
            .arg(&d_psi)
            .arg(&d_diag)
            .arg(&d_upper)
            .arg(&d_lower)
            .arg(&d_owner)
            .arg(&d_neighbour)
            .arg(&d_cf_offset)
            .arg(&d_cf_face)
            .arg(&d_cf_own)
            .arg(&n_cells)
            .launch(cfg)?;
    }

    let got = stream.clone_dtoh(&d_out)?;

    // ---- compare ---------------------------------------------------------
    let mut worst = 0.0 as Scalar;
    println!("\n  cell        gpu                 cpu              diff");
    for c in 0..n_cells as usize {
        let d = (got[c] - expect[c]).abs();
        worst = worst.max(d);
        println!("  {c:>4}   {:>16.12}   {:>16.12}   {d:.3e}", got[c], expect[c]);
    }

    println!("\nmax |gpu - cpu| = {worst:.3e}");

    if worst == 0.0 {
        println!("PASS - bitwise identical");
        Ok(())
    } else {
        Err(ofgpu::Error::Config(format!("MISMATCH: worst difference {worst:.3e}")))
    }
}
