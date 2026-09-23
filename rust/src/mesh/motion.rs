// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The prescribed motion of a mesh's points - SPEC-LIT 105.9.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md 105.9 (the rule table, the refusals, and the
//!     parameters of the weight, which are this module's own choices)
//!   Luke, Collins & Blades (2012), DOI 10.1016/j.jcp.2011.09.021 - the
//!     two-term inverse-distance weight; this module's parameters are its
//!     own, stated in SPEC-LIT 105.9
//! No GPL-licensed source was consulted.
//!
//! The rule table below says which points the laws move (`Move`), which
//! points a plane holds while the smoother moves them along it (`Slide`),
//! and which stay where they are as the smoother's control points with zero
//! displacement (`Fixed`). Every patch of the mesh must be named exactly
//! once; the classification walks the boundary faces patch by patch and
//! refuses the assignments no per-point state can carry.
//!
//! NOT here: rotations, the paper's per-node area weights, its
//! boundary-node reduction, and any device smoother - the weight is
//! evaluated on the host, `O(N_free N_control)` per call, and its result
//! reaches the mesh through `AleMesh::set_points`.

use crate::error::{Error, Result};
use crate::mesh::{HostMesh, PatchKind};
use crate::{Label, Scalar, Vec3};

/// A displacement from the rest position, as a function of time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Displacement {
    /// `d(t) = velocity * t`.
    Linear { velocity: Vec3 },
    /// `d(t) = amplitude * sin(2 pi t / period)`.
    Sine { amplitude: Vec3, period: Scalar },
}

impl Displacement {
    pub fn at(&self, t: Scalar) -> Vec3 {
        match *self {
            Self::Linear { velocity } => velocity * t,
            Self::Sine { amplitude, period } => {
                let pi = std::f64::consts::PI as Scalar;
                amplitude * (2.0 * pi * t / period).sin()
            }
        }
    }
    /// The one direction the law moves along.
    pub fn direction(&self) -> Vec3 {
        match *self {
            Self::Linear { velocity } => velocity,
            Self::Sine { amplitude, .. } => amplitude,
        }
    }
}

/// What one patch's points do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchMotion {
    /// The points stay where they are: control points with zero displacement.
    Fixed,
    /// The points are moved by the smoother, like interior points - a plane
    /// the mesh slides along. A point stays in its plane exactly when no
    /// control displacement has a component along the plane's normal.
    Slide,
    /// Every point of the patch is displaced by the law.
    Move(Displacement),
}

/// The inverse-distance weight of Luke, Collins & Blades (2012), with the
/// parameters of SPEC-LIT 105.9.
#[derive(Debug, Clone)]
pub struct Idw {
    control: Vec<Vec3>,
    length: Scalar,
}

impl Idw {
    pub fn new(control: Vec<Vec3>) -> Result<Self> {
        if control.is_empty() {
            return Err(Error::Mesh(
                "Idw::new: the control point list is empty; a smoother with no \
                 control point moves nothing"
                    .to_string(),
            ));
        }
        let (mut lo, mut hi) = (control[0], control[0]);
        for p in &control {
            lo.x = lo.x.min(p.x);
            lo.y = lo.y.min(p.y);
            lo.z = lo.z.min(p.z);
            hi.x = hi.x.max(p.x);
            hi.y = hi.y.max(p.y);
            hi.z = hi.z.max(p.z);
        }
        let length = (hi - lo).mag();
        if length == 0.0 {
            return Err(Error::Mesh(
                "Idw::new: the control points have zero extent; the weight's \
                 length scale would divide every distance by zero"
                    .to_string(),
            ));
        }
        Ok(Self { control, length })
    }

    pub fn length(&self) -> Scalar {
        self.length
    }

    /// `d.len() == self.control.len()` is the caller's promise: both lists
    /// describe the same control points, rest positions here, displacements
    /// there.
    pub fn displacement(&self, x: Vec3, d: &[Vec3]) -> Vec3 {
        let dmax = d.iter().fold(0.0 as Scalar, |a, v| a.max(v.mag()));
        if dmax == 0.0 {
            return Vec3::ZERO;
        }
        let alpha = 5.0 * dmax / self.length;
        let mut num = Vec3::ZERO;
        let mut den = 0.0 as Scalar;
        for (xi, di) in self.control.iter().zip(d) {
            let r = (x - *xi).mag();
            if r == 0.0 {
                return *di;
            }
            let q = self.length / r;
            let qa = alpha * q;
            let w = q * q * q + qa * qa * qa * qa * qa;
            num = num + *di * w;
            den += w;
        }
        num / den
    }
}

/// A prescribed motion: which points the laws move, which the smoother moves.
#[derive(Debug, Clone)]
pub struct MeshMotion {
    rest: Vec<Vec3>,
    control: Vec<usize>,
    control_law: Vec<Option<Displacement>>,
    free: Vec<usize>,
    idw: Idw,
    patch_names: Vec<String>,
    patch_rules: Vec<PatchMotion>,
}

/// One point's classification, mid-walk.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PointState<'a> {
    Untouched,
    Fixed,
    Slide,
    Move(&'a Displacement),
}

impl MeshMotion {
    pub fn new(
        m: &HostMesh,
        rest: &[Vec3],
        faces: &[Vec<Label>],
        rules: &[(&str, PatchMotion)],
    ) -> Result<Self> {
        if m.n_points != 0 && rest.len() != m.n_points {
            return Err(Error::Mesh(format!(
                "MeshMotion::new: {} rest points for a mesh of {} points",
                rest.len(),
                m.n_points
            )));
        }
        if faces.len() != m.n_internal_faces + m.n_boundary_faces {
            return Err(Error::Mesh(format!(
                "MeshMotion::new: {} face point lists for a mesh of {} internal and {} \
                 boundary faces",
                faces.len(),
                m.n_internal_faces,
                m.n_boundary_faces
            )));
        }
        let patch_of = |name: &str| m.patches.iter().position(|p| p.name == name);
        for (name, _) in rules {
            if patch_of(name).is_none() {
                return Err(Error::Config(format!(
                    "MeshMotion::new: no patch named {name} in this mesh"
                )));
            }
        }
        for i in 0..rules.len() {
            for j in i + 1..rules.len() {
                if rules[i].0 == rules[j].0 {
                    return Err(Error::Config(format!(
                        "MeshMotion::new: patch {} is named twice in the rule table",
                        rules[i].0
                    )));
                }
            }
        }
        for p in &m.patches {
            if !rules.iter().any(|(n, _)| *n == p.name) {
                return Err(Error::Config(format!(
                    "MeshMotion::new: patch {} has no motion rule; every patch of the \
                     mesh must be named",
                    p.name
                )));
            }
        }
        for (name, rule) in rules {
            let p = &m.patches[patch_of(name).expect("named, checked above")];
            if p.kind == PatchKind::Cyclic && *rule != PatchMotion::Fixed {
                return Err(Error::Config(format!(
                    "MeshMotion::new: a cyclic patch can only be `Fixed`, and {name} was \
                     given another rule"
                )));
            }
        }
        let mut state: Vec<PointState> = vec![PointState::Untouched; rest.len()];
        for (name, rule) in rules {
            let p = &m.patches[patch_of(name).expect("named, checked above")];
            let rm = match rule {
                PatchMotion::Fixed => PointState::Fixed,
                PatchMotion::Slide => PointState::Slide,
                PatchMotion::Move(l) => PointState::Move(l),
            };
            for k in 0..p.size {
                let fi = m.n_internal_faces + p.start + k;
                for &pt in &faces[fi] {
                    let pt = pt as usize;
                    if pt >= rest.len() {
                        return Err(Error::Mesh(format!(
                            "MeshMotion::new: face {fi} uses point {pt}, but only {} rest \
                             points were given",
                            rest.len()
                        )));
                    }
                    state[pt] = match (state[pt], rm) {
                        (PointState::Untouched, s) => s,
                        (PointState::Slide, s) => s,
                        (PointState::Move(l), PointState::Slide) => PointState::Move(l),
                        (PointState::Fixed, PointState::Fixed) => PointState::Fixed,
                        (PointState::Move(l), PointState::Move(l2)) => {
                            if l != l2 {
                                return Err(Error::Config(format!(
                                    "MeshMotion::new: patches prescribe different motions for \
                                     point {pt}"
                                )));
                            }
                            PointState::Move(l)
                        }
                        (PointState::Move(_), PointState::Fixed)
                        | (PointState::Fixed, PointState::Move(_)) => {
                            return Err(Error::Config(format!(
                                "MeshMotion::new: patches prescribe different motions for point \
                                 {pt}: a point cannot be both moved by a law and fixed"
                            )));
                        }
                        // a later Untouched or Slide never overrides anything
                        (settled, PointState::Slide) | (settled, PointState::Untouched) => {
                            settled
                        }
                    };
                }
            }
        }
        let mut control = Vec::new();
        let mut control_law = Vec::new();
        let mut free = Vec::new();
        for (pt, s) in state.iter().enumerate() {
            match s {
                PointState::Move(l) => {
                    control.push(pt);
                    control_law.push(Some(**l));
                }
                PointState::Fixed => {
                    control.push(pt);
                    control_law.push(None);
                }
                _ => free.push(pt),
            }
        }
        let idw = Idw::new(control.iter().map(|&p| rest[p]).collect())?;
        Ok(Self {
            rest: rest.to_vec(),
            control,
            control_law,
            free,
            idw,
            patch_names: rules.iter().map(|(n, _)| (*n).to_string()).collect(),
            patch_rules: rules.iter().map(|(_, r)| *r).collect(),
        })
    }

    pub fn rest(&self) -> &[Vec3] {
        &self.rest
    }

    pub fn n_control(&self) -> usize {
        self.control.len()
    }

    pub fn n_free(&self) -> usize {
        self.free.len()
    }

    /// The points at time `t`: prescribed points exactly where their laws put
    /// them, free points where the smoother - measured from the REST
    /// position, never incrementally - puts them.
    pub fn points_at(&self, t: Scalar) -> Vec<Vec3> {
        let mut out = self.rest.clone();
        let mut d = Vec::with_capacity(self.control.len());
        for (k, &p) in self.control.iter().enumerate() {
            let dk = self.control_law[k].map_or(Vec3::ZERO, |l| l.at(t));
            d.push(dk);
            if self.control_law[k].is_some() {
                out[p] = self.rest[p] + dk;
            }
        }
        for &p in &self.free {
            out[p] = self.rest[p] + self.idw.displacement(self.rest[p], &d);
        }
        out
    }

    /// The boundary faces of the named patches, in the order the names were
    /// given, faces ascending - the moving wall's face list (SPEC-LIT 105.8).
    pub fn wall_faces(&self, m: &HostMesh, patches: &[&str]) -> Result<Vec<Label>> {
        let mut out = Vec::new();
        for name in patches {
            let pi = m
                .patches
                .iter()
                .position(|p| &p.name == name)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "MeshMotion::wall_faces: no patch named {name} in this mesh"
                    ))
                })?;
            let p = &m.patches[pi];
            let ri = self
                .patch_names
                .iter()
                .position(|n| n == name)
                .expect("the rule table named every patch, checked at new");
            match self.patch_rules[ri] {
                PatchMotion::Move(l) => {
                    let dir = l.direction();
                    for k in 0..p.size {
                        let i = p.start + k;
                        if dir.cross(m.b_sf[i]).mag() > 1e-12 * dir.mag() * m.b_sf[i].mag() {
                            return Err(Error::Config(format!(
                                "MeshMotion::wall_faces: patch {name} face {i}: a tangential \
                                 wall velocity is not implemented"
                            )));
                        }
                        out.push(i as Label);
                    }
                }
                _ => {
                    return Err(Error::Config(format!(
                        "MeshMotion::wall_faces: patch {name} does not move; only a `Move` \
                         patch has a moving wall"
                    )));
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
