// Fixture generators for the dataset tests (and anyone who needs a small,
// realistic result tree without the Rust binaries): a JSONC case with its
// `<stem>_jsonc/` output, an OpenFOAM directory case with a blockgen-layout
// polyMesh, and a VTU/PVD series. Fields are analytic so tests can check
// statistics against closed forms.
import fs from 'node:fs/promises'
import path from 'node:path'
import { buildCartesianGrid, cartesianCellCenters, cartesianToPolyMesh, type CartesianGrid, type CartesianSpec } from '../../formats/cartesian.js'
import { writeFoamField } from '../../formats/foam.js'
import { writePolyMesh } from '../../formats/polymesh.js'
import { writePvd } from '../../formats/pvd.js'
import { writeVtuFromPolyMesh } from '../../formats/vtu.js'

export const FIXTURE_SPEC: CartesianSpec = {
  bounds: { min: [0, 0, 0], max: [4, 2, 1] },
  cells: [8, 4, 2],
  grading: { y: { expansion: 3, twoSided: true } },
  boundaries: { xmin: 'inlet', xmax: 'outlet', ymin: 'floor', ymax: 'ceiling', zmin: 'sideMin', zmax: 'sideMax' },
  regions: [{ name: 'vent', on: 'ymin', shape: { kind: 'box', min: [1, -1, -1], max: [2, 1, 2] } }],
  cyclic: [],
  patchTypes: { inlet: 'patch', outlet: 'patch', floor: 'wall', ceiling: 'wall', sideMin: 'wall', sideMax: 'wall', vent: 'patch' },
}

export const FIXTURE_JSONC = `// generated fixture
{
  "$schema": "https://meteor-cfd.msimul.com/schema/case-1.json",
  "name": "fixtureCase",
  "mesh": {
    "kind": "cartesian",
    "bounds": { "min": [0, 0, 0], "max": [4, 2, 1] },
    "cells": [8, 4, 2],
    "grading": { "y": { "expansion": 3, "twoSided": true } },
    "boundaries": { "xmin": "inlet", "xmax": "outlet", "ymin": "floor", "ymax": "ceiling", "zmin": "sideMin", "zmax": "sideMax" },
    "regions": [ { "name": "vent", "on": "ymin", "shape": { "kind": "box", "min": [1, -1, -1], "max": [2, 1, 2] } } ],
  },
  "physics": { "gravity": [0, -9.81, 0], "fluid": { "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }, "buoyancy": "densityRatio" },
  "turbulence": { "kind": "RAS", "model": "kEpsilon", "wallFunctions": { "kappa": 0.41, "E": 9.8 } },
  "patches": [
    { "match": "inlet|vent", "kind": "inlet" },
    { "match": "outlet", "kind": "open" },
    { "match": ".*", "kind": "wall" },
  ],
  "initial": { "U": [1, 0, 0], "p": 0, "k": 0.1, "epsilon": 0.01 },
  "numerics": { "algorithm": { "kind": "SIMPLE" } },
  "run": { "endTime": 100, "deltaT": 1 },
}
`

/** U = (x, 0.5 y, 0), p = -x, k = 0.1 + 0.01 t (uniform), epsilon = z. */
export function analyticFields(grid: CartesianGrid, t: number): { U: Float64Array; p: Float64Array; k: Float64Array; epsilon: Float64Array } {
  const c = cartesianCellCenters(grid)
  const n = c.length / 3
  const U = new Float64Array(3 * n)
  const p = new Float64Array(n)
  const k = new Float64Array(n)
  const epsilon = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    U[3 * i] = c[3 * i]
    U[3 * i + 1] = 0.5 * c[3 * i + 1]
    U[3 * i + 2] = 0
    p[i] = -c[3 * i]
    k[i] = 0.1 + 0.01 * t
    epsilon[i] = c[3 * i + 2]
  }
  return { U, p, k, epsilon }
}

const PATCHES = [
  { name: 'inlet', type: 'fixedValue', entries: { value: 'uniform 0' } },
  { name: 'outlet', type: 'zeroGradient' },
  { name: 'vent', type: 'fixedValue', entries: { value: 'uniform 0' } },
  { name: 'floor', type: 'zeroGradient' },
  { name: 'ceiling', type: 'zeroGradient' },
  { name: 'sideMin', type: 'zeroGradient' },
  { name: 'sideMax', type: 'zeroGradient' },
]

async function writeTime(dir: string, time: string, fields: Partial<ReturnType<typeof analyticFields>>): Promise<void> {
  const t = path.join(dir, time)
  if (fields.U) await writeFoamField(path.join(t, 'U'), { name: 'U', class: 'volVectorField', dimensions: '[0 1 -1 0 0 0 0]', time, data: fields.U, components: 3, patches: PATCHES })
  if (fields.p) await writeFoamField(path.join(t, 'p'), { name: 'p', class: 'volScalarField', dimensions: '[0 2 -2 0 0 0 0]', time, data: fields.p, components: 1, patches: PATCHES })
  if (fields.k) await writeFoamField(path.join(t, 'k'), { name: 'k', class: 'volScalarField', dimensions: '[0 2 -2 0 0 0 0]', time, data: fields.k, components: 1, patches: PATCHES })
  if (fields.epsilon) await writeFoamField(path.join(t, 'epsilon'), { name: 'epsilon', class: 'volScalarField', dimensions: '[0 2 -3 0 0 0 0]', time, data: fields.epsilon, components: 1, patches: PATCHES })
}

/**
 * `<dir>/fixture.jsonc` + `<dir>/fixture_jsonc/{0,50,100}`: U only in 0/ (frozen-U solver),
 * p/k/epsilon at 50 and 100, k uniform, plus a surface field `phi` that must be ignored.
 */
export async function makeJsoncCase(dir: string): Promise<{ jsonc: string; out: string; grid: CartesianGrid }> {
  await fs.mkdir(dir, { recursive: true })
  const jsonc = path.join(dir, 'fixture.jsonc')
  await fs.writeFile(jsonc, FIXTURE_JSONC)
  const out = path.join(dir, 'fixture_jsonc')
  const grid = buildCartesianGrid(FIXTURE_SPEC)
  const f0 = analyticFields(grid, 0)
  await writeTime(out, '0', { U: f0.U, p: f0.p, k: f0.k, epsilon: f0.epsilon })
  const f50 = analyticFields(grid, 50)
  await writeTime(out, '50', { p: f50.p, k: f50.k, epsilon: f50.epsilon })
  const f100 = analyticFields(grid, 100)
  await writeTime(out, '100', { p: f100.p, k: f100.k, epsilon: f100.epsilon })
  await fs.writeFile(path.join(out, '100', 'phi'), 'FoamFile\n{\n    class       surfaceScalarField;\n    object      phi;\n}\ndimensions [0 3 -1 0 0 0 0];\ninternalField nonuniform List<scalar> 2(1 2);\n')
  return { jsonc, out, grid }
}

/** OpenFOAM directory case: constant/polyMesh in blockgen layout, constant/g (y down), 0/U and 0/p, 200/p. */
export async function makeFoamCase(dir: string): Promise<{ grid: CartesianGrid }> {
  const grid = buildCartesianGrid(FIXTURE_SPEC)
  const mesh = cartesianToPolyMesh(grid, FIXTURE_SPEC)
  await writePolyMesh(path.join(dir, 'constant', 'polyMesh'), mesh)
  await fs.writeFile(path.join(dir, 'constant', 'g'), 'FoamFile { class uniformDimensionedVectorField; object g; }\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n')
  await fs.mkdir(path.join(dir, 'system'), { recursive: true })
  await fs.writeFile(path.join(dir, 'system', 'controlDict'), 'application ofgpu;\n')
  const f0 = analyticFields(grid, 0)
  await writeTime(dir, '0', { U: f0.U, p: f0.p })
  await writeTime(dir, '200', { p: analyticFields(grid, 200).p })
  return { grid }
}

/** `<dir>/VTK/series.pvd` + two VTU steps written exactly like rust/src/io/vtu.rs. */
export async function makeVtuSeries(dir: string): Promise<{ pvd: string; grid: CartesianGrid }> {
  const grid = buildCartesianGrid(FIXTURE_SPEC)
  const mesh = cartesianToPolyMesh(grid, FIXTURE_SPEC)
  const vtk = path.join(dir, 'VTK')
  const entries: Array<{ time: number; file: string }> = []
  for (const [step, t] of [[0, 0], [10, 0.5]] as Array<[number, number]>) {
    const f = analyticFields(grid, step)
    const file = `series_${String(step).padStart(6, '0')}.vtu`
    await writeVtuFromPolyMesh(path.join(vtk, file), mesh, { time: t, step, cellData: [{ name: 'p', components: 1, data: f.p }, { name: 'U', components: 3, data: f.U }] })
    entries.push({ time: t, file })
  }
  const pvd = path.join(vtk, 'series.pvd')
  await writePvd(pvd, entries)
  return { pvd, grid }
}
