// File-format readers and writers for the meteor-cfd on-disk formats. This
// file fixes the API; the implementations live in the sibling modules:
//   foam.ts       OpenFOAM ASCII vol*Field reader/writer (rust/src/io/fields.rs)
//   polymesh.ts   constant/polyMesh reader/writer (rust/src/io/polymesh.rs, blockgen.rs)
//   polycut.ts    exact plane cut of a polyhedral polyMesh, triangulated per cell (SPEC-LIT 148.1)
//   pointLocate.ts  the cell of a polyMesh that holds a world point (docs/17 STU-13)
//   cartesian.ts  JSONC `mesh` block -> graded nodes, cell centres, boundary surface (blockgen.rs)
//   vtu.ts        VTK XML UnstructuredGrid appended-raw reader/writer, VTK_POLYHEDRON (rust/src/io/vtu.rs)
//   pvd.ts        .pvd collections
//   casejsonc.ts  JSONC case reading (comments stripped), mesh spec extraction
//   results.ts    time-directory discovery with OpenFOAM time fallback

export * from './geometry.js'
export * from './foam.js'
export * from './polymesh.js'
export * from './polycut.js'
export * from './pointLocate.js'
export * from './cartesian.js'
export * from './vtu.js'
export * from './pvd.js'
export * from './casejsonc.js'
export * from './results.js'
