// File-format readers and writers for the meteor-cfd on-disk formats. This
// file fixes the API; the implementations live in the sibling modules:
//   foam.ts       OpenFOAM ASCII vol*Field reader/writer (rust/src/io/fields.rs)
//   polymesh.ts   constant/polyMesh reader/writer (rust/src/io/polymesh.rs, blockgen.rs)
//   cartesian.ts  JSONC `mesh` block -> graded nodes, cell centres, boundary surface (blockgen.rs)
//   vtu.ts        VTK XML UnstructuredGrid appended-raw reader/writer, VTK_POLYHEDRON (rust/src/io/vtu.rs)
//   pvd.ts        .pvd collections
//   casejsonc.ts  JSONC case reading (comments stripped), mesh spec extraction
//   results.ts    time-directory discovery with OpenFOAM time fallback

export * from './geometry.js'
export * from './foam.js'
export * from './polymesh.js'
export * from './cartesian.js'
export * from './vtu.js'
export * from './pvd.js'
export * from './casejsonc.js'
export * from './results.js'
