# OpenFOAM model catalogue

OpenFOAM-12 (OpenFOAM Foundation) 소스 트리에 들어 있는 모든 런타임 선택 가능 모델·스킴·솔버·메쉬 구성요소의 전수 목록입니다.

Every entry was read out of `[Foundation-12] src`. The *Keyword* column is what you write in a dictionary to select the component; an empty keyword means it is a base class, a helper, or selected implicitly.

ESI(v2606)에만 있는 모델은 `03-esi-vs-foundation.md`를 보세요. GPU 이식성 등급은 `02-gpu-portability.md`에 있습니다.

## Contents

| Subsystem | Components |
|---|---:|
| [Linear algebra: lduMatrix solvers, preconditioners, smoothers, AMG](#linear-algebra-ldumatrix-solvers,-preconditioners,-smoothers,-amg) | 91 |
| [Momentum transport / turbulence](#momentum-transport--turbulence) | 129 |
| [Thermophysical and chemistry](#thermophysical-and-chemistry) | 212 |
| [Multiphase, two-phase, lagrangian, waves](#multiphase,-two-phase,-lagrangian,-waves) | 289 |
| [finiteVolume discretisation schemes, surface interpolation and fvMatrix](#finitevolume-discretisation-schemes,-surface-interpolation-and-fvmatrix) | 317 |
| [Mesh: Mesh generation and manipulation](#mesh-mesh-generation-and-manipulation) | 145 |
| [Mesh: Mesh tools and searching](#mesh-mesh-tools-and-searching) | 187 |
| [Mesh: Mesh motion, run-time topology change, decomposition](#mesh-mesh-motion,-run-time-topology-change,-decomposition) | 187 |
| [Mesh: Core mesh data structures](#mesh-core-mesh-data-structures) | 204 |
| **Total** | **1761** |

---

## Linear algebra: lduMatrix solvers, preconditioners, smoothers, AMG

> **Subsystem notes**
>
> IMPORTANT CHECKOUT ARTEFACT: upstream OpenFOAM-12 has TWO sibling directories that differ only by the case of one letter — src/OpenFOAM/matrices/lduMatrix (the scalar lduMatrix hierarchy) and src/OpenFOAM/matrices/LduMatrix (the templated LduMatrix<Type,DType,LUType> hierarchy).
> On this Windows (case-insensitive) checkout they have been merged into a single directory, src/OpenFOAM/matrices/LduMatrix, and where two files' names differed only by case one of the pair was lost.
> Confirmed missing on disk but present upstream and referenced by the build: LduMatrix.H / LduMatrix.C / LduMatrixTemplates.C, DiagonalSolver.H/.C, SmoothSolver.H/.C, NoPreconditioner.H/.C, DiagonalPreconditioner.H/.C, SolverPerformance.H/.C.
> src/OpenFOAM/Make/files still lists both trees separately (lduMatrix = matrices/lduMatrix at line 333, LduMatrix = matrices/LduMatrix at line 432) and is the authoritative record of which files belong where.
> Paths in this catalogue are the ACTUAL on-disk paths in this checkout; upstream the scalar-hierarchy ones live under .../matrices/lduMatrix/{solvers,smoothers,preconditioners,lduAddressing,lduMatrix}/. TWO PARALLEL HIERARCHIES.
> (1) lduMatrix (scalar coefficients) with lduMatrix::solver / ::preconditioner / ::smoother, each declaring exactly two RTS tables — symMatrix and asymMatrix — registered with lduMatrix::solver::addsymMatrixConstructorToTable<X> / addasymMatrixConstructorToTable<X>, NOT with addToRunTimeSelectionTable.
> (2) LduMatrix<Type,DType,LUType> (block/coupled) with three tables each (generic, symMatrix, asymMatrix) populated by the makeLduSolver/makeLduSymSolver/makeLduAsymSolver, makeLduPreconditioner/..., makeLduSmoother/...
> macros expanded in Solvers/lduSolvers.C, Preconditioners/lduPreconditioners.C and Smoothers/lduSmoothers.C for five field types (scalar, vector, sphericalTensor, symmTensor, tensor). So e.g. PCICG exists in 5 instantiations.
> Only GAMG-related classes and the agglomerations use plain addToRunTimeSelectionTable.
> SYMMETRIC / ASYMMETRIC AVAILABILITY MATRIX (this is what users hit as "Unknown symmetric matrix smoother"): solvers — PCG sym only, PBiCG asym only, PBiCGStab both, smoothSolver both, GAMG both, diagonal auto-selected for diagonal matrices; preconditioners — DIC sym, FDIC sym, DILU asym, diagonal both, GAMG both, none both; smoothers — GaussSeidel both, symGaussSeidel both, nonBlockingGaussSeidel both, DIC sym, FDIC sym, DICGaussSeidel sym, DILU asym, DILUGaussSeidel asym.
> SELECTION KEYWORD SOURCES: 'solver' is read with a plain lookup in lduMatrix::solver::New.
> 'preconditioner' and 'smoother' are read via lookupEntry(..., false, false) and accept EITHER a primitive entry (preconditioner DIC;) OR a sub-dictionary whose own 'preconditioner'/'smoother' entry names the type — this is how GAMG-as-preconditioner takes its own nVcycles/agglomerator settings.
> GAMG AGGLOMERATION HAS THREE SEPARATE RTS TABLES, all keyed on the same 'agglomerator' entry (default faceAreaPair): lduMesh (pure geometric: faceAreaPair, dummy, MGridGen), lduMatrix (matrix/mixed: algebraicPair), and geometry (mesh + cellVolumes + faceAreas: faceAreaPair only).
> GAMGAgglomeration::New(matrix, dict) tries the lduMatrix table first and silently falls back to the lduMesh table if the name is not found there. Extra libraries are dlopened from the dict entries geometricGAMGAgglomerationLibs and algebraicGAMGAgglomerationLibs.
> pairGAMGAgglomeration is an abstract intermediate (TypeName "pair") that is NOT registered — algebraicPair and faceAreaPair are its concrete children.
> Note faceAreaPair, the DEFAULT agglomerator, is not in src/OpenFOAM at all: it lives in libfiniteVolume (src/finiteVolume/fvMatrices/solvers/GAMGSymSolver/GAMGAgglomerations/).
> GAMG COARSEST-LEVEL SOLVE: with directSolveCoarsest yes; the coarsest matrix is gathered (procLduMatrix/procLduInterface) into a dense LUscalarMatrix and solved directly; otherwise GAMGSolverSolve.C hard-codes two internal dictionaries — "solver PCG; preconditioner DIC;" for symmetric and "solver PBiCGStab; preconditioner DILU;" for asymmetric coarsest solves.
> These are not user-selectable.
> GAMGProcAgglomeration keyword note: noneGAMGProcAgglomeration is registered with addNamedToRunTimeSelectionTable under the name 'none', which differs from its TypeName ('noneGAMGProcAgglomeration'); all the others (masterCoarsest, eager, manual, procFaces) use their TypeName.
> GAMG interface/interfaceField selection is not user-facing: GAMGInterface and GAMGInterfaceField are keyed on the coupled patch type word (cyclic, processor, processorCyclic) taken from the fine-level interface, and each concrete class registers in two tables (lduInterface + Istream for interfaces; lduInterfaceField + lduInterface for interface fields) so coarse levels can be both agglomerated locally and reconstructed on a receiving processor after processor agglomeration.
> PARALLEL COUPLING: every solver's Amul/Tmul goes through lduMatrix::initMatrixInterfaces/updateMatrixInterfaces, which branches on UPstream::defaultCommsType (blocking | scheduled | nonBlocking, from the commsType optimisation switch).
> nonBlockingGaussSeidelSmoother exists specifically to overlap that exchange with interior smoothing and REQUIRES the mesh cells to be renumbered so coupled cells are last. Other relevant optimisation switches: floatTransfer (0), nProcsSimpleSum (16), nPollProcInterfaces (0).
> NOT FOUND / DOES NOT EXIST in OpenFOAM-12 Foundation despite being common in other forks: no ILU(k)/ICCG-by-that-name entries (ICCG and BICCG were removed long ago — PCG+DIC and PBiCG+DILU replace them), no AMGCL/PETSc/Hypre bindings, no PBiCGStab preconditioned-transpose variant, no Jacobi smoother class (use diagonal preconditioner or GaussSeidel), no red-black or ILU smoothers, no algebraic multigrid other than the pair/MGridGen agglomeration-based GAMG.
> Total user-selectable runtime entries in this subsystem: 6 scalar linear solvers + 5 templated/coupled solvers, 6 scalar preconditioners + 3 templated preconditioners, 8 scalar smoothers + 1 templated smoother, 4 concrete GAMG agglomerators (across 3 tables), 5 GAMG processor agglomerators, 3 GAMG interfaces and 3 GAMG interface fields.

### AMG agglomeration (abstract intermediate)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pairGAMGAgglomeration` | `(TypeName "pair", but abstract — use algebraicPair or faceAreaPair. Controls: mergeLevels (default 1), nCellsInCoarsestLevel (default 10))` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGAgglomerations/pairGAMGAgglomeration/pairGAMGAgglomeration.C` | Implements the pair (face-weight based) agglomeration algorithm shared by algebraicPair and faceAreaPair. Alternates cell-loop direction each level (forward_ flag) and can merge mergeLevels levels at a time. Not itself registered in any RTS table. | For each cell in (alternating) order pick the neighbour connected by the largest face weight w_f that is not yet clustered and pair them; unmatched cells join their strongest already-formed cluster. Repeat until nCells <= nCellsInCoarsestLevel or maxLevels reached. |

### AMG agglomeration (algebraic)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `algebraicPairGAMGAgglomeration` | `algebraicPair` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGAgglomerations/algebraicPairGAMGAgglomeration/algebraicPairGAMGAgglomeration.C` | Pair agglomeration whose face weights come from the matrix coefficients rather than the geometry. Registered in the GAMGAgglomeration lduMatrix table. | faceWeights_f = mag(upper_f) (for symmetric) or mag(upper_f + lower_f)/2 for asymmetric, normalised by the maximum; then the pair algorithm. |

### AMG agglomeration (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GAMGAgglomeration` | `(base; declares three RTS tables: lduMesh, lduMatrix and geometry, all keyed on the 'agglomerator' entry, default faceAreaPair; also opens libs listed under geometricGAMGAgglomerationLibs / algebraicGAMGAgglomerationLibs)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGAgglomerations/GAMGAgglomeration/GAMGAgglomeration.C` | Abstract base holding the multilevel restriction addressing (restrictAddressing_, faceRestrictAddressing_, faceFlipMap_, nCells_/nFaces_ per level), the mesh hierarchy and the restrictField/prolongField operators. A DemandDrivenMeshObject, so agglomeration is cached on the mesh registry. | restrictField: c^H_I = sum_{i in cluster I} c^h_i (summation restriction). prolongField: c^h_i = c^H_{I(i)} (injection). agglomerateLduAddressing builds coarse owner/neighbour from faceRestrictAddressing. |

### AMG agglomeration (external library)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MGridGenGAMGAgglomeration` | `MGridGen (controls: nCellsInCoarsestLevel, mergeLevels; requires geometricGAMGAgglomerationLibs ("libMGridGenGAMGAgglomeration.so"))` | `[Foundation-12] src/fvAgglomerationMethods/MGridGenGamgAgglomeration/MGridGenGAMGAgglomeration.C` | Agglomeration delegated to the third-party MGridGen library (MGridGen_f77 / MGRIDGEN). Built as a separate optional library (libMGridGenGAMGAgglomeration). Registered in the lduMesh table. | Calls MGridGen with cell volumes, face areas and the cell-cell adjacency to produce clusters bounded by minSize/maxSize and optimised for aspect ratio; agglomeration then proceeds level by level. |

### AMG agglomeration (geometric — default)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `faceAreaPairGAMGAgglomeration` | `faceAreaPair (this is the default value of the 'agglomerator' entry)` | `[Foundation-12] src/finiteVolume/fvMatrices/solvers/GAMGSymSolver/GAMGAgglomerations/faceAreaPairGAMGAgglomeration/faceAreaPairGAMGAgglomeration.C` | Default GAMG agglomerator. Pair agglomeration using face areas as weights. Registered in BOTH the lduMesh and the geometry GAMGAgglomeration tables (lives in libfiniteVolume because it needs fvMesh face areas). | faceWeights_f = mag(Sf_f)/max(mag(Sf)); then the pairGAMGAgglomeration pairing. |

### AMG agglomeration (testing)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dummyAgglomeration` | `dummy (control: nLevels)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGAgglomerations/dummyAgglomeration/dummyAgglomeration.C` | Agglomerates without actually combining any cells — produces nLevels identical levels. Used for testing the multigrid machinery. Registered in the lduMesh table. | restrictAddressing[i] = i on every level (identity restriction). |

### AMG coupled interface  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicGAMGInterface` | `cyclic` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaces/cyclicGAMGInterface/cyclicGAMGInterface.C` | Agglomerated cyclic interface; also implements cyclicLduInterface (neighbPatchID, transform). Registered in both the lduInterface and Istream GAMGInterface tables. |  |
| `processorCyclicGAMGInterface` | `processorCyclic` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaces/processorCyclicGAMGInterface/processorCyclicGAMGInterface.C` | Agglomerated processor-cyclic interface (a cyclic that has been split across processors); derives from processorGAMGInterface and additionally stores the referring patch index. |  |
| `processorGAMGInterface` | `processor` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaces/processorGAMGInterface/processorGAMGInterface.C` | Agglomerated processor interface; implements processorLduInterface (myProcNo, neighbProcNo, tag, comm, send/receive). Registered in the lduInterface and Istream tables. |  |

### AMG coupled interface (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GAMGInterface` | `(base; two RTS tables — lduInterface (from a fine interface + restriction) and Istream (reconstruct on a receiving processor))` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaces/GAMGInterface/GAMGInterface.C` | Abstract base for agglomerated (coarse-level) coupled interfaces; carries faceCells_ and faceRestrictAddressing_ and provides agglomerateCoeffs (restriction of interface coefficients) and interfaceInternalField. | coarseCoeffs_I = sum_{f in coarse face I} fineCoeffs_f. |

### AMG coupled interface field  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicGAMGInterfaceField` | `cyclic` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaceFields/cyclicGAMGInterfaceField/cyclicGAMGInterfaceField.C` | Coarse-level cyclic interface field; applies the cyclic transform to the neighbour internal field before the matrix update. Registered in both GAMGInterfaceField tables. | pnf = psiInternal[nbrFaceCells]; transform applied if doTransform; result[faceCells] -= coeffs*pnf. |
| `processorCyclicGAMGInterfaceField` | `processorCyclic` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaceFields/processorCyclicGAMGInterfaceField/processorCyclicGAMGInterfaceField.C` | Coarse-level processor-cyclic interface field, derived from processorGAMGInterfaceField. Registered in both GAMGInterfaceField tables. |  |
| `processorGAMGInterfaceField` | `processor` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaceFields/processorGAMGInterfaceField/processorGAMGInterfaceField.C` | Coarse-level processor interface field; does the send/receive (blocking, scheduled or non-blocking per UPstream::commsTypes) of the neighbour internal field and applies the matrix update. Supports outstandingSendRequest/RecvRequest for non-blocking. | scalarSendBuf = psiInternal[faceCells]; exchange; result[faceCells] -= coeffs*scalarReceiveBuf. |

### AMG coupled interface field (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GAMGInterfaceField` | `(base; RTS table lduInterfaceField keyed on the coupled patch type)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/interfaceFields/GAMGInterfaceField/GAMGInterfaceField.C` | Abstract base for the coarse-level interface field that performs updateInterfaceMatrix on agglomerated interfaces. Two selectors (New from a fine lduInterfaceField, New from a doTransform/rank pair). | result[faceCell] -= coeffs*psi_neighbour, with sign/transform handling. |

### AMG processor agglomeration  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `eagerGAMGProcAgglomeration` | `eager (control: mergeLevels)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGProcAgglomerations/eagerGAMGProcAgglomeration/eagerGAMGProcAgglomeration.C` | 'Eager' agglomeration: at every level combines mergeLevels processors onto the lowest-numbered processor of each group. |  |
| `manualGAMGProcAgglomeration` | `manual (control: processorAgglomeration ( (3 ((0 1)(3 2))) (6 ((0 1))) ))` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGProcAgglomerations/manualGAMGProcAgglomeration/manualGAMGProcAgglomeration.C` | User-specified processor agglomeration: an explicit list of (level, list-of-processor-clusters) entries; each cluster is merged onto its first element. |  |
| `masterCoarsestGAMGProcAgglomeration` | `masterCoarsest` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGProcAgglomerations/masterCoarsestGAMGProcAgglomeration/masterCoarsestGAMGProcAgglomeration.C` | Collapses the coarsest level onto the master processor only (all cells gathered onto proc 0), so the coarsest solve becomes serial. |  |
| `noneGAMGProcAgglomeration` | `none (registered via addNamedToRunTimeSelectionTable with the name 'none', although its TypeName is noneGAMGProcAgglomeration)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGProcAgglomerations/noneGAMGProcAgglomeration/noneGAMGProcAgglomeration.C` | No processor agglomeration — every level keeps the original processor decomposition. agglomerate() returns false. |  |
| `procFacesGAMGProcAgglomeration` | `procFaces (controls: nAgglomeratingCells, mergeLevels)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGProcAgglomerations/procFacesGAMGProcAgglomeration/procFacesGAMGProcAgglomeration.C` | Agglomerates processors once a level drops below nAgglomeratingCells, by building a one-cell-per-processor mesh whose faces are the processor interfaces and running the pairGAMGAgglomeration algorithm on it with the number of interface faces as weight. | Weight of the pseudo-face between processors p,q = number of real faces on that processor interface; pair agglomeration on that reduced graph. |

### AMG processor agglomeration (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GAMGProcAgglomeration` | `(base; RTS table GAMGAgglomeration keyed on the 'processorAgglomerator' entry of the GAMG solver dict)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGProcAgglomerations/GAMGProcAgglomeration/GAMGProcAgglomeration.C` | Abstract base for redistributing coarse levels onto fewer processors (reduces communication cost at coarse levels). Provides agglomerate(levelIndex, procAgglomMap, ...) which rebuilds the coarse lduMesh, the communicator and the distributionMap. |  |

### bounded explicit solver / limiter  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MULES` | `(dict controls in the field's solver dict: nLimiterIter, smoothLimiter, extremaCoeff, applyPrevCorr)` | `[Foundation-12] src/finiteVolume/fvMatrices/solvers/MULES/MULES.C` | Multidimensional Universal Limiter for Explicit Solution — solves a convective-only transport equation explicitly with a multidimensional FCT-style limiter that keeps the solution within [psiMin, psiMax]. Lives under fvMatrices/solvers alongside the linear solvers. | phiPsi = phiBD + lambda*phiCorr, where phiBD is the bounded (upwind) flux and phiCorr the higher-order correction. lambda_f in [0,1] is found by nLimiterIter sweeps of Zalesak's algorithm using the per-cell sums of incoming/outgoing corrections against psiMax/psiMin extrema. Then psi^{n+1} = (rho psi^n - dt*div(phiPsi) + dt*Su)/(rho + dt*Sp). |

### bounded semi-implicit solver / limiter  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CMULES` | `(used via MULES::correct; controls MULESCorr, nLimiterIter, alphaApplyPrevCorr in the alpha solver dict)` | `[Foundation-12] src/finiteVolume/fvMatrices/solvers/MULES/CMULES.C` | Corrected MULES: applies the explicit MULES limiter as a correction on top of a rigorously bounded implicit solution (Euler-implicit in time, upwind in space), enabling larger Courant numbers than pure explicit MULES. | Solve the implicit bounded equation for psi (Euler + upwind), then correct: psi += limited explicit correction flux, with the limiter computed as in MULES against the implicit solution's bounds. |

### dense decomposition  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LLTMatrix<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/LLTMatrix/LLTMatrix.C` | Cholesky decomposition of a symmetric positive-definite SquareMatrix, with solve() for linear systems. | A = L L^T; forward substitution L y = b then back substitution L^T x = y. |
| `QRMatrix<MatrixType>` |  | `[Foundation-12] src/OpenFOAM/matrices/QRMatrix/QRMatrix.C` | QR decomposition by Householder reflections for square or rectangular matrices; supports full/economy output and column pivoting, with solve() and inv() built on it. | A = Q R with Q orthogonal built from successive Householder reflectors H_k = I - 2 v v^T/(v^T v); solve by R x = Q^T b back-substitution. |
| `SVD` |  | `[Foundation-12] src/OpenFOAM/matrices/scalarMatrices/SVD/SVD.C` | Singular value decomposition of a rectangular matrix; stores U_, V_, S_ and nZeros_. Backs SVDinv and the least-squares gradient/interpolation schemes. | A = U S V^T via Householder bidiagonalisation plus QR iteration; pseudo-inverse VSinvUt() = V diag(1/s_i) U^T with s_i below the condition threshold set to zero. |

### dense linear algebra  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `scalarMatrices (free functions)` |  | `[Foundation-12] src/OpenFOAM/matrices/scalarMatrices/scalarMatrices.C` | Typedefs scalarRectangularMatrix / scalarSquareMatrix / scalarSymmetricSquareMatrix / scalarDiagonalMatrix and the dense algorithms: solve (Gaussian elimination with pivoting), LUDecompose (with pivoting, and Cholesky for symmetric, U = L^T, from JAMA/NIST), LUBacksubstitute, LUsolve, multiply (2- and 3-factor, incl. diagonal middle factor) and SVDinv. | Gaussian elimination with partial pivoting; PA = LU; for symmetric SPD A = L L^T; SVDinv(A) = V diag(1/s_i) U^T with small singular values zeroed by minCondition. |
| `simpleMatrix<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/simpleMatrix/simpleMatrix.C` | A small square matrix with scalar coefficients and a Field<Type> source; solve() via Gaussian elimination with pivoting, LUsolve() via LU. Used for e.g. small coupled ODE/thermo systems and boundary-condition coupling. | solve(): Foam::solve(scalarSquareMatrix, source) -> Gaussian elimination with pivoting. LUsolve(): LU decomposition then back-substitution. |

### dense matrix container  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DiagonalMatrix<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/DiagonalMatrix/DiagonalMatrix.C` | n x n diagonal matrix stored as a List<Type>; constructible from any Matrix (extracts the diagonal) and has inv(). |  |
| `Matrix<Form, Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/Matrix/Matrix.C` | CRTP base for all dense (m x n) matrices — storage, subscripting, block access, arithmetic operators, T() transpose, IO. |  |
| `MatrixBlock<MatrixType> / ConstMatrixBlock<MatrixType>` |  | `[Foundation-12] src/OpenFOAM/matrices/MatrixBlock/MatrixBlock.C` | Non-owning views of a rectangular block of a dense matrix, assignable to/from another block, a VectorSpace/MatrixSpace (e.g. tensor) or a Field<T> for a column block. |  |
| `RectangularMatrix<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/RectangularMatrix/RectangularMatrix.C` | m x n dense matrix; the type used by SVD and least-squares fits. |  |
| `SquareMatrix<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/SquareMatrix/SquareMatrix.C` | n x n dense matrix with bounds checking; supports construction from an Identity and inv()/det() helpers. |  |
| `SymmetricSquareMatrix<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/SymmetricSquareMatrix/SymmetricSquareMatrix.C` | n x n symmetric dense matrix storing only the lower triangle; the Cholesky LUDecompose/LUBacksubstitute overloads in scalarMatrices are specialised for it. | A = L L^T with U = L^T so no pivoting is required in back-substitution. |

### direct dense solver  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LUscalarMatrix` | `(activated by directSolveCoarsest yes; in the GAMG dict)` | `[Foundation-12] src/OpenFOAM/matrices/LUscalarMatrix/LUscalarMatrix.C` | Converts an lduMatrix (optionally gathered from all processors) into a dense scalarSquareMatrix and LU-decomposes it. Used by GAMGSolver when directSolveCoarsest is true, and by lagrangian/ODE code. | PA = LU with partial pivoting (pivotIndices_); solve by forward/back substitution. In parallel the whole coarsest matrix is gathered onto the master via procLduMatrix/procLduInterface and solved there. |

### direct dense solver support  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `procLduInterface` |  | `[Foundation-12] src/OpenFOAM/matrices/LUscalarMatrix/procLduInterface.C` | Serialisable per-processor copy of an lduInterface (faceCells, coeffs, myProcNo, neighbProcNo, tag, comm) accompanying procLduMatrix. |  |
| `procLduMatrix` |  | `[Foundation-12] src/OpenFOAM/matrices/LUscalarMatrix/procLduMatrix.C` | Serialisable per-processor copy of an lduMatrix (upper/lower/diag + addressing) used to gather a distributed matrix onto one processor for the direct coarsest-level solve. |  |

### ldu addressing  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lduAddressing` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduAddressing.C` | Holds the upper (owner) and lower (neighbour) face addressing plus the lazily built losort addressing and the ownerStart/losortStart offset lists used by the Gauss-Seidel and *IC/ILU sweeps. | Owner labels ascending with grouped identical labels; losort orders faces by neighbour so the lower-triangle contribution to a cell can be gathered contiguously. |
| `lduScheduleEntry / lduSchedule` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduSchedule/lduSchedule.H` | Struct {label patch; bool init;} and List typedef describing the deterministic patch communication schedule used when commsType is 'scheduled'. |  |

### ldu coupled interface (base)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicLduInterface` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduInterface/cyclicLduInterface.C` | Abstract base for cyclic coupled interfaces; adds neighbPatchID(), owner() and transform(). |  |
| `lduInterface` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduInterface/lduInterface.C` | Abstract base class for implicitly-coupled interfaces (processor, cyclic patches). Declares faceCells(), interfaceInternalField() and the transfer/internalFieldTransfer protocol. |  |
| `processorLduInterface` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduInterface/processorLduInterface.C` | Abstract base for processor coupled interfaces; adds myProcNo(), neighbProcNo(), tag(), comm(), transform() and templated send/receive/compressedSend/compressedReceive honouring UPstream::commsTypes and UPstream::floatTransfer. |  |

### ldu coupled interface field (base)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicLduInterfaceField` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduInterfaceFields/cyclicLduInterfaceField/cyclicLduInterfaceField.C` | Abstract base for cyclic coupled interface fields; supplies doTransform(), rank() and the transformCoupleField helpers. |  |
| `lduInterfaceField` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduInterfaceFields/lduInterfaceField/lduInterfaceField.C` | Abstract base for implicitly-coupled interface fields; declares initInterfaceMatrixUpdate/updateInterfaceMatrix which the solvers call inside Amul/Tmul, plus the transform-based sign handling. |  |
| `processorLduInterfaceField` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/lduAddressing/lduInterfaceFields/processorLduInterfaceField/processorLduInterfaceField.C` | Abstract base for processor coupled interface fields; supplies doTransform(), rank() and transformCoupleField for the parallel halo exchange. |  |

### ldu coupled interface field (templated)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LduInterfaceField<Type>` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/LduInterfaceField/LduInterfaceField.C` | Templated interface-field base used by the LduMatrix<Type,DType,LUType> hierarchy; instantiated for scalar, vector, sphericalTensor, symmTensor, tensor by LduInterfaceFields.C. |  |

### ldu mesh framework  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lduMesh` |  | `[Foundation-12] src/OpenFOAM/meshes/lduMesh/lduMesh.C` | Abstract base for any mesh that can supply LDU addressing (lduAddr(), interfaces(), comm(), thisDb()) for lduMatrix construction and the LDU solvers. |  |
| `lduPrimitiveMesh` |  | `[Foundation-12] src/OpenFOAM/meshes/lduMesh/lduPrimitiveMesh.C` | Simplest concrete lduMesh, storing owner/neighbour lists directly. Used for the GAMG coarse levels and for the gathered/agglomerated processor-level matrices. |  |

### linear algebra kernels  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lduMatrix ATmul / Amul / Tmul / H / faceH / residual / sumA` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/lduMatrixATmul.C` | The core sparse kernels: Amul (A.psi), Tmul (A^T.psi), sumA (row sums), H (off-diagonal product used for the SIMPLE/PISO H operator), faceH, residual, sumMagOffDiag. All initialise/update the coupled interfaces around the interior loop. | Amul: ApsiPtr[i] = D_i psi_i; then for each face f: Apsi[u_f] += L_f psi[l_f]; Apsi[l_f] += U_f psi[u_f]. Tmul swaps L and U. H(psi)_i = -(sum_f U_f psi[u_f] + sum_f L_f psi[l_f]). |

### linear solver (AMG, symmetric + asymmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GAMGSolver` | `GAMG (controls: agglomerator, nCellsInCoarsestLevel, mergeLevels, cacheAgglomeration, nPreSweeps, preSweepsLevelMultiplier, maxPreSweeps, nPostSweeps, postSweepsLevelMultiplier, maxPostSweeps, nFinestSweeps, interpolateCorrection, scaleCorrection, directSolveCoarsest, smoother, processorAgglomerator)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/GAMG/GAMGSolver.C` | Geometric-agglomerated algebraic multigrid solver. Requires positive-definite, diagonally dominant matrix. Restriction = summation, prolongation = injection, V-cycle with optional pre-smoothing, coarse-level correction scaling by steepest descent, coarsest level by direct LU or PCG/PBiCGStab. | V-cycle: restrict residual r^H = R r^h (cluster summation); smooth coarse correction; optionally scale correction by alpha = (Ac, s)/(Ac, Ac) style steepest-descent factor; prolongate by injection with optional interpolateCorrection; post-smooth. Coarse matrices built by summing fine diag + intra-cluster face coeffs and summing inter-cluster face coeffs. |

### linear solver (asymmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PBiCG` | `PBiCG` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/PBiCG/PBiCG.C` | Preconditioned bi-conjugate-gradient solver for asymmetric lduMatrices with a run-time selectable preconditioner. Registered in asymMatrix table only. | Classic BiCG: maintains shadow residual rT and shadow search direction pT using A^T (Tmul) and the transpose preconditioner (preconditionT); wArT = (w,rT); beta = wArT/wArTold; alpha = wArT/(w,pT); psi += alpha p; r -= alpha w; rT -= alpha wT. |

### linear solver (diagonal)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `diagonalSolver` | `diagonal` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/DiagonalSolver/diagonalSolver.C` | Trivial direct solver for purely diagonal matrices. Note: lduMatrix::solver::New short-circuits to this whenever matrix.diagonal() is true, regardless of the solver keyword. | psi = b / diag(A); returns solverPerformance with 0 iterations, converged = true. |

### linear solver (symmetric + asymmetric)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PBiCGStab` | `PBiCGStab` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/PBiCGStab/PBiCGStab.C` | Preconditioned bi-conjugate-gradient stabilised solver; registered in BOTH symMatrix and asymMatrix tables, so usable for pressure as well as momentum. References Van der Vorst (1992) and Barrett et al. (1994). | BiCGStab: rho = (rw0, r); beta = (rho/rho_old)(alpha/omega); p = r + beta(p - omega AyA); y = M^-1 p; AyA = A.y; alpha = rho/(rw0, AyA); s = r - alpha AyA; z = M^-1 s; t = A.z; omega = (t,s)/(t,t); psi += alpha y + omega z; r = s - omega t. |
| `smoothSolver` | `smoothSolver (requires sub-key smoother; optional nSweeps, default 1)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/SmoothSolver/smoothSolver.C` | Iterative solver that repeatedly applies a run-time selected smoother (GaussSeidel, symGaussSeidel, DIC, DILU, ...) until tolerance is met; residual only evaluated every nSweeps sweeps. Registered sym and asym. | Repeat: smoother.smooth(psi, b, cmpt, nSweeps); then residual res = b - A.psi, \|res\|/normFactor tested against tolerance/relTol. If nSweeps < 0 a fixed \|nSweeps\| sweeps are done with no residual check. |

### linear solver (symmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PCG` | `PCG (solver keyword in system/fvSolution solvers dict)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/PCG/PCG.C` | Preconditioned conjugate-gradient solver for symmetric lduMatrices with a run-time selectable preconditioner. Registered in symMatrix table only. | r0 = b - A.psi; loop: w = M^-1 r; wArA = (w,r); beta = wArA/wArAold; p = w + beta p; w = A.p; alpha = wArA/(w,p); psi += alpha p; r -= alpha w. Residual normalised by normFactor; singularity test on \|(w,p)\|/normFactor. |

### linear solver (templated/coupled, asymmetric)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PBiCCCG<Type, DType, LUType>` | `PBiCCCG` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/PBiCCCG/PBiCCCG.C` | Preconditioned bi-conjugate-gradient solver for asymmetric LduMatrix using a run-time selectable preconditioner, with cmpt-coupled (fully coupled component) inner products. | BiCG as PBiCICG but inner products are formed over all components of Type simultaneously (coupled), rather than per-component. |
| `PBiCICG<Type, DType, LUType>` | `PBiCICG` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/PBiCICG/PBiCICG.C` | Preconditioned bi-conjugate-gradient solver for asymmetric LduMatrix<Type,DType,LUType> using a run-time selectable preconditioner; component-coupled variant. | BiCG with shadow residual/direction using the transpose matrix-vector product (Tmul) and transpose preconditioning. |

### linear solver (templated/coupled, diagonal)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DiagonalSolver<Type, DType, LUType>` | `diagonal` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/DiagonalSolver/ (DiagonalSolver.H/.C — see notes on file-name collision)` | Templated diagonal direct solver for LduMatrix; registered by makeLduSolvers in the generic, sym and asym tables. | psi = inv(D) & b component-wise. |

### linear solver (templated/coupled, symmetric + asymmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SmoothSolver<Type, DType, LUType>` | `smoothSolver` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/SmoothSolver/ (SmoothSolver.H/.C — see notes on file-name collision)` | Templated smoothSolver for LduMatrix: applies a run-time selected LduMatrix smoother (e.g. TGaussSeidelSmoother) nSweeps at a time until convergence. Registered by makeLduSolvers in lduSolvers.C for all five field types. | Same fixed-point iteration as smoothSolver, on the block system. |

### linear solver (templated/coupled, symmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PCICG<Type, DType, LUType>` | `PCICG` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Solvers/PCICG/PCICG.C` | Preconditioned conjugate-gradient solver for symmetric LduMatrix<Type,DType,LUType>; the block/coupled analogue of PCG. Instantiated for scalar, vector, sphericalTensor, symmTensor, tensor. | Component-wise CG on the coupled system: wArA = sumProd(w, r); beta = wArA/wArAold; p = w + beta p; alpha = wArA/sumProd(w, p); psi += alpha p; r -= alpha w. |

### linear solver framework  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lduMatrix::preconditioner (base)` | `(dict key: preconditioner)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/lduMatrixPreconditioner.C` | Base class with symMatrix/asymMatrix RTS tables, getName(dict) and New(solver, dict) keyed on the 'preconditioner' entry (accepts either a primitive entry or a sub-dictionary). Declares precondition() and preconditionT(). | w = M^-1 r (and w = M^-T r for the transpose used by BiCG). |
| `lduMatrix::smoother (base)` | `(dict key: smoother)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/lduMatrixSmoother.C` | Base class with symMatrix/asymMatrix RTS tables, getName(dict) and New(...) keyed on the 'smoother' entry (primitive or sub-dictionary). Declares smooth(psi, source, cmpt, nSweeps) and scalarSmooth. |  |
| `lduMatrix::solver (base controls)` | `(dict keys: solver, tolerance (default 1e-6... read via readControls), relTol, maxIter (defaultMaxIter_ = 1000), minIter)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/lduMatrixSolver.C` | Base class with the two RTS tables (symMatrix, asymMatrix), the New() selector keyed on the 'solver' entry, readControls() and the residual normalisation factor. New() bypasses the tables and returns diagonalSolver when matrix.diagonal(). | normFactor = gSum(\|A.psi - A.psiRef\| + \|b - A.psiRef\|) + solverPerformance::small_, with psiRef = average(psi); residual = gSumMag(b - A.psi)/normFactor. |

### matrix container  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LduMatrix<Type, DType, LUType>` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/ (LduMatrix.H, LduMatrixI.H, lduMatrixTemplates.C, lduMatrices.C)` | Templated/coupled (block) counterpart of lduMatrix: field Type, diagonal type DType, off-diagonal type LUType. Instantiated for scalar, vector, sphericalTensor, symmTensor, tensor by lduMatrices.C. | Same LDU form but coefficients are DType on the diagonal and LUType off-diagonal, allowing coupled/segregated-block systems (e.g. vector psi with scalar D/LU). |
| `lduMatrix` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/lduMatrix.H` | Sparse face-addressed matrix storing three scalar arrays: diag(), upper(), lower(); hosts the solver/smoother/preconditioner abstract classes and their RTS tables. | A.psi = b with A stored as D (diag, nCells) + U (upper, nFaces) + L (lower, nFaces); Amul: y_i = D_i psi_i + sum_f U_f psi_{nbr(f)} + sum_f L_f psi_{own(f)} plus interface contributions. Tags itself diagonal()/symmetric()/asymmetric() from which of L/U are allocated. |

### parallel communications  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Pstream MPI implementation` | `(build-time: WM_MPLIB; runtime switches floatTransfer, nProcsSimpleSum (16), commsType, nPollProcInterfaces)` | `[Foundation-12] src/Pstream/mpi/ (UPstream.C, UIPread.C, UOPwrite.C, PstreamGlobals.C, allReduce.H)` | Real MPI back-end (libPstream in $FOAM_LIBBIN/$FOAM_MPI): MPI_Init_thread, communicator splitting into MPI_COMM_FOAM, allocate/free communicators for the GAMG processor-agglomerated levels, and the blocking/non-blocking send/receive used by processorLduInterface. | allReduce implements gSum/gMin/gMax as MPI_Allreduce, or as a linear gather/scatter tree when nProcs > nProcsSimpleSum. |
| `Pstream dummy implementation` |  | `[Foundation-12] src/Pstream/dummy/ (UPstream.C, UIPread.C, UOPwrite.C)` | Serial stub Pstream library (libPstream in $FOAM_LIBBIN/dummy) that satisfies the same interface with no MPI, so the LDU solvers link and run in serial. |  |
| `UPstream::commsTypes` | `blocking \| scheduled \| nonBlocking (etc/controlDict OptimisationSwitches/commsType)` | `[Foundation-12] src/OpenFOAM/db/IOstreams/Pstreams/UPstream.C` | NamedEnum of the three communication strategies used throughout the linear solvers' interface updates: blocking, scheduled, nonBlocking. Default read from the 'commsType' optimisation switch in etc/controlDict. |  |
| `allReduce` |  | `[Foundation-12] src/Pstream/mpi/allReduce.H` | Templated global reduction used for every inner product and residual norm in PCG/PBiCG/PBiCGStab (gSum, gSumProd, gSumMag, gMin, gMax). | For nProcs <= nProcsSimpleSum: MPI_Allreduce. Otherwise a hand-rolled linear gather to the master, apply the binary op, then a linear scatter (deterministic ordering). |

### parallel linear algebra  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lduMatrix::updateMatrixInterfaces / initMatrixInterfaces` | `(controlled by the 'commsType' optimisation switch: blocking \| scheduled \| nonBlocking)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/lduMatrixUpdateMatrixInterfaces.C` | Drives the coupled-interface contribution to the matrix-vector product for each UPstream::commsTypes: blocking, scheduled (uses the lduSchedule patchSchedule) and nonBlocking (init all sends/receives, do the interior, then waitRequests). Also honours UPstream::nPollProcInterfaces. | result[faceCells] -= interfaceBouCoeffs * psi_neighbour, summed over all interfaces. |

### preconditioner (asymmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DILUPreconditioner` | `DILU` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/DILUPreconditioner/DILUPreconditioner.C` | Simplified diagonal-based incomplete LU preconditioner for asymmetric matrices; stores reciprocal preconditioned diagonal rD_. Provides both precondition() and preconditionT() (transpose) for BiCG-type solvers. | calcReciprocalD: rD[u_f] -= U_f*L_f/rD[l_f]; rD = 1/rD. Apply: forward w[u] -= rD[u]*L_f*w[l]; w *= rD; backward w[l] -= rD[l]*U_f*w[u]. Transpose swaps the roles of upper and lower. |

### preconditioner (symmetric + asymmetric)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GAMGPreconditioner` | `GAMG (extra control: nVcycles, default 2)` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/GAMGPreconditioner/GAMGPreconditioner.C` | Geometric agglomerated algebraic multigrid used as a preconditioner (derives from both GAMGSolver and lduMatrix::preconditioner). Performs nVcycles V-cycles per preconditioning call. | w = 0; for cycle in 0..nVcycles-1: Vcycle(smoothers, w, r, ...); if not last cycle, update AwA = A.w and residual r - AwA for the next cycle. |
| `diagonalPreconditioner` | `diagonal` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/DiagonalPreconditioner/diagonalPreconditioner.C` | Jacobi (diagonal) preconditioner for both symmetric and asymmetric matrices. Stores 1/diag once because multiply is faster than divide. | w = rD * r with rD = 1/diag(A). |
| `noPreconditioner` | `none` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/NoPreconditioner/noPreconditioner.C` | Null preconditioner for both symmetric and asymmetric matrices — turns PCG/PBiCG/PBiCGStab into unpreconditioned CG/BiCG/BiCGStab. | w = r (identity). |

### preconditioner (symmetric)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DICPreconditioner` | `DIC` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/DICPreconditioner/DICPreconditioner.C` | Simplified diagonal-based incomplete Cholesky preconditioner for symmetric matrices (symmetric equivalent of DILU). Stores the reciprocal of the preconditioned diagonal rD_. | calcReciprocalD: rD[u_f] -= U_f*U_f/rD[l_f] over faces, then rD = 1/rD. Apply: forward sweep w[u] -= rD[u]*U_f*w[l]; scale w *= rD; backward sweep w[l] -= rD[l]*U_f*w[u]. |
| `FDICPreconditioner` | `FDIC` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/FDICPreconditioner/FDICPreconditioner.C` | Faster variant of DIC: additionally precomputes and stores rDuUpper_ = rD*upper and rDlUpper_ = rD*upper indexed by lower, trading memory for arithmetic in the sweeps. | Same DIC factorisation as DICPreconditioner; the sweeps use the pre-scaled coefficient arrays so no multiply by rD is needed inside the loops. |

### preconditioner (templated/coupled)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DiagonalPreconditioner<Type, DType, LUType>` | `diagonal` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/DiagonalPreconditioner/ (DiagonalPreconditioner.H/.C — see notes on file-name collision)` | Templated Jacobi preconditioner for LduMatrix, registered generic/sym/asym for all five field types. | w = inv(D) & r. |
| `NoPreconditioner<Type, DType, LUType>` | `none` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/NoPreconditioner/ (NoPreconditioner.H/.C — see notes on file-name collision)` | Templated null preconditioner registered in the generic, sym and asym LduMatrix preconditioner tables for all five field types. | w = r. |

### preconditioner (templated/coupled, asymmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `TDILUPreconditioner<Type, DType, LUType>` | `DILU` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Preconditioners/DILUPreconditioner/TDILUPreconditioner.C` | Templated DILU for LduMatrix<Type,DType,LUType>; stores the inverse (reciprocal for scalar) of the preconditioned diagonal. Registered for all five field types by lduPreconditioners.C. | rD[u] -= inv(rD[l]) applied to U_f*L_f products; then inverted. Forward/backward substitution as DILU with DType inverse. |

### smoother (asymmetric)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DILUGaussSeidelSmoother` | `DILUGaussSeidel` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/DILUGaussSeidel/DILUGaussSeidelSmoother.C` | Combined DILU then GaussSeidel smoother for asymmetric matrices — DILU sweeps followed by Gauss-Seidel to remove DILU-induced spikes. | smooth() = diluSmoother_.smooth(nSweeps) followed by gsSmoother_.smooth(nSweeps). |
| `DILUSmoother` | `DILU` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/DILU/DILUSmoother.C` | Simplified diagonal-based incomplete LU smoother for asymmetric matrices; stores the reciprocal preconditioned diagonal. | Per sweep: rA = b - A.psi; DILU forward/backward substitution on rA; psi += rA. |

### smoother (symmetric + asymmetric)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GaussSeidelSmoother` | `GaussSeidel` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/GaussSeidel/GaussSeidelSmoother.C` | Forward Gauss-Seidel sweep smoother, registered for both symmetric and asymmetric matrices. Used as the default GAMG smoother and by smoothSolver. | For each sweep: bPrime = b - interface contributions; then for cells in order psi_i = (bPrime_i - sum_{j<i} L psi_j - sum_{j>i} U psi_j)/D_i, implemented with the losort addressing for the lower triangle. |
| `nonBlockingGaussSeidelSmoother` | `nonBlockingGaussSeidel` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/nonBlockingGaussSeidel/nonBlockingGaussSeidelSmoother.C` | Variant of GaussSeidel that expects processor-boundary cells to be sorted last, so the interface halo exchange can be initiated first and only waited on when those cells are actually visited — overlaps MPI with interior smoothing. | Identical Gauss-Seidel update; only the ordering/communication schedule differs (blockStart_ marks the first coupled cell; interfaces are updated non-blocking and completed before the coupled block is swept). |
| `symGaussSeidelSmoother` | `symGaussSeidel` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/symGaussSeidel/symGaussSeidelSmoother.C` | Symmetric Gauss-Seidel: a forward sweep immediately followed by a reverse sweep, making the smoother symmetric (needed when used inside a symmetric preconditioner/AMG cycle). | Forward sweep over cells 0..n-1 then reverse sweep n-1..0 of the Gauss-Seidel update above. |

### smoother (symmetric)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DICGaussSeidelSmoother` | `DICGaussSeidel` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/DICGaussSeidel/DICGaussSeidelSmoother.C` | Combined DIC then GaussSeidel smoother for symmetric matrices: DIC smoothing followed by Gauss-Seidel to smooth out the 'spikes' the DIC sweeps create. Holds a DICSmoother and a GaussSeidelSmoother member. | smooth() = dicSmoother_.smooth(nSweeps) followed by gsSmoother_.smooth(nSweeps). |
| `DICSmoother` | `DIC` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/DIC/DICSmoother.C` | Simplified diagonal-based incomplete Cholesky smoother for symmetric matrices; residual evaluated after every nSweeps sweeps. | Per sweep: rA = b - A.psi; apply DIC forward/backward substitution to rA; psi += rA. |
| `FDICSmoother` | `FDIC` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/FDIC/FDICSmoother.C` | Faster DIC smoother for symmetric matrices using precomputed rD*upper arrays (rDuUpper_, rDlUpper_) as in FDICPreconditioner. | Same as DICSmoother but with pre-scaled coefficients in the substitution sweeps. |

### smoother (templated/coupled, symmetric + asymmetric)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `TGaussSeidelSmoother<Type, DType, LUType>` | `GaussSeidel` | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/Smoothers/GaussSeidel/TGaussSeidelSmoother.C` | The only templated LduMatrix smoother; registered generic/sym/asym for scalar, vector, sphericalTensor, symmTensor and tensor by lduSmoothers.C. Provides both smooth() and a static smooth() helper. | psi_i = inv(D_i) & (bPrime_i - sum L psi - sum U psi), in forward cell order with losort addressing. |

### solution control dictionary  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `solution` | `(dictionary keys: solvers, relaxationFactors/fields, relaxationFactors/equations, cache, select)` | `[Foundation-12] src/OpenFOAM/matrices/solution/solution.C` | IOdictionary wrapper for system/fvSolution: holds the solvers sub-dictionary (per-field solver controls, with optional 'select' to pick a named sub-dictionary), relaxationFactors (fields{} and equations{}, with an optional 'default'), and the cache{} switch controlling temporary-field caching. Also implements upgradeSolverDict for the older solver-controls syntax. | Field relaxation: phi = phi_old + alpha (phi_new - phi_old). Equation relaxation applied to the matrix diagonal: D /= alpha, source += (1/alpha - 1) D phi_prev. |
| `tolerances` |  | `[Foundation-12] src/OpenFOAM/matrices/tolerances/tolerances.C` | IOdictionary holding relaxationFactors_ and solverTolerances_/solverRelativeTolerances_ read from a 'tolerances' file — the legacy per-field tolerance selector. |  |

### solver diagnostics  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SolverPerformance<Type> / solverPerformance` |  | `[Foundation-12] src/OpenFOAM/matrices/LduMatrix/LduMatrix/solverPerformance.C` | Records solverName, fieldName, initialResidual, finalResidual, nIterations, converged and singular flags; provides checkConvergence(tolerance, relTol), checkSingularity(residual) and the printing/IO of the 'Solving for p, Initial residual = ...' lines. typedef solverPerformance = SolverPerformance<scalar>. | Converged if finalResidual < tolerance OR (relTol > 0 && finalResidual <= relTol*initialResidual). Singular if residual < vsmall. |

### surface/patch agglomeration  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pairPatchAgglomeration` | `(dict controls: mergeLevels, maxLevels, nFacesInCoarsestLevel, featureAngle)` | `[Foundation-12] src/fvAgglomerationMethods/pairPatchAgglomeration/pairPatchAgglomeration.C` | Pair agglomeration applied to a primitive (surface) patch rather than a volume mesh — used by view-factor radiation and surface coarsening. Built as libpairPatchAgglomeration. | Edge weight between neighbouring faces = shared edge length; faces are paired greedily by largest weight subject to a featureAngle constraint; repeated until nFacesInCoarsestLevel or maxLevels is reached, mergeLevels at a time. |

---

## Momentum transport / turbulence

> **Subsystem notes**
>
> STRUCTURE / SELECTION MECHANICS 1. Two-level runtime selection.
> The whole subsystem is driven by `constant/momentumTransport` (name comes from `momentumTransportModel::typeName`; `momentumTransportModel::readModelDict` falls back to the legacy `constant/turbulenceProperties`, and supports `<phaseName>` group suffixes).
> The top level key is `simulationType` -> one of `laminar`, `RAS`, `LES` (these are the TypeNames of laminarModel/RASModel/LESModel, registered by the `makeBaseMomentumTransportModel` macro).
> The second level is `laminar { model X; }` / `RAS { model X; }` / `LES { model X; }` — each uses `lookupBackwardsCompatible<word>({"model", "<laminar|RAS|LES>Model"})`, so the old `RASModel`, `LESModel`, `laminarModel` keys still work. Coefficients go in `<modelName>Coeffs` via `optionalSubDict`. 2.
> Template instantiation, not inheritance, gives the four flavours.
> Each model is a class template over the momentum-transport base; the same source is compiled four times through the macros in `momentumTransportModels/makeMomentumTransportModel.H` (`makeMomentumTransportModelTypes`, `makeBaseMomentumTransportModel`, `makeMomentumTransportModel`, `makeTemplatedMomentumTransportModel`, `makeTemplatedLaminarModel`, and the per-library `makeRASModel`/`makeLESModel`/`makeLaminarModel` wrappers).
> The four instantiations, with their (alpha, rho) template arguments, are: - incompressible: (geometricOneField, geometricOneField) -> libincompressibleMomentumTransportModels - compressible: (geometricOneField, volScalarField) -> libcompressibleMomentumTransportModels - phaseIncompressible: (volScalarField, geometricOneField)-> libphaseIncompressibleMomentumTransportModels - phaseCompressible: (volScalarField, volScalarField) -> libphaseCompressibleMomentumTransportModels Availability differs per library.
> Full RAS/LES sets are registered only for incompressible and compressible. phaseCompressible registers only kEpsilon, RNGkEpsilon, kOmegaSST, Smagorinsky, kEqn (+ all six laminar models); phaseIncompressible only kEpsilon, kOmegaSST, Smagorinsky, kEqn (+ laminar).
> `buoyantKEpsilon` exists only in the compressible library. 3. Models physically located under src/MomentumTransportModels but compiled ELSEWHERE.
> The four phaseCompressible RAS models (LaheyKEpsilon, continuousGasKEpsilon, kOmegaSSTSato, mixtureKEpsilon) and the three phaseCompressible LES models (SmagorinskyZhang, NicenoKEqn, continuousGasKEqn) are NOT in `phaseCompressible/Make/files`.
> They are registered in `applications/modules/multiphaseEuler/momentumTransportModels/momentumTransportModels.C`, i.e. they are only available to the multiphaseEuler solver module. 4.
> The six extra incompressible-only RAS models (qZeta, kkLOmega, LamBremhorstKE, LienLeschziner, ShihQuadraticKE, LienCubicKE) are in the `Foam::incompressible::RASModels` namespace and use `makeMomentumTransportModelTypes(geometricOneField, geometricOneField, incompressibleMomentumTransportModel)` + explicit `defineTypeNameAndDebug` / `addToRunTimeSelectionTable(RASincompressibleMomentumTransportModel, ...)` rather than the `makeRASModel` macro.
> They are listed individually in `incompressible/Make/files`. 5.
> Class hierarchy of stress closures (all in momentumTransportModels/): momentumTransportModel -> {incompressible,compressible,phase*}MomentumTransportModel -> laminarModel|RASModel|LESModel -> linearViscousStress -> eddyViscosity (holds nut_) -> concrete eddy-viscosity models.
> Two side branches: nonlinearEddyViscosity (adds an explicit `nonlinearStress_`, used by ShihQuadraticKE and LienCubicKE) and ReynoldsStress (transports the full `R_` tensor, used by LRR, SSG and the LES DeardorffDiffStress). LESeddyViscosity sits between LESModel and the SGS models.
> kOmegaSSTBase (`Foam::kOmegaSST`, under Base/kOmegaSST/) is shared by RASModels::kOmegaSST, kOmegaSSTSAS, kOmegaSSTLM, LESModels::kOmegaSSTDES and RASModels::kOmegaSSTSato. 6. Generalised-Newtonian viscosity is orthogonal to turbulence.
> RASModel and LESModel both construct a `generalisedNewtonianViscosityModel` if the entry `viscosityModel` is found in the model's coeffDict (otherwise a plain Newtonian one), so any RAS/LES model can be combined with any of the 7 viscosity models.
> IMPORTANT: in this checkout the two base-class directories `.../generalisedNewtonianViscosityModels/generalisedNewtonianViscosityModel/` and `.../strainRateViscosityModels/strainRateViscosityModel/` exist but are EMPTY — the .C/.H files listed in `momentumTransportModels/Make/files` (generalisedNewtonianViscosityModel.C, generalisedNewtonianViscosityModelNew.C, strainRateViscosityModel.C) are missing.
> These are the only two empty directories in either subsystem tree, so the checkout is incomplete there; the library as listed would not build. 7. Wall functions.
> `nutWallFunctionFvPatchScalarField` and `wallCellWallFunctionFvPatchScalarField` are base classes only — they have TypeName/defineTypeNameAndDebug but no `makePatchTypeField`, so they cannot be used as a patch `type`.
> The six concrete nut variants, epsilonWallFunction, omegaWallFunction, kLowReWallFunction, v2WallFunction, fWallFunction and (in phaseCompressible) epsilonmWallFunction are all registered with makePatchTypeField. kqRWallFunction is registered via `makePatchFields(kqRWallFunction)`, i.e.
> for scalar, vector, sphericalTensor, symmTensor AND tensor fields.
> All nut-derived wall functions share the laminar/turbulent switch-over `yPlusLam` derived from kappa and E in the nut wall function, and epsilon/omega/k/v2/f wall functions read those coefficients from the corresponding nut BC on the same patch — so nut and the scalar wall functions must be chosen consistently.
> The v2f model documentation explicitly warns that kLowReWallFunction must be paired with a velocity-based nut wall function (nutUWallFunction), not a k-based one (nutk*). 8. Thermophysical transport mirrors the momentum structure exactly.
> Dictionary `constant/thermophysicalTransport`, sub-dicts `laminar`/`RAS`/`LES` with a `model` entry (no backwards-compatible alias here — plain `lookup("model")`).
> Six libraries: libthermophysicalTransportModel (root abstract class), libfluidThermophysicalTransportModel (base + BCs), and four instantiation libraries formed from the cross product {ThermophysicalTransportModel, PhaseThermophysicalTransportModel} x {fluidThermo, fluidMulticomponentThermo}: - fluidThermo: ThermophysicalTransportModel<compressibleMomentumTransportModel, fluidThermo> - fluidMulticomponentThermo: ThermophysicalTransportModel<compressibleMomentumTransportModel, fluidMulticomponentThermo> - phaseFluidThermo: PhaseThermophysicalTransportModel<phaseCompressibleMomentumTransportModel, fluidThermo> - phaseFluidMulticomponentThermo: PhaseThermophysicalTransportModel<phaseCompressibleMomentumTransportModel, fluidMulticomponentThermo> plus libsolidThermophysicalTransportModels / libphaseSolidThermophysicalTransportModels (isotropic, anisotropic) and libcoupledThermophysicalTransportModels (coupledTemperature, externalTemperature, lumpedMassTemperature).
> This is the "thermo package combination pattern" for this subsystem: the model set available depends on the thermo template argument.
> Fourier and unityLewisFourier (laminar) and eddyDiffusivity and unityLewisEddyDiffusivity (RAS and LES) are registered in all four; FickianFourier, MaxwellStefanFourier (laminar) and nonUnityLewisEddyDiffusivity (RAS and LES) only in the two multicomponent libraries; FickianEddyDiffusivity only in the two multicomponent libraries and ONLY for RAS (deliberately not registered for LES).
> 9.
> Common implementation idiom for heat flux: every fluid thermophysical transport model implements `divq(he)` as an implicit energy correction — an explicit temperature-gradient laplacian minus an implicit `fvm::laplacian(alphahe, he)` plus its explicit counterpart — so the correction is exactly zero at convergence while keeping the energy equation diagonally dominant.
> The Fickian/MaxwellStefan models add sum_i h_i*j_i to the heat flux and support Soret thermal diffusion through the optional `DT` dictionary. 10.
> Wall distance is a hard dependency for SpalartAllmaras (and DES/DDES/IDDES), v2f, all epsilon/omega/k/v2/f wall functions, vanDriestDelta, PrandtlDelta and IDDESDelta — they all use `wallDist::New(mesh).y()` or `nearWallDist`. 11.
> Scope note: the task brief also mentioned "div schemes / limiters, linear solvers, preconditioners, smoothers, thermo package combinations, mesh generators/movers/topo-changers".
> None of those live under src/MomentumTransportModels or src/ThermophysicalTransportModels — limitedSurfaceInterpolationScheme limiters are in src/finiteVolume/interpolation/surfaceInterpolation, linear solvers/preconditioners/smoothers are in src/OpenFOAM/matrices/lduMatrix/solvers|preconditioners|smoothers, thermo packages are in src/thermophysicalModels, and mesh generation/motion/topology change are in applications/utilities/mesh, src/fvMeshMovers, src/fvMeshTopoChangers and src/motionSolvers.
> Nothing of that kind was found in the paths assigned to this subsystem, so it is not included above. The only scheme-like selectable items in this subsystem are the LESdelta (7 entries) and LESfilter (4 entries) tables.
> TOTALS in this catalogue: 128 components — 15 framework/base classes, 6 laminar stress models, 7 generalised-Newtonian viscosity models, 14 shared RAS models + 6 incompressible-only RAS models + 4 multiphase RAS models, 10 shared LES/DES models + 3 multiphase LES models, 7 LES deltas (incl.
> base + helper), 4 LES filters (incl. base), 15 wall functions (incl. 2 base classes), 6 other momentum BCs/field sources, 9 thermophysical base classes, 6 laminar thermophysical models (incl.
> 2 mixin bases), 4 turbulent thermophysical models, 4 solid thermophysical entries, 10 thermophysical/CHT boundary conditions.

### LES SGS Reynolds-stress model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DeardorffDiffStress` | `LES { model DeardorffDiffStress; delta <deltaName>; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/DeardorffDiffStress/DeardorffDiffStress.H` | Deardorff differential SGS stress-equation model: transports the full SGS stress tensor B rather than assuming an eddy viscosity; Donaldson triple-correlation replaced by Daly-Harlow generalised gradient diffusion. | ddt(a,rho,R)+div(aRhoPhi,R)-laplacian(I*nu + Cs*(k/epsilon)*R, R) + Sp(Cm*a*rho*sqrt(k)/delta, R) = a*rho*P + (4/5)... pressure-strain and (2/3)*(1 + Cm/Ce)*I*a*rho*epsilon terms; k = 0.5*tr(R), epsilon = Ce*k^(3/2)/delta. Coeffs Ck, Ce, Cm, Cs (default Cs 0.25). |

### LES SGS base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LESeddyViscosity` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESeddyViscosity/LESeddyViscosity.H` | Eddy-viscosity LES SGS model base class; supplies Ce_ coefficient and the equilibrium epsilon estimate. | epsilon = Ce*k^(3/2)/delta; omega = epsilon/(Cmu*k). |

### LES SGS model  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Smagorinsky` | `LES { model Smagorinsky; } (all four variants)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/Smagorinsky/Smagorinsky.H` | Classical Smagorinsky SGS model, formulated from the k-equation model under local equilibrium so that both k and epsilon are available. | B = (2/3)k I - 2*nuSgs*dev(D), D = symm(grad U); k obtained from D:B + Ce*k^(3/2)/delta = 0 giving k = (2*Ck/Ce)*delta^2*magSqr(dev(D))... ; nuSgs = Ck*sqrt(k)*delta. Coeffs Ck 0.094, Ce 1.048. |
| `WALE` | `LES { model WALE; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/WALE/WALE.H` | Wall-Adapting Local Eddy-viscosity SGS model (Nicoud & Ducros 1999) giving the correct near-wall nuSgs ~ y^3 scaling without damping functions. | Sd = dev(symm(sqr(grad U))); k = sqr(sqr(Cw)*delta/Ck)*(pow3(magSqr(Sd))/sqr(pow(magSqr(symm(grad U)),5/2) + pow(magSqr(Sd),5/4))); nuSgs = Ck*sqrt(k)*delta. Coeffs Ck 0.094, Ce 1.048, Cw 0.325. |
| `dynamicKEqn` | `LES { model dynamicKEqn; dynamicKEqnCoeffs { filter simple; } } — requires an LESfilter` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/dynamicKEqn/dynamicKEqn.H` | Dynamic one-equation eddy-viscosity SGS model (Kim & Menon 1995): the Ck and Ce coefficients are computed dynamically using a test filter. | Same k transport as kEqn but with Ck(D,KK) and Ce(D,KK) evaluated from the Germano-type identity with KK = 0.5*(filter(magSqr(U)) - magSqr(filter(U))); nut = Ck(D,KK)*sqrt(k)*delta. |
| `dynamicLagrangian` | `LES { model dynamicLagrangian; } — requires an LESfilter; coefficients theta, flm0, fmm0` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/dynamicLagrangian/dynamicLagrangian.H` | Dynamic Smagorinsky SGS model with Lagrangian averaging of the model coefficient along fluid pathlines (Meneveau, Lund & Cabot 1996). | Transports flm_ and fmm_: ddt(a,rho,flm)+div(aRhoPhi,flm) = a*rho*invT*LM - Sp(a*rho*invT, flm) (and analogously fmm with MM); relaxation time invT = 1/(theta*delta*(flm*fmm)^(1/8)); nut = flm/fmm * delta^2 * \|D\|. |
| `kEqn` | `LES { model kEqn; } (all four variants)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/kEqn/kEqn.H` | One-equation eddy-viscosity SGS model (Yoshizawa 1986) solving a modelled balance equation for the SGS kinetic energy. | ddt(a,rho,k)+div(aRhoPhi,k)-laplacian(a*rho*DkEff,k) = a*rho*G - SuSp((2/3)a*rho*divU, k) - Sp(Ce*a*rho*sqrt(k)/delta, k); nut = Ck*sqrt(k)*delta. Coeffs Ck 0.094, Ce 1.048. |

### LES delta  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `IDDESDelta` | `delta IDDESDelta; with IDDESDeltaCoeffs { deltaCoeff 1; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/IDDESDelta/IDDESDelta.H` | Composite delta required by the IDDES model; min/max deltas are formed from twice the min/max face-centre-to-cell-centre distances. | delta = deltaCoeff*min(max(Cw*y, Cw*hmax, hwn), hmax) with hmax/hmin from the doubled face-to-cell distances. |
| `PrandtlDelta` | `delta Prandtl; with PrandtlCoeffs { delta cubeRootVol; Cdelta 0.158; kappa 0.41; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/PrandtlDelta/PrandtlDelta.H` | Applies Prandtl mixing-length near-wall damping to an underlying geometric delta. | delta = min(geometricDelta, (kappa/Cdelta)*y). |
| `cubeRootVolDelta` | `delta cubeRootVol; with cubeRootVolCoeffs { deltaCoeff 1; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/cubeRootVolDelta/cubeRootVolDelta.H` | Simple cube-root-of-cell-volume filter width. | delta = deltaCoeff * V^(1/3) (2D meshes use the appropriate reduced power). |
| `maxDeltaxyz` | `delta maxDeltaxyz; with maxDeltaxyzCoeffs { deltaCoeff 2; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/maxDeltaxyz/maxDeltaxyz.H` | Filter width from the maximum cell-centre-to-face-centre distance (half the cell width for a regular hex, so deltaCoeff should be 2). | delta = deltaCoeff * max_f \|Cf - C\|. |
| `smoothDelta` | `delta smooth; with smoothCoeffs { delta cubeRootVol; maxDeltaRatio 1.1; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/smoothDelta/smoothDelta.H` | Smooths a given geometric delta with an FvFaceCellWave so the ratio of deltas between neighbouring cells never exceeds a specified factor (typically 1.15). | delta_neighbour <= maxDeltaRatio * delta_cell, enforced by wave propagation (deltaData transport type). |
| `vanDriestDelta` | `delta vanDriest; with vanDriestCoeffs { delta cubeRootVol; Aplus 26; Cdelta 0.158; kappa 0.41; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/vanDriestDelta/vanDriestDelta.H` | Applies the van Driest damping function to an underlying geometric delta to improve near-wall behaviour; uses a FvFaceCellWave with WallLocationYPlus to propagate y+ up to a cut-off. | delta = min(geometricDelta, (kappa/Cdelta)*y*(1 - exp(-yPlus/Aplus))). |

### LES delta base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LESdelta` | `TypeName "LESdelta"; selected by the 'delta' entry inside the LES sub-dictionary` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/LESdelta/LESdelta.H` | Abstract base class and runtime-selection table for LES filter-width (delta) models. | delta() returns a volScalarField used as the SGS length scale. |

### LES delta helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `WallLocationYPlus` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESdeltas/vanDriestDelta/WallLocationYPlus.H` | FvFaceCellWave transport type holding the nearest-wall point and yStar, so van Driest damping only needs y+ computed up to a cut-off (e.g. y+ < 200). |  |

### LES filter  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `anisotropicFilter` | `filter anisotropic; with widthCoeff` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESfilters/anisotropicFilter/anisotropicFilter.H` | Direction-dependent (anisotropic) LES test filter using a per-direction coefficient field derived from the cell geometry. | filter(phi) = phi + sum over directions of coeff_i * d2/dx_i2(phi) with coeff derived from the cell's directional widths. |
| `laplaceFilter` | `filter laplace; with widthCoeff (default 2)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESfilters/laplaceFilter/laplaceFilter.H` | Laplace (diffusion-based) LES test filter with an isotropic filter coefficient. | filter(phi) = phi + coeff*laplacian(phi), coeff = delta^2/widthCoeff (box: delta2/24, spherical box: delta2/64, Gaussian: delta2/24; as test filter with ratio 2 the widths become delta2/6, delta2/16, delta2/6). |
| `simpleFilter` | `filter simple;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESfilters/simpleFilter/simpleFilter.H` | Simple top-hat filter implemented as the surface integral of the face interpolate of the field. | filter(phi) = fvc::surfaceSum(mesh.magSf()*fvc::interpolate(phi))/fvc::surfaceSum(mesh.magSf()). |

### LES filter base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LESfilter` | `TypeName "LESfilter"; selected via the 'filter' entry in the dynamic model's coefficient dictionary` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESfilters/LESfilter/LESfilter.H` | Abstract base class and runtime-selection table for LES test filters used by the dynamic SGS models. | operator() applied to vol scalar/vector/tensor fields returns the filtered field. |

### LES multiphase SGS model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NicenoKEqn` | `LES { model NicenoKEqn; } — registered in the multiphaseEuler module` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/LES/Niceno/NicenoKEqn.H` | One-equation SGS model for the continuous phase of a two-phase system including bubble-generated turbulence (Niceno, Dhotre & Deen 2008). | kEqn k transport plus the bubble-generated production term and the Cmub bubble-induced viscosity; blended out above alphaInversion. Coeffs Ck 0.094, Ce 1.048, alphaInversion 0.3, Cp = Ck, Cmub 0.6. |
| `SmagorinskyZhang` | `LES { model SmagorinskyZhang; } — registered in the multiphaseEuler module` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/LES/SmagorinskyZhang/SmagorinskyZhang.H` | Smagorinsky SGS model with the Zhang et al. (2006) bubble-induced turbulence contribution for Euler-Euler bubble columns. | Smagorinsky nut plus nutb = Cmub*d_bubble*alpha_gas*mag(Ur) added to the effective viscosity. Coeffs Ck 0.094, Ce 1.048, Cmub 0.6. |
| `continuousGasKEqn` | `LES { model continuousGasKEqn; } — registered in the multiphaseEuler module` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/LES/continuousGasKEqn/continuousGasKEqn.H` | One-equation SGS model for the gas phase of a two-phase system supporting phase inversion; blends in the liquid-phase k as the gas fraction approaches zero. | kEqn k transport with a blended source from the other phase up to alphaInversion. Coeffs Ck 0.094, Ce 1.048, alphaInversion 0.7. |

### LES/DES hybrid model  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SpalartAllmarasDDES` | `LES { model SpalartAllmarasDDES; delta <deltaName>; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/SpalartAllmarasDDES/SpalartAllmarasDDES.H` | Delayed DES variant of the Spalart-Allmaras model (Spalart et al. 2006), resistant to ambiguous grid densities / modelled-stress depletion. | Adds the shielding function fd = 1 - tanh(pow3(8*rd)) with rd = (nut+nu)/(sqrt(magSqr(grad U))*sqr(kappa*y)); dTilda = y - fd*max(0, y - CDES*delta). |
| `SpalartAllmarasDES` | `LES { model SpalartAllmarasDES; delta <deltaName>; } — coefficient CDES (default 0.65), ck` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/SpalartAllmarasDES/SpalartAllmarasDES.H` | Spalart-Allmaras DES (Spalart et al. 1997): the SA destruction length scale y is replaced by dTilda = min(CDES*delta, y). | ddt(a,rho,nuTilda)+div(aRhoPhi,nuTilda)-laplacian(a*rho*DnuTildaEff,nuTilda) - Cb2/sigmaNut*magSqr(grad nuTilda) = Cb1*a*rho*Stilda(chi,fv1,Omega,dTilda)*nuTilda - Sp(Cw1*a*rho*fw*nuTilda/sqr(dTilda), nuTilda); LESRegion() diagnostic field; k estimated as sqr(nut/ck/dTilda). |
| `SpalartAllmarasIDDES` | `LES { model SpalartAllmarasIDDES; delta IDDESDelta; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/SpalartAllmarasIDDES/SpalartAllmarasIDDES.H` | Improved DDES (Shur et al. 2008): hybrid RANS-LES with delayed-DES plus wall-modelled LES capability. | dTilda = fdTilda*(1+fe)*lRAS + (1-fdTilda)*lLES using the alpha/fB/fe1/fe2/fdt elevating and blending functions; normally used with the IDDESDelta filter width. Coeffs Cdt1, Cdt2, Cl, Ct. |
| `kOmegaSSTDES` | `LES { model kOmegaSSTDES; delta <deltaName>; } — coefficient CDES` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/kOmegaSSTDES/kOmegaSSTDES.H` | k-omega-SST DES (Menter, Kuntz & Langtry 2003) with optional F1/F2-based zonal shielding. | Uses the kOmegaSSTBase equations but replaces the k-destruction length scale: epsilonByk = betaStar*omega*max(1, CDES*Lt/delta) with Lt = sqrt(k)/(betaStar*omega); optional multiplication by (1-F1) or (1-F2) for zonal filtering. |

### RAS Reynolds-stress model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LRR` | `RAS { model LRR; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/LRR/LRR.H` | Launder-Reece-Rodi Reynolds-stress transport model with the Daly-Harlow generalised gradient diffusion and optional Gibson-Launder wall reflection. | ddt(a,rho,R)+div(aRhoPhi,R)-laplacian(a*rho*DREff,R) + Sp(C1*a*rho*eps/k, R) = a*rho*P - (2/3)(1-C1)*I*a*rho*eps - C2*a*rho*dev(P) + wall-reflection terms; epsilon eq. with Ceps1*G*eps/k production and Sp(Ceps2*a*rho*eps/k, eps). DREff = Cs*(k/eps)*R + nu*I (Daly-Harlow). Coeffs Cmu 0.09, C1 1.8, C2 0.6, Ceps1 1.44, Ceps2 1.92, Cs 0.25, Ceps 0.15, wallReflection yes, kappa 0.41, Cref1 0.5, Cref2 0.3, couplingFactor 0. |
| `SSG` | `RAS { model SSG; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/SSG/SSG.H` | Speziale-Sarkar-Gatski quadratic pressure-strain Reynolds-stress model with Daly-Harlow generalised gradient diffusion. | Same R and epsilon transport skeleton as LRR but with the SSG quadratic pressure-strain correlation in terms of the anisotropy tensor b and its invariants. Coeffs Cmu 0.09, C1 3.4, C1s 1.8, C2 4.2, C3 0.8, C3s 1.3, C4 1.25, C5 0.4, Ceps1 1.44, Ceps2 1.92, Cs 0.25, Ceps 0.15, couplingFactor 0. |

### RAS base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `v2fBase` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/v2f/v2fBase.H` | Abstract base giving the v2WallFunction and fWallFunction boundary conditions access to the v2 and f fields of any v2-f model. |  |

### RAS multiphase turbulence model  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LaheyKEpsilon` | `RAS { model LaheyKEpsilon; } — registered in applications/modules/multiphaseEuler/momentumTransportModels/momentumTransportModels.C` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/RAS/LaheyKEpsilon/LaheyKEpsilon.H` | Continuous-phase k-epsilon model including bubble-generated (pseudo) turbulence for Euler-Euler dispersed flows. | Standard k-epsilon plus bubble-induced source terms and the Sato-like bubble eddy viscosity nutAlphaEff; phase-inversion blending at alphaInversion. Coeffs Cmu 0.09, C1 1.44, C2 1.92, C3 0, C4 1.92, sigmak 1.0, sigmaEps 1.3, Cp 0.25, Cmub 0.6, alphaInversion 0.3. |
| `continuousGasKEpsilon` | `RAS { model continuousGasKEpsilon; } — registered in the multiphaseEuler module` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/RAS/continuousGasKEpsilon/continuousGasKEpsilon.H` | k-epsilon for the gas phase of a two-phase system supporting phase inversion; blends in the liquid-phase contribution as the gas fraction approaches zero. | Standard k-epsilon with a blended nutEff_ and blended k/epsilon sources up to the alphaInversion phase fraction. Coeffs Cmu 0.09, C1 1.44, C2 1.92, C3 0, sigmak 1.0, sigmaEps 1.3, alphaInversion 0.7. |
| `kOmegaSSTSato` | `RAS { model kOmegaSSTSato; } — registered in the multiphaseEuler module` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/RAS/kOmegaSSTSato/kOmegaSSTSato.H` | k-omega-SST for dispersed bubbly flow with the Sato (1981) bubble-induced turbulent viscosity contribution. | Standard SST k and omega equations; nutEff = nut + Cmub*d_bubble*alpha_dispersed*mag(Ur) (Sato bubble-induced viscosity) used in the momentum stress. Coeffs alphaK1 0.85034, alphaK2 1.0, alphaOmega1 0.5, alphaOmega2 0.85616, Prt 1.0, beta1 0.075, beta2 0.0828, betaStar 0.09, gamma1 0.5532, ..., Cmub 0.6. |
| `mixtureKEpsilon` | `RAS { model mixtureKEpsilon; } — registered in the multiphaseEuler module; uses epsilonmWallFunction` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/RAS/mixtureKEpsilon/mixtureKEpsilon.H` | Mixture k-epsilon model for two-phase gas-liquid systems (Behzadi/Issa/Rusche) using an effective gas density in the averaging, with Lahey bubble-generated turbulence. | Solves km/epsilonm on the mixture with rhom, Ur and the response-coefficient Ct; k and epsilon of each phase reconstructed from the mixture values. Coeffs Cmu 0.09, C1 1.44, C2 1.92, C3 = C2, Cp 0.25, alphap 0.3, sigmak 1.0, sigmaEps 1.3. |

### RAS non-linear turbulence model (incompressible only)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LienCubicKE` | `RAS { model LienCubicKE; } — incompressible library only` | `[Foundation-12] src/MomentumTransportModels/incompressible/RAS/LienCubicKE/LienCubicKE.H` | Lien-Chen-Leschziner cubic non-linear low-Reynolds k-epsilon model for incompressible flow with both low-Re damping and wall-function support; derives from nonlinearEddyViscosity. | k-epsilon transport plus quadratic and cubic explicit non-linear stress contributions in S and W with strain/vorticity-dependent Cmu and Cbeta coefficients. |
| `ShihQuadraticKE` | `RAS { model ShihQuadraticKE; } — incompressible library only` | `[Foundation-12] src/MomentumTransportModels/incompressible/RAS/ShihQuadraticKE/ShihQuadraticKE.H` | Shih's quadratic algebraic Reynolds-stress k-epsilon model for incompressible flow; derives from nonlinearEddyViscosity. | k-epsilon transport plus an explicit quadratic non-linear stress: nonlinearStress = Cbeta1*(S&S - (1/3)I tr(S&S)) + Cbeta2*(W&S + S&W) + Cbeta3*(W&W - (1/3)I tr(W&W)) scaled by k^3/eps^2. |

### RAS thermophysical transport model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FickianEddyDiffusivity` | `RAS { model FickianEddyDiffusivity; } — multicomponent libraries only (registered for RAS only, not LES)` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/turbulence/FickianEddyDiffusivity/FickianEddyDiffusivity.H` | Multi-component Fickian laminar diffusion combined with turbulent eddy diffusivity for RAS, with optional Soret thermal diffusion. | j(Yi) = -(rho*Dm_i + rho*nut/Sct)*grad(Yi) - DT_i*grad(T)/T; q = -(kappa + Cp*rho*nut/Prt)*grad(T) + sum_i hi*j_i. Entries mixtureDiffusionCoefficients, Prt, Sct, Dm{...} or D{...}, optional DT{...}. |

### RAS transition model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kOmegaSSTLM` | `RAS { model kOmegaSSTLM; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/kOmegaSSTLM/kOmegaSSTLM.H` | Langtry-Menter 4-equation correlation-based transitional SST model (k, omega, gammaInt, ReThetat). | ddt(a,rho,ReThetat)+div(aRhoPhi,ReThetat)-laplacian(a*rho*DReThetatEff,ReThetat) = Pthetat*ReThetat0(Us,dUsds,nu) - Sp(Pthetat, ReThetat); ddt(a,rho,gammaInt)+div(aRhoPhi,gammaInt)-laplacian(a*rho*DgammaIntEff,gammaInt) = Pgamma - Sp(ce1*Pgamma, gammaInt) + Egamma - Sp(ce2*Egamma, gammaInt); k production/destruction multiplied by gammaIntEff. Coeffs ca1 2, ca2 0.06, ce1 1, ce2 50, cThetat 0.03, sigmaThetat 2, lambdaErr 1e-6, maxLambdaIter 10. |

### RAS transition model (incompressible only)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kkLOmega` | `RAS { model kkLOmega; } — incompressible library only` | `[Foundation-12] src/MomentumTransportModels/incompressible/RAS/kkLOmega/kkLOmega.H` | Walters & Cokljat three-equation k-kL-omega transition-sensitive eddy-viscosity model, with the Furst (2013) correction and the Lopez & Walters (2016) improvement. | Transports kt_ (turbulent KE), kl_ (laminar kinetic energy) and omega_ (epsilon_ derived); includes bypass/natural transition production terms, shear-sheltering damping, small/large-scale energy splitting. Coeffs A0 4.04, As 2.12, Av 6.75, Abp 0.6, Anat 200, Ats 200, CbpCrit 1.2, Cnc 0.1, CnatCrit 1250, Cint 0.75, CtsCrit 1000, CrNat 0.02, C11 3.4e-6, C12 1e-10, CR 0.12, CalphaTheta 0.035, Css 1.5, CtauL 4360, Cw1 0.44, Cw2 0.92, Cw3 0.3, CwR 1.5, Clambda 2.495, CmuStd 0.09. |

### RAS turbulence model  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LaunderSharmaKE` | `RAS { model LaunderSharmaKE; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/LaunderSharmaKE/LaunderSharmaKE.H` | Launder-Sharma low-Reynolds-number k-epsilon with damping functions, for near-wall-resolved flows, incl. RDT compression term. | Solves for the reduced dissipation epsilonTilda; fMu = exp(-3.4/sqr(1 + sqr(k)/(nu*eps)/50)); f2 = 1 - 0.3exp(-min(sqr(sqr(k)/(nu*eps)),50)); extra terms E = 2*nu*nut*magSqr(grad grad U) and D = 2*nu*magSqr(grad sqrt(k)). Coeffs Cmu 0.09, C1 1.44, C2 1.92, C3 0, alphah 1.0, alphahk 1.0, alphaEps 0.76923. |
| `RNGkEpsilon` | `RAS { model RNGkEpsilon; } (incompressible, compressible, phaseCompressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/RNGkEpsilon/RNGkEpsilon.H` | Renormalisation-group k-epsilon (Yakhot et al. 1992) for incompressible/compressible flow. | As kEpsilon but with the strain-dependent C1 correction: C1RNG = C1 - eta*(1 - eta/eta0)/(1 + beta*eta^3), eta = S*k/eps. Coeffs Cmu 0.0845, C1 1.42, C2 1.68, C3 0, sigmak 0.71942, sigmaEps 0.71942, eta0 4.38, beta 0.012. |
| `SpalartAllmaras` | `RAS { model SpalartAllmaras; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/SpalartAllmaras/SpalartAllmaras.H` | Spalart-Allmaras one-equation mixing-length model for external aerodynamic flows; implemented without the trip term (ft2 omitted), with Spalart's Stilda clipping at Cs*Omega. | ddt(a,rho,nuTilda)+div(aRhoPhi,nuTilda)-laplacian(a*rho*DnuTildaEff,nuTilda) - Cb2/sigmaNut*a*rho*magSqr(grad nuTilda) = Cb1*a*rho*Stilda*nuTilda - fvm::Sp(Cw1*a*rho*fw*nuTilda/sqr(y), nuTilda); nut = nuTilda*fv1, fv1 = chi^3/(chi^3+Cv1^3). Coeffs Cb1 0.1355, Cb2 0.622, Cw2 0.3, Cw3 2.0, Cv1 7.1, Cs 0.3, sigmaNut 0.66666, kappa 0.41. |
| `kEpsilon` | `RAS { model kEpsilon; } (all four variants incl. phaseIncompressible, phaseCompressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/kEpsilon/kEpsilon.H` | Standard Launder-Spalding k-epsilon model for incompressible and compressible flow, including an RDT-based compression term (El Tahry). | ddt(a,rho,eps)+div(aRhoPhi,eps)-laplacian(a*rho*DepsilonEff,eps) = C1*a*rho*G*eps/k - SuSp(((2/3)C1-C3)*a*rho*divU, eps) - Sp(C2*a*rho*eps/k, eps); ddt(a,rho,k)+div(aRhoPhi,k)-laplacian(a*rho*DkEff,k) = a*rho*G - SuSp((2/3)a*rho*divU,k) - Sp(a*rho*eps/k, k); nut = Cmu*k^2/eps. Coeffs Cmu 0.09, C1 1.44, C2 1.92, C3 0, sigmak 1.0, sigmaEps 1.3. |
| `kOmega` | `RAS { model kOmega; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/kOmega/kOmega.H` | Standard high-Reynolds-number Wilcox (1998) k-omega model. | ddt(a,rho,omega)+div(aRhoPhi,omega)-laplacian(a*rho*DomegaEff,omega) = gamma*a*rho*G*omega/k - SuSp((2/3)gamma*a*rho*divU, omega) - Sp(beta*a*rho*omega, omega); k equation with betaStar*k*omega sink; nut = k/omega. Coeffs betaStar 0.09, gamma 0.52, beta 0.072, alphak 0.5, alphaOmega 0.5. |
| `kOmega2006` | `RAS { model kOmega2006; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/kOmega2006/kOmega2006.H` | Wilcox (2006/2008) revised high-Reynolds k-omega model with cross-diffusion, vortex-stretching-dependent beta and a stress limiter. | Adds sigmaDo cross-diffusion term max(grad k & grad omega, 0)/omega, beta = beta0*fbeta(chiOmega) vortex-stretching function, and the stress limiter nut = k/max(omega, Clim*sqrt(2*S:S/betaStar)). Coeffs Cmu 0.09, beta0 0.0708, gamma 0.52, Clim 0.875, alphak 0.6, alphaOmega 0.5. |
| `kOmegaSST` | `RAS { model kOmegaSST; } (incompressible, compressible, phaseIncompressible, phaseCompressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/kOmegaSST/kOmegaSST.H` | RAS specialisation of the generic Menter k-omega-SST base class (kOmegaSSTBase). | See kOmegaSSTBase: blended k-omega/k-epsilon with F1, the F2/F23 shear-stress limiter nut = a1*k/max(a1*omega, b1*F23*sqrt(2)\|S\|), optional F3 rough-wall term. |
| `realizableKE` | `RAS { model realizableKE; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/realizableKE/realizableKE.H` | Shih realizable k-epsilon; enforces realizability of the normal stresses via a variable Cmu. | Cmu = 1/(A0 + As*U*k/eps) with As = sqrt(6)cos(phi), phi from the strain-rate invariants; epsilon equation uses the C1 = max(eta/(eta+5), 0.43) production form and the eps^2/(k+sqrt(nu*eps)) sink. Coeffs A0 4.0, C2 1.9, sigmak 1.0, sigmaEps 1.2. |
| `v2f` | `RAS { model v2f; } — requires k kLowReWallFunction, epsilon epsilonWallFunction, v2 v2WallFunction, f fWallFunction` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/v2f/v2f.H` | Lien-Kalitzin v2-f four-equation model (k, epsilon, v2, f) with the Davidson et al. limit on nut; N=6 variant so f=0 at walls. | epsilon eq. with time scale Ts and Ceps1 = 1.4*(1 + 0.05*sqrt(k/v2)); k eq. standard; elliptic relaxation: -laplacian(f) - Sp(1/L^2, f) = (C1-N)*v2/(k*Ts) - (C1-1)/Ts ... ; v2 eq: ddt+div-laplacian(a*rho*DkEff, v2) = min(k*f, -(C1-N)*v2/Ts + ...) - Sp(N*a*rho*eps/k, v2); nut = min(CmuKEps*k^2/eps, Cmu*v2*Ts). Coeffs Cmu 0.22, CmuKEps 0.09, C1 1.4, C2 0.3, CL 0.23, Ceta 70, Ceps2 1.9, Ceps3 -0.33, sigmaEps 1.3, sigmaK 1. |

### RAS turbulence model (compressible only)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `buoyantKEpsilon` | `RAS { model buoyantKEpsilon; } (compressible library only)` | `[Foundation-12] src/MomentumTransportModels/compressible/RAS/buoyantKEpsilon/buoyantKEpsilon.H` | Adds a density-gradient-based buoyancy generation/dissipation term to the standard k-epsilon k and epsilon equations (Henkes et al. 1991), applicable to compositional as well as thermal stratification. | Gcoef = Cg*(g & grad rho)/rho*nut/... added as a source to the k equation and, via the C3 blending, to the epsilon equation. Coeff Cg 1.0. |

### RAS turbulence model (incompressible only)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LamBremhorstKE` | `RAS { model LamBremhorstKE; } — incompressible library only` | `[Foundation-12] src/MomentumTransportModels/incompressible/RAS/LamBremhorstKE/LamBremhorstKE.H` | Lam & Bremhorst (1981) low-Reynolds-number k-epsilon model for incompressible flow. | Standard k-epsilon transport with wall-distance-based damping functions fMu = sqr(1 - exp(-0.0165*Ry))*(1 + 20.5/Rt), f1 = 1 + (0.05/fMu)^3, f2 = 1 - exp(-sqr(Rt)); nut = Cmu*fMu*k^2/eps. |
| `LienLeschziner` | `RAS { model LienLeschziner; } — incompressible library only` | `[Foundation-12] src/MomentumTransportModels/incompressible/RAS/LienLeschziner/LienLeschziner.H` | Lien & Leschziner (1993) low-Reynolds-number k-epsilon model for incompressible flow, with wall-function support for high-Re operation. | k-epsilon transport with yStar-based damping functions fMu, f2 and additional near-wall E/D source terms; nut = Cmu*fMu*k^2/eps. |
| `qZeta` | `RAS { model qZeta; } — Foam::incompressible::RASModels::qZeta, incompressible library only` | `[Foundation-12] src/MomentumTransportModels/incompressible/RAS/qZeta/qZeta.H` | Gibson & Dafa'Alla q-zeta two-equation low-Re model for incompressible flow (a q = sqrt(k), zeta = epsilon/(2 sqrt(k)) reformulation of k-epsilon). | Transports q_ and zeta_ (with k = q^2, epsilon = 2*q*zeta); boundZeta() enforces positivity; low-Re damping functions fMu, f2 and the extra E term using fvc::magSqrGradGrad. |

### RAS/LES shared base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kOmegaSST (kOmegaSSTBase)` | `used via the derived models; coefficients in kOmegaSSTCoeffs (alphaK1, alphaK2, alphaOmega1, alphaOmega2, beta1, beta2, betaStar, gamma1, gamma2, a1, b1, c1, F3)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/Base/kOmegaSST/kOmegaSSTBase.H` | Generic Menter k-omega-SST implementation shared by the RAS kOmegaSST, kOmegaSSTSAS, kOmegaSSTLM, the LES kOmegaSSTDES and the phase kOmegaSSTSato models; written in alpha (=1/sigma) diffusion coefficients so blending is consistent. | F1/F2/F3 blending functions; ddt(a,rho,k)+div(aRhoPhi,k)-laplacian(a*rho*DkEff(F1),k) = min(G, c1*betaStar*k*omega) - (2/3)a rho divU k - a rho betaStar k omega; ddt(a,rho,omega)+div(aRhoPhi,omega)-laplacian(a*rho*DomegaEff(F1),omega) = gamma*rho*GbyNu - (2/3)gamma a rho divU omega - a rho beta omega^2 + (1-F1)*CDkOmega cross-diffusion. nut = a1*k/max(a1*omega, b1*F23*sqrt(2)*\|symm(grad U)\|). |

### RAS/LES thermophysical transport model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `eddyDiffusivity` | `RAS { model eddyDiffusivity; } or LES { model eddyDiffusivity; } — all four thermo library variants` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/turbulence/eddyDiffusivity/eddyDiffusivity.H` | Eddy-diffusivity temperature-gradient heat-flux model for single-specie RAS or LES turbulent flow. | alphat = rho*nut/Prt; kappaEff = kappa + Cp*alphat; q = -kappaEff*grad(T); divq is the implicit energy-corrected laplacian. Entry Prt (default 0.85). |
| `nonUnityLewisEddyDiffusivity` | `RAS { model nonUnityLewisEddyDiffusivity; } / LES { ... } — multicomponent libraries only` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/turbulence/nonUnityLewisEddyDiffusivity/nonUnityLewisEddyDiffusivity.H` | Eddy-diffusivity model allowing independent turbulent Prandtl and Schmidt numbers (unity laminar Lewis number assumed). | Thermal: alphaEff from Prt; specie: DEff(Yi) = rho*(alphahe + nut/Sct). Entries Prt (0.85), Sct (0.7). |
| `unityLewisEddyDiffusivity` | `RAS { model unityLewisEddyDiffusivity; } / LES { ... } — all four thermo library variants` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/turbulence/unityLewisEddyDiffusivity/unityLewisEddyDiffusivity.H` | Eddy-diffusivity energy-gradient heat-flux model for RAS/LES assuming unity turbulent Lewis number for the specie fluxes. | alphaEff = alphahe + alphat with alphat = rho*nut/Prt; q = -alphaEff*grad(he); j(Yi) = -rho*alphaEff*grad(Yi). Entry Prt. |

### RAS/URAS turbulence model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kOmegaSSTSAS` | `RAS { model kOmegaSSTSAS; delta cubeRootVol; } (incompressible, compressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/kOmegaSSTSAS/kOmegaSSTSAS.H` | Scale-Adaptive Simulation (Egorov & Menter 2008) variant of k-omega-SST; resolves unsteady structures by adding the von Karman length-scale source Q_SAS to the omega equation. | Adds Qsas = max(zeta2*kappa*S^2*(L/Lvk)^2 - 2*C*k/sigmaPhi*max(magSqr(grad omega)/omega^2, magSqr(grad k)/k^2), 0) with L = sqrt(k)/(Cmu^0.25*omega), Lvk from \|grad U\|/\|lapl U\| bounded by the LES delta. Coeffs Cs 0.11, kappa 0.41, zeta2 3.51, sigmaPhi 2/3, C 2; requires a 'delta' specification. |

### conjugate heat transfer boundary condition  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `coupledTemperature` | `type coupledTemperature;` | `[Foundation-12] src/ThermophysicalTransportModels/coupledThermophysicalTransportModels/coupledTemperature/coupledTemperatureFvPatchScalarField.H` | Mixed temperature BC for region-to-region conjugate heat transfer (CHT), with optional thin thermal-layer resistances and radiative/source heat fluxes; works on either fluid or solid regions because kappa comes from the region's thermophysicalTransportModel. | refGradient = qs/kappa; refValue = neighbour T; valueFraction = kappaByDeltaNbr/(kappaByDeltaNbr + kappaByDelta) with kappaByDelta = kappa*deltaCoeffs. Entries Tnbr, thicknessLayers, kappaLayers, qs or Qs, qrNbr, qr. Requires a mapped(Wall) patch. |

### external heat transfer boundary condition  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `externalTemperature` | `type externalTemperature; (Q \| q \| h+Ta, thicknessLayers, kappaLayers, relaxation, emissivity, qr, qrRelaxation)` | `[Foundation-12] src/ThermophysicalTransportModels/coupledThermophysicalTransportModels/externalTemperature/externalTemperatureFvPatchScalarField.H` | Applies a heat-flux condition on an external wall specified as fixed power Q, fixed flux q, or a heat-transfer coefficient h with ambient temperature Ta (all Function1 of time), plus optional thin-layer resistances, radiative flux and surface emissivity. | Mixed condition combining q/(A) + h*(Ta - Tw) (+ emissivity*sigma*(Ta^4 - Tw^4) + qr); layer resistance folded into an effective h via thicknessLayers/kappaLayers; wall temperature relaxable. kappa taken from the region thermophysicalTransportModel. |

### framework base class  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `compressibleMomentumTransportModel` | `not directly selectable; library libcompressibleMomentumTransportModels` | `[Foundation-12] src/MomentumTransportModels/compressible/compressibleMomentumTransportModel.H` | Base for single-phase compressible turbulence models; alpha=geometricOneField, rho=volScalarField. | Density-weighted (Favre) form: alphaRhoPhi-based convection, divDevTau in kg/m^2/s^2. |
| `incompressibleMomentumTransportModel` | `not directly selectable (template instantiation base); library libincompressibleMomentumTransportModels` | `[Foundation-12] src/MomentumTransportModels/incompressible/incompressibleMomentumTransportModel.H` | Base for single-phase incompressible turbulence models; instantiated with alpha=geometricOneField, rho=geometricOneField. | Kinematic form: stresses in m^2/s^2; divDevTau = -div(nuEff*dev2(T(grad U))) - laplacian(nuEff, U). |
| `momentumTransportModel` | `TypeName "momentumTransport"; top-level selection is by the 'simulationType' entry in constant/momentumTransport (backwards compatible with constant/turbulenceProperties)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/momentumTransportModel.H` | Abstract base for all RAS, LES and laminar momentum transport (turbulence) models; owns U, alphaRhoPhi, phi and the viscosity object, and is itself the IOdictionary constant/momentumTransport. | Defines the interface: nut(), nuEff(), sigma() (Reynolds/SGS stress), devTau(), divDevTau(U) which supplies the momentum-equation stress term. |
| `phaseCompressibleMomentumTransportModel` | `not directly selectable; library libphaseCompressibleMomentumTransportModels` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/phaseCompressibleMomentumTransportModel.H` | Base for per-phase compressible turbulence models used by multiphaseEuler; alpha=volScalarField, rho=volScalarField. | alpha*rho weighted transport equations; adds pPrime()/pPrimef() particle-pressure interface for dispersed phases. |
| `phaseIncompressibleMomentumTransportModel` | `not directly selectable; library libphaseIncompressibleMomentumTransportModels` | `[Foundation-12] src/MomentumTransportModels/phaseIncompressible/phaseIncompressibleMomentumTransportModel.H` | Base for per-phase incompressible turbulence models; alpha=volScalarField, rho=geometricOneField. | Phase-fraction weighted transport: ddt(alpha,k) + div(alphaPhi,k) - laplacian(alpha*DkEff, k) = ... |

### fvModel field source  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `turbulentMixingLengthDissipationRate (fvScalarFieldSource)` | `type turbulentMixingLengthDissipationRate; (registered via makeTypeFieldSource)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/derivedFvFieldSources/turbulentMixingLengthDissipationRate/turbulentMixingLengthDissipationRateFvScalarFieldSource.H` | Source-condition (fvModel inflow) value for epsilon based on a specified mixing length; for a sink the local cell values are used instead. | epsilon = Cmu^0.75 * k^1.5 / L. Entries mixingLength, k. |
| `turbulentMixingLengthFrequency (fvScalarFieldSource)` | `type turbulentMixingLengthFrequency; (registered via makeTypeFieldSource)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/derivedFvFieldSources/turbulentMixingLengthFrequency/turbulentMixingLengthFrequencyFvScalarFieldSource.H` | Source-condition (fvModel inflow) value for omega based on a specified mixing length; for a sink the local cell values are used instead. | omega = k^0.5 / (Cmu^0.25 * L). Entries mixingLength, k. |

### generalised-Newtonian viscosity model  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BirdCarreau` | `viscosityModel BirdCarreau; (nuInf, k or tauStar, n, optional a)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/strainRateViscosityModels/BirdCarreau/BirdCarreau.H` | Bird-Carreau (and Bird-Carreau-Yasuda when 'a' is given) shear-thinning viscosity. | nu = nuInf + (nu0 - nuInf)*(1 + (k*gammaDot)^a)^((n-1)/a), or with tauStar: (1 + (nu0*gammaDot/tauStar)^a)^((n-1)/a); a defaults to 2. |
| `Casson` | `viscosityModel Casson; (m, tau0, nuMin, nuMax)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/strainRateViscosityModels/Casson/Casson.H` | Casson viscosity model for yield-stress fluids (typical use: blood rheology). | nu = min(nuMax, max(nuMin, (sqrt(tau0/gammaDot) + sqrt(m))^2)). |
| `CrossPowerLaw` | `viscosityModel CrossPowerLaw; (nuInf, m or tauStar, n)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/strainRateViscosityModels/CrossPowerLaw/CrossPowerLaw.H` | Cross power-law shear-thinning viscosity. | nu = nuInf + (nu0 - nuInf)/(1 + (m*gammaDot)^n), or with tauStar: nu = nuInf + (nu0-nuInf)/(1 + (nu0*gammaDot/tauStar)^n). |
| `HerschelBulkley` | `viscosityModel HerschelBulkley; (tau0, k, n)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/strainRateViscosityModels/HerschelBulkley/HerschelBulkley.H` | Herschel-Bulkley model combining Bingham-plastic yield stress with power-law behaviour. | nu = min(nu0, tau0/gammaDot + k*gammaDot^(n-1)). |
| `Newtonian (generalisedNewtonianViscosityModel)` | `viscosityModel Newtonian;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/Newtonian/NewtonianViscosityModel.H` | Returns the constant fluid Newtonian viscosity; the default when no viscosityModel is specified. | nu = nu (from physicalProperties). |
| `powerLaw` | `viscosityModel powerLaw; (k, n, nuMin, nuMax)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/strainRateViscosityModels/powerLaw/powerLaw.H` | Standard power-law viscosity with min/max clipping. | nu = max(nuMin, min(nuMax, k*gammaDot^(n-1))). |
| `strainRateFunction` | `viscosityModel strainRateFunction; (function <Function1>)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/strainRateViscosityModels/strainRateFunction/strainRateFunction.H` | Viscosity given by an arbitrary runtime-selected Function1 of the strain rate. | nu = Function1(gammaDot), e.g. 'function polynomial ((0 0.1) (1 1.3));'. |

### inlet turbulence BC  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `turbulentMixingLengthDissipationRateInlet` | `type turbulentMixingLengthDissipationRateInlet; mixingLength <L>;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/derivedFvPatchFields/turbulentMixingLengthDissipationRateInlet/turbulentMixingLengthDissipationRateInletFvPatchScalarField.H` | Inlet condition for epsilon derived from a specified turbulent mixing length; inletOutlet-based so reverse flow reverts to zeroGradient. | epsilon_p = Cmu^0.75 * k^1.5 / L, with Cmu = 0.09. Entries mixingLength, phi, k. |
| `turbulentMixingLengthFrequencyInlet` | `type turbulentMixingLengthFrequencyInlet; mixingLength <L>;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/derivedFvPatchFields/turbulentMixingLengthFrequencyInlet/turbulentMixingLengthFrequencyInletFvPatchScalarField.H` | Inlet condition for omega derived from a specified turbulent mixing length; inletOutlet-based. | omega_p = k^0.5 / (Cmu^0.25 * L), with Cmu = 0.09. Entries mixingLength, phi, k. |

### laminar stress model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Stokes` | `laminar { model Stokes; } (registered for incompressible, compressible, phaseIncompressible, phaseCompressible)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/Stokes/Stokes.H` | Plain Newtonian/Stokes laminar flow: no turbulence, constant molecular viscosity from the physicalProperties viscosity model. | nut = 0; devTau = -rho*nu*dev2(T(grad U)); divDevTau = -div(a*rho*nu*dev2(T(grad U))) - laplacian(a*rho*nu, U). |
| `generalisedNewtonian` | `laminar { model generalisedNewtonian; viscosityModel <name>; ... }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/generalisedNewtonian/generalisedNewtonian.H` | Shear-dependent non-Newtonian laminar momentum transport; wraps a runtime-selected generalisedNewtonianViscosityModel that returns nu(strainRate). | nu = f(gammaDot) where gammaDot = sqrt(2)*mag(symm(grad U)); stress as for Stokes but with the variable nu. |
| `lambdaThixotropic` | `laminar { model lambdaThixotropic; }` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/lambdaThixotropic/lambdaThixotropic.H` | Thixotropic viscosity model transporting a structural parameter lambda (Barnes 1997) from which the viscosity is derived. | D(lambda)/Dt = a(1-lambda)^b - c*lambda*gammaDot^d, solved as ddt(alpha,rho,lambda)+div(alphaRhoPhi,lambda) = source - fvm::Sp(...); nu = nuInf/(1 - K*lambda)^2 with K = 1 - sqrt(nuInf/nu0). |

### laminar thermophysical base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MaxwellStefan` | `not selectable directly — mixed into MaxwellStefanFourier` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/MaxwellStefan/MaxwellStefan.H` | Base class implementing Maxwell-Stefan generalised Fick's-law diffusion coefficients (Taylor & Krishna 1993; Merk 1959) with optional Soret thermal diffusion; binary coefficients are Function2<scalar> of (p, T). | Inverts the Maxwell-Stefan matrix to obtain a generalised Fickian diffusion matrix Dij; j(Yi) = -rho*sum_j Dij*grad(Yj) - DT_i*grad(T)/T. |

### laminar thermophysical transport model  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FickianFourier` | `laminar { model FickianFourier; } — multicomponent libraries only (fluidMulticomponentThermo, phaseFluidMulticomponentThermo)` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/FickianFourier/FickianFourier.H` | Multi-component Fickian mass diffusion combined with Fourier heat conduction for laminar flow, with optional Soret thermal diffusion. | Fickian j(Yi) (mixture coefficients Dm, or binary D mixed to mixture coefficients) plus Fourier q = -kappa*grad(T) and the specie-enthalpy flux. Entries mixtureDiffusionCoefficients (yes/no), Dm{...} or D{...}, optional DT{...}. |
| `Fourier` | `laminar { model Fourier; } — fluidThermo, fluidMulticomponentThermo, phaseFluidThermo, phaseFluidMulticomponentThermo` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/Fourier/Fourier.H` | Fourier temperature-gradient heat-flux model for single-specie laminar flow; the heat flux source is an implicit energy correction to the temperature-gradient flux so the correction vanishes at convergence. | q = -kappa*grad(T); divq(he) = -fvc::laplacian(kappa, T) - fvm::laplacian(alphahe, he) + fvc::laplacian(alphahe, he) (implicit energy correction form). |
| `MaxwellStefanFourier` | `laminar { model MaxwellStefanFourier; } — multicomponent libraries only` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/MaxwellStefanFourier/MaxwellStefanFourier.H` | Maxwell-Stefan generalised Fick's-law multi-component diffusion combined with Fourier conduction for laminar flow, with optional Soret thermal diffusion. | As MaxwellStefan for the specie fluxes plus q = -kappa*grad(T) and the implicit energy correction. Entries D{ A-B ... } binary coefficients, optional DT{...}. |
| `unityLewisFourier` | `laminar { model unityLewisFourier; } — all four thermo library variants` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/unityLewisFourier/unityLewisFourier.H` | Energy-gradient heat-flux model for laminar flow assuming unity Lewis number, so specie fluxes follow the thermal diffusivity directly. | q = -alphahe*grad(he); j(Yi) = -rho*alphahe*grad(Yi); divq(he) = -fvm::laplacian(alphahe, he). |

### laminar viscoelastic model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Giesekus` | `laminar { model Giesekus; } — extra coefficient alphaG` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/Giesekus/Giesekus.H` | Giesekus viscoelastic model (deformation-dependent tensional mobility) built on the Maxwell multi-mode framework. | Adds the quadratic mobility term to the Maxwell relaxation: fvm source = -(alphaG/nuM_i)*(sigma_i & sigma_i) - fvm::Sp(alpha*rho/lambda_i, sigma_i). |
| `Maxwell` | `laminar { model Maxwell; } — coefficients nuM/lambda, optional 'modes' list for multi-mode` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/Maxwell/Maxwell.H` | Generalised (multi-mode) Maxwell viscoelastic model using the upper-convected derivative of the stress tensor; equivalent to Oldroyd-B when a non-zero solvent viscosity nu is present. | Per mode i: ddt(alpha,rho,sigma_i) + div(alphaRhoPhi,sigma_i) - twoSymm(sigma_i & grad U) = alpha*rho*nuM_i/lambda_i * twoSymm(grad U) - fvm::Sp(alpha*rho/lambda_i, sigma_i); sigma = sum_i sigma_i; momentum contribution -div(alpha*rho*sigma) with implicit laplacian(alpha*rho*nu0, U) stabilisation. |
| `PTT` | `laminar { model PTT; } — extra coefficient epsilonPTT` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/PTT/PTT.H` | Phan-Thien/Tanner viscoelastic model built on the Maxwell multi-mode framework. | Relaxation source multiplied by the PTT exponential/linear function of the stress trace: -fvm::Sp(alpha*rho/lambda_i * f(epsilonPTT*lambda_i/nuM_i * tr(sigma_i)), sigma_i). |

### laminar/turbulent thermophysical base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Fickian` | `not selectable directly — mixed into FickianFourier and FickianEddyDiffusivity` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/Fickian/Fickian.H` | Base class implementing multi-component Fickian mass diffusion with optional Soret thermal diffusion; diffusion coefficients are Function2<scalar> of (p, T) but independent of composition. | j(Yi) = -rho*Dm_i*grad(Yi) - DT_i*grad(T)/T (Soret); heat flux gains sum_i hi*j_i; energy correction implemented implicitly. |

### lumped-mass heat transfer boundary condition  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lumpedMassTemperature` | `type lumpedMassTemperature;` | `[Foundation-12] src/ThermophysicalTransportModels/coupledThermophysicalTransportModels/lumpedMassTemperature/lumpedMassTemperatureFvPatchScalarField.H` | Represents a bounded solid body as a single lumped mass at a uniform temperature fixed across the patch; the temperature evolves from the applied power and the integrated boundary heat transfer. | dT/dt = (Q + Q_b)/(rho*Cv*V), where Q_b is the total heat transferred across the boundary (positive into the mass) and V is user-specified or computed for a closed patch. Entries rho, Cv, T, Q, volume. |

### momentum boundary condition  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fixedShearStress` | `type fixedShearStress; tau (vector)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/fixedShearStress/fixedShearStressFvPatchVectorField.H` | Imposes a constant wall shear stress on a velocity patch. | tau0 = -nuEff*dU/dn; the patch value/gradient are set so that this stress is realised. |
| `porousBafflePressure` | `type porousBafflePressure; patchType cyclic; (instantiated for scalar patch fields)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/porousBafflePressure/porousBafflePressureFvPatchField.H` | Pressure-jump condition representing a thin porous baffle, built on the cyclic (jump) patch. | deltaP = -(D*mu*U + 0.5*I*rho*\|U\|^2)*L, applied as a relaxable jump. Entries: D (Darcy), I (inertial), length, relaxation, phi, rho, patchType cyclic. |

### simulation-type base class  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LESModel` | `TypeName "LES"; simulationType LES; then LES { model <name>; delta <deltaName>; } (backwards compatible key 'LESModel')` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/LES/LESModel/LESModel.H` | Templated abstract base for LES SGS models; owns the runtime-selected LESdelta and optional generalised-Newtonian viscosity model. | Provides delta() (filter width), turbulence()/printCoeffs, kMin_ bounding. |
| `RASModel` | `TypeName "RAS"; simulationType RAS; then RAS { model <name>; } (backwards compatible key 'RASModel'); optional 'viscosityModel' entry inside <model>Coeffs selects a generalisedNewtonianViscosityModel` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/RAS/RASModel/RASModel.H` | Templated abstract base for RAS turbulence models, with optional generalised-Newtonian (strain-rate dependent) molecular viscosity support. | Provides turbulence()/printCoeffs switches, kMin_/epsilonMin_/omegaMin_ bounding, nu() overridden by the selected viscosityModel. |
| `laminarModel` | `simulationType laminar; then laminar { model <name>; } (backwards compatible key 'laminarModel')` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/laminar/laminarModel/laminarModel.H` | Templated abstract base for laminar (non-turbulent) stress models; reads the 'laminar' sub-dictionary and the <type>Coeffs sub-dictionary. | nut() == 0; k(), epsilon(), omega() return zero fields; sigma() supplied by the derived stress model. |

### solid thermophysical transport base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `phaseSolidThermophysicalTransportModel` | `'model' entry; same isotropic/anisotropic selection as solid` | `[Foundation-12] src/ThermophysicalTransportModels/phaseSolid/phaseSolidThermophysicalTransportModel/phaseSolidThermophysicalTransportModel.H` | Abstract base for per-phase solid thermophysical transport models (solid phases in multiphaseEuler). |  |
| `solidThermophysicalTransportModel` | `selected by the top-level 'model' entry in the region's constant/thermophysicalTransport` | `[Foundation-12] src/ThermophysicalTransportModels/solid/solidThermophysicalTransportModel/solidThermophysicalTransportModel.H` | Abstract base and runtime-selection table for solid-region thermophysical transport models; defaults to isotropic if constant/thermophysicalTransport is absent. | q = -kappa*grad(T); divq(e) as the implicit energy-corrected laplacian. |

### solid thermophysical transport model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `anisotropic (solid)` | `model anisotropic; with coordinateSystem{...}, zones{...}, boundaryAligned (registered for both solid and phaseSolid)` | `[Foundation-12] src/ThermophysicalTransportModels/solid/anisotropic/anisotropic.H` | Solid transport model for anisotropic thermal conductivity; the material Kappa tensor is transformed into the global frame by a default coordinateSystem and optional per-zone coordinate systems, with optional boundary alignment enforcement. | q = -(Kappa & grad(T)) with Kappa = R & diag(kappa) & R^T; boundaryAligned switch forces aligned kappa handling at patches. |
| `isotropic (solid)` | `model isotropic; (registered for both solidThermophysicalTransportModel and phaseSolidThermophysicalTransportModel)` | `[Foundation-12] src/ThermophysicalTransportModels/solid/isotropic/isotropic.H` | Solid transport model for isotropic thermal conductivity; the default when no thermophysicalTransport dictionary is present. | q = -kappa*grad(T) with scalar kappa from the solid thermo. |

### stress-model base class  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ReynoldsStress` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/ReynoldsStress/ReynoldsStress.H` | Reynolds-stress (second-moment closure) base class; transports the full symmetric tensor R_ and provides realizability bounding of normal stresses and wall-shear-stress correction. | Transports R (6 components); divDevTau uses div(alpha*rho*R) with an optional couplingFactor blending an implicit laplacian(nuEff,U) contribution. |
| `eddyViscosity` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/eddyViscosity/eddyViscosity.H` | Eddy-viscosity turbulence model base class (linear Boussinesq closure on top of linearViscousStress); holds nut_. | sigma = (2/3)k I - 2 nut dev(symm(grad U)); R = (2/3)k I - nut*twoSymm(grad U). |
| `linearViscousStress` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/linearViscousStress/linearViscousStress.H` | Base class implementing a linear (Boussinesq) viscous stress closure and the corresponding divDevTau momentum source. | devTau = -rho*nuEff*dev2(T(grad U)); divDevTau(U) = -div(alpha*rho*nuEff*dev2(T(grad U))) - laplacian(alpha*rho*nuEff, U). |
| `nonlinearEddyViscosity` |  | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/nonlinearEddyViscosity/nonlinearEddyViscosity.H` | Eddy-viscosity base class with an additional explicit non-linear stress correction term nonlinearStress_ (used by the quadratic/cubic k-epsilon models). | sigma = eddyViscosity part + nonlinearStress; divDevTau adds div(alpha*rho*nonlinearStress). |

### thermophysical boundary condition  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `compressible::thermalBaffle1D<eConstSolidThermoPhysics>` | `type compressible::thermalBaffle1D<eConstSolidThermoPhysics>; neighbourPatch, thickness, qs, qr, qrRelaxation, specie/transport/thermodynamics/equationOfState sub-dicts` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/thermalBaffle1D/thermalBaffle1DFvPatchScalarField.H` | Solves a steady 1-D thermal baffle across a pair of coupled patches using constIsoSolidTransport + eConstThermo + rhoConst solid physics; optional radiative flux qr and source flux qs. | Steady 1-D conduction through a wall of specified thickness: q = kappa_s*(T_master - T_slave)/thickness, coupled to both fluid sides plus qs and (optionally, relaxed) qr. |
| `compressible::thermalBaffle1D<ePowerSolidThermoPhysics>` | `type compressible::thermalBaffle1D<ePowerSolidThermoPhysics>;` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/thermalBaffle1D/thermalBaffle1DFvPatchScalarFields.C` | Same 1-D thermal baffle but instantiated with exponentialSolidTransport + ePowerThermo + rhoConst solid physics (temperature-dependent conductivity and heat capacity). | As above with kappa(T) exponential and Cv(T) power-law. |
| `convectiveHeatTransfer` | `type convectiveHeatTransfer; L <length>;` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/convectiveHeatTransfer/convectiveHeatTransferFvPatchScalarField.H` | Computes a convective heat-transfer coefficient field on a patch from flat-plate correlations. | Re > 5e5: htc = 0.037*Re^0.8*Pr^0.333*kappa/L; else htc = 0.664*Re^0.5*Pr^0.333*kappa/L. Entry L. |
| `externalCoupledTemperature` | `type externalCoupledTemperature;` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/externalCoupledTemperatureMixed/externalCoupledTemperatureMixedFvPatchScalarField.H` | Mixed temperature BC that exchanges magSf, value, qDot and htc with an external application via plain-text files and a lock file in $FOAM_CASE/<commsDir>. | Writes (magSf, T, qDot, htc) and reads back (refValue, refGrad, valueFraction) for the mixed condition. Entries commsDir, file, waitInterval, timeOut, calcFrequency, log. |

### thermophysical simulation-type base  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LESThermophysicalTransportModel` | `TypeName "LES"; LES { model <name>; }` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/turbulence/LES/LESThermophysicalTransportModel/LESThermophysicalTransportModel.H` | Templated abstract base for LES thermophysical transport models; reads the 'LES' sub-dictionary of constant/thermophysicalTransport. |  |
| `RASThermophysicalTransportModel` | `TypeName "RAS"; RAS { model <name>; }` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/turbulence/RAS/RASThermophysicalTransportModel/RASThermophysicalTransportModel.H` | Templated abstract base for RAS thermophysical transport models; reads the 'RAS' sub-dictionary of constant/thermophysicalTransport. |  |
| `laminarThermophysicalTransportModel` | `TypeName "laminar"; laminar { model <name>; }` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/laminar/laminarThermophysicalTransportModel/laminarThermophysicalTransportModel.H` | Templated abstract base for laminar thermophysical transport models; reads the 'laminar' sub-dictionary of constant/thermophysicalTransport. |  |

### thermophysical transport base class  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PhaseThermophysicalTransportModel` | `instantiated for (phaseCompressibleMomentumTransportModel, fluidThermo) and (…, fluidMulticomponentThermo)` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/PhaseThermophysicalTransportModel/PhaseThermophysicalTransportModel.H` | Templated base for multiphase (per-phase) thermophysical transport models. |  |
| `ThermophysicalTransportModel` | `instantiated for (compressibleMomentumTransportModel, fluidThermo), (compressibleMomentumTransportModel, fluidMulticomponentThermo)` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/ThermophysicalTransportModel/ThermophysicalTransportModel.H` | Templated abstract base parameterised on the momentum transport model and the thermo package; carries the runtime-selection table for laminar/RAS/LES thermophysical transport. |  |
| `fluidThermophysicalTransportModel` |  | `[Foundation-12] src/ThermophysicalTransportModels/fluid/fluidThermophysicalTransportModel/fluidThermophysicalTransportModel.H` | Abstract base for fluid (RAS, LES and laminar) thermophysical transport models; provides the common j()/divj() specie-flux interface and alphaEff. | q = -kappaEff*grad(T); j(Yi) = -rho*DEff(Yi)*grad(Yi). |
| `thermophysicalTransportModel` | `TypeName "thermophysicalTransport"; dictionary constant/thermophysicalTransport` | `[Foundation-12] src/ThermophysicalTransportModels/thermophysicalTransportModel/thermophysicalTransportModel.H` | Abstract base for all fluid and solid thermophysical transport models; the root of the kappaEff/q/divq interface used by heat-transfer BCs. | Interface: kappaEff(), q() (heat flux), divq(he) (energy-equation source). |

### thermophysical wall function  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `compressible::alphatJayatillekeWallFunction` | `type compressible::alphatJayatillekeWallFunction;` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/alphatWallFunctions/alphatJayatillekeWallFunction/alphatJayatillekeWallFunctionFvPatchScalarField.H` | Thermal wall function for alphat based on the Jayatilleke P-function, accounting for the thermal sublayer resistance at high molecular Prandtl number. | P = 9.24*((Pr/Prt)^0.75 - 1)*(1 + 0.28*exp(-0.007*Pr/Prt)); alphat obtained by matching the thermal law of the wall; Cmu, kappa and E are taken from the corresponding nut wall function. Entry Prt (0.85). |
| `compressible::alphatWallFunction` | `type compressible::alphatWallFunction;` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/alphatWallFunctions/alphatWallFunction/alphatWallFunctionFvPatchScalarField.H` | Turbulent thermal diffusivity wall function (replicates OpenFOAM v1.5 behaviour) derived directly from nut and a constant turbulent Prandtl number. | alphat = mut/Prt (i.e. rho*nut/Prt). Entries nut (default nut), Prt (default 0.85). |

### thermophysical/specie boundary condition  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `totalFlowRateAdvectiveDiffusive` | `type totalFlowRateAdvectiveDiffusive;` | `[Foundation-12] src/ThermophysicalTransportModels/fluid/derivedFvPatchFields/totalFlowRateAdvectiveDiffusive/totalFlowRateAdvectiveDiffusiveFvPatchScalarField.H` | Species inlet condition that balances the advective and diffusive fluxes to determine the patch specie mass fraction, given the fraction of the total mass flux carried by that specie. | Mixed condition with valueFraction derived from phi/(phi + rho*DEff*deltaCoeffs*magSf); refValue set from massFluxFraction. Entries massFluxFraction, phi, rho. |

### wall function (epsilon)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `epsilonWallFunction` | `type epsilonWallFunction;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/epsilonWallFunctions/epsilonWallFunction/epsilonWallFunctionFvPatchScalarField.H` | Wall constraint on the turbulence dissipation rate for low- and high-Re models; sets the near-wall cell epsilon and the turbulence generation G, switching between laminar and turbulent forms at yPlusLam (from the corresponding nutWallFunction). Derives from wallCellWallFunction so it modifies cell values. | Turbulent: epsilon = Cmu^0.75*k^1.5/(kappa*y), G = (nutw+nuw)*magGradUw*Cmu^0.25*sqrt(k)/(kappa*y); laminar: epsilon = 2*k*nuw/sqr(y), G = 0. Cell values are weighted by the number of wall faces of the cell. |

### wall function (f)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fWallFunction` | `type fWallFunction; (used with RAS model v2f)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/fWallFunctions/fWallFunction/fWallFunctionFvPatchScalarField.H` | Wall function for the v2-f model's elliptic damping function f, operating in laminar/turbulent modes based on yPlusLam. | f = -N*v2*nuw/(sqr(y)*epsilon) type expression evaluated from the v2fBase fields; applied as a fixedValue (f = 0 at the wall for the N=6 variant). |

### wall function (k, low-Re)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kLowReWallFunction` | `type kLowReWallFunction; (Ceps2) — pair with a velocity-based nut wall function, not nutk*` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/kqRWallFunctions/kLowReWallFunction/kLowReWallFunctionFvPatchScalarField.H` | Turbulence kinetic energy wall function valid for both low- and high-Reynolds-number meshes, switching at the nutWallFunction's yPlusLam. | Turbulent: k = Ck/sqrt(Cmu)*log(yPlus) + Bk (Ck = -0.416, Bk = 8.366 form); laminar: k = C*sqr(yPlus)/(Ceps2) scaling; k is a fixedValue on the patch. Coeff Ceps2 1.9. |

### wall function (k, q, R)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kqRWallFunction` | `type kqRWallFunction;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/kqRWallFunctions/kqRWallFunction/kqRWallFunctionFvPatchField.H` | High-Reynolds-number wall condition for the k, q and R fields — a thin wrapper around zeroGradient. Instantiated for all field types (scalar, vector, sphericalTensor, symmTensor, tensor) via makePatchFields. | dk/dn = 0 at the wall. |

### wall function (multiphase epsilon)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `epsilonmWallFunction` | `type epsilonmWallFunction; (libphaseCompressibleMomentumTransportModels)` | `[Foundation-12] src/MomentumTransportModels/phaseCompressible/derivedFvPatchFields/wallFunctions/epsilonWallFunctions/epsilonmWallFunction/epsilonmWallFunctionFvPatchScalarField.H` | Wall constraint on the mixture turbulence dissipation epsilonm used by the mixtureKEpsilon model. | As epsilonWallFunction but applied to the mixture epsilonm field of the two-phase mixture k-epsilon model. |

### wall function (nut)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `nutUWallFunction` | `type nutUWallFunction;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutUWallFunction/nutUWallFunctionFvPatchScalarField.H` | Velocity-based turbulent viscosity wall function (recommended companion for kLowReWallFunction / v2f). | yPlus solved from the log law using Re = magUp*y/nu; nut = max(0, sqr(yPlus)*nu/(Re) - nu) in the turbulent region. |
| `nutkWallFunction` | `type nutkWallFunction;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutkWallFunction/nutkWallFunctionFvPatchScalarField.H` | Turbulent viscosity wall function for high-Reynolds-number flow based on the near-wall turbulence kinetic energy. | yPlus = Cmu^0.25*sqrt(k)*y/nu; nut = nu*(yPlus*kappa/log(max(E*yPlus,1+1e-4)) - 1) where yPlus > yPlusLam, else 0. |

### wall function (nut, continuous)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `nutUSpaldingWallFunction` | `type nutUSpaldingWallFunction;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutUSpaldingWallFunction/nutUSpaldingWallFunctionFvPatchScalarField.H` | Velocity-based nut wall function using Spalding's law of the wall, giving a continuous nut profile all the way to y+ = 0 (all-y+ treatment). | y+ = u+ + (1/E)[exp(kappa u+) - 1 - kappa u+ - 0.5(kappa u+)^2 - (1/6)(kappa u+)^3], solved by Newton iteration for uTau; nut = max(0, sqr(uTau)/magGradU - nu). |

### wall function (nut, low-Re)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `nutLowReWallFunction` | `type nutLowReWallFunction;` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutLowReWallFunction/nutLowReWallFunctionFvPatchScalarField.H` | Zero turbulent viscosity at the wall for wall-resolved (low-Re) meshes, while still providing a yPlus() accessor for post-processing. | nut = 0; yPlus = y*sqrt(nu*magGradUw)/nu. |

### wall function (nut, rough wall)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `nutURoughWallFunction` | `type nutURoughWallFunction; roughnessHeight/Ks, roughnessConstant/Cs, roughnessFactor` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutURoughWallFunction/nutURoughWallFunctionFvPatchScalarField.H` | Velocity-based nut wall function for rough walls; manipulates E to account for sand-grain roughness (Cebeci & Bradshaw 1977). | yPlus obtained by iterative solution of the rough-wall log law with the modified E(KsPlus, Cs); nut = max(0, sqr(yPlus)*nu/Re - nu). |
| `nutkRoughWallFunction` | `type nutkRoughWallFunction; Ks <field>; Cs <field>; (Cs typically 0.5-1.0)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutkRoughWallFunction/nutkRoughWallFunctionFvPatchScalarField.H` | k-based nut wall function for rough walls; manipulates the E parameter for roughness (Cebeci & Bradshaw 1977). | E modified by the roughness Reynolds number KsPlus = Cmu^0.25*sqrt(k)*Ks/nu, with the three regimes (hydraulically smooth / transitional / fully rough) blended. |

### wall function (omega)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `omegaWallFunction` | `type omegaWallFunction; (beta1, blended)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/omegaWallFunctions/omegaWallFunction/omegaWallFunctionFvPatchScalarField.H` | Wall constraint on the specific dissipation rate omega for low- and high-Re models, using either switching (default) or blending between the viscous- and log-region values. | omegaVis = 6*nuw/(beta1*sqr(y)); omegaLog = sqrt(k)/(Cmu^0.25*kappa*y); switched at yPlusLam or blended as sqrt(sqr(omegaVis)+sqr(omegaLog)); G computed as for epsilonWallFunction. Coeffs beta1 0.075, blended false. |

### wall function (v2)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `v2WallFunction` | `type v2WallFunction; (used with RAS model v2f)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/v2WallFunctions/v2WallFunction/v2WallFunctionFvPatchScalarField.H` | Wall function for the v2-f model's wall-normal stress v2, valid for low- and high-Re meshes. | Turbulent: v2 = Cv2/sqrt(Cmu)*log(yPlus) + Bv2; laminar: v2 = Cv2*sqr(yPlus) form; applied as a fixedValue. |

### wall function base class  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `nutWallFunctionFvPatchScalarField` | `not directly selectable as a patch type (base class only)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutWallFunction/nutWallFunctionFvPatchScalarField.H` | Base class for all nut wall functions; holds Cmu, kappa, E, computes yPlusLam (the laminar-turbulent switch-over y+) and provides yPlus() and blending logic. | yPlusLam solved from yPlusLam = log(max(E*yPlusLam,1))/kappa by fixed-point iteration. Defaults Cmu 0.09, kappa 0.41, E 9.8. |
| `wallCellWallFunctionFvPatchScalarField` | `not directly selectable (base class)` | `[Foundation-12] src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/wallCellWallFunction/wallCellWallFunctionFvPatchScalarField.H` | Base class for wall functions that overwrite near-wall CELL values (epsilon, omega); handles the bookkeeping of cells with multiple wall faces and the master/slave patch averaging. |  |

---

## Thermophysical and chemistry

> **Subsystem notes**
>
> Repo root for all paths: C:/Users/sdd32/Documents/GitHub/open_cfd/OpenFOAM-Foundation-12 STRUCTURAL POINT 1 - thermo packages are combinatorial, not a flat list.
> A thermo package name is built by string concatenation and looked up in a run-time table populated by macro expansion (src/thermophysicalModels/basic/basicThermo/basicThermoTemplates.C lines 150-225).
> The 7-part dictionary form is thermoType { type; mixture; transport; thermo; equationOfState; specie; energy; } assembled into the key `type<mixture<transport<thermo<equationOfState<specie>>,energy>>>`.
> A 4-part form thermoType { type; mixture; properties; energy; } -> `type<mixture<properties,energy>>` is used by liquidThermo (properties = a liquidProperties name such as H2O, or the generic `liquid`).
> If the key is missing from the table and dynamicCode is permitted, basicThermo compiles the template on the fly via compileTemplate.
> The `type` values are basicThermo::derivedThermoName: hePsiThermo (psiThermo.C:36), heRhoThermo (rhoFluidThermo.C:36), heheuPsiThermo (psiuMulticomponentThermo.C:40), heSolidThermo (solidThermo.C:37).
> liquidThermo, psiMulticomponentThermo and rhoFluidMulticomponentThermo inherit derivedThermoName from psiThermo/rhoFluidThermo so they also use hePsiThermo/heRhoThermo.
> The combination sets are macro cascades in src/thermophysicalModels/specie/include/: - forGases.H: {Boussinesq, perfectGas} x {(sensibleEnthalpy,hConst),(sensibleEnthalpy,janaf),(sensibleInternalEnergy,eConst),(sensibleInternalEnergy,hConst),(sensibleInternalEnergy,janaf)} x {const, sutherland} = 20 packages.
> forCoeffGases restricts the equation of state to perfectGas only = 10. - forLiquids.H: {adiabaticPerfectFluid, rhoConst, rPolynomial} x {(sensibleEnthalpy,hConst),(sensibleInternalEnergy,eConst),(sensibleInternalEnergy,hConst)} x {const} = 9. forCoeffLiquids drops adiabaticPerfectFluid = 6.
> - forAbsoluteGases.H: perfectGas x {(absoluteEnthalpy,hConst),(absoluteEnthalpy,janaf)} x {const, sutherland} = 4, used only by psiuMulticomponentThermo.
> - forTabulated.H: rhoTabulated x {hTabulated,eTabulated} x tabulatedTransport plus icoTabulated x {hIcoTabulated,eIcoTabulated} x icoTabulatedTransport = 4.
> - solidSpecie/include/forSolids.H: rhoConst x sensibleInternalEnergy x {eConst, ePolynomial, ePower, eIcoTabulated} x {constIsoSolid, constAnisoSolid, exponentialSolid, polynomialSolid, tabulatedSolid} = 20.
> These are crossed with mixtures in psiThermos.C, rhoFluidThermos.C, liquidThermos.C, psiMulticomponentThermos.C, psiuMulticomponentThermos.C, rhoFluidMulticomponentThermos.C and solidThermos.C.
> NOTE: several equation-of-state, thermo and transport classes present in the tree are NOT pre-instantiated by any for* macro and are only reachable via on-the-fly template compilation or from other libraries (lagrangian, twoPhaseModels): PengRobinsonGas, icoPolynomial, incompressiblePerfectGas, linear, perfectFluid, hPolynomial, hPower, ePolynomial (with fluid transports), Andrade, WLF, logPolynomial, polynomial.
> STRUCTURAL POINT 2 - chemistry selection is also a concatenated key. basicChemistryModelNew.C builds `solver<method<thermoName>>` from constant/chemistryProperties: chemistryType { solver ode|EulerImplicit|none; method chemistryModel; }.
> makeChemistrySolver.H crosses {noChemistrySolver, EulerImplicit, ode} x chemistryModel x (forCoeffGases + forCoeffLiquids). Reduction (TDAC) and tabulation (ISAT) are nested sub-dictionaries of chemistryModel, each with its own templated selection table over the same thermo packages.
> STRUCTURAL POINT 3 - reaction type keywords are composed as ReactionType::typeName_() + ReactionRate::type().capitalise() (makeReaction.H). The dictionary keywords are therefore e.g.
> irreversibleArrhenius, reversibleArrhenius, nonEquilibriumReversibleArrhenius, reversibleThirdBodyArrhenius, irreversibleArrheniusLindemannFallOff, reversibleArrheniusTroeFallOff, irreversibleArrheniusSRIChemicallyActivated, reversibleLangmuirHinshelwood, irreversibleMichaelisMenten (liquids only), irreversiblefluxLimitedLangmuirHinshelwood (gases only), irreversiblesurfaceArrhenius.
> FallOff and ChemicallyActivated are only instantiated with ArrheniusReactionRate as the inner rate, and with the three fall-off functions Lindemann/Troe/SRI.
> Reactions using surface or flux-limited rates are registered in the objectRegistry table rather than the dictionary table because they need field access.
> STRUCTURAL POINT 4 - AntoineExtended is present in the tree but its source file is commented out of src/thermophysicalModels/saturationModels/Make/files, so it is not compiled in this release.
> Also note constantPressure and constantTemperature BOTH register under the keyword `constant` but in different tables (saturationPressureModel vs saturationTemperatureModel), and Antoine registers in both tables.
> STRUCTURAL POINT 5 - libraries produced by this subsystem: libspecie, libthermophysicalProperties, libfluidThermophysicalModels, libmulticomponentThermophysicalModels, libsolidThermo, libchemistryModel, libsaturationModels, libODE, libphysicalProperties, libspecieTransfer.
> WHAT IS NOT IN THIS SUBSYSTEM (so this catalogue deliberately contains no RAS/LES/wall-function/divergence-scheme/limiter/linear-solver/preconditioner/smoother/mesh entries): turbulence RAS and LES models and wall functions live in src/MomentumTransportModels and src/ThermophysicalTransportModels; interpolation/divergence schemes, limitedSurfaceInterpolationScheme limiters, linear solvers, preconditioners and smoothers live in src/finiteVolume and src/OpenFOAM; mesh generation, motion and topology changes live in src/mesh, src/motionSolvers, src/fvMotionSolver, src/fvMeshMovers, src/fvMeshTopoChangers, src/fvMeshStitchers, src/fvMeshDistributors and src/polyTopoChange.
> Within the six assigned paths the only "solvers" are the 13 ODESolvers and the 3 chemistrySolvers listed above; there are no scheme limiters, no linear-algebra solvers and no mesh classes at all.

### ODE solver (Rosenbrock, stiff)  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Rosenbrock12` | `Rosenbrock12` | `src/ODE/ODESolvers/Rosenbrock12` | L-stable embedded Rosenbrock solver of order (1)2 (Verwer et al. 1999) | 2-stage Rosenbrock: (I/(gamma*dx) - J) k_i = f(...) + sum_j c_ij*k_j/dx + d_i*dfdx |
| `Rosenbrock23` | `Rosenbrock23` | `src/ODE/ODESolvers/Rosenbrock23` | L-stable embedded Rosenbrock solver of order (2)3 (Sandu et al. 1997) | 3-stage Rosenbrock with one linear solve per stage using the analytic Jacobian |
| `Rosenbrock34` | `Rosenbrock34` | `src/ODE/ODESolvers/Rosenbrock34` | L-stable embedded Rosenbrock solver of order (3)4; default Shampine constants (L-stable set left commented out in the .C) | 4-stage Rosenbrock; embedded 3rd order solution for error estimation |
| `rodas23` | `rodas23` | `src/ODE/ODESolvers/rodas23` | L-stable, stiffly-accurate embedded Rosenbrock solver of order (2)3 | Stiffly-accurate 3-stage Rosenbrock (RODAS family) |
| `rodas34` | `rodas34` | `src/ODE/ODESolvers/rodas34` | L-stable, stiffly-accurate embedded Rosenbrock solver of order (3)4 (Hairer & Wanner) | 6-stage stiffly-accurate Rosenbrock (RODAS) |

### ODE solver (explicit Runge-Kutta)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RKCK45` | `RKCK45` | `src/ODE/ODESolvers/RKCK45` | 4/5th order Cash-Karp embedded Runge-Kutta solver | 6-stage explicit RK with Cash-Karp coefficients; embedded 4th/5th order pair |
| `RKDP45` | `RKDP45` | `src/ODE/ODESolvers/RKDP45` | 4/5th order Dormand-Prince embedded Runge-Kutta solver | 7-stage explicit RK with Dormand-Prince coefficients; embedded 4th/5th order pair |
| `RKF45` | `RKF45` | `src/ODE/ODESolvers/RKF45` | 4/5th order Runge-Kutta-Fehlberg embedded pair; the 4th order step is embedded in the 5th order step so no re-evaluation is needed for error control | 6-stage explicit RK; err = y5 - y4 |

### ODE solver (explicit)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Euler` | `Euler` | `src/ODE/ODESolvers/Euler` | First-order explicit Euler with embedded 0th-order error estimate | y_{n+1} = y_n + dx*f(y_n); err = y_{n+1} - y_n |
| `Trapezoid` | `Trapezoid` | `src/ODE/ODESolvers/Trapezoid` | Trapezoidal (Heun) solver of order (1)2 with embedded error estimate | y_{n+1} = y_n + dx/2*(f(y_n) + f(y_n + dx*f(y_n))) |

### ODE solver (extrapolation, stiff)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SIBS` | `SIBS` | `src/ODE/ODESolvers/SIBS` | Semi-implicit Bader-Deuflhard mid-point rule with polynomial (Bulirsch-Stoer) extrapolation for stiff systems | Semi-implicit midpoint rule + extrapolation to the dx->0 limit |
| `seulex` | `seulex` | `src/ODE/ODESolvers/seulex` | Extrapolation algorithm based on the linearly implicit Euler method with step-size control and automatic order selection | Linearly implicit Euler + extrapolation tableau with order selection |

### ODE solver (semi-implicit)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `EulerSI` | `EulerSI` | `src/ODE/ODESolvers/EulerSI` | Semi-implicit (linearly implicit) Euler of order (0)1 for stiff systems | y_{n+1} = y_n + dx*[I - dx*df/dy]^-1 . [f(y_n) + dx*df/dx] |

### ODE solver base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ODESolver` | `solver (keyword read in ODESolverNew.C from the odeCoeffs/ODE dictionary)` | `src/ODE/ODESolvers/ODESolver` | Abstract base class + runtime selection table for ODE system solvers; holds absTol/relTol, maxSteps and the adaptive step-size driver solve(x, y, dxTry) | Integrates dy/dx = f(x,y) with local error control \|\|err/(absTol + relTol*\|y\|)\|\| |

### ODE step-size controller  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `adaptiveSolver` |  | `src/ODE/ODESolvers/adaptiveSolver` | Mix-in providing adaptive step-size control (safeScale, alphaInc, alphaDec, minScale, maxScale) used by the embedded solvers | scale = safeScale*err^-alpha; dx_new = dx*scale |

### ODE system base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ODESystem` |  | `src/ODE/ODESystem/ODESystem.H` | Abstract interface for a system of ODEs supplying nEqns(), derivatives() and jacobian() | dy/dx = f(x,y); J = df/dy, dfdx = df/dx |

### chemistry mechanism reduction  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DAC` | `DAC` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/DAC` | Dynamic Adaptive Chemistry using the DRGEP rAB matrix seeded from a user search-initiating set (fuel, HO2, CO) | rAB = \|sum_i nu_Ai*w_i*d_Bi\| / max(PA,CA); PA = sum max(0, nu_Ai*w_i); CA = sum max(0, -nu_Ai*w_i) |
| `DRG` | `DRG` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/DRG` | Directed Relation Graph mechanism reduction | rAB = sum_i \|nu_Ai*w_i*d_Bi\| / sum_i \|nu_Ai*w_i\|; species retained if graph-reachable above the tolerance |
| `DRGEP` | `DRGEP` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/DRGEP` | Directed Relation Graph with Error Propagation; O(Nr) implementation with element-exchange-based target coefficients (uses SortableListDRGEP) | rAB = \|sum_i nu_Ai*w_i*d_Bi\| / max(PA, CA); R_V0(V) = max over paths of the product of edge weights |
| `EFA` | `EFA` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/EFA` | Element Flux Analysis reduction (uses the SortableListEFA helper) | Element flux between species pairs per reaction; species retained above the flux tolerance |
| `PFA` | `PFA` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/PFA` | Path Flux Analysis mechanism reduction | First- and second-generation production/consumption flux ratios r1/r2 combined into the interaction coefficient |
| `chemistryReductionMethods::none` | `none` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/noChemistryReduction` | No reduction; retains the full mechanism |  |

### chemistry mechanism reduction base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `chemistryReductionMethod` | `reduction { active on; method <...>; tolerance ...; }` | `src/thermophysicalModels/chemistryModel/chemistryModel/reduction/chemistryReductionMethod` | Abstract base plus a templated (per thermo package) selection table for dynamic mechanism reduction; maintains the active-species map and tolerance |  |

### chemistry model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `chemistryModel` | `chemistryModel (the method entry of chemistryType)` | `src/thermophysicalModels/chemistryModel/chemistryModel` | Full finite-rate chemistry: builds the reaction system, evaluates omega and its Jacobian, with optional TDAC mechanism reduction and ISAT tabulation (Contino et al.) | dc_i/dt = sum_r nu_ir*(kf_r*prod c^v' - kr_r*prod c^v''); dT/dt = -sum_i h_i*dc_i/dt/(rho*Cp) |

### chemistry model base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `basicChemistryModel` | `chemistryType { solver ode\|EulerImplicit\|none; method chemistryModel; } plus jacobian fast\|exact` | `src/thermophysicalModels/chemistryModel/basicChemistryModel` | Base class for chemistry models; owns the chemistryProperties selection, the chemistry on/off switch, deltaTChem and the Jacobian type; New() assembles the key solver<method<thermoName>> | RR_i = specie reaction rate [kg/m^3/s]; tc = chemical time scale |
| `odeChemistryModel` | `odeChemistryModel` | `src/thermophysicalModels/chemistryModel/odeChemistryModel` | Extends basicChemistryModel with an ODESystem interface and the species/reaction reduction maps needed for TDAC and tabulation | Supplies derivatives() and jacobian() of the composition + T (+ p) state vector |

### chemistry solver  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `EulerImplicit` | `EulerImplicit (cTauChem, eqRateLimiter)` | `src/thermophysicalModels/chemistryModel/chemistrySolver/EulerImplicit` | Euler-implicit integration of composition using the reaction-rate Jacobian with Euler-explicit temperature integration, which is much more stable for exothermic systems | (I/dt - dR/dc) dc = R(c); T then updated explicitly from the energy release |
| `noChemistrySolver` | `none` | `src/thermophysicalModels/chemistryModel/chemistrySolver/noChemistrySolver` | Dummy chemistry solver returning zero reaction rates | RR = 0 |
| `ode` | `ode (with odeCoeffs { solver; absTol; relTol; } and eqRateLimiter)` | `src/thermophysicalModels/chemistryModel/chemistrySolver/ode` | Integrates the chemistry ODE system with any of the 13 ODESolvers, sub-cycling with adaptive step size and an optional equilibrium rate limiter | dY/dt = omega(Y,T,p)/rho integrated over the flow time step |

### chemistry solver base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `chemistrySolver` | `solver (entry of the chemistryType dictionary)` | `src/thermophysicalModels/chemistryModel/chemistrySolver/chemistrySolver` | Abstract base wrapping a chemistry model with a time-integration strategy solve(c, T, p, deltaT, subDeltaT) |  |

### chemistry tabulation  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ISAT` | `ISAT` | `src/thermophysicalModels/chemistryModel/chemistryModel/tabulation/ISAT` | In-Situ Adaptive Tabulation (Pope 1997): binary tree of chemPoints with ellipsoids of accuracy, supporting retrieve, grow, add and balance | R(phi) ~ R(phi0) + A(phi0)*(phi - phi0), accepted while phi lies inside the EOA |
| `chemistryTabulationMethods::none` | `none` | `src/thermophysicalModels/chemistryModel/chemistryModel/tabulation/noChemistryTabulation` | No tabulation; every cell integrates the chemistry directly |  |

### chemistry tabulation base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `chemistryTabulationMethod` | `tabulation { method <...>; tolerance ...; }` | `src/thermophysicalModels/chemistryModel/chemistryModel/tabulation/chemistryTabulationMethod` | Abstract base plus selection table for in-situ chemistry tabulation |  |

### chemistry tabulation data structure  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `binaryTree / binaryNode / chemPointISAT` |  | `src/thermophysicalModels/chemistryModel/chemistryModel/tabulation/ISAT` | Binary search tree, splitting node (hyperplane test) and leaf storing the composition, mapping, mapping gradient A and the EOA matrix L | Node test v.(phi - phi_split) > 0; EOA test \|\|L.(phi - phi0)\|\| <= 1 |

### energy boundary condition  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `energyJumpFvPatchScalarField` | `energyJump` | `src/thermophysicalModels/basic/derivedFvPatchFields/energyJump` | Cyclic jump condition for energy computed from the corresponding temperature jump; constructed automatically when the T condition is a jump cyclic | jump(he) = he(p, T + jump(T)) - he(p, T) |
| `fixedEnergyFvPatchScalarField` | `fixedEnergy (set internally by the thermo, not by the user)` | `src/thermophysicalModels/basic/derivedFvPatchFields/fixedEnergy` | Fixed-value energy condition, selected automatically when the corresponding temperature condition is fixedValue | he = he(p, T_patch) |
| `gradientEnergyFvPatchScalarField` | `gradientEnergy` | `src/thermophysicalModels/basic/derivedFvPatchFields/gradientEnergy` | Fixed-gradient energy condition, selected when the temperature condition is zeroGradient, fixedGradient or gradientEnergyCalculatedTemperature | snGrad(he) = Cpv*snGrad(T) by linearisation, or taken directly from the temperature condition |
| `mixedEnergyFvPatchScalarField` | `mixedEnergy` | `src/thermophysicalModels/basic/derivedFvPatchFields/mixedEnergy` | Mixed (Robin) energy condition, selected when the temperature condition is mixed or mixedEnergyCalculatedTemperature | refValue/refGrad/valueFraction either linearised from the T condition or supplied directly |

### energy field source  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `energyFvScalarFieldSource` | `energy` | `src/thermophysicalModels/basic/derivedFvFieldSources/energy` | Field source giving the energy value derived from the corresponding temperature source condition; constructed automatically by the thermo | he_source = he(p, T_source) |

### energy variable  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `absoluteEnthalpy` | `absoluteEnthalpy` | `src/thermophysicalModels/specie/thermo/absoluteEnthalpy` | Maps the thermo package onto the absolute enthalpy (includes heat of formation); energy field named ha; used by psiuMulticomponentThermo | he = ha = hs + hf |
| `absoluteInternalEnergy` | `absoluteInternalEnergy` | `src/thermophysicalModels/specie/thermo/absoluteInternalEnergy` | Maps the thermo package onto the absolute internal energy; energy field named ea | he = ea = es + hf |
| `sensibleEnthalpy` | `sensibleEnthalpy` | `src/thermophysicalModels/specie/thermo/sensibleEnthalpy` | Maps the thermo package onto the sensible enthalpy; energy field named h | he = hs = ha - hf; Cpv = Cp |
| `sensibleInternalEnergy` | `sensibleInternalEnergy` | `src/thermophysicalModels/specie/thermo/sensibleInternalEnergy` | Maps the thermo package onto the sensible internal energy; energy field named e | he = es = ea - hf; Cpv = Cv |

### equation of state  <sub>(12)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Boussinesq` | `Boussinesq` | `src/thermophysicalModels/specie/equationOfState/Boussinesq` | Boussinesq approximation: density a linear function of temperature only; coefficient mixing unsupported so not applicable to mixtures | rho = rho0*(1 - beta*(T - T0)) |
| `PengRobinsonGas` | `PengRobinsonGas` | `src/thermophysicalModels/specie/equationOfState/PengRobinsonGas` | Peng-Robinson cubic equation of state for real gases, from Tc, Vc, Pc and the acentric factor omega | p = R*T/(v-b) - a(T)/(v^2 + 2bv - b^2); a = 0.457235*R^2*Tc^2/Pc*alpha(T,omega); b = 0.077796*R*Tc/Pc |
| `adiabaticPerfectFluid` | `adiabaticPerfectFluid` | `src/thermophysicalModels/specie/equationOfState/adiabaticPerfectFluid` | Adiabatic perfect fluid (equivalent to Murnaghan-Tait) for liquids | rho = rho0*((p + B)/(p0 + B))^(1/gamma) |
| `icoPolynomial` | `icoPolynomial` | `src/thermophysicalModels/specie/equationOfState/icoPolynomial` | Incompressible density as a polynomial in T (templated order, default 8); no coefficient mixing | rho = sum_i rhoCoeffs[i]*T^i |
| `icoTabulated` | `icoTabulated` | `src/thermophysicalModels/specie/equationOfState/icoTabulated` | Incompressible equation of state from a non-uniform table of rho vs T | rho = f_table(T) |
| `incompressiblePerfectGas` | `incompressiblePerfectGas` | `src/thermophysicalModels/specie/equationOfState/incompressiblePerfectGas` | Perfect gas evaluated at a fixed reference pressure so density varies with temperature only | rho = pRef/(R*T) |
| `linear` | `linear` | `src/thermophysicalModels/specie/equationOfState/linear` | Linear equation of state with constant compressibility | rho = rho0 + psi*p |
| `perfectFluid` | `perfectFluid` | `src/thermophysicalModels/specie/equationOfState/perfectFluid` | Perfect gas extended to liquids by a constant density offset (documented as superseded by rPolynomial) | rho = rho0 + p/(R*T) |
| `perfectGas` | `perfectGas` | `src/thermophysicalModels/specie/equationOfState/perfectGas` | Ideal gas density/compressibility; R comes from the molecular weight so no coefficient sub-dictionary is required | rho = p/(R*T); psi = 1/(R*T); Z = 1 |
| `rPolynomial` | `rPolynomial` | `src/thermophysicalModels/specie/equationOfState/rPolynomial` | Reciprocal-density polynomial in p and T for liquids/solids; fits better than a density polynomial and supports coefficient mixing | 1/rho = C0 + C1*T + C2*T^2 - C3*p - C4*p*T |
| `rhoConst` | `rhoConst` | `src/thermophysicalModels/specie/equationOfState/rhoConst` | Constant density equation of state for liquids and solids | rho = const; psi = 0 |
| `rhoTabulated` | `rhoTabulated` | `src/thermophysicalModels/specie/equationOfState/rhoTabulated` | Uniformly-tabulated rho vs (p,T); must be paired with hTabulated/eTabulated because h, Cp, e, Cv, sp, sv and CpMCv are deliberately not implemented from it | rho = f_table(p,T) |

### fall-off function  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LindemannFallOffFunction` | `Lindemann` | `src/thermophysicalModels/specie/reaction/reactionRate/fallOffFunctions/LindemannFallOffFunction` | Simplest (no broadening) fall-off function | F = 1 |
| `SRIFallOffFunction` | `SRI` | `src/thermophysicalModels/specie/reaction/reactionRate/fallOffFunctions/SRIFallOffFunction` | SRI broadening factor with coefficients a, b, c, d, e | F = d*(a*exp(-b/T) + exp(-T/c))^X * T^e with X = 1/(1 + log10(Pr)^2) |
| `TroeFallOffFunction` | `Troe` | `src/thermophysicalModels/specie/reaction/reactionRate/fallOffFunctions/TroeFallOffFunction` | Troe broadening factor with coefficients alpha, Tsss, Ts, Tss | Fcent = (1-alpha)exp(-T/Tsss) + alpha*exp(-T/Ts) + exp(-Tss/T); log F = log Fcent/(1 + ((logPr + c)/(n - d(logPr + c)))^2) |

### function object (chemistry)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `functionObjects::adjustTimeStepToChemistry` | `adjustTimeStepToChemistry (phase)` | `src/thermophysicalModels/chemistryModel/functionObjects/adjustTimeStepToChemistry` | Limits the solver time step to the minimum chemical time scale deltaTChem; only active if adjustTimeStep is on | deltaT <= min(deltaTChem) |
| `functionObjects::specieReactionRates` | `specieReactionRates` | `src/thermophysicalModels/chemistryModel/functionObjects/specieReactionRates` | Writes the domain-averaged reaction rate for each specie in each reaction to <timeDir>/specieReactionRates.dat | <RR_ir> = (1/V) integral(nu_ir*omega_r dV) |

### function object (thermo)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `functionObjects::massFractions` | `massFractions (optional phase)` | `src/thermophysicalModels/multicomponentThermo/functionObjects/massFractions` | Initialisation helper computing mass fractions from X_ or n_ fields on disk; errors out if mass-fraction fields other than Ydefault already exist | Y_i = X_i*W_i/sum_j(X_j*W_j) |
| `functionObjects::moleFractions` | `moleFractions (optional phase)` | `src/thermophysicalModels/multicomponentThermo/functionObjects/moleFractions` | Computes X_<specie> mole-fraction fields from the mass fractions of a multicomponent thermo | X_i = (Y_i/W_i)/sum_j(Y_j/W_j) |

### liquid properties (generic)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `liquid` | `liquid` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/liquid` | Generic liquid whose 13 properties (rho, pv, hl, Cp, h, Cpg, B, mu, mug, kappa, kappag, sigma, D) are each an independently run-time-selected Function1 | each property = user-selected Function1<scalar>(T) |

### liquid properties (mixture)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `liquidMixtureProperties` | `list of liquid names in a sub-dictionary` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/liquidMixtureProperties` | Mixture of liquidProperties components with mole/mass weighted mixing rules (Li critical mixing, Chueh-Prausnitz) | rho_mix, mu_mix, sigma_mix from mole-fraction weighted correlations |

### liquid properties (specie)  <sub>(31)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Ar` | `Ar` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/Ar` | Liquid argon property set |  |
| `C10H22` | `C10H22` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C10H22` | n-Decane liquid property set |  |
| `C12H26` | `C12H26` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C12H26` | n-Dodecane liquid property set |  |
| `C13H28` | `C13H28` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C13H28` | n-Tridecane liquid property set |  |
| `C14H30` | `C14H30` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C14H30` | n-Tetradecane liquid property set |  |
| `C16H34` | `C16H34` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C16H34` | n-Hexadecane liquid property set |  |
| `C2H5OH` | `C2H5OH` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C2H5OH` | Ethanol liquid property set |  |
| `C2H6` | `C2H6` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C2H6` | Ethane liquid property set |  |
| `C2H6O` | `C2H6O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C2H6O` | Di-methyl ether liquid property set |  |
| `C3H6O` | `C3H6O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C3H6O` | Acetone liquid property set |  |
| `C3H8` | `C3H8` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C3H8` | Propane liquid property set |  |
| `C4H10O` | `C4H10O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C4H10O` | Di-ethyl ether liquid property set |  |
| `C6H14` | `C6H14` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C6H14` | n-Hexane liquid property set |  |
| `C6H6` | `C6H6` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C6H6` | Benzene liquid property set |  |
| `C7H16` | `C7H16` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C7H16` | n-Heptane liquid property set |  |
| `C7H8` | `C7H8` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C7H8` | Toluene liquid property set |  |
| `C8H10` | `C8H10` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C8H10` | Ethylbenzene liquid property set |  |
| `C8H18` | `C8H18` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C8H18` | n-Octane liquid property set |  |
| `C9H20` | `C9H20` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/C9H20` | n-Nonane liquid property set |  |
| `CH3OH` | `CH3OH` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/CH3OH` | Methanol liquid property set |  |
| `CH4N2O` | `CH4N2O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/CH4N2O` | Urea; some properties are copied from water where literature data is unavailable |  |
| `H2O` | `H2O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/H2O` | Water liquid property set built from NSRDS correlations |  |
| `IC8H18` | `IC8H18` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/IC8H18` | iso-Octane liquid property set |  |
| `IDEA` | `IDEA` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/IDEA` | IDEA diesel surrogate: 70% n-decane + 30% alpha-methylnaphthalene |  |
| `MB` | `MB` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/MB` | Methyl butyrate (nC3H7COOCH3), a biodiesel surrogate |  |
| `N2` | `N2` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/N2` | Liquid nitrogen property set |  |
| `NH3` | `NH3` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/NH3` | Liquid ammonia property set |  |
| `aC10H7CH3` | `aC10H7CH3` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/aC10H7CH3` | alpha-Methylnaphthalene liquid property set |  |
| `bC10H7CH3` | `bC10H7CH3` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/bC10H7CH3` | beta-Methylnaphthalene liquid property set |  |
| `iC3H8O` | `iC3H8O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/iC3H8O` | iso-Propanol liquid property set |  |
| `nC3H8O` | `nC3H8O` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/nC3H8O` | n-Propanol liquid property set |  |

### liquid properties base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `liquidProperties` | `the liquid name itself, or a type entry inside its sub-dictionary` | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/liquidProperties` | Base for liquid property models supplying rho, pv, hl, Cp, h, Cpg, B, mu, mug, kappa, kappag, sigma, D plus critical and triple point data; two selection tables (bare name and dictionary) |  |

### liquid properties wrapper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `liquidPropertiesSelector` |  | `src/thermophysicalModels/thermophysicalProperties/liquidProperties/liquidProperties/liquidPropertiesSelector.H` | Run-time-selectable liquidProperties presented as a compile-time thermo type; the basis of liquidThermo |  |

### mixture (thermo mixing)  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `coefficientMulticomponentMixture` | `multicomponentMixture` | `src/thermophysicalModels/multicomponentThermo/mixtures/coefficientMulticomponentMixture` | Mass-fraction weighted mixing of the thermodynamic AND transport coefficients themselves | coeff_mix = sum_i Y_i*coeff_i |
| `coefficientWilkeMulticomponentMixture` | `coefficientWilkeMulticomponentMixture` | `src/thermophysicalModels/multicomponentThermo/mixtures/coefficientWilkeMulticomponentMixture` | Mass-fraction weighted mixing of thermodynamic coefficients plus Wilke's (1950) equation for transport properties | mu_mix = sum_i X_i*mu_i/(sum_j X_j*phi_ij) |
| `pureMixture` | `pureMixture` | `src/thermophysicalModels/basic/mixtures/pureMixture` | No mixing at all: returns the single underlying thermo model | cellMixture = the single specie thermo |
| `singleComponentMixture` | `pureMixture` | `src/thermophysicalModels/multicomponentThermo/mixtures/singleComponentMixture` | Multicomponent interface backed by a single specie (Y = 1); registers under the pureMixture keyword |  |
| `valueMulticomponentMixture` | `valueMulticomponentMixture` | `src/thermophysicalModels/multicomponentThermo/mixtures/valueMulticomponentMixture` | Mass-fraction weighted mixing of thermodynamic property VALUES and mole-fraction weighted mixing of transport property values; used for liquids and tabulated data | h_mix = sum_i Y_i*h_i(p,T); mu_mix = sum_i X_i*mu_i(p,T) |

### mixture (thermo mixing) base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `multicomponentMixture` |  | `src/thermophysicalModels/multicomponentThermo/mixtures/multicomponentMixture` | Base for multi-species mixtures; reads the species list and the per-specie thermo data |  |

### multicomponent thermo utility  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FieldListSlice / GeometricFieldListSlicer` |  | `src/thermophysicalModels/multicomponentThermo/include` | Lightweight per-cell and per-face slices through a list of species fields so mixture evaluation avoids copying data |  |

### physical properties base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `physicalProperties` | `constant/physicalProperties dictionary` | `src/physicalProperties/physicalProperties` | Base class for the physicalProperties IOdictionary, with backwards-compatible reading of transportProperties/thermophysicalProperties |  |
| `viscosity` |  | `src/physicalProperties/viscosity` | Abstract base class for all fluid physical properties, providing nu() | nu [m^2/s] |

### reaction base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Reaction` | `type <ReactionType><ReactionRate> in the reactions sub-dictionary` | `src/thermophysicalModels/specie/reaction/Reactions/Reaction` | Templated reaction base holding the lhs/rhs specieCoeffs and the reaction thermo; supplies kf, kr, Kc and the ddc Jacobian contributions; declares two selection tables (dictionary and objectRegistry) | kr = kf/Kc; Kc = Kp*(p0/(R*T))^(sum nu) |

### reaction base (non-templated)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `reaction` |  | `src/thermophysicalModels/specie/reaction/reaction` | Non-templated base holding specie names and stoichiometric coefficients, with the reaction-string parser |  |

### reaction bookkeeping  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `specieCoeffs` |  | `src/thermophysicalModels/specie/reaction/specieCoeffs` | Specie index plus stoichiometric coefficient and the forward/reverse exponents used in the rate expression |  |
| `specieExponent` |  | `src/thermophysicalModels/specie/reaction/specieExponent` | Exponent type storing either an integer or a scalar power, optimising pow() when the exponent is integral |  |

### reaction container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ReactionList` |  | `src/thermophysicalModels/specie/reaction/Reactions/ReactionList` | PtrList of templated reactions read from the reactions dictionary or objectRegistry |  |

### reaction rate  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ArrheniusReactionRate` | `Arrhenius (keywords irreversibleArrhenius, reversibleArrhenius, nonEquilibriumReversibleArrhenius)` | `src/thermophysicalModels/specie/reaction/reactionRate/ArrheniusReactionRate` | Standard modified Arrhenius rate; instantiated irreversible, reversible and non-equilibrium-reversible for gases and liquids | k = A*T^beta*exp(-Ta/T) |
| `JanevReactionRate` | `Janev` | `src/thermophysicalModels/specie/reaction/reactionRate/JanevReactionRate` | Janev-Langer-Evans-Post rate for plasma/ionised reactions; instantiated irreversible and reversible | k = A*T^beta*exp(-Ta/T)*exp(sum_{n=0..8} b_n*(ln T)^n) |
| `LandauTellerReactionRate` | `LandauTeller` | `src/thermophysicalModels/specie/reaction/reactionRate/LandauTellerReactionRate` | Landau-Teller rate for vibrational relaxation; instantiated I/R/N | k = A*T^beta*exp(-Ta/T + B/T^(1/3) + C/T^(2/3)) |
| `LangmuirHinshelwoodReactionRate` | `LangmuirHinshelwood` | `src/thermophysicalModels/specie/reaction/reactionRate/LangmuirHinshelwood` | Langmuir-Hinshelwood rate for gaseous reactions on surfaces; instantiated irreversible and reversible for gases and liquids | k = A*T^beta*exp(-Ta/T)/(1 + sum_j K_j*c_j)^m with Arrhenius-form adsorption constants |
| `MichaelisMentenReactionRate` | `MichaelisMenten` | `src/thermophysicalModels/specie/reaction/reactionRate/MichaelisMenten` | Michaelis-Menten enzymatic kinetics; instantiated for liquid mechanisms only, irreversible only | v = Vmax*c/(Km + c) |
| `fluxLimitedLangmuirHinshelwoodReactionRate` | `fluxLimitedLangmuirHinshelwood` | `src/thermophysicalModels/specie/reaction/reactionRate/fluxLimitedLangmuirHinshelwoodReactionRate` | Langmuir-Hinshelwood with the optional Waletzko-Schmidt mass-transfer flux limiter; registered only for gases via the objectRegistry table, irreversible only | 1/k_eff = 1/k_LH + 1/k_mt, k_mt from the specie mass-transfer coefficient |
| `powerSeriesReactionRate` | `powerSeries` | `src/thermophysicalModels/specie/reaction/reactionRate/powerSeries` | Power-series rate in 1/T; instantiated irreversible and reversible | k = A*T^beta*exp(sum_{n=0..3} e_n/T^(n+1)) |
| `surfaceArrheniusReactionRate` | `surfaceArrhenius` | `src/thermophysicalModels/specie/reaction/reactionRate/surfaceArrheniusReactionRate` | Arrhenius rate scaled by a user-named surface-area-per-unit-volume field; objectRegistry table, irreversible only | k = (A*T^beta*exp(-Ta/T))*a with a the surface area per unit volume |
| `thirdBodyArrheniusReactionRate` | `thirdBodyArrhenius` | `src/thermophysicalModels/specie/reaction/reactionRate/thirdBodyArrheniusReactionRate` | Arrhenius rate enhanced by the third-body concentration; instantiated I/R/N | k = M*A*T^beta*exp(-Ta/T) with M = sum_i alpha_i*c_i |

### reaction rate (composite)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ChemicallyActivatedReactionRate` | `<Rate><Function>ChemicallyActivated, i.e. ArrheniusLindemannChemicallyActivated, ArrheniusTroeChemicallyActivated, ArrheniusSRIChemicallyActivated` | `src/thermophysicalModels/specie/reaction/reactionRate/ChemicallyActivatedReactionRate` | Chemically-activated bimolecular reactions, the inverse of the fall-off form | Pr = k0*M/kInf; k = k0*(1/(1 + Pr))*F(Pr,T) |
| `FallOffReactionRate` | `<Rate><Function>FallOff, i.e. ArrheniusLindemannFallOff, ArrheniusTroeFallOff, ArrheniusSRIFallOff (prefixed irreversible/reversible)` | `src/thermophysicalModels/specie/reaction/reactionRate/FallOffReactionRate` | Unimolecular/recombination fall-off reactions combining low- and high-pressure Arrhenius rates through a fall-off function | Pr = k0*M/kInf; k = kInf*(Pr/(1 + Pr))*F(Pr,T) |

### reaction rate helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `thirdBodyEfficiencies` | `coeffs list inside the reaction sub-dictionary` | `src/thermophysicalModels/specie/reaction/reactionRate/thirdBodyEfficiencies` | Per-specie collision efficiencies used to form the third-body concentration M | M = sum_i alpha_i*c_i |

### reaction type  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `IrreversibleReaction` | `irreversible<Rate>, e.g. irreversibleArrhenius` | `src/thermophysicalModels/specie/reaction/Reactions/IrreversibleReaction` | Forward-only reaction | omega = kf*prod(c_i^v'_i) |
| `NonEquilibriumReversibleReaction` | `nonEquilibriumReversible<Rate>, e.g. nonEquilibriumReversibleArrhenius` | `src/thermophysicalModels/specie/reaction/Reactions/NonEquilibriumReversibleReaction` | Independent forward and reverse rate expressions not tied by Kc | omega = kf*prod(c^v') - kr*prod(c^v'') with kf and kr specified separately |
| `ReversibleReaction` | `reversible<Rate>, e.g. reversibleArrhenius` | `src/thermophysicalModels/specie/reaction/Reactions/ReversibleReaction` | Reverse rate derived from the equilibrium constant | omega = kf*prod(c^v') - (kf/Kc)*prod(c^v'') |

### saturation model  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Antoine` | `Antoine` | `src/thermophysicalModels/saturationModels/Antoine` | Antoine vapour pressure equation; the only model registered in BOTH the pressure and temperature tables | ln(p) = A + B/(C + T); T = B/(ln(p) - A) - C |
| `AntoineExtended` | `AntoineExtended` | `src/thermophysicalModels/saturationModels/AntoineExtended` | Extended Antoine equation with extra log and power terms; source file is commented out of Make/files so it is NOT compiled in this release | ln(p) = A + B/(C + T) + D*ln(T) + E*T^F |
| `ArdenBuck` | `ArdenBuck` | `src/thermophysicalModels/saturationModels/ArdenBuck` | Arden Buck equation for the saturation vapour pressure of moist air | pSat = 611.21*exp((18.678 - Tc/234.5)*Tc/(257.14 + Tc)), Tc = T - 273.15 |
| `constantPressure` | `constant (in the saturationPressureModel table)` | `src/thermophysicalModels/saturationModels/constantPressure` | Constant saturation pressure | pSat = const; dpSat/dT = 0 |
| `constantTemperature` | `constant (in the saturationTemperatureModel table)` | `src/thermophysicalModels/saturationModels/constantTemperature` | Constant saturation temperature | Tsat = const; dTsat/dp = 0 |
| `function1Temperature` | `function1` | `src/thermophysicalModels/saturationModels/function1Temperature` | Saturation temperature as an arbitrary Function1 of pressure (constant, polynomial, table, csv, ...) | Tsat = Function1(p) |
| `polynomialTemperature` | `polynomial` | `src/thermophysicalModels/saturationModels/polynomialTemperature` | Saturation temperature as a polynomial in pressure | Tsat = sum_i C_i*p^i |

### saturation model base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `saturationPressureModel` |  | `src/thermophysicalModels/saturationModels/saturationPressureModel` | Base class and selection table for models giving p_sat(T) and dp_sat/dT | pSat = f(T) |
| `saturationTemperatureModel` |  | `src/thermophysicalModels/saturationModels/saturationTemperatureModel` | Base class and selection table for models giving T_sat(p) and dT_sat/dp | Tsat = f(p) |

### solid properties (mixture)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `solidMixtureProperties` | `list of solid names in a sub-dictionary` | `src/thermophysicalModels/thermophysicalProperties/solidProperties/solidMixtureProperties` | Mixture of solidProperties components with mass/volume-fraction mixing rules | rho_mix = 1/sum(Y_i/rho_i); Cp_mix = sum(Y_i*Cp_i) |

### solid properties (specie)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `C (graphite)` | `C` | `src/thermophysicalModels/thermophysicalProperties/solidProperties/C` | Graphite solid property set |  |
| `CaCO3` | `CaCO3` | `src/thermophysicalModels/thermophysicalProperties/solidProperties/CaCO3` | Calcium carbonate (limestone) solid property set |  |
| `ash` | `ash` | `src/thermophysicalModels/thermophysicalProperties/solidProperties/ash` | Coal ash solid property set |  |

### solid properties base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `solidProperties` | `the solid name itself, or a type entry inside its sub-dictionary` | `src/thermophysicalModels/thermophysicalProperties/solidProperties/solidProperties` | Base for solid property models supplying rho, Cp, kappa, Hf and emissivity; two selection tables (bare name and dictionary) |  |

### solid transport model  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `constAnisoSolidTransport` | `constAnisoSolid` | `src/thermophysicalModels/solidThermo/solidSpecie/transport/constAniso` | Constant anisotropic (diagonal tensor) solid thermal conductivity | Kappa = diag(kx, ky, kz) = const |
| `constIsoSolidTransport` | `constIsoSolid` | `src/thermophysicalModels/solidThermo/solidSpecie/transport/constIso` | Constant isotropic solid thermal conductivity | kappa = const (scalar) |
| `exponentialSolidTransport` | `exponentialSolid` | `src/thermophysicalModels/solidThermo/solidSpecie/transport/exponential` | Exponential/power temperature dependence of the solid conductivity | kappa = kappa0*(T/Tref)^n0 |
| `polynomialSolidTransport` | `polynomialSolid` | `src/thermophysicalModels/solidThermo/solidSpecie/transport/polynomial` | Solid kappa as a polynomial in T (default order 8) | kappa = sum_i kappaCoeffs[i]*T^i |
| `tabulatedSolidTransport` | `tabulatedSolid` | `src/thermophysicalModels/solidThermo/solidSpecie/transport/tabulated` | Non-uniformly-tabulated solid kappa vs T | kappa = f_table(T) |

### specie transfer BC base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `specieTransferMassFractionFvPatchScalarField` | `property massFraction \| moleFraction \| molarConcentration \| partialPressure` | `src/specieTransfer/derivedFvPatchFields/specieTransferMassFraction` | Abstract mixed-BC base for specie-transferring walls; derived classes compute the specie flux and this base adds a corrective diffusive flux so the right amounts are transported | phi_Yi supplied by the derived class; corrective diffusive flux enforces the specie transport |

### specie transfer boundary condition  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `adsorptionMassFractionFvPatchScalarField` | `adsorptionMassFraction (c default 0, property, phi, U)` | `src/specieTransfer/derivedFvPatchFields/adsorptionMassFraction` | Mass-fraction condition for an adsorbing wall; adsorbed species have c > 0, non-adsorbed have c = 0. Used with specieTransferVelocity and specieTransferTemperature | phi_Yi = c*A*psi_i |
| `semiPermeableBaffleMassFractionFvPatchScalarField` | `semiPermeableBaffleMassFraction (c default 0, property, phi, U)` | `src/specieTransfer/derivedFvPatchFields/semiPermeableBaffleMassFraction` | Mass-fraction condition for a semi-permeable baffle; flux proportional to the property jump across the baffle, zero coefficient for impermeable species | phi_Yi = c*A*(psi_i - psi_i,neighbour) |
| `specieTransferTemperatureFvPatchScalarField` | `specieTransferTemperature (phi, U)` | `src/specieTransfer/derivedFvPatchFields/specieTransferTemperature` | Temperature condition summing the species energy fluxes from the mass-fraction conditions to set the energy flux into or out of the domain | q = sum_i phi_Yi*h_i(T) |
| `specieTransferVelocityFvPatchVectorField` | `specieTransferVelocity (rho default rho)` | `src/specieTransfer/derivedFvPatchFields/specieTransferVelocity` | Velocity condition summing the species mass fluxes generated by the mass-fraction conditions to set the wall-normal velocity | U.n = sum_i phi_Yi/(rho*A) |

### temperature boundary condition base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `gradientEnergyCalculatedTemperatureFvPatchScalarField` | `gradientEnergyCalculatedTemperature` | `src/thermophysicalModels/basic/derivedFvPatchFields/gradientEnergy` | Base class for temperature boundary conditions that supply the energy gradient directly, so no linearisation is needed |  |
| `mixedEnergyCalculatedTemperatureFvPatchScalarField` | `mixedEnergyCalculatedTemperature` | `src/thermophysicalModels/basic/derivedFvPatchFields/mixedEnergy` | Base class for temperature conditions that supply energy refValue/refGrad/valueFraction directly |  |

### temperature field source  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `uniformFixedEnergyTemperatureFvScalarFieldSource` | `uniformFixedEnergyTemperature (uniformHe)` | `src/thermophysicalModels/basic/derivedFvFieldSources/uniformFixedEnergyTemperature` | Applied to the temperature field but injects a uniform fixed energy into the energy equation | he_source = uniformHe |
| `uniformInletOutletEnergyTemperatureFvScalarFieldSource` | `uniformInletOutletEnergyTemperature (uniformInletHe)` | `src/thermophysicalModels/basic/derivedFvFieldSources/uniformInletOutletEnergyTemperature` | Injects a uniform fixed energy when the source is positive and the internal value when it is a sink | he_source = uniformInletHe for source > 0; he_internal for source < 0 |

### temperature field source base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `energyCalculatedTemperatureFvScalarFieldSource` | `energyCalculatedTemperature` | `src/thermophysicalModels/basic/derivedFvFieldSources/energy` | Base for temperature source conditions that set the parameters of the corresponding energy source directly |  |

### thermo bookkeeping  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `specieElement` |  | `src/thermophysicalModels/specie/specieElement` | Element name plus atom count; used for elemental balances and mechanism reduction |  |
| `speciesTable` |  | `src/thermophysicalModels/specie/speciesTable` | Hash-indexed word list of species names giving the name->index lookup used throughout chemistry |  |

### thermo data table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `atomicWeightTable` |  | `src/thermophysicalModels/specie/atomicWeights` | Static table of atomic weights of all the elements, used to derive specie molecular weights from chemical formulae |  |

### thermo initialisation  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `hydrostaticInitialisation` | `hydrostaticInitialisation switch in fvSolution/PIMPLE\|SIMPLE, with nHydrostaticCorrectors (default 5)` | `src/thermophysicalModels/basic/fluidThermo/hydrostaticInitialisation.C` | Optional iterative hydrostatic initialisation of ph_rgh, p and rho at the start of a run (not on restart) | p = ph_rgh + rho*gh + pRef, solved iteratively when rho depends on p |

### thermo package (solid)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `constAnisoSolidThermo` | `constAnisoSolidThermo` | `src/thermophysicalModels/solidThermo/constAnisoSolidThermo` | As constSolidThermo but with an anisotropic diagonal-tensor conductivity Kappa given as a vector or vector field | Kappa = diag(kx,ky,kz); q = -Kappa . grad T |
| `constSolidThermo` | `constSolidThermo (as thermoType, not a thermoType sub-dictionary)` | `src/thermophysicalModels/solidThermo/constSolidThermo` | Uniform, zonal or file-based constant solid properties (rho, Cv, kappa), each entry specified as type uniform \| zonal \| file | rho, Cv, kappa constant, optionally per cell-zone or read as fields |

### thermo package base  <sub>(13)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `basicThermo` | `thermoType { type; mixture; transport; thermo; equationOfState; specie; energy; } or the 4-part { type; mixture; properties; energy; } form` | `src/thermophysicalModels/basic/basicThermo` | Pure-virtual base for all fluid and solid thermodynamics; owns the selection tables, assembles/splits the thermoType name, maps T boundary types onto he boundary types and triggers on-the-fly template compilation |  |
| `fluidMulticomponentThermo` | `fluidMulticomponentThermo (selection table name)` | `src/thermophysicalModels/multicomponentThermo/fluidMulticomponentThermo` | Multi-species fluid thermo interface adding species diffusivity/mass transfer; this is the table chemistry models are constructed from |  |
| `fluidThermo` | `fluidThermo (selection table name)` | `src/thermophysicalModels/basic/fluidThermo` | Base for fluid thermodynamics adding p, psi, mu, nu and its own selection table | nu = mu/rho |
| `liquidThermo` | `thermoType { type heRhoThermo; mixture pureMixture; properties <liquidName>\|liquid; energy sensibleEnthalpy\|sensibleInternalEnergy; }` | `src/thermophysicalModels/basic/liquidThermo` | Liquid thermo built on liquidPropertiesSelector adding surface tension sigma(); registered into the basicThermo, fluidThermo, rhoFluidThermo and liquidThermo tables for both sensible energy forms | All properties come from the selected liquidProperties model; sigma = f(T) |
| `multicomponentThermo` |  | `src/thermophysicalModels/multicomponentThermo/multicomponentThermo` | Adds species mass-fraction fields Y, per-specie properties and the composition interface on top of basicThermo | sum_i Y_i = 1 |
| `psiMulticomponentThermo` | `hePsiThermo with mixture multicomponentMixture \| coefficientWilkeMulticomponentMixture \| pureMixture` | `src/thermophysicalModels/multicomponentThermo/psiMulticomponentThermo` | Compressibility-based multi-species thermo; instantiated with coefficientMulticomponentMixture and coefficientWilkeMulticomponentMixture over forCoeffGases and singleComponentMixture over forGases | rho = psi*p with mixture-averaged R |
| `psiThermo` | `hePsiThermo (thermoType type entry); table name psiThermo` | `src/thermophysicalModels/basic/psiThermo` | Compressibility-based fluid thermo; psiThermos.C instantiates it over pureMixture for the full forGases set (20 packages) | rho = psi*p with psi = 1/(R*T); correctRho is a no-op |
| `psiuMulticomponentThermo` | `heheuPsiThermo with mixture egrMixture \| homogeneousMixture \| inhomogeneousMixture \| veryInhomogeneousMixture` | `src/thermophysicalModels/multicomponentThermo/psiuMulticomponentThermo` | Compressibility thermo carrying both burnt and unburnt states (Tu, heu, psiu, muu, alphau) for premixed/partially-premixed Xi combustion; instantiated over forAbsoluteGases with the four regress-variable mixtures | psiu = 1/(Ru*Tu); heu solved with fixed/gradient/mixedUnburntEnthalpy boundary conditions |
| `pureThermo` |  | `src/thermophysicalModels/basic/pureThermo` | Interface layer for single-component thermodynamics used by the pureMixture instantiations |  |
| `rhoFluidMulticomponentThermo` | `heRhoThermo with mixture multicomponentMixture \| coefficientWilkeMulticomponentMixture \| valueMulticomponentMixture \| pureMixture` | `src/thermophysicalModels/multicomponentThermo/rhoFluidMulticomponentThermo` | Density-based multi-species thermo; instantiated over gases (coefficient/Wilke/single), liquids (coefficient/value/single) and tabulated data (value/single) | rho = f(p,T,Y) from the mixed equation of state |
| `rhoFluidThermo` | `heRhoThermo (thermoType type entry); table name rhoFluidThermo` | `src/thermophysicalModels/basic/rhoFluidThermo` | Density-based fluid thermo; rhoFluidThermos.C instantiates it over pureMixture for forGases + forLiquids + forTabulated | rho = f(p,T) from the equation of state; correctRho(deltaRho) adds the density correction after the pressure solution |
| `rhoThermo` | `rhoThermo (selection table name)` | `src/thermophysicalModels/basic/rhoThermo` | Base for density-based thermodynamic properties, shared by fluid and solid density thermos | rho stored as a field |
| `solidThermo` | `heSolidThermo (thermoType type entry); table name solidThermo` | `src/thermophysicalModels/solidThermo/solidThermo` | Base for solid thermodynamics adding isotropic kappa() and anisotropic Kappa(); solidThermos.C instantiates the full forSolids set over pureMixture (20 packages) | rho*Cv*dT/dt = div(kappa grad T) |

### thermo package wrapper  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NamedThermo` |  | `src/thermophysicalModels/basic/basicThermo/NamedThermo.H` | Final wrapper adding run-time type info; defineThermo names the instantiation derivedThermoName<Mixture<ThermoPhysics>> |  |
| `PhysicalPropertiesThermo` |  | `src/thermophysicalModels/basic/PhysicalPropertiesThermo.H` | Template wrapper that also constructs and registers the physicalProperties dictionary around any thermo |  |

### thermo specie base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `specie` | `specie (entry in the thermoType dictionary)` | `src/thermophysicalModels/specie/specie` | Base of all thermophysical property types: molecular weight W and number of moles Y, plus the mass/mole mixing operators | R = RR/W |

### thermodynamics (Cp/Cv model)  <sub>(11)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `eConstThermo` | `eConst` | `src/thermophysicalModels/specie/thermo/eConst` | Internal-energy-based thermo with constant heat capacity at constant volume | Cv = const; es = Cv*(T - Tref) + esRef; ea = es + hf |
| `eIcoTabulatedThermo` | `eIcoTabulated` | `src/thermophysicalModels/specie/thermo/eIcoTabulated` | Non-uniformly-tabulated Cv vs T with integrated internal energy; part of the solid thermo set | Cv = f_table(T); es = integral(Cv dT) |
| `ePolynomialThermo` | `ePolynomial` | `src/thermophysicalModels/specie/thermo/ePolynomial` | Internal-energy-based thermo with Cv a polynomial in T (default order 8); used for solids | Cv = sum_i CvCoeffs[i]*T^i; es = integral(Cv dT) |
| `ePowerThermo` | `ePower` | `src/thermophysicalModels/specie/thermo/ePower` | Power-law Cv particularly suited to solids at low temperature | Cv = c0*(T/Tref)^n0 |
| `eTabulatedThermo` | `eTabulated` | `src/thermophysicalModels/specie/thermo/eTabulated` | Uniformly-tabulated sensible internal energy es and Cv vs (p,T) | es = f_table(p,T); Cv = f_table(p,T) |
| `hConstThermo` | `hConst` | `src/thermophysicalModels/specie/thermo/hConst` | Enthalpy-based thermo with constant heat capacity at constant pressure | Cp = const; hs = Cp*(T - Tref) + hsRef; ha = hs + hf |
| `hIcoTabulatedThermo` | `hIcoTabulated` | `src/thermophysicalModels/specie/thermo/hIcoTabulated` | Non-uniformly-tabulated Cp vs T with analytically integrated enthalpy (incompressible) | Cp = f_table(T); hs = integral(Cp dT) |
| `hPolynomialThermo` | `hPolynomial` | `src/thermophysicalModels/specie/thermo/hPolynomial` | Enthalpy-based thermo with Cp a polynomial in T (templated order, default 8) | Cp = sum_i CpCoeffs[i]*T^i; hs = integral(Cp dT) |
| `hPowerThermo` | `hPower` | `src/thermophysicalModels/specie/thermo/hPower` | Power-law Cp particularly suited to solids at low temperature | Cp = c0*(T/Tref)^n0 |
| `hTabulatedThermo` | `hTabulated` | `src/thermophysicalModels/specie/thermo/hTabulated` | Uniformly-tabulated sensible enthalpy hs and Cp vs (p,T) | hs = f_table(p,T); Cp = f_table(p,T) |
| `janafThermo` | `janaf` | `src/thermophysicalModels/specie/thermo/janaf` | JANAF/NASA 7-coefficient polynomials with low and high temperature ranges split at Tcommon | Cp = (((a4*T + a3)*T + a2)*T + a1)*T + a0; ha = ((((a4/5*T + a3/4)*T + a2/3)*T + a1/2)*T + a0)*T + a5 |

### thermodynamics wrapper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `species::thermo` |  | `src/thermophysicalModels/specie/thermo/thermo` | Composes a Thermo model with an energy Type mapping; adds the THE/THs/THa inverse Newton solves, entropy, Gibbs free energy and the equilibrium constant | T from he by Newton iteration T_{n+1} = T_n - (he(T_n) - he)/Cpv(T_n); Kc from -dG/(R*T) |

### thermophysical Function1  <sub>(10)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NSRDS0` | `NSRDS0` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS0` | NSRDS-AICHE function 100: fifth-order polynomial in T (registered via addScalarFunction1) | f = ((((f*T + e)*T + d)*T + c)*T + b)*T + a |
| `NSRDS1` | `NSRDS1` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS1` | NSRDS-AICHE function 101, typically vapour pressure or liquid viscosity | f = exp(a + b/T + c*ln(T) + d*T^e) |
| `NSRDS14` | `NSRDS14` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS14` | NSRDS-AICHE function 114, saturated liquid heat capacity near the critical point | t = 1 - T/Tc; f = a^2/t + b - t*(2ac + t*(ad + t*(c^2/3 + t*(cd/2 + d^2*t/5)))) |
| `NSRDS2` | `NSRDS2` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS2` | NSRDS-AICHE function 102, gas viscosity/conductivity form | f = a*T^b/(1 + c/T + d/T^2) |
| `NSRDS3` | `NSRDS3` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS3` | NSRDS-AICHE function 103 | f = a + b*exp(-c/T^d) |
| `NSRDS4` | `NSRDS4` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS4` | NSRDS-AICHE function 104, second virial coefficient form | f = a + b/T + c/T^3 + d/T^8 + e/T^9 |
| `NSRDS5` | `NSRDS5` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS5` | NSRDS-AICHE function 105, Rackett-type saturated liquid density | f = a/b^(1 + (1 - T/c)^d) |
| `NSRDS6` | `NSRDS6` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS6` | NSRDS-AICHE function 106, heat of vaporisation / surface tension form | Tr = T/Tc; f = a*(1 - Tr)^(((e*Tr + d)*Tr + c)*Tr + b) |
| `NSRDS7` | `NSRDS7` | `src/thermophysicalModels/specie/thermophysicalFunctions/NSRDS/NSRDS7` | NSRDS-AICHE function 107, Aly-Lee ideal gas heat capacity | f = a + b*((c/T)/sinh(c/T))^2 + d*((e/T)/cosh(e/T))^2 |
| `integratedNonUniformTable` | `integratedNonUniformTable` | `src/thermophysicalModels/specie/thermophysicalFunctions/integratedNonUniformTable1` | Non-uniform tabulated property with linear interpolation plus pre-computed integral and integral of f/x, accelerated by a uniform jump table | f = linear interpolation of (x,y) pairs; integrals evaluated analytically per interval |

### thermophysical Function2  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `APIdiffCoefFunc` | `APIdiffCoef` | `src/thermophysicalModels/specie/thermophysicalFunctions/APIdiffCoef` | American Petroleum Institute correlation for vapour mass diffusivity, implemented as a Function2 of (p,T) | D = 3.6059e-3*(1.8*T)^1.75*sqrt(1/Wa + 1/Wb)/(p*(Va^(1/3) + Vb^(1/3))^2) |

### thermophysical properties base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `thermophysicalProperties` |  | `src/thermophysicalModels/thermophysicalProperties/thermophysicalProperties` | Base class for solid/liquid/gas property models presenting an interface compatible with the templated thermo packages |  |

### thermophysical properties wrapper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `thermophysicalPropertiesSelector` |  | `src/thermophysicalModels/thermophysicalProperties/thermophysicalProperties/thermophysicalPropertiesSelector.H` | Wrapper giving run-time selection of a thermophysicalProperties model inside a compile-time-templated thermo package |  |

### transport model  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `AndradeTransport` | `Andrade` | `src/thermophysicalModels/specie/transport/Andrade` | Andrade function for log(mu) and log(kappa) of liquids (Andrade 1934) | log(mu) = c0 + c1*T + c2*T^2 + c3/(c4 + T); same form for log(kappa) |
| `WLFTransport` | `WLF` | `src/thermophysicalModels/specie/transport/WLF` | Williams-Landel-Ferry viscosity for polymer melts; kappa from a constant Prandtl number | mu = mu0*exp(-C1*(T - Tr)/(C2 + T - Tr)); kappa = mu*Cp/Pr |
| `constTransport` | `const` | `src/thermophysicalModels/specie/transport/const` | Constant dynamic viscosity with constant Prandtl number | mu = const; kappa = mu*Cp/Pr |
| `icoTabulatedTransport` | `icoTabulated` | `src/thermophysicalModels/specie/transport/icoTabulated` | Non-uniformly-tabulated mu and kappa vs T (incompressible) | mu = f_table(T); kappa = f_table(T) |
| `logPolynomialTransport` | `logPolynomial` | `src/thermophysicalModels/specie/transport/logPolynomial` | log(mu) and log(kappa) as polynomials in log(T) (default order 8) | log(mu) = sum_i muCoeffs[i]*log(T)^i; log(kappa) = sum_i kappaCoeffs[i]*log(T)^i |
| `polynomialTransport` | `polynomial` | `src/thermophysicalModels/specie/transport/polynomial` | mu and kappa as polynomials in T (templated order, default 8) | mu = sum_i muCoeffs[i]*T^i; kappa = sum_i kappaCoeffs[i]*T^i |
| `sutherlandTransport` | `sutherland` | `src/thermophysicalModels/specie/transport/sutherland` | Sutherland's law for viscosity with modified-Eucken thermal conductivity for gases | mu = As*sqrt(T)/(1 + Ts/T); kappa = mu*Cv*(1.32 + 1.77*R/Cv) |
| `tabulatedTransport` | `tabulated` | `src/thermophysicalModels/specie/transport/tabulated` | Uniformly-tabulated mu and kappa vs (p,T) | mu = f_table(p,T); kappa = f_table(p,T) |

### viscosity model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `viscosityModels::constant` | `constant (also registered as Newtonian)` | `src/physicalProperties/viscosityModels/constant` | Uniform constant Newtonian kinematic viscosity; registered under two keywords via addToRunTimeSelectionTable plus addNamedToRunTimeSelectionTable | nu = nu0 = const |

### viscosity model base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `viscosityModel` | `viscosityModel, backwards compatible with transportModel, in physicalProperties` | `src/physicalProperties/viscosityModels/viscosityModel` | Abstract base and selection table for Newtonian viscosity models (non-Newtonian generalised models live in MomentumTransportModels) | nu = f(...) |

---

## Multiphase, two-phase, lagrangian, waves

> **Subsystem notes**
>
> STRUCTURE AND SELECTION MECHANICS 1.
> Libraries built from these paths: - src/multiphaseModels -> libmultiphaseProperties (only the alphaContactAngle BC + correctContactAngle helper; the Euler-Euler multiphase modelling that used to live here is not in this tree) - src/twoPhaseModels -> libtwoPhaseMixture, libtwoPhaseProperties, libincompressibleTwoPhases, libcompressibleTwoPhases, libinterfaceProperties, libcompressibleInterfaceProperties, libinterfaceCompression, libincompressibleCavitationModels, libcompressibleCavitationModels - src/waves -> libwaves - src/randomProcesses -> librandomProcesses - src/lagrangian -> liblagrangian (basic), libsolidParticle, liblagrangianParcel, libDSMC, libmolecularDynamics, liblagrangianFunctionObjects 2.
> Duplicated model family: cavitation exists TWICE with the same keywords in separate namespaces/libraries - Foam::cavitationModels::{Kunz, Merkle, SchnerrSauer} (incompressible, constant rho) and Foam::compressible::cavitationModels::{Kunz, Merkle, Saito, SchnerrSauer} (rhoFluidThermo phases).
> Saito is compressible-only. Both select via `model <name>;` and read `<name>Coeffs` as an optional sub-dict (dict.optionalSubDict). 3.
> The six interface-compression schemes are NOT ordinary limitedSurfaceInterpolationSchemes - they register through surfaceInterpolationScheme<scalar>::addMeshFluxConstructorToTable and are usable only for scalar (phase-fraction) fields.
> Two class names differ from their keywords: interfaceCompressionNew -> "interfaceCompression" and noInterfaceCompressionNew -> "noInterfaceCompression". interfaceCompression, noInterfaceCompression, PLIC and PLICU WRAP a base scheme supplied as the next token (e.g.
> `Gauss PLIC interfaceCompression vanLeer 1;`); MPLIC/MPLICU are self-contained (`Gauss MPLIC;`). The canonical keyword list is the Foam::compressionSchemes wordHashSet in interfaceCompression.C. 4.
> Lagrangian parcel sub-model availability is per cloud type, fixed at compile time by parcels/derived/*/make*Submodels.C: - cloud (momentumCloud): cloudFunctionObjects, 14 parcel forces, dispersion, momentum injection (incl.
> momentumLookupTableInjection), patchInteraction, stochasticCollision (none only), surfaceFilm - collidingCloud: as momentum plus collision models (pairCollision + PairModel/WallModel) - MPPICCloud: as momentum plus damping, isotropy and packing models - thermoCloud: thermoParcelForces (adds BrownianMotion), thermo injection (adds thermoLookupTableInjection), heatTransfer, composition - reactingCloud: as thermo plus reacting injection (reactingLookupTableInjection) and phaseChange - reactingMultiphaseCloud: as reacting plus reactingMultiphase injection, suppressionCollision, multiphase composition, devolatilisation, surfaceReaction - sprayCloud: thermo forces + distortedSphereDrag (spray-only), spray injection, ORourke/trajectory stochastic collision, phaseChange, atomisation, breakup Consequently `liquidEvaporationBoil` is invalid for a plain `cloud`, `distortedSphereDrag` only exists for `sprayCloud`, and momentumLookupTableInjection is only registered for momentumCloud/collidingCloud/mppicCloud.
> 5. Cloud selection keywords do not match the class names: momentumCloud registers as "cloud" and mppicCloud as "MPPICCloud". Two selection tables exist - "viscosity" (incompressible carrier, constructed from mu) and "thermo" (compressible carrier, constructed from a fluidThermo).
> Only cloud, collidingCloud and MPPICCloud are in both; thermoCloud, reactingCloud, reactingMultiphaseCloud and sprayCloud are thermo-only. 6.
> Keyword casing is inconsistent and must be copied exactly: SaffmanMeiLiftForce (literally ends in "Force", unlike TomiyamaLift); sphereDrag/nonSphereDrag/distortedSphereDrag are lower-camel while SchillerNaumannDrag/WenYuDrag/ErgunWenYuDrag/PlessisMasliyahDrag are upper-camel.
> Every model family has a "none" placeholder registered under exactly "none". 7. Flux.H is one header defining three separately registered classes (NumberFlux -> numberFlux, VolumeFlux -> volumeFlux, MassFlux -> massFlux) although the directory is called Flux. 8.
> AveragingMethod is registered separately for scalar and vector via defineTemplateRunTimeSelectionTable in makeAveragingMethods.C, so `basic` and `dual` appear in two distinct tables. 9. src/waves has no turbulence models of its own.
> The three wave BCs (waveAlpha, waveVelocity, waveInletOutlet) and fv::waveForcing all read from ONE centrally registered waveSuperposition object built from constant/waveProperties, whose `type` entry selects wave (waveSuperposition) or waveAtmBoundaryLayer (waveAtmBoundaryLayerSuperposition).
> The individual wave models are entries in that dict's `waves` list, and the irregular model additionally selects a waveSpectrum. 10. src/randomProcesses provides the only fvModel for turbulence forcing in this subsystem (OUForce).
> It is serial-only and requires an isotropic power-of-2 mesh because of the in-house FFT. It also hosts noiseFFT (used by the noise post-processing utility) and turbGen/Ek (used by boxTurb). 11.
> Nothing under these five paths defines a linear solver, preconditioner, smoother, agglomeration method, wall function, RAS model, LES SGS model, mesh generator, mesh mover or topology changer - those live in src/OpenFOAM, src/finiteVolume, src/MomentumTransportModels, src/fvMeshMovers, src/fvMeshTopoChangers, src/fvAgglomerationMethods and src/mesh.
> The only "schemes" here are the six VoF interface-compression / geometric-advection surface-interpolation schemes and the two Lagrangian semi-implicit ODE integration schemes (Euler, analytical). 12.
> src/twoPhaseModels/VoF compiles nothing - it is three solver #include files defining the user-facing MULES controls (nAlphaCorr, nAlphaSubCycles, MULESCorr, alphaApplyPrevCorr) plus the interface Courant number and the maxAlphaCo-based adjustable time step. 13.
> src/lagrangian/basic/Make/files does not compile passiveParticle.C or InteractionLists.C (header-only/templated); only particle, IOPositionName, cloud, passiveParticleCloud and referredWallFace are compiled into liblagrangian.

### DEM collision base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CollisionModel` | `collisionModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/CollisionModel/CollisionModel.H` | Base for deterministic particle collision handling in colliding clouds; defines the required number of collision sub-cycles. |  |

### DEM collision model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NoCollision` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/NoCollision/NoCollision.H` | Placeholder for the 'none' option. |  |
| `PairCollision` | `pairCollision` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/PairCollision.H` | Full DEM pair-and-wall collision handling using InteractionLists for parallel neighbour search; delegates to a PairModel and a WallModel. | Contact when \|r_ij\| < 0.5*(d_i+d_j); forces accumulated on both parcels with torque from the tangential components |

### DEM contact history  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CollisionRecordList / PairCollisionRecord / WallCollisionRecord` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/CollidingParcel/CollisionRecordList` | Per-parcel persistent record of active pair and wall contacts carrying the accumulated tangential spring displacement between time steps. |  |

### DEM helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `WallSiteData` |  | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/WallSiteData` | Stores per-wall-contact site data (patch index and templated payload) used by the wall collision models. |  |

### DEM pair force base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PairModel` | `pairModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/PairModel/PairModel/PairModel.H` | Base for particle-particle contact force models. |  |

### DEM pair force model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PairSpringSliderDashpot` | `pairSpringSliderDashpot` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/PairModel/PairSpringSliderDashpot/PairSpringSliderDashpot.H` | Linear spring-slider-dashpot (Hookean) particle-particle contact with Coulomb friction and tangential history via CollisionRecordList. | Fn = -kn*delta - eta_n*vn; Ft = min(-kt*xi_t - eta_t*vt, mu*\|Fn\|); kn from Young's modulus/Poisson ratio, eta from the restitution coefficient |

### DEM wall force base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `WallModel` | `wallModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/WallModel/WallModel/WallModel.H` | Base for particle-wall contact force models. |  |

### DEM wall force model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `WallLocalSpringSliderDashpot` | `wallLocalSpringSliderDashpot` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/WallModel/WallLocalSpringSliderDashpot/WallLocalSpringSliderDashpot.H` | As WallSpringSliderDashpot but with per-patch material properties (E, nu, e, mu). |  |
| `WallSpringSliderDashpot` | `wallSpringSliderDashpot` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/CollisionModel/PairCollision/WallModel/WallSpringSliderDashpot/WallSpringSliderDashpot.H` | Spring-slider-dashpot particle-wall contact with globally uniform wall material properties. | As PairSpringSliderDashpot with the wall as an infinite-mass, infinite-radius partner |

### DSMC binary collision base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BinaryCollisionModel` | `BinaryCollisionModel` | `[Foundation-12] src/lagrangian/DSMC/submodels/BinaryCollisionModel/BinaryCollisionModel/BinaryCollisionModel.H` | Base for DSMC molecule-molecule collision models; supplies sigmaTcR (cross-section times relative speed) and the post-collision state. |  |

### DSMC binary collision model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LarsenBorgnakkeVariableHardSphere` | `LarsenBorgnakkeVariableHardSphere` | `[Foundation-12] src/lagrangian/DSMC/submodels/BinaryCollisionModel/LarsenBorgnakkeVariableHardSphere/LarsenBorgnakkeVariableHardSphere.H` | VHS collisions with Larsen-Borgnakke internal (rotational) energy redistribution, following Bird's INELRS routine in DSMC0R.FOR. | With probability 1/relaxationCollisionNumber, translational and internal energy are redistributed from the total collision energy Ec by acceptance-rejection sampling of the LB distribution |
| `NoBinaryCollision` | `none` | `[Foundation-12] src/lagrangian/DSMC/submodels/BinaryCollisionModel/NoBinaryCollision/NoBinaryCollision.H` | Collisionless (free-molecular) option. | sigmaTcR = 0 |
| `VariableHardSphere` | `VariableHardSphere` | `[Foundation-12] src/lagrangian/DSMC/submodels/BinaryCollisionModel/VariableHardSphere/VariableHardSphere.H` | Bird's Variable Hard Sphere elastic collision model (no internal energy exchange). | sigmaT = pi*dRef^2*((2*kB*Tref/(mR*cR^2))^(omega-0.5))/gamma(2.5-omega); post-collision velocities from isotropic scattering in the centre-of-mass frame |

### DSMC cloud  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dsmcCloud / DSMCCloud` |  | `[Foundation-12] src/lagrangian/DSMC/clouds/Templates/DSMCCloud/DSMCCloud.H` | Direct Simulation Monte Carlo cloud: per-cell collision partner selection, sub-models for binary collisions, wall interaction and inflow, and accumulation of the extensive DSMC fields. | No-time-counter selection nCandidates = 0.5*N*(N-1)*Fn*sigmaTcRMax*dt/V; accumulates rhoN, rhoM, momentum, linearKE, internalE and iDof fields |

### DSMC inflow base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `InflowBoundaryModel` | `InflowBoundaryModel` | `[Foundation-12] src/lagrangian/DSMC/submodels/InflowBoundaryModel/InflowBoundaryModel/InflowBoundaryModel.H` | Base for inserting new DSMC particles across boundaries. |  |

### DSMC inflow model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FreeStream` | `FreeStream` | `[Foundation-12] src/lagrangian/DSMC/submodels/InflowBoundaryModel/FreeStream/FreeStream.H` | Inserts particles across every patch of type 'patch' from a free stream with number density, temperature and velocity sourced face-by-face from the cloud's boundaryT and boundaryU fields. | Flux N = n*A*dt/(2*sqrt(pi))*(exp(-sM^2) + sqrt(pi)*sM*(1 + erf(sM)))*sqrt(2*kB*T/m), sM = (U & n)/sqrt(2*kB*T/m) |
| `NoInflow` | `none` | `[Foundation-12] src/lagrangian/DSMC/submodels/InflowBoundaryModel/NoInflow/NoInflow.H` | Inserts no particles. |  |

### DSMC parcel  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dsmcParcel / DSMCParcel` | `DSMCParcel` | `[Foundation-12] src/lagrangian/DSMC/parcels/Templates/DSMCParcel/DSMCParcel.H` | DSMC parcel carrying microscopic velocity U, internal energy Ei and typeId; moved ballistically between stochastic collisions. | x_{n+1} = x_n + U*dt free flight, then in-cell stochastic collisions |

### DSMC wall interaction base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `WallInteractionModel` | `WallInteractionModel` | `[Foundation-12] src/lagrangian/DSMC/submodels/WallInteractionModel/WallInteractionModel/WallInteractionModel.H` | Base for DSMC particle-wall (gas-surface accommodation) interaction. |  |

### DSMC wall interaction model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MaxwellianThermal` | `MaxwellianThermal` | `[Foundation-12] src/lagrangian/DSMC/submodels/WallInteractionModel/MaxwellianThermal/MaxwellianThermal.H` | Fully diffuse wall: post-collision velocity drawn from a Maxwellian at the local wall temperature plus the wall velocity. | Un' = sqrt(-2*kB*Tw/m*ln(rand)); Ut' ~ N(0, sqrt(kB*Tw/m)); internal energy resampled at Tw |
| `MixedDiffuseSpecular` | `MixedDiffuseSpecular` | `[Foundation-12] src/lagrangian/DSMC/submodels/WallInteractionModel/MixedDiffuseSpecular/MixedDiffuseSpecular.H` | Blends diffuse (Maxwellian) and specular reflection with a prescribed diffuse fraction. | With probability diffuseFraction apply MaxwellianThermal, otherwise SpecularReflection |
| `SpecularReflection` | `SpecularReflection` | `[Foundation-12] src/lagrangian/DSMC/submodels/WallInteractionModel/SpecularReflection/SpecularReflection.H` | Fully specular (mirror) wall reflection - reverses the wall-normal velocity component. | U' = U - 2*(U & n)*n |

### MD electrostatics  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `electrostaticPotential` |  | `[Foundation-12] src/lagrangian/molecularDynamics/potential/electrostaticPotential/electrostaticPotential.H` | Tabulated Coulomb energy/force between charged sites, shared by the pairPotentialList. | U(r) = 1/(4*pi*eps0*r); F(r) = 1/(4*pi*eps0*r^2) |

### MD energy scaling base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `energyScalingFunction` | `energyScalingFunction` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/energyScalingFunction/basic/energyScalingFunction.H` | Base for functions that modify a pair potential near the cut-off radius. |  |

### MD energy scaling function  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `energyScalingFunctions::doubleSigmoid` | `doubleSigmoid` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/energyScalingFunction/derived/doubleSigmoid/doubleSigmoid.H` | Multiplies the potential by the product of two sigmoid switching functions (inner and outer switch). | e' = e*S(r; shift1, scale1)*S(r; shift2, scale2) |
| `energyScalingFunctions::noScaling` | `noScaling` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/energyScalingFunction/derived/noScaling/noScaling.H` | Leaves the potential unmodified (simple truncation at rCut). | e' = e |
| `energyScalingFunctions::shifted` | `shifted` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/energyScalingFunction/derived/shifted/shifted.H` | Shifted-potential truncation so the energy is continuous at rCut. | e' = e(r) - e(rCut) |
| `energyScalingFunctions::shiftedForce` | `shiftedForce` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/energyScalingFunction/derived/shiftedForce/shiftedForce.H` | Shifted-force truncation so both energy and force are continuous at rCut. | e' = e(r) - (e(rCut) + de/dr\|rCut*(r - rCut)) |
| `energyScalingFunctions::sigmoid` | `sigmoid` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/energyScalingFunction/derived/sigmoid/sigmoid.H` | Multiplies the potential by a sigmoid switching function. | e' = e*0.5*(1 - tanh(scale*(r - shift))) |

### MD pair potential  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pairPotentials::azizChen` | `azizChen` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/azizChen/azizChen.H` | Aziz-Chen (1977) accurate argon intermolecular potential. | U(x) = epsilon*(A*x^gamma*exp(-alpha*x) - F(x)*(C6/x^6 + C8/x^8 + C10/x^10)), x = r/rm, F = exp(-((D/x)-1)^2) for x<D else 1 |
| `pairPotentials::coulomb` | `coulomb` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/coulomb/coulomb.H` | Bare electrostatic (Coulomb) site-site potential. | U(r) = 1/(4*pi*eps0*r) |
| `pairPotentials::dampedCoulomb` | `dampedCoulomb` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/dampedCoulomb/dampedCoulomb.H` | Screened / damped Coulomb potential. | U(r) = erfc(alpha*r)/(4*pi*eps0*r) |
| `pairPotentials::exponentialRepulsion` | `exponentialRepulsion` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/exponentialRepulsion/exponentialRepulsion.H` | Purely repulsive exponential potential. | U(r) = epsilon*exp(-r/rm) |
| `pairPotentials::lennardJones` | `lennardJones` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/lennardJones/lennardJones.H` | 12-6 Lennard-Jones site-site potential. | U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6) |
| `pairPotentials::maitlandSmith` | `maitlandSmith` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/maitlandSmith/maitlandSmith.H` | Maitland-Smith n(r)-6 simplified intermolecular potential (1973; parameters for other monoatomics from Maitland et al. 1981). | n(r) = m + gamma*(r/rm - 1); U(r) = epsilon*((6/(n-6))*(r/rm)^-n - (n/(n-6))*(r/rm)^-6) |
| `pairPotentials::noInteraction` | `noInteraction` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/derived/noInteraction/noInteraction.H` | Zero pair interaction. | U(r) = 0 |

### MD pair potential base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pairPotential` | `pairPotential` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/basic/pairPotential.H` | Base for site-site pair potentials; builds tabulated force/energy lookup tables between rMin and rCut and applies an energyScalingFunction. | energy(r) = scaled(unscaledEnergy(r)); force(r) = -d(energy)/dr, both tabulated |

### MD potential container  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pairPotentialList` |  | `[Foundation-12] src/lagrangian/molecularDynamics/potential/pairPotential/pairPotentialList/pairPotentialList.H` | Holds one pairPotential per site-pair combination plus the electrostatic potential; determines rCutMax and the pair lookup index. |  |
| `tetherPotentialList` |  | `[Foundation-12] src/lagrangian/molecularDynamics/potential/tetherPotential/tetherPotentialList/tetherPotentialList.H` | Holds one tetherPotential per tethered site id. |  |

### MD solver includes  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `mdTools` |  | `[Foundation-12] src/lagrangian/molecularDynamics/mdTools` | Header include set for MD solvers: createMDFields, calculateMDFields, averageMDFields, resetMDFields, createRefUnits, meanMomentumEnergyAndNMols, temperatureAndPressure(+Variables), temperatureEquilibration, createAutoCorrelationFunctions, calculateAutoCorrelationFunctions, calculateTransportProperties. | Instantaneous T = 2*KE_translational/(3*N*kB); p = (N*kB*T + virial/3)/V; velocity rescaling for temperature equilibration |

### MD statistics  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `correlationFunction / bufferedAccumulator` |  | `[Foundation-12] src/lagrangian/molecularDynamics/correlationFunction/correlationFunction.H` | Time auto-correlation function machinery with a buffered accumulator, used to compute transport properties (diffusion, viscosity, thermal conductivity) via Green-Kubo integrals. | C(tau) = <A(t).A(t+tau)>; transport coefficient = integral of C(tau) dtau |

### MD tether potential  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `tetherPotentials::harmonicSpring` | `harmonicSpring` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/tetherPotential/derived/harmonicSpring/harmonicSpring.H` | Linear harmonic spring tether. | U = 0.5*k*\|r\|^2; F = -k*r |
| `tetherPotentials::pitchForkRing` | `pitchForkRing` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/tetherPotential/derived/pitchForkRing/pitchForkRing.H` | Pitchfork-bifurcation potential with a circular minimum of radius rOrbit in the x-y plane and harmonic confinement in z. | U = -0.5*mu*(p - rOrbit)^2 + 0.25*(p - rOrbit)^4 + 0.5*alpha*z^2 with p = sqrt(x^2 + y^2) |
| `tetherPotentials::restrainedHarmonicSpring` | `restrainedHarmonicSpring` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/tetherPotential/derived/restrainedHarmonicSpring/restrainedHarmonicSpring.H` | Harmonic spring that becomes linear (constant force) beyond a restraint radius rR. | \|r\|<rR: U = 0.5*k*\|r\|^2; else U = 0.5*k*rR^2 + k*rR*(\|r\| - rR), F = -k*rR*rhat |

### MD tether potential base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `tetherPotential` | `tetherPotential` | `[Foundation-12] src/lagrangian/molecularDynamics/potential/tetherPotential/basic/tetherPotential.H` | Base for potentials tethering a molecule site to a fixed lattice position (e.g. solid walls). |  |

### MD unit system  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `reducedUnits` |  | `[Foundation-12] src/lagrangian/molecularDynamics/reducedUnits/reducedUnits.H` | Conversion between SI and Lennard-Jones reduced units (refLength, refTime, refMass and all derived reference quantities). | refEnergy = refMass*refLength^2/refTime^2; refTemp = refEnergy/kB; refPressure = refMass/(refLength*refTime^2) |

### MPPIC averaging base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `AveragingMethod` | `averagingMethod (AveragingMethod<scalar> and AveragingMethod<vector>)` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/AveragingMethods/AveragingMethod/AveragingMethod.H` | Base for mapping Lagrangian point quantities onto the Eulerian mesh and back; templated on scalar and vector with separate RTS tables. |  |

### MPPIC averaging method  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `AveragingMethods::Basic` | `basic` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/AveragingMethods/Basic/Basic.H` | Cell-volume based averaging: point values summed over cells and divided by the cell volume; piecewise-constant interpolation with fvc::grad gradients. | phi_cell = sum(w_p)/V_cell |
| `AveragingMethods::Dual` | `dual` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/AveragingMethods/Dual/Dual.H` | Dual-mesh (cell + point-tet) averaging using the tetrahedral decomposition; linear interpolation across the tet with first-order finite-element gradients. | Barycentric weights within the tet distribute a point value between the cell sum and the point (dual-cell) sum |

### MPPIC correction limiter  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CorrectionLimitingMethods::absolute` | `absolute` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/CorrectionLimitingMethods/absolute/absolute.H` | Limits the correction to a rebound with restitution coefficient e, using the absolute particle velocity for the magnitude and the relative velocity for the direction. | \|dU\| <= (1+e)*\|u\|, direction from (u - uMean) |
| `CorrectionLimitingMethods::noCorrectionLimiting` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/CorrectionLimitingMethods/noCorrectionLimiting/noCorrectionLimiting.H` | No limiting of the packing velocity correction. |  |
| `CorrectionLimitingMethods::relative` | `relative` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/CorrectionLimitingMethods/relative/relative.H` | Limits the correction to a rebound with restitution coefficient e using the velocity relative to the local mean for both magnitude and direction. | \|dU\| <= (1+e)*\|u - uMean\| |

### MPPIC correction limiter base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CorrectionLimitingMethod` | `correctionLimitingMethod` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/CorrectionLimitingMethods/CorrectionLimitingMethod/CorrectionLimitingMethod.H` | Base for limiting the velocity correction applied by the explicit packing model. |  |

### MPPIC damping base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DampingModel` | `dampingModel` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/DampingModels/DampingModel/DampingModel.H` | Base for collisional damping of parcel velocity fluctuations in MP-PIC. |  |

### MPPIC damping model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DampingModels::NoDamping` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/DampingModels/NoDamping/NoDamping.H` | No collisional damping. |  |
| `DampingModels::Relaxation` | `relaxation` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/DampingModels/Relaxation/Relaxation.H` | Relaxes particle velocities towards the local mean over a time scale from a TimeScaleModel (O'Rourke & Snider 2010). | du/dt = -(u - uMean)/tau with tau from the selected timeScaleModel |

### MPPIC inter-particle stress base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ParticleStressModel` | `particleStressModel` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/ParticleStressModels/ParticleStressModel/ParticleStressModel.H` | Base for the solids-pressure (inter-particle stress) closure tau_p(alpha, rho, sigma) and its derivative with respect to alpha. |  |

### MPPIC inter-particle stress model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ParticleStressModels::HarrisCrighton` | `HarrisCrighton` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/ParticleStressModels/HarrisCrighton/HarrisCrighton.H` | Harris & Crighton (1994) solids pressure with a numerical-stability floor on the denominator. | tau = Ps*alpha^beta / max(alphaPacked - alpha, eps*(1 - alpha)) |
| `ParticleStressModels::Lun` | `Lun` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/ParticleStressModels/Lun/Lun.H` | Kinetic-theory granular solids pressure of Lun et al. (1984) with restitution coefficient e. | tau = (alpha*rho + alpha^2*rho*(1+e)*(3/5)*(1 - (alpha/alphaPacked)^(1/3)))*(1/3)*sigma^2 |
| `ParticleStressModels::exponential` | `exponential` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/ParticleStressModels/exponential/exponential.H` | Exponential solids-pressure closure of the same form used in the Euler-Euler (twoPhaseEuler) solvers. | tau ~ g0*exp(preAlphaExp*(alpha - alphaPacked)), limited by expMax |

### MPPIC isotropy model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `IsotropyModels::NoIsotropy` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/IsotropyModels/NoIsotropy/NoIsotropy.H` | No return-to-isotropy. |  |
| `IsotropyModels::Stochastic` | `stochastic` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/IsotropyModels/Stochastic/Stochastic.H` | Stochastic return-to-isotropy (O'Rourke & Snider 2012): samples a Gaussian-plus-delta distribution so some parcel velocities are randomised, then corrects to conserve momentum and granular temperature. | P(resample) = 1 - exp(-dt/tau); resampled u ~ uMean + N(0, sigma) with sigma^2 the granular temperature; global momentum and Theta conservation correction applied |

### MPPIC packing base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PackingModel` | `packingModel` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/PackingModels/PackingModel/PackingModel.H` | Base for applying the inter-particle stress to parcels; owns the ParticleStressModel, the correction limiter and the averaging methods. | du/dt = -(1/(alpha_p*rho_p))*grad(tau_p) |

### MPPIC packing model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PackingModels::Explicit` | `explicit` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/PackingModels/Explicit/Explicit.H` | Explicit inter-particle stress (Snider 2001) evaluated at current particle positions and applied only to particles moving towards close pack, with the correction velocity limited. | dU = -dt*grad(tau)/(alpha_p*rho_p) applied when the particle moves into the packing gradient; magnitude limited by the correctionLimitingMethod |
| `PackingModels::Implicit` | `implicit` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/PackingModels/Implicit/Implicit.H` | Solves the particulate volume-fraction transport implicitly on the Eulerian mesh and maps the resulting flux back onto the parcels; can optionally apply gravity here (applyGravity). | ddt(alpha_p) + div(phi_p) - laplacian(D_tau, alpha_p) = 0 with D_tau from dtau/dalpha; parcel velocity corrected by the resulting flux |
| `PackingModels::NoPacking` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/PackingModels/NoPacking/NoPacking.H` | No inter-particle stress. |  |

### MPPIC return-to-isotropy base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `IsotropyModel` | `isotropyModel` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/IsotropyModels/IsotropyModel/IsotropyModel.H` | Base for collisional return-to-isotropy of the parcel velocity distribution. |  |

### MPPIC time-scale base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `TimeScaleModel` | `timeScaleModel` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/TimeScaleModels/TimeScaleModel/TimeScaleModel.H` | Base for the collisional relaxation time scale used by the damping and isotropy models. |  |

### MPPIC time-scale model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `TimeScaleModels::equilibrium` | `equilibrium` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/TimeScaleModels/equilibrium/equilibrium.H` | Equilibrium collisional exchange time scale (O'Rourke, Zhao & Snider 2009). | 1/tau proportional to (alpha/alphaPacked)*(12/d)*sqrt(2*Theta/(3*pi)), e-weighted collision frequency |
| `TimeScaleModels::isotropic` | `isotropic` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/TimeScaleModels/isotropic/isotropic.H` | Time scale over which the dispersed-phase velocity field returns to an isotropic distribution (O'Rourke & Snider 2012). | tau_iso from the collision frequency and the restitution coefficient e |
| `TimeScaleModels::nonEquilibrium` | `nonEquilibrium` | `[Foundation-12] src/lagrangian/parcel/submodels/MPPIC/TimeScaleModels/nonEquilibrium/nonEquilibrium.H` | Improved (non-equilibrium) collision damping time for dense polydisperse flows (O'Rourke & Snider 2010). | Modified collision frequency including the mean relative velocity as well as the granular temperature |

### VoF solution controls (solver include)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `alphaControls.H` | `solvers { alpha.<phase> { nAlphaCorr; nAlphaSubCycles; MULESCorr; alphaApplyPrevCorr; } }` | `[Foundation-12] src/twoPhaseModels/VoF/alphaControls.H` | Reads the MULES phase-fraction solution controls from the alpha solver dict. | Entries: nAlphaCorr, nAlphaSubCycles, MULESCorr (semi-implicit MULES), alphaApplyPrevCorr |
| `alphaCourantNo.H` |  | `[Foundation-12] src/twoPhaseModels/VoF/alphaCourantNo.H` | Computes the mean/max interface Courant number restricted to near-interface cells. | alphaCoNum = 0.5*max(nearInterface*surfaceSum(\|phi\|)/V)*deltaT |
| `setDeltaT.H (VoF)` |  | `[Foundation-12] src/twoPhaseModels/VoF/setDeltaT.H` | Adjustable time-step calculation using both the flow and the interface (alpha) Courant numbers. | deltaT = min(maxCo/CoNum, maxAlphaCo/alphaCoNum)*deltaT0, clipped by maxDeltaT and a growth limit |

### acoustic post-processing  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `noiseFFT` |  | `[Foundation-12] src/randomProcesses/noise/noiseFFT.H` | FFT of a pressure-time history producing pressure spectra, PSD, one-third-octave bands and dB / dBA weighted levels. | P(f) from FFT of p(t); SPL = 20*log10(p_rms/pRef); PSD = \|P\|^2/df; octave-band summation and A-weighting |

### boundary condition (alpha)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `contactAngleFvPatchScalarField` | `contactAngle` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/contactAngle/contactAngleFvPatchScalarField.H` | General alpha contact-angle wall BC; hosts a contactAngleModel and applies a limiter to the wall gradient of alpha1. | grad(alpha1)\|wall = (nf & nHat)*\|grad(alpha)_f\|; limiter options none / gradient / alpha / zeroGradient |

### boundary condition (alpha, multiphase)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `alphaContactAngleFvPatchScalarField (multiphase)` | `alphaContactAngle` | `[Foundation-12] src/multiphaseModels/multiphaseProperties/alphaContactAngle/alphaContactAngleFvPatchScalarField.H` | Multi-phase contact-angle BC; a contactAngleProperties sub-dict holds per-other-phase theta0 and optional dynamic entries uTheta/thetaA/thetaR. | theta = theta0 + (thetaA-theta0)*max(uwall/uTheta,0) + (thetaR-theta0)*min(uwall/uTheta,0) per phase pair |

### boundary condition (generic, templated)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveInletOutletFvPatchField` | `waveInletOutlet` | `[Foundation-12] src/waves/derivedFvPatchFields/waveInletOutlet/waveInletOutletFvPatchField.H` | Inlet-outlet BC with different inlet values above and below the wave interface (e.g. for k, epsilon, omega, T). | Inflow: value = levelSetAverage(inletValueBelow, inletValueAbove); outflow: zero gradient |

### boundary condition (p_rgh)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `alphaFixedPressureFvPatchScalarField` | `alphaFixedPressure` | `[Foundation-12] src/twoPhaseModels/twoPhaseProperties/alphaFixedPressure/alphaFixedPressureFvPatchScalarField.H` | Fixed-pressure companion BC for alphaContactAngle/contactAngle walls; sets p_rgh from a fixed p with the hydrostatic contribution removed. | p_rgh = p - rho*(g & (h - hRef)) |

### boundary condition (wave alpha)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveAlphaFvPatchScalarField` | `waveAlpha` | `[Foundation-12] src/waves/derivedFvPatchFields/waveAlpha/waveAlphaFvPatchScalarField.H` | Sets phase fraction on a patch from the registered waveSuperposition using a level-set/sub-cell integration of the wave surface across each face; inlet-outlet behaviour based on phi. | alpha_p = levelSetFraction(eta_wave - z) per face; negated when liquid=false |

### boundary condition (wave velocity)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveVelocityFvPatchVectorField` | `waveVelocity` | `[Foundation-12] src/waves/derivedFvPatchFields/waveVelocity/waveVelocityFvPatchVectorField.H` | Sets patch velocity from the registered waveSuperposition with level-set blending of liquid and gas velocities; zero-gradient on outflow. | U_p = levelSetAverage(U_liquid_wave, U_gas_wave) about the wave surface |

### cavitation base class / runtime-selection table  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cavitationModel (compressible)` | `selected by 'model <name>;'` | `[Foundation-12] src/twoPhaseModels/compressibleCavitation/cavitationModel` | Abstract base for compressible two-phase (rhoFluidThermo per phase) cavitation models; supplies pSatl/pSatv from a saturationPressureModel. | Returns Pair<mDotcvAlphal()> (alpha-linearised) and Pair<mDotcvP()> (p-linearised) condensation/vaporisation rates so both alpha and p equations get implicit contributions. |
| `cavitationModel (incompressible)` | `selected by 'model <name>;' in the cavitation dict; coeffs in <model>Coeffs` | `[Foundation-12] src/twoPhaseModels/incompressibleCavitation/cavitationModel` | Abstract base for incompressible VoF cavitation mass-transfer models; owns the saturationPressureModel and the two phases. | Provides Pair<mDotcvAlphal> and Pair<mDotcvP>: condensation/vaporisation rates split into alpha-implicit and p-implicit parts, mDot = f(p - pSat); alpha equation source (mDotvAlphal - mDotcAlphal) and pressure equation source. |

### cavitation model  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Kunz (compressible)` | `Kunz` | `[Foundation-12] src/twoPhaseModels/compressibleCavitation/Kunz/Kunz.H` | Kunz cavitation model for compressible two-phase VoF (rhol, rhov, thermo per phase). | Vaporisation ~ Cv*rhov*alphal^2*(1-alphal)/(0.5*rhol*UInf^2*tInf); condensation ~ Cc*rhov*alphal*max(p-pSatv,0)/(0.5*rhol*UInf^2*tInf) |
| `Kunz (incompressible)` | `Kunz` | `[Foundation-12] src/twoPhaseModels/incompressibleCavitation/Kunz/Kunz.H` | Kunz cavitation model for constant-density two-phase VoF; condensation term switched off below pSat so it can be implicit in p. | mDotv = Cv*rhov*alphal^2*(1-alphal)/(0.5*rhol*UInf^2*tInf); mDotc = Cc*rhov*alphal*max(p-pSat,0)/(0.5*rhol*UInf^2*tInf) |
| `Merkle (compressible)` | `Merkle` | `[Foundation-12] src/twoPhaseModels/compressibleCavitation/Merkle/Merkle.H` | Merkle cavitation model for compressible two-phase VoF. | mcCoeff = Cc/(0.5*UInf^2*tInf); mvCoeff = Cv*rhov/(0.5*rhol*UInf^2*tInf); rates proportional to max/min(p - pSat, 0) |
| `Merkle (incompressible)` | `Merkle` | `[Foundation-12] src/twoPhaseModels/incompressibleCavitation/Merkle/Merkle.H` | Merkle sheet-cavitation mass transfer model for incompressible VoF. | mDotc = Cc*alphal*max(p-pSat,0)/(0.5*UInf^2*tInf); mDotv = Cv*rhov/rhol*(1-alphal)*min(p-pSat,0)/(0.5*UInf^2*tInf) |
| `Saito (compressible)` | `Saito` | `[Foundation-12] src/twoPhaseModels/compressibleCavitation/Saito/Saito.H` | Saito cloud-cavitation model based on interfacial-area concentration and kinetic (Hertz-Knudsen) evaporation/condensation; compressible-only. | Rate ~ Ca*alphal*(1-alphal)*fT(thermo)*(p - pSat), with fT = 1/sqrt(2*pi*R*T); alphaNuc nucleation-site volume fraction added on the vapour side |
| `SchnerrSauer (compressible)` | `SchnerrSauer` | `[Foundation-12] src/twoPhaseModels/compressibleCavitation/SchnerrSauer/SchnerrSauer.H` | Schnerr-Sauer bubble-number-density cavitation model for compressible two-phase VoF. | mDotc = Cc*alphal_lim*pCoeff(p,pSatv)*max(p-pSatv,p0); mDotv = -Cv*(1+alphaNuc-alphal_lim)*pCoeff(p,pSatl)*min(p-pSatl,p0); pCoeff from Rayleigh bubble growth with R from n and dNuc |
| `SchnerrSauer (incompressible)` | `SchnerrSauer` | `[Foundation-12] src/twoPhaseModels/incompressibleCavitation/SchnerrSauer/SchnerrSauer.H` | Bubble-number-density (Rayleigh-Plesset based) cavitation model for incompressible VoF. | R = (3/(4*pi*n))^(1/3)*(alphav/alphal)^(1/3); pCoeff = 3*rhol*rhov/rho*(1/R)*sqrt(2/(3*rhol))/sqrt(\|p-pSat\|); mDot = Cc or Cv weighted * pCoeff * (p - pSat) |

### cloud function object  <sub>(13)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FacePostProcessing` | `facePostProcessing` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/FacePostProcessing/FacePostProcessing.H` | Records accumulated parcel mass and average mass flux crossing user-specified face zones. | mass = sum(nParticle*m) over the zone; massFlux = mass/totalTime |
| `MassFlux (class in Flux.H)` | `massFlux` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/Flux/Flux.H` | Generates the surface field of particle mass flux through faces. | phi_m = sum(nParticle*m_p)/dt per face |
| `NumberFlux (class in Flux.H)` | `numberFlux` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/Flux/Flux.H` | Generates the surface field of particle number flux through faces. | phi_n = sum(nParticle)/dt per face |
| `ParticleCollector` | `particleCollector` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/ParticleCollector/ParticleCollector.H` | Collects parcel mass and mass flow rate over user-defined polygons or a concentric-circle arrangement; parcels can be removed on collection. | Per-polygon mass and mDot accumulation; modes 'polygon' and 'concentricCircle'; negateParcelsOppositeNormal option |
| `ParticleErosion` | `particleErosion` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/ParticleErosion/ParticleErosion.H` | Creates a field of eroded volume Q on specified patches using the Finnie et al. ductile erosion model (review by Yadav et al.). | Q = m*U^2/(p*psi*K)*f(a), f(a) = sin(2a) - 3*sin^2(a) for tan(a)<=K/6, else (K/6)*cos^2(a) |
| `ParticleTracks` | `particleTracks` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/ParticleTracks/ParticleTracks.H` | Records the full parcel state at each face crossing to build particle tracks for post-processing. |  |
| `ParticleTrap` | `particleTrap` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/ParticleTrap/ParticleTrap.H` | Traps parcels within a given phase fraction for multi-phase cases (e.g. keeps droplets in the liquid). | Active where alpha < threshold; parcel velocity corrected along -grad(alpha) to push it back into the phase |
| `PatchCollisionDensity` | `patchCollisionDensity` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/PatchCollisionDensity/PatchCollisionDensity.H` | Generates patch fields of the number and mass of collisions per unit area and their rates, optionally above a minimum impact speed. | n'' = sum(nParticle)/A; m'' = sum(nParticle*m)/A; rates by division by the accumulated time |
| `PatchPostProcessing` | `patchPostProcessing` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/PatchPostProcessing/PatchPostProcessing.H` | Writes the state of every parcel hitting the specified patches (standard patch post-processing). |  |
| `RelativeVelocity` | `relativeVelocity` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/RelativeVelocity/RelativeVelocity.H` | Creates a Lagrangian field of each parcel's velocity relative to the surrounding fluid. | Urel = u - Uc(x_p) |
| `SizeDistribution` | `sizeDistribution` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/SizeDistribution/SizeDistribution.H` | Produces graphs of the cloud size distribution over nPoints bins in a chosen setFormat. | Number/volume histogram of parcel diameters |
| `VolumeFlux (class in Flux.H)` | `volumeFlux` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/Flux/Flux.H` | Generates the surface field of particle volume flux through faces. | phi_V = sum(nParticle*V_p)/dt per face |
| `VolumeFraction` | `volumeFraction` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/VolumeFraction/VolumeFraction.H` | Creates the particle volume fraction field on the carrier phase mesh. | alpha_p = sum(nParticle*V_p)/V_cell |

### cloud function object base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CloudFunctionObject` | `cloudFunctionObject; listed in the cloudFunctions sub-dict` | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/CloudFunctionObject/CloudFunctionObject.H` | Templated base for post-processing hooks called at preEvolve, postEvolve, postMove, postPatch and postFace. |  |

### cloud function object container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CloudFunctionObjectList` |  | `[Foundation-12] src/lagrangian/parcel/submodels/CloudFunctionObjects/CloudFunctionObjectList/CloudFunctionObjectList.H` | Runtime-constructed list of cloud function objects for a cloud. |  |

### cloud solution controls  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cloudSolution` | `solution { ... } in constant/<cloud>Properties` | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/MomentumCloud/cloudSolution/cloudSolution.H` | Holds the 'solution' sub-dict of <cloud>Properties: active, coupled, transient/steady, cellValueSourceCorrection, maxCo, calcFrequency, maxTrackTime, source relaxation/semi-implicit switches and the interpolation schemes. | Source relaxation phi_new = phi_old + relax*(phi - phi_old); sourceTerms { schemes { U semiImplicit 1; } } |

### composition base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CompositionModel` | `compositionModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/CompositionModel/CompositionModel/CompositionModel.H` | Base for parcel composition: carrier species from the thermo package plus additional liquid and solid tables; provides per-phase cp, h, rho, L and species mapping to the carrier. | rho_mix = 1/sum(Y_i/rho_i); cp_mix = sum(Y_i*cp_i) |

### composition model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NoComposition` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/CompositionModel/NoComposition/NoComposition.H` | Dummy 'none' option; errors if composition data is requested. |  |
| `SinglePhaseMixture` | `singlePhaseMixture` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/CompositionModel/SinglePhaseMixture/SinglePhaseMixture.H` | Single-phase, multi-component parcel composition (e.g. a multi-component liquid droplet). |  |

### composition model (multiphase)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SingleMixtureFraction` | `singleMixtureFraction` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/CompositionModel/SingleMixtureFraction/SingleMixtureFraction.H` | Multi-phase, multi-component composition with a fixed overall gas/liquid/solid mass split; used for coal and biomass parcels. | Y = (YGasTot, YLiquidTot, YSolidTot) with per-phase component fractions |

### contact angle base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `contactAngleModel` | `contactAngle { type <model>; ... }` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/contactAngleModels/contactAngleModel` | Abstract base returning the cosine of the wall contact angle field used to reorient nHat at walls. | cos(theta) supplied to correctContactAngle so that nHat_wall satisfies the prescribed angle |

### contact angle model  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `contactAngleModels::constant` | `constant` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/contactAngleModels/constant/constantContactAngle.H` | Uniform equilibrium contact angle theta0. | cos(theta) = cos(theta0) |
| `contactAngleModels::dynamic` | `dynamic` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/contactAngleModels/dynamic/dynamicContactAngle.H` | Velocity-dependent dynamic contact angle limited between advancing and receding values. | theta = theta0 + (thetaAdv-theta0)*max(uwall/uTheta,0) + (thetaRec-theta0)*min(uwall/uTheta,0), uwall = wall-parallel velocity along the interface |
| `contactAngleModels::gravitational` | `gravitational` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/contactAngleModels/gravitational/gravitationalContactAngle.H` | Contact angle biased by the gravity component along the interface at the wall. | theta = theta0 + (thetaAdv-theta0)*max(g_par/gTheta,0) + (thetaRec-theta0)*min(g_par/gTheta,0) |
| `contactAngleModels::temperatureDependent` | `temperatureDependent` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/contactAngleModels/temperatureDependent/temperatureDependentContactAngle.H` | Contact angle from a Function1 of a looked-up temperature field. | theta = f(T) |

### data type  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `forceSuSp` |  | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/forceSuSp/forceSuSp.H` | Tuple of explicit vector Su and implicit scalar Sp used to accumulate particle forces. |  |

### devolatilisation base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DevolatilisationModel` | `devolatilisationModel` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/DevolatilisationModel/DevolatilisationModel/DevolatilisationModel.H` | Base for the release of volatile matter from solid fuel parcels; tracks when devolatilisation is complete so char oxidation may start. | dm_volatile/dt with a residual coal fraction criterion |

### devolatilisation model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ConstantRateDevolatilisation` | `constantRateDevolatilisation` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/DevolatilisationModel/ConstantRateDevolatilisation/ConstantRateDevolatilisation.H` | Constant-rate volatile release above a vaporisation temperature (typically 600 K). | dm_i/dt = A0*m_volatile0*Y_i for T > Tvap |
| `NoDevolatilisation` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/DevolatilisationModel/NoDevolatilisation/NoDevolatilisation.H` | Placeholder for the 'none' option. |  |
| `SingleKineticRateDevolatilisation` | `singleKineticRateDevolatilisation` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/DevolatilisationModel/SingleKineticRateDevolatilisation/SingleKineticRateDevolatilisation.H` | Per-species single-kinetic-rate (Arrhenius) devolatilisation; volatileData gives (name A1 E) per volatile. | kappa = A1*exp(-E/(R*T)); dm_i/dt = kappa*m_i,remaining |

### div scheme / surface interpolation scheme (VoF geometric)  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MPLIC` | `MPLIC (e.g. div(phi,alpha) Gauss MPLIC;)` | `[Foundation-12] src/twoPhaseModels/interfaceCompression/MPLIC/MPLIC.H` | Multicut PLIC using three progressively more complex cut algorithms (single cut of all cell faces, topological face-edge-face walk producing multiple sub-volumes, tetrahedron-decomposition cut) so the cell volume fraction is always reproduced. Needs no base scheme. | Root-find the cut-plane offset d such that Vol(cell intersect {n.x < d}) = alpha*V; alpha_f = wetted face-area fraction |
| `MPLICU` | `MPLICU` | `[Foundation-12] src/twoPhaseModels/interfaceCompression/MPLIC/MPLICU.H` | Velocity-weighted multicut PLIC; as MPLIC but uses face point velocities for the flux. | As MPLIC with point-velocity weighting of the face flux |
| `PLIC` | `PLIC (e.g. div(phi,alpha) Gauss PLIC interfaceCompression vanLeer 1;)` | `[Foundation-12] src/twoPhaseModels/interfaceCompression/PLIC/PLIC.H` | Piecewise-Linear Interface Calculation: single planar cut per cell matching the cell volume fraction; face alpha from the cut face areas, falling back to a specified default scheme where the single cut fails. | Find plane with normal from the point-interpolated alpha gradient and offset d such that Vol(cell n.x<d) = alpha*V; alpha_f = A_cut_side/A_face |
| `PLICU` | `PLICU` | `[Foundation-12] src/twoPhaseModels/interfaceCompression/PLIC/PLICU.H` | Velocity-weighted PLIC: as PLIC but the face flux is built from face point velocities, more accurate under high shear. | As PLIC, with alpha_f weighted by the point-velocity-resolved face flux distribution |

### div scheme / surface interpolation scheme (VoF)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `interfaceCompression (class interfaceCompressionNew)` | `interfaceCompression (e.g. div(phi,alpha) Gauss interfaceCompression vanLeer 1;)` | `[Foundation-12] src/twoPhaseModels/interfaceCompression/interfaceCompression/interfaceCompression.H` | Counter-gradient interface-compression corrected surface interpolation applied on top of a base NVD/TVD scheme (e.g. vanLeer, vanAlbada) to keep VoF interfaces sharp. | alpha_f = alpha_f,base + cAlpha*\|phi/\|Sf\|\|*alpha_f*(1-alpha_f) projected onto nHatf; cAlpha=1 typical |
| `noInterfaceCompression (class noInterfaceCompressionNew)` | `noInterfaceCompression (e.g. div(phi,alpha) Gauss noInterfaceCompression vanLeer;)` | `[Foundation-12] src/twoPhaseModels/interfaceCompression/noInterfaceCompression/noInterfaceCompression.H` | Wrapper forwarding to the base scheme with zero compression, so VoF solvers run efficiently without compression (e.g. cavitation cases). | alpha_f = alpha_f,base (no counter-gradient term) |

### function object (DSMC)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `functionObjects::dsmcFields` | `dsmcFields` | `[Foundation-12] src/lagrangian/functionObjects/dsmcFields/dsmcFields.H` | Converts averaged extensive DSMC fields into intensive fields UMean, translationalT, internalT and overallT. | translationalT = (linearKEMean - 0.5*rhoNMean*mass*\|UMean\|^2)/(1.5*kB*rhoNMean); internalT = internalEMean/(0.5*iDofMean*kB); overallT weighted by degrees of freedom |

### function object (lagrangian)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `functionObjects::cloudInfo` | `cloudInfo` | `[Foundation-12] src/lagrangian/functionObjects/cloudInfo/cloudInfo.H` | Writes total parcel count and total parcel mass for the listed clouds to a file. |  |
| `functionObjects::particles` | `particles` | `[Foundation-12] src/lagrangian/functionObjects/particles/particles.H` | Tracks a parcel cloud in a given incompressible velocity field without affecting the flow (one-way coupled post-processing tracking); rho from constant/phaseProperties, g from constant/g. | Cloud evolved each time step against the looked-up U field |
| `functionObjects::stopAtEmptyClouds` | `stopAtEmptyClouds` | `[Foundation-12] src/lagrangian/functionObjects/stopAtEmptyClouds/stopAtEmptyClouds.H` | Stops the run when all parcel clouds are empty; supported actions noWriteNow, writeNow, nextWrite (default). |  |

### fvModel (forced isotropic turbulence)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::OUForce` | `OUForce` | `[Foundation-12] src/randomProcesses/OUForce/OUForce.H` | Applies a random Ornstein-Uhlenbeck stochastic force to the momentum equation for DNS of forced isotropic turbulence in a periodic box; writes the energy spectrum at write times (useful for comparing LES SGS models). Serial-only, needs an isotropic power-of-2 mesh. | dF_k = -alpha*F_k*dt + sigma*sqrt(dt)*N(0,1) applied in wavenumber shells kLower<=\|k\|<=kUpper; force projected solenoidal via FFT |

### fvModel (lagrangian coupling)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::clouds` | `clouds` | `[Foundation-12] src/lagrangian/parcel/fvModels/clouds/clouds.H` | Adds any number of Lagrangian clouds to a single-phase solver, tracking particles and adding their momentum, energy and species sources to the Eulerian fields when solution/coupled is true. Reads constant/clouds for multiple cloud names and g from constant/g. | addSup contributes cloud SU(U), Sh(he), SYi(Yi) and Srho to the corresponding equations |

### fvModel (wave absorption)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::isotropicDamping` | `isotropicDamping` | `[Foundation-12] src/waves/fvModels/isotropicDamping/isotropicDamping.H` | Implicit relaxation of all velocity components towards a specified uniform value in an outlet zone to prevent wave reflection. | S(U) = -rho*forceCoeff*(U - value), added implicitly to the momentum equation |
| `fv::verticalDamping` | `verticalDamping` | `[Foundation-12] src/waves/fvModels/verticalDamping/verticalDamping.H` | Explicit damping of the gravity-aligned velocity component in an outlet zone. | d(m*u_z)/dt = -lambda*m*u_z => u_z = u_z0*exp(-lambda*t) |

### fvModel (wave generation/absorption)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::waveForcing` | `waveForcing` | `[Foundation-12] src/waves/fvModels/waveForcing/waveForcing.H` | Forces both the liquid phase fraction and all velocity components towards the current wave solution inside a forcing zone; supports a stronger coefficient adjacent to boundaries. | S(phi) = -forceCoeff*(phi - phi_wave) for alpha and U; lambda = lambdaCoeff*maxWaveSpeed/regionLength, lambdaBoundary from lambdaBoundaryCoeff |

### fvModel base (wave damping/forcing)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::forcing` | `(abstract; not directly selectable)` | `[Foundation-12] src/waves/fvModels/forcing/forcing.H` | Base fvModel providing the graded forcing-zone scale field built from origin(s)/direction(s) and a Function1 'scale', plus the forceCoeff (lambda) field and optional writeForceFields output. | forceCoeff = lambda*scale(x); scale graded along the specified line(s), e.g. halfCosineRamp; writes forcing:scale and forcing:forceCoeff |

### geometric VoF cutting algorithm (support classes)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MPLICcell / MPLICface / MPLICcellStorage` |  | `[Foundation-12] src/twoPhaseModels/interfaceCompression/MPLIC/MPLICcell.H` | Implementation of the cell/face cutting machinery for MPLIC/MPLICU: single-cut, topological multi-cut face-edge-face walk, tetrahedral-decomposition cut, plus per-cell geometric storage. | Volume-of-cut root find on the plane offset; face intersection polygon area accumulation |

### heat transfer base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `HeatTransferModel` | `heatTransferModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Thermodynamic/HeatTransferModel/HeatTransferModel/HeatTransferModel.H` | Base for parcel-carrier convective heat transfer; provides the Nusselt number and optional Bird correction for blowing. | htc = Nu*kappa_c/d; BirdCorrection modifies Nu at high mass-transfer rates |

### heat transfer model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NoHeatTransfer` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Thermodynamic/HeatTransferModel/NoHeatTransfer/NoHeatTransfer.H` | Placeholder for the 'none' option. | Nu = 0 |
| `RanzMarshall` | `RanzMarshall` | `[Foundation-12] src/lagrangian/parcel/submodels/Thermodynamic/HeatTransferModel/RanzMarshall/RanzMarshall.H` | Ranz-Marshall convective heat transfer correlation for a sphere. | Nu = 2 + 0.6*Re^0.5*Pr^(1/3) |

### injection base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `injectionModel (non-templated base)` |  | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/InjectionModel/injectionModel.H` | Non-templated base holding the injector name and common bookkeeping so injectors can be listed generically. |  |

### injection base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `InjectionModel` | `injectionModel; models listed in the 'injectors' sub-dict` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/InjectionModel` | Templated base for parcel injection; handles SOI, duration, mass/number flow rate, parcels-per-second, parcel basis (mass/number/fixed) and the size distribution model. | nParcels and injected mass per time step from massFlowRate(t) and parcelBasisType; nParticle per parcel follows from the basis |

### injection container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `InjectionModelList` |  | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/InjectionModel/InjectionModelList.H` | List of injection models, allowing multiple simultaneous injectors per cloud. |  |

### injection helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `patchInjectionBase` |  | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/PatchInjection/patchInjectionBase.H` | Shared machinery for patch-based injectors: triangulates patch faces, accumulates area fractions and selects a random injection point/face. | Face chosen with probability proportional to its triangulated area fraction |

### injection model  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CellZoneInjection` | `cellZoneInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/CellZoneInjection/CellZoneInjection.H` | Fills a cell zone at SOI with parcels at a specified effective number density and initial velocity; diameters from a distribution model. | nParticles = numberDensity*V(cellZone) |
| `ConeInjection` | `coneInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/ConeInjection/ConeInjection.H` | Injects into one or more cones from a point or a disc (inner/outer diameter), with constant, pressure-driven or flow-rate-and-discharge velocity. | U = Uconstant; or U = sqrt(2*(p_inj - p)/rho); or U = mDot/(rho*Cd*A) |
| `FieldActivatedInjection` | `fieldActivatedInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/FieldActivatedInjection/FieldActivatedInjection.H` | Injects at listed positions only when a reference field exceeds a threshold field; limited to nParcelsPerInjector injections per site. | Injects when factor*referenceField[celli] >= thresholdField[celli] |
| `ManualInjection` | `manualInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/ManualInjection/ManualInjection.H` | All parcels introduced at SOI at positions read from a positionsFile with uniform initial velocity; diameters from a distribution. |  |
| `MomentumLookupTableInjection` | `momentumLookupTableInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/MomentumLookupTableInjection/MomentumLookupTableInjection.H` | Injection sites read from a table, one row per site: (x y z) (u v w) d rho mDot. |  |
| `NoInjection` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/NoInjection/NoInjection.H` | Placeholder for the 'none' option. |  |
| `PatchFlowRateInjection` | `patchFlowRateInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/PatchFlowRateInjection/PatchFlowRateInjection.H` | Injects at a named patch using the local carrier flow rate to set concentration and velocity; parcels take the local flow velocity. | mDot_parcels = concentration*carrierVolumetricFlowRate through the patch |
| `PatchInjection` | `patchInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/InjectionModel/PatchInjection/PatchInjection.H` | Injects a total mass over a duration randomly across a named patch with specified initial velocity and volume flow rate. |  |

### injection model (multiphase reacting clouds)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ReactingMultiphaseLookupTableInjection` | `reactingMultiphaseLookupTableInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/InjectionModel/ReactingMultiphaseLookupTableInjection/ReactingMultiphaseLookupTableInjection.H` | Tabulated injection sites with gas/liquid/solid composition: (x y z) (u v w) d rho mDot T cp (Y0..Y2) (Yg) (Yl) (Ys). |  |

### injection model (reacting clouds)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ReactingLookupTableInjection` | `reactingLookupTableInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/InjectionModel/ReactingLookupTableInjection/ReactingLookupTableInjection.H` | Tabulated injection sites with composition: (x y z) (u v w) d rho mDot T cp (Y0..YN). |  |

### injection model (thermo clouds)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ThermoLookupTableInjection` | `thermoLookupTableInjection` | `[Foundation-12] src/lagrangian/parcel/submodels/Thermodynamic/InjectionModel/ThermoLookupTableInjection/ThermoLookupTableInjection.H` | Tabulated injection sites with thermal data: (x y z) (u v w) d rho mDot T cp. |  |

### interface capturing / CSF surface tension  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `interfaceProperties` |  | `[Foundation-12] src/twoPhaseModels/interfaceProperties/interfaceProperties.H` | Computes interface unit normal, curvature and the CSF surface tension force for two-phase VoF; corrects wall nHat for contact angle. | nHatf = (grad(alpha)_f/(\|grad(alpha)_f\|+deltaN)) & Sf; K = -div(nHatf); surfaceTensionForce = sigmaK*snGrad(alpha1); deltaN = 1e-8/avg(V)^(1/3) |

### interface capturing helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `correctContactAngle (multiphase)` |  | `[Foundation-12] src/multiphaseModels/multiphaseProperties/correctContactAngle/correctContactAngle.H` | Multiphase version of the wall-normal correction that reorients nHat on walls to reproduce the (optionally dynamic) contact angle between each phase pair. | nHat_wall rebuilt from cos(theta) via a = (b1 - a12*b2)/det, b = (b2 - a12*b1)/det with a12 = nHat & nf, det = 1 - a12^2 |

### lagrangian I/O  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `IOPosition` |  | `[Foundation-12] src/lagrangian/basic/IOPosition/IOPosition.H` | IO wrapper writing/reading the cloud positions/coordinates file (barycentric coordinates plus topology). |  |

### lagrangian ODE integration base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `integrationScheme` | `integrationSchemes { U Euler; T analytical; } in constant/<cloud>Properties` | `[Foundation-12] src/lagrangian/parcel/integrationScheme/integrationScheme/integrationScheme.H` | Base for schemes integrating the semi-implicit rate equation used by all parcel state variables; supports integration in stages when A and B are sums of contributions. | dphi/dt = A - B*phi; dphi = (A - B*phi^n)*dtEff(dt, B), dtEff defined per scheme |

### lagrangian ODE integration scheme  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `integrationSchemes::Euler` | `Euler` | `[Foundation-12] src/lagrangian/parcel/integrationScheme/Euler/Euler.H` | Euler-implicit integration of the semi-implicit rate equation. | dphi = (A - B*phi^n)*dt/(1 + B*dt); dtEff = dt/(1 + B*dt) |
| `integrationSchemes::analytical` | `analytical` | `[Foundation-12] src/lagrangian/parcel/integrationScheme/analytical/analytical.H` | Exact (exponential) integration of the semi-implicit rate equation over the time step. | dphi = (A - B*phi^n)*(1/B)*(1 - exp(-B*dt)); dtEff = (1 - exp(-B*dt))/B |

### lagrangian base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `particle` | `particle` | `[Foundation-12] src/lagrangian/basic/particle/particle.H` | Base Lagrangian particle: barycentric position within a tet, cell/tetFace/tetPt topology, and the barycentric tracking algorithm across faces and coupled patches. | Tracking solves for the fraction of the displacement to the next tet-face crossing using barycentric coordinates, applying a transformer for cyclic/processor transforms |

### lagrangian cloud  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `passiveParticleCloud` | `passiveParticleCloud` | `[Foundation-12] src/lagrangian/basic/passiveParticle/passiveParticleCloud.H` | Cloud of passiveParticles. |  |

### lagrangian cloud base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cloud / Cloud<ParticleType>` | `cloud` | `[Foundation-12] src/lagrangian/basic/Cloud/cloud.H` | Base registry object for a collection of Lagrangian particles; Cloud<T> adds IDLList storage, parallel transfer, the move() loop and mesh-change mapping. |  |

### lagrangian parallel interaction machinery  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `InteractionLists / referredWallFace` |  | `[Foundation-12] src/lagrangian/basic/InteractionLists/InteractionLists.H` | Builds direct (local) and referred (across cyclic/processor patches) cell interaction lists for pairwise particle interaction, with non-blocking send/receive of referred data and wall faces. | Cells interact when within the interaction distance; referred cells carry the coupling transform |

### lagrangian particle + cloud (simple)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `solidParticle / solidParticleCloud` |  | `[Foundation-12] src/lagrangian/solidParticle/solidParticle.H` | Simple solid spherical particle with one-way coupling to the continuous phase (libsolidParticle). | du/dt = 0.75*Cd*rhoc*\|Uc-u\|*(Uc-u)/(rhop*d) + g*(1-rhoc/rhop); Cd = 24/Re*(1+0.15*Re^0.687) for Re<1000 else 0.44 |

### lagrangian particle type  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `passiveParticle` |  | `[Foundation-12] src/lagrangian/basic/passiveParticle/passiveParticle.H` | Copy of the base particle with no additional state; used for tracking-only clouds (streamlines etc.). |  |

### model energy spectrum  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Ek` |  | `[Foundation-12] src/randomProcesses/turbulence/Ek.H` | Analytic model energy spectrum used to seed synthetic isotropic turbulence. | E(k) = Ea*(k/k0)^4*exp(-2*(k/k0)^2) |

### molecular dynamics particle + cloud  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `molecule / moleculeCloud` |  | `[Foundation-12] src/lagrangian/molecularDynamics/molecule/molecule.H` | Rigid multi-site molecule (position, velocity, orientation Q, angular momentum, site references) and its cloud with velocity-Verlet integration and pair/tether force evaluation via InteractionLists. | Velocity Verlet: v(t+dt/2) = v + a*dt/2; x(t+dt) = x + v(t+dt/2)*dt; a from the sum of pair, tether and electrostatic forces; rotation via the rotation-tensor scheme |

### molecular dynamics potential manager  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `potential` |  | `[Foundation-12] src/lagrangian/molecularDynamics/potential/potential/potential.H` | Reads potentialDict and constructs the pairPotentialList, tetherPotentialList, electrostatic potential, molecule id lists and site properties. |  |

### parcel cloud (selectable)  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `collidingCloud` | `collidingCloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/collidingCloud/collidingCloud.H` | Concrete cloud of colliding (DEM) parcels; registered in both viscosity and thermo tables. |  |
| `momentumCloud` | `cloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/momentumCloud/momentumCloud.H` | Concrete cloud of momentum parcels; registered in both the viscosity and thermo selection tables. |  |
| `mppicCloud` | `MPPICCloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/mppicCloud/mppicCloud.H` | Concrete cloud of MP-PIC parcels; registered in both viscosity and thermo tables. |  |
| `reactingCloud` | `reactingCloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/reactingCloud/reactingCloud.H` | Concrete cloud of evaporating multi-component parcels; thermo table only. |  |
| `reactingMultiphaseCloud` | `reactingMultiphaseCloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/reactingMultiphaseCloud/reactingMultiphaseCloud.H` | Concrete cloud of coal/biomass parcels with gas, liquid and solid phases; thermo table only. |  |
| `sprayCloud` | `sprayCloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/sprayCloud/sprayCloud.H` | Concrete cloud of spray parcels with atomisation and breakup; thermo table only. |  |
| `thermoCloud` | `thermoCloud` | `[Foundation-12] src/lagrangian/parcel/clouds/derived/thermoCloud/thermoCloud.H` | Concrete cloud of heated parcels; thermo selection table only. |  |

### parcel cloud base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `parcelCloudBase` | `parcelCloudBase` | `[Foundation-12] src/lagrangian/parcel/parcelCloud/parcelCloudBase.H` | Virtualises most cloud methods (sources, info, mesh-change hooks) at the base of the cloud template hierarchy. |  |

### parcel cloud base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `parcelCloud` | `type <cloudType>; in constant/<cloudName>Properties` | `[Foundation-12] src/lagrangian/parcel/parcelCloud/parcelCloud.H` | Virtual abstract base for all parcel clouds; owns two selection tables (viscosity-based for incompressible carriers, thermo-based for compressible), selected by the 'type' entry of constant/<cloudName>Properties. |  |

### parcel cloud container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `parcelCloudList` |  | `[Foundation-12] src/lagrangian/parcel/parcelCloud/parcelCloudList.H` | List of parcel clouds presenting a single-cloud interface; constructed by an fvModel. Cloud names read from constant/clouds. |  |

### parcel cloud container (mesh object)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `parcelClouds` |  | `[Foundation-12] src/lagrangian/parcel/parcelCloud/parcelClouds.H` | Mesh-object version of parcelCloudList with mesh-change hooks; constructed by a solver rather than an fvModel. |  |

### parcel cloud template  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CollidingCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/CollidingCloud/CollidingCloud.H` | Adds a deterministic pair/wall CollisionModel and sub-cycled collision time-stepping to a momentum cloud (DEM). | Collision sub-cycling with nSubCycles; angular momentum I*dOmega/dt = T from tangential contact forces |
| `MPPICCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/MPPICCloud/MPPICCloud.H` | Adds MP-PIC modelling (packing, damping, return-to-isotropy) to a momentum cloud for dense particle flow. | du/dt includes -(1/(alpha_p*rho_p))*grad(tau_p) from the inter-particle stress model plus damping and isotropy corrections |
| `MomentumCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/MomentumCloud/MomentumCloud.H` | Templated base cloud providing cloud function objects, the particle force list and dispersion/injection/patch-interaction/stochastic-collision/surface-film sub-models, plus the Eulerian momentum source. | Parcel: m*du/dt = Su + Sp*(Uc - u); Eulerian source SU = -sum(Su + Sp*u)/V |
| `ReactingCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/ReactingCloud/ReactingCloud.H` | Adds variable single-phase composition and phase change to the thermo cloud. | dm_i/dt from the PhaseChangeModel; species sources Srho(i) to the carrier with latent-heat coupling in the parcel enthalpy equation |
| `ReactingMultiphaseCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/ReactingMultiphaseCloud/ReactingMultiphaseCloud.H` | Adds gas/liquid/solid multiphase composition, devolatilisation and surface reactions to the reacting cloud (coal/biomass combustion). | dm_gas/dt (devolatilisation), dm_solid/dt (char oxidation) and liquid evaporation combined |
| `SprayCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/SprayCloud/SprayCloud.H` | Adds atomisation and secondary breakup sub-models to the reacting cloud for liquid fuel sprays. | Primary atomisation sets the initial SMD and liquid core length; secondary breakup evolves the parcel diameter d(t) |
| `ThermoCloud` |  | `[Foundation-12] src/lagrangian/parcel/clouds/Templates/ThermoCloud/ThermoCloud.H` | Adds heat transfer and radiation coupling to the momentum cloud. | m*cp*dT/dt = h*A*(Tc - T) + A*eps*(G/4 - sigma*T^4); Eulerian energy source Sh |

### parcel composition data  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `phaseProperties / phasePropertiesList` | `phase types: gas, liquid, solid` | `[Foundation-12] src/lagrangian/parcel/phaseProperties/phaseProperties/phaseProperties.H` | Per-phase (gas/liquid/solid/unknown) component names and mass fractions used by the composition models. |  |

### parcel template  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CollidingParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/CollidingParcel/CollidingParcel.H` | Wrapper adding collision force/torque accumulators and pair/wall collision records to a parcel. |  |
| `MPPICParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/MPPICParcel/MPPICParcel.H` | Wrapper adding the MP-PIC correction velocity UCorrect to a parcel. |  |
| `MomentumParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/MomentumParcel/MomentumParcel.H` | Momentum parcel with rotational motion (spherical) and one/two-way coupling; carries d, U, rho, nParticle, angularMomentum. | Integrates m*du/dt = Su + Sp*(Uc-u) with the selected integrationScheme; rotation I*dOmega/dt = T |
| `ReactingMultiphaseParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/ReactingMultiphaseParcel/ReactingMultiphaseParcel.H` | Adds YGas, YLiquid, YSolid composition arrays with devolatilisation and surface reaction to the reacting parcel. | dm/dt = dm_devol + dm_evap + dm_srf; canCombust state machine controls the sequence |
| `ReactingParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/ReactingParcel/ReactingParcel.H` | Adds mass fractions Y and mass transfer (phase change) to the thermo parcel. | dm_i/dt from the PhaseChangeModel; an enthalpy retention factor splits the latent heat between parcel and carrier |
| `SprayParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/SprayParcel/SprayParcel.H` | Adds atomisation/breakup state (d0, liquidCore, KHindex, y, yDot, tc, ms, injector) to the reacting parcel. | TAB oscillator: y'' + (5*mu_l/(rho_l*r^2))*y' + (8*sigma/(rho_l*r^3))*y = (2/3)*rho_g*u^2/(rho_l*r) |
| `ThermoParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/Templates/ThermoParcel/ThermoParcel.H` | Adds temperature, cp and heat transfer (with radiation) to the momentum parcel. | m*cp*dT/dt = htc*A*(Tc-T) + radiation, solved with the integrationScheme |

### parcel typedefs (concrete)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `momentumParcel / collidingParcel / mppicParcel / thermoParcel / reactingParcel / reactingMultiphaseParcel / sprayParcel` |  | `[Foundation-12] src/lagrangian/parcel/parcels/derived` | Concrete parcel typedefs composing the templates onto Foam::particle, one per selectable cloud type. |  |

### particle body force  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GravityForce` | `gravity` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Gravity/GravityForce.H` | Buoyancy-corrected gravity force on a parcel. | Su = mass*g*(1 - rho_c/rho_p) |
| `NonInertialFrameForce` | `nonInertialFrame` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/NonInertialFrame/NonInertialFrameForce.H` | Fictitious forces in a non-inertial reference frame (Landau & Lifshitz Mechanics p126-129): linear acceleration, Euler, centrifugal and Coriolis. | Su = -mass*(W + dOmega/dt x r + Omega x (Omega x r) + 2*Omega x u) |
| `ParamagneticForce` | `paramagnetic` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Paramagnetic/ParamagneticForce.H` | Magnetic force on a paramagnetic particle from the gradient of the magnetic field derived from an electric potential field. | Su = 3*V*mu0*chi/(chi+3)*grad(0.5*\|H\|^2) |
| `PressureGradientForce` | `pressureGradient` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/PressureGradient/PressureGradientForce.H` | Force from the carrier-phase pressure gradient / fluid acceleration along the particle path. | Su = mass*(rho_c/rho_p)*(DUc/Dt) |
| `VirtualMassForce` | `virtualMass` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/VirtualMass/VirtualMassForce.H` | Added-mass force from the accelerated carrier fluid displaced by the particle. | Su = Cvm*mass*(rho_c/rho_p)*(DUc/Dt - du/dt), Cvm default 0.5 |

### particle drag force  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NonSphereDragForce` | `nonSphereDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/NonSphereDrag/NonSphereDragForce.H` | Haider & Levenspiel (1989) drag for non-spherical particles with sphericity phi in (0,1]. | Cd = 24/Re*(1 + a*Re^b) + Re*c/(Re + d), with a,b,c,d functions of phi |
| `SchillerNaumannDragForce` | `SchillerNaumannDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/SchillerNaumannDrag/SchillerNaumannDragForce.H` | Schiller-Naumann (1935) sphere drag correlation. | CdRe = 24*(1 + 0.15*Re^0.687) for Re<=1000, else 0.44*Re |
| `SphereDragForce` | `sphereDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/SphereDrag/SphereDragForce.H` | Standard sphere drag (KIVA-II / Amsden, Butler & O'Rourke 1987). | CdRe = 24*(1 + (1/6)*Re^(2/3)) for Re<1000, else 0.424*Re; Sp = 0.75*muc*CdRe/d^2 * mass/rho_p |

### particle drag force (dense)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ErgunWenYuDragForce` | `ErgunWenYuDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/ErgunWenYuDrag/ErgunWenYuDragForce.H` | Blended Ergun (dense, alphac<0.8) / Wen-Yu (dilute) drag for fluidised beds (Gidaspow 1994). | alphac<0.8: CdRe = (4/3)*(150*(1-alphac)/(alphac*Re) + 1.75)*Re*alphac; else Wen-Yu |
| `PlessisMasliyahDragForce` | `PlessisMasliyahDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/PlessisMasliyahDrag/PlessisMasliyahDragForce.H` | Du Plessis & Masliyah (1988) porous-media-based dense drag. | CdRe built from the tortuosity/porosity functions A(alphac), B(alphac) of the consolidated isotropic porous-medium model |
| `WenYuDragForce` | `WenYuDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/WenYuDrag/WenYuDragForce.H` | Wen-Yu dilute/dense sphere drag with voidage correction (Gidaspow 1994). | CdRe = alphac*24*(1 + 0.15*(alphac*Re)^0.687)*alphac^-2.65 for alphac*Re<1000, else 0.44*alphac*Re*alphac^-2.65 |

### particle drag force (spray only)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DistortedSphereDragForce` | `distortedSphereDrag` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/DistortedSphereDrag/DistortedSphereDragForce.H` | Liu, Mather & Reitz (1993) drag for droplets distorted by the TAB oscillator; registered only for sprayCloud. | Cd = Cd_sphere*(1 + 2.632*y), y the TAB distortion parameter |

### particle drag force base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DenseDragForce` | `(abstract)` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Drag/DenseDrag/DenseDragForce.H` | Base for dense-flow drag models; caches and interpolates the carrier volume-fraction field alphac. |  |

### particle force base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ParticleForce` | `particleForce` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/ParticleForce/ParticleForce.H` | Abstract base for all particle forces; returns forceSuSp (explicit Su, implicit Sp) for coupled and non-coupled contributions. | F = Su + Sp*(Uc - u) |

### particle force container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ParticleForceList` | `particleForces { <forceName> { ... } }` | `[Foundation-12] src/lagrangian/parcel/submodels/ForceTypes/ParticleForceList/ParticleForceList.H` | Runtime-constructed list of ParticleForce models read from the 'particleForces' sub-dict of the cloud properties. | Total F = sum of each model's forceSuSp |

### particle force wrapper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ScaledForce` | `scaled` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Scaled/ScaledForce.H` | Wraps another particle force model and multiplies its contribution by a constant factor. | (Su, Sp) -> factor*(Su, Sp) |

### particle lift force  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SaffmanMeiLiftForce` | `SaffmanMeiLiftForce` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Lift/SaffmanMeiLift/SaffmanMeiLiftForce.H` | Saffman-Mei shear lift applicable to rigid spherical particles. | Cl = 6.46*f(Re, Rew)/(pi^1.5*sqrt(Rew)), f the Mei correction to Saffman's low-Re result |
| `TomiyamaLiftForce` | `TomiyamaLift` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Lift/TomiyamaLift/TomiyamaLiftForce.H` | Tomiyama lift for deformable bubbles including sign reversal at large Eotvos number. | Cl = min(0.288*tanh(0.121*Re), f(Eo')) for Eo'<4; f(Eo') for 4<=Eo'<=10; -0.27 for Eo'>10; f(Eo') = 0.00105*Eo'^3 - 0.0159*Eo'^2 - 0.0204*Eo' + 0.474 |

### particle lift force base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LiftForce` | `(abstract)` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/ParticleForces/Lift/LiftForce/LiftForce.H` | Base class for lift forces; caches the carrier velocity curl (vorticity). | F_L = Cl*(mass/rho_p)*rho_c*(Uc - u) x curl(Uc) |

### particle stochastic force (thermo clouds)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BrownianMotionForce` | `BrownianMotion` | `[Foundation-12] src/lagrangian/parcel/submodels/Thermodynamic/ParticleForces/BrownianMotion/BrownianMotionForce.H` | Brownian (thermal) random force on sub-micron particles after Li & Ahmadi (1992), including the Cunningham slip correction. | F_i = zeta_i*sqrt(pi*S0/dt), S0 = 216*mu*kB*T/(pi^2*rho_c*d^5*(rho_p/rho_c)^2*Cc); Cc = 1 + (2*lambda/d)*(1.257 + 0.4*exp(-1.1*d/(2*lambda))) |

### patch interaction base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PatchInteractionModel` | `patchInteractionModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/PatchInteractionModel/PatchInteractionModel/PatchInteractionModel.H` | Base for models defining what happens when a parcel hits a patch; interaction types none/rebound/stick/escape. |  |

### patch interaction model  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LocalInteraction` | `localInteraction` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/PatchInteractionModel/LocalInteraction/LocalInteraction.H` | Per-patch specification of the interaction type (rebound/stick/escape) with per-patch e and mu; optionally writes escaped/stuck mass fields. |  |
| `NoInteraction` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/PatchInteractionModel/NoInteraction/NoInteraction.H` | Dummy 'none' option; errors if a return value is required. |  |
| `Rebound` | `rebound` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/PatchInteractionModel/Rebound/Rebound.H` | Simple specular rebound off every patch. | U' = U - (1 + UFactor)*(U & n)*n |
| `StandardWallInteraction` | `standardWallInteraction (sub-entry type: rebound\|stick\|escape)` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/PatchInteractionModel/StandardWallInteraction/StandardWallInteraction.H` | Single interaction type applied to all walls: rebound (elasticity e, restitution mu), stick (zero velocity) or escape (removal). | rebound: Un' = -e*Un, Ut' = (1-mu)*Ut |

### phase change base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PhaseChangeModel` | `phaseChangeModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/PhaseChangeModel/PhaseChangeModel/PhaseChangeModel.H` | Base for parcel evaporation/condensation; provides the enthalpy retention coefficient controlling how the latent heat is split between parcel and carrier. | dm_i/dt with latent heat L(T); enthalpyTransfer options latentHeat / enthalpyDifference |

### phase change model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LiquidEvaporation` | `liquidEvaporation` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/PhaseChangeModel/LiquidEvaporation/LiquidEvaporation.H` | Diffusion-controlled evaporation of the liquid components using the ideal-gas assumption and a Sherwood-number mass transfer correlation. | dm_i/dt = -pi*d*Sh*Dab*rho_c*ln(1 + Bm); Sh = 2 + 0.6*Re^0.5*Sc^(1/3); surface mole fraction Xs from pv(T) of the liquid |
| `LiquidEvaporationBoil` | `liquidEvaporationBoil` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/PhaseChangeModel/LiquidEvaporationBoil/LiquidEvaporationBoil.H` | As LiquidEvaporation but adds a boiling regime (Zuo, Gomes & Rutland 2000) once the droplet reaches saturation temperature. | Sub-boiling as LiquidEvaporation; boiling: dd/dt = 4*kappa_c/(rho_l*cp_c*d)*(1 + 0.23*sqrt(Re))*ln(1 + cp_c*(Tc - T)/hv) |
| `NoPhaseChange` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Reacting/PhaseChangeModel/NoPhaseChange/NoPhaseChange.H` | Placeholder for the 'none' option. |  |

### scheme registry  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `compressionSchemes (wordHashSet)` |  | `[Foundation-12] src/twoPhaseModels/interfaceCompression/interfaceCompression/interfaceCompression.C` | Named set of the six interface-compression scheme keywords {interfaceCompression, noInterfaceCompression, PLIC, PLICU, MPLIC, MPLICU} used by solvers to detect a compression scheme. |  |

### spectral mesh  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Kmesh` |  | `[Foundation-12] src/randomProcesses/Kmesh/Kmesh.H` | Builds the wavenumber vector field corresponding to a uniform power-of-2 finite-volume mesh for FFT-based operations. | k_i = 2*pi*n_i/L_i with n_i wrapped to [-N/2, N/2) |

### spectral post-processing  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `kShellIntegration` |  | `[Foundation-12] src/randomProcesses/fft/kShellIntegration.H` | Integrates a spectral field over spherical shells in wavenumber space to produce the 1-D energy spectrum E(k). | E(k) = integral over \|k'\|=k of 0.5*\|u_hat(k')\|^2 dS |
| `writeEk` |  | `[Foundation-12] src/randomProcesses/fft/writeEk.H` | Computes and writes the turbulent kinetic energy spectrum E(k) of a velocity field on a Kmesh. | FFT of U followed by kShellIntegration |

### spectral transform  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fft` |  | `[Foundation-12] src/randomProcesses/fft/fft.H` | Multi-dimensional fast Fourier transform derived from Numerical Recipes in C; direction 1 = forward, -1 = reverse, dimensionality via an nn index array. | X_k = sum_n x_n*exp(-2*pi*i*k*n/N) |

### spectral transform helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fftRenumber` |  | `[Foundation-12] src/randomProcesses/fft/fftRenumber.H` | Renumbers (fftshift) a complex field so the zero wavenumber sits at the array centre. |  |

### spray atomisation base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `AtomisationModel` | `atomisationModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/AtomisationModel/AtomisationModel/AtomisationModel.H` | Base for primary breakup / atomisation of the injected liquid core. | Returns the liquid core length scale and the initial SMD of the primary parcels |

### spray atomisation model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BlobsSheetAtomisation` | `blobsSheetAtomisation` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/AtomisationModel/BlobsSheetAtomisation/BlobsSheetAtomisation.H` | Blobs-sheet primary breakup for pressure-swirl (hollow-cone) atomisers, after Han/Parrish/Farrell/Reitz (1997) and Allocca et al. (2002). | Sheet thickness from the discharge coefficient and cone angle; parcel diameter set from the sheet thickness and the B coefficient |
| `LISAAtomisation` | `LISA` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/AtomisationModel/LISAAtomisation/LISAAtomisation.H` | Linearised Instability Sheet Atomisation (Senecal et al. 1999; Schmidt et al. 1999) for pressure-swirl atomisers; two SMD calculation methods provided. | Sheet dispersion relation gives max growth rate Omega and wavenumber Ks; ligament dL = sqrt(8h/Ks); droplet dD = (3*pi*dL^2*U/Omega)^(1/3); SMD from We and Oh |
| `NoAtomisation` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/AtomisationModel/NoAtomisation/NoAtomisation.H` | Placeholder for the 'none' option. |  |

### spray breakup model  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ETAB` | `ETAB` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/ETAB/ETAB.H` | Enhanced TAB cascade breakup model (Tanner 1997; Tanner & Weisser 1998). | Same oscillator as TAB but child size follows r_child/r = exp(-K_br*t), with K_br differing between the bag and stripping regimes at the We transition |
| `NoBreakup` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/NoBreakup/NoBreakup.H` | Placeholder for the 'none' option. |  |
| `PilchErdman` | `PilchErdman` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/PilchErdman/PilchErdman.H` | Pilch & Erdman (1987) regime-based secondary atomisation with a fragment velocity model (see also Guildenbecher et al. 2009). | Non-dimensional breakup time T from the We regime (vibrational, bag, bag-and-stamen, sheet stripping, wave crest stripping, catastrophic); Vd = V*sqrt(epsilon)*(B1*T + B2*T^2) |
| `ReitzDiwakar` | `ReitzDiwakar` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/ReitzDiwakar/ReitzDiwakar.H` | Reitz & Diwakar (1986/87) bag and stripping secondary breakup. | Bag: We > Cbag*6, tau_bag = Cb*sqrt(rho_l*r^3/(2*sigma)); Stripping: We/sqrt(Re) > Cstrip, tau_strip = Cs*(r/U)*sqrt(rho_l/rho_g); dr/dt = -(r - r_stable)/tau |
| `ReitzKHRT` | `ReitzKHRT` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/ReitzKHRT/ReitzKHRT.H` | Competing Kelvin-Helmholtz (stripping) and Rayleigh-Taylor (catastrophic) instability breakup. | KH: Lambda, Omega from the Reitz correlations, r_stable = B0*Lambda, tau_KH = 3.726*B1*r/(Lambda*Omega); RT: growth rate from the droplet acceleration, breakup when the RT wave persists longer than Ctau/OmegaRT |
| `SHF` | `SHF` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/SHF/SHF.H` | Schmehl/Maier/Wittig (2000) regime-dependent secondary breakup covering bag, multimode and shear regimes. | Regime selected from We and Oh; child size drawn from Rosin-Rammler or a log-normal distribution with regime-specific coefficients (weConst, weCrit1..3, cInit, coeffA/B etc.) |
| `TAB` | `TAB` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/TAB/TAB.H` | Taylor Analogy Breakup (O'Rourke & Amsden 1987), KIVA implementation; two SMD calculation methods. | y'' + (5*mu_l/(rho_l*r^2))*y' + (8*sigma/(rho_l*r^3))*y = (2/3)*rho_g*u^2/(rho_l*r); breakup when y>1; r32 = r/(1 + 0.4*K*y^2 + rho_l*r^3*yDot^2*(6K-5)/(120*sigma)) |

### spray secondary breakup base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BreakupModel` | `breakupModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/BreakupModel/BreakupModel/BreakupModel.H` | Base for secondary droplet breakup; owns solveOscillationEq, the y/yDot state, TABComega/TABCmu/TABtwoWeCrit and the child-parcel spawning logic. |  |

### stochastic collision base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `StochasticCollisionModel` | `collisionModel (stochasticCollisionModel entry)` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/StochasticCollision/StochasticCollisionModel/StochasticCollisionModel.H` | Base for sub-grid probabilistic parcel-parcel collision models, as opposed to deterministic DEM. |  |

### stochastic collision model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NoStochasticCollision` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/StochasticCollision/NoStochasticCollision/NoStochasticCollision.H` | Placeholder for the 'none' option. |  |

### stochastic collision model (multiphase reacting)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SuppressionCollision` | `suppressionCollision` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/StochasticCollision/SuppressionCollision/SuppressionCollision.H` | Inter-cloud collision model that toggles the canReact flag to inhibit devolatilisation and surface reactions where a suppressing cloud is present. | canReact = false where the suppression cloud volume fraction exceeds the threshold |

### stochastic collision model (spray)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ORourkeCollision` | `ORourke` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/StochasticCollision/ORourkeCollision/ORourkeCollision.H` | O'Rourke droplet collision/coalescence model; same-cell parcel pairs collide with a Poisson probability, outcome coalescence or grazing/stretching separation. | nu = n2*pi*(r1+r2)^2*\|U1-U2\|*dt/V; P_nocoll = exp(-nu); Weber number We = rho*\|U1-U2\|^2*d/sigma decides coalescence vs separation |
| `TrajectoryCollision` | `trajectory` | `[Foundation-12] src/lagrangian/parcel/submodels/Spray/StochasticCollision/TrajectoryCollision/TrajectoryCollision.H` | Nordin's trajectory-based extension of O'Rourke: collisions detected from actual parcel trajectories rather than same-cell probability. | Collision if the closest approach along the relative trajectory within dt is less than r1+r2; cSpace/cTime scale the detection window |

### stochastic process  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `OUprocess` |  | `[Foundation-12] src/randomProcesses/processes/OUprocess/OUprocess.H` | Random Ornstein-Uhlenbeck process generator on a Kmesh producing a complex vector force field in wavenumber space. | dX = -alpha*X*dt + sigma*dW; band-limited between kLower and kUpper and projected solenoidal |

### surface film base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SurfaceFilmModel` | `surfaceFilm` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/SurfaceFilmModel/SurfaceFilmModel/SurfaceFilmModel.H` | Templated base for transfer of parcel mass/momentum/energy into a wall surface film region. |  |

### surface film model  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NoSurfaceFilm` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/SurfaceFilmModel/NoSurfaceFilm/NoSurfaceFilm.H` | Placeholder for the 'none' option - the only surface film model registered in this release. |  |

### surface reaction base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SurfaceReactionModel` | `surfaceReactionModel` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/SurfaceReactionModel/SurfaceReactionModel.H` | Base for heterogeneous char/solid surface reactions on parcels; returns the heat of reaction and the gas/solid mass source split. |  |

### surface reaction model  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `COxidationDiffusionLimitedRate` | `COxidationDiffusionLimitedRate` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/COxidationDiffusionLimitedRate/COxidationDiffusionLimitedRate.H` | Diffusion-limited char oxidation for coal parcels, C(s) + Sb*O2 -> CO2. | dm_C/dt = -D*Sb*pi*d*(rho_c*Y_O2/W_O2)*W_C with D = D0*(Tmean/1500)^0.75 |
| `COxidationHurtMitchell` | `COxidationHurtMitchell` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/COxidationHurtMitchell/COxidationHurtMitchell.H` | Hurt & Mitchell (1992) unified high-temperature char combustion kinetics; valid Tc>1500 K, 75-200 um particles, pO2>0.3 atm. | Burn-out-dependent intrinsic rate with thermal-annealing deactivation and a mode-of-burning exponent on the char density |
| `COxidationIntrinsicRate` | `COxidationIntrinsicRate` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/COxidationIntrinsicRate/COxidationIntrinsicRate.H` | Intrinsic (pore-diffusion corrected) char surface reaction with effectiveness factor and Thiele modulus, C(s) + Sb*O2 -> CO2. | ki = A*exp(-E/(R*T)); Thiele phi = (d/2)*sqrt(Sb*rho_c*Ag*ki/De); eta = (3/phi^2)*(phi/tanh(phi) - 1); Rk = eta*(d/6)*rho_c*Ag*ki; dm/dt = -pi*d^2*pO2*Rd*Rk/(Rd+Rk) |
| `COxidationKineticDiffusionLimitedRate` | `COxidationKineticDiffusionLimitedRate` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/COxidationKineticDiffusionLimitedRate/COxidationKineticDiffusionLimitedRate.H` | Combined kinetic/diffusion limited char oxidation (Baum & Street form), C(s) + Sb*O2 -> CO2. | Rd = C1/d*sqrt((T+Tc)/2); Rk = C2*exp(-E/(R*T)); dm/dt = -pi*d^2*pO2*Rd*Rk/(Rd + Rk) |
| `COxidationMurphyShaddix` | `COxidationMurphyShaddix` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/COxidationMurphyShaddix/COxidationMurphyShaddix.H` | Murphy & Shaddix (2006) char oxidation in oxygen-enriched environments, C(s) + O2 -> CO2; iteratively solves for the surface O2 concentration. | Iterate q = kc*(C_inf - C_s) = A*exp(-E/(R*T))*C_s^n with Stefan-flow-corrected mass transfer |
| `NoSurfaceReaction` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/ReactingMultiphase/SurfaceReactionModel/NoSurfaceReaction/NoSurfaceReaction.H` | Placeholder for the 'none' option. |  |

### surface tension base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `surfaceTensionModel` | `sigma { type <model>; ... }` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/surfaceTensionModels/surfaceTensionModel` | Abstract base returning the surface-tension coefficient field sigma; also accepts the backward-compatible scalar 'sigma 0.07;' form. | sigma = sigma(x,t) [N/m], used in the CSF force sigma*kappa*grad(alpha) |

### surface tension model  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `surfaceTensionModels::constant` | `constant` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/surfaceTensionModels/constant/constantSurfaceTension.H` | Uniform constant surface tension coefficient. | sigma = const |
| `surfaceTensionModels::temperatureDependent` | `temperatureDependent` | `[Foundation-12] src/twoPhaseModels/interfaceProperties/surfaceTensionModels/temperatureDependent/temperatureDependentSurfaceTension.H` | Surface tension evaluated from a Function1 of a looked-up temperature field. | sigma = f(T), Function1 evaluated cell-wise on field T |

### surface tension model (compressible)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `surfaceTensionModels::liquidProperties` | `liquidProperties` | `[Foundation-12] src/twoPhaseModels/compressibleInterfaceProperties/surfaceTensionModels/liquidProperties/liquidPropertiesSurfaceTension.H` | Surface tension taken from the liquidProperties thermophysical function of the named phase. | sigma = liquidProperties::sigma(p,T) of the named phase |

### synthetic turbulence generator  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `turbGen` |  | `[Foundation-12] src/randomProcesses/turbulence/turbGen.H` | Generates a divergence-free random velocity field conforming to a prescribed energy spectrum (used by the boxTurb initialisation utility). | Random phases with amplitude sqrt(E(k)); projection u_hat -= k*(k.u_hat)/\|k\|^2 enforces div(U)=0 |

### thermo package (lagrangian)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `parcelThermo` |  | `[Foundation-12] src/lagrangian/parcel/parcelThermo/parcelThermo.H` | Registered thermo package giving parcels access to carrier species plus optional liquid (liquidProperties) and solid (solidProperties) component tables; nullptr if absent. | Provides rho, cp, h and L for each of the S/L/G component sets |

### turbulent dispersion base class  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DispersionRASModel` | `dispersionRASModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/DispersionModel/DispersionRASModel/DispersionRASModel.H` | Base for dispersion models that read k and epsilon from a RAS momentum transport model. | Eddy time scale tTurb = min(k/epsilon, cps*k^1.5/epsilon/\|Uc-u\|); sigma = sqrt(2*k/3) |

### turbulent dispersion base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DispersionModel` | `dispersionModel` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/DispersionModel/DispersionModel/DispersionModel.H` | Base for models that perturb the carrier velocity seen by a parcel to represent unresolved turbulence. | Uturb added to Uc for the drag calculation |

### turbulent dispersion model  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GradientDispersionRAS` | `gradientDispersionRAS` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/DispersionModel/GradientDispersionRAS/GradientDispersionRAS.H` | Perturbs the velocity in the -grad(k) direction with a Gaussian random magnitude of variance sigma, resampled each eddy interaction time. | Uturb = sigma*N(0,1)*(-grad(k)/\|grad(k)\|), sigma = sqrt(2*k/3) |
| `NoDispersion` | `none` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/DispersionModel/NoDispersion/NoDispersion.H` | Placeholder for the 'none' option. | Uturb = 0 |
| `StochasticDispersionRAS` | `stochasticDispersionRAS` | `[Foundation-12] src/lagrangian/parcel/submodels/Momentum/DispersionModel/StochasticDispersionRAS/StochasticDispersionRAS.H` | Perturbs the velocity in a random isotropic direction with a Gaussian random magnitude of variance sigma. | Uturb = sigma*N(0,1)*randomUnitVector, sigma = sqrt(2*k/3) |

### two-phase interface class  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `compressibleTwoPhases` |  | `[Foundation-12] src/twoPhaseModels/compressibleTwoPhases/compressibleTwoPhases.H` | Interface to two rhoFluidThermo-based phases, giving per-phase rho, thermo and alpha for compressible VoF/cavitation. | rho = alpha1*rho1(p,T) + alpha2*rho2(p,T) |
| `incompressibleTwoPhases` |  | `[Foundation-12] src/twoPhaseModels/incompressibleTwoPhases/incompressibleTwoPhases.H` | Interface to two constant-density phases (rho1, rho2 dimensionedScalars) for incompressible VoF and cavitation. | rho = alpha1*rho1 + alpha2*rho2 |
| `twoPhases` |  | `[Foundation-12] src/twoPhaseModels/twoPhaseMixture/twoPhases.H` | Abstract interface giving indexed access to the two phases' alpha and names. |  |

### two-phase property container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `twoPhaseMixture` |  | `[Foundation-12] src/twoPhaseModels/twoPhaseMixture/twoPhaseMixture.H` | Represents a mixture of two named phases; holds alpha1/alpha2 and the phase names. | alpha2 = 1 - alpha1 |

### utility  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `demandDrivenEntry` |  | `[Foundation-12] src/lagrangian/basic/demandDrivenEntry/demandDrivenEntry.H` | Dictionary entry read lazily on first access; used widely by parcel sub-models for optional coefficients. |  |

### wave model  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveModels::Airy` | `Airy` | `[Foundation-12] src/waves/waveModels/Airy/Airy.H` | First-order (linear, Stokes 1847) progressive wave. | eta = a*cos(k*x - omega*t + phi); u = a*omega*cosh(k(z+h))/sinh(kh)*cos(theta), w = a*omega*sinh(k(z+h))/sinh(kh)*sin(theta); omega^2 = g*k*tanh(kh) |
| `waveModels::Stokes2` | `Stokes2` | `[Foundation-12] src/waves/waveModels/Stokes2/Stokes2.H` | Second-order Stokes wave (Stokes 1847, eqns 18-19), adds the second harmonic to Airy. | eta = a*cos(t1) + (k*a^2/4)*(3/tanh^3(kh) - 1/tanh(kh))*cos(2*t1), velocity gains a matching second-harmonic term |
| `waveModels::Stokes5` | `Stokes5` | `[Foundation-12] src/waves/waveModels/Stokes5/Stokes5.H` | Fifth-order Stokes wave using the Fenton (1985) coefficient set. | eta and velocity expanded to fifth order in ka with Fenton's A_ij, B_ij, C_i coefficient tables |
| `waveModels::irregular` | `irregular` | `[Foundation-12] src/waves/waveModels/irregular/irregular.H` | Irregular sea built from n first-order (Airy) components sampled from a selectable spectrum over a fractional span, with random phases; can plot the spectrum via setFormat. | eta = sum_i a_i*cos(k_i*x - omega_i*t + phi_i), a_i = sqrt(2*S(omega_i)*dOmega_i) |
| `waveModels::solitary` | `solitary` | `[Foundation-12] src/waves/waveModels/solitary/solitary.H` | Single solitary (soliton) wave after Dean & Dalrymple (1991) pp.314-317. | eta = a*sech^2(sqrt(3a/(4h^3))*(x - c*t - x0)), c = sqrt(g*(h+a)) |

### wave model base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveModel` | `waveModel (entries in the constant/waveProperties 'waves' list)` | `[Foundation-12] src/waves/waveModels/waveModel/waveModel.H` | Generic base for water-wave models; derived classes return surface elevation and velocity as functions of position and time. | elevation(t,x) and velocity(t,x) in the wave frame; dispersion omega^2 = g*k*tanh(k*h) solved for length or period |

### wave model helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `AiryCoeffs` |  | `[Foundation-12] src/waves/waveModels/Airy/AiryCoeffs.C` | Shared coefficient/dispersion solver used by Airy, Stokes2, Stokes5 and irregular; converts between length, period, depth and celerity. | Iterative solution of omega^2 = g*k*tanh(k*h); celerity c = omega/k |

### wave spectrum  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GodaJONSWAP` | `GodaJONSWAP` | `[Foundation-12] src/waves/waveModels/irregular/waveSpectra/GodaJONSWAP/GodaJONSWAP.H` | Goda's approximate JONSWAP re-parameterisation using significant wave height Hs and peak period Tp instead of wind speed and fetch. | S(f) = beta_J*Hs^2*Tp^-4*f^-5*exp(-1.25*(Tp*f)^-4)*gamma^exp(-(Tp*f-1)^2/(2*sigma^2)) |
| `JONSWAP` | `JONSWAP` | `[Foundation-12] src/waves/waveModels/irregular/waveSpectra/JONSWAP/JONSWAP.H` | JONSWAP fetch-limited spectrum parameterised by U10 and fetch F, with peak-enhancement factor gamma (default 3.3). | S(f) = PM(f)*gamma^r, r = exp(-(f-fp)^2/(2*sigma^2*fp^2)); fp from U10 and F |
| `PiersonMoskowitz` | `PiersonMoskowitz` | `[Foundation-12] src/waves/waveModels/irregular/waveSpectra/PiersonMoskowitz/PiersonMoskowitz.H` | Fully-developed wind-sea spectrum parameterised by the wind speed 19.5 m above the surface. | S(omega) = alpha*g^2/omega^5*exp(-beta*(omega0/omega)^4), alpha=8.1e-3, beta=0.74, omega0 = g/U19.5 |

### wave spectrum base class / runtime-selection table  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveSpectrum` | `spectrum <name>; with a <name>Coeffs sub-dict` | `[Foundation-12] src/waves/waveModels/irregular/waveSpectra/waveSpectrum/waveSpectrum.H` | Base class for one-dimensional wave energy spectra S(f); provides equal-energy sampling and integral moments. | S(f) [m^2/Hz]; integral, mean and significant statistics |

### wave superposition  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveAtmBoundaryLayerSuperposition` | `waveAtmBoundaryLayer` | `[Foundation-12] src/waves/waveSuperpositions/waveAtmBoundaryLayerSuperposition/waveAtmBoundaryLayerSuperposition.H` | waveSuperposition extended with an atmospheric boundary layer for the gas velocity above the wave; surface roughness derived from user-supplied hWaveMin/hWaveMax. | U_gas(z) = (u*/kappa)*ln((z+z0)/z0), matched to UGasRef at hRef; z0 from the wave height range |

### wave superposition (registered mesh object)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `waveSuperposition` | `wave` | `[Foundation-12] src/waves/waveSuperpositions/waveSuperposition/waveSuperposition.H` | Wraps a list of waveModels, defines the wave coordinate system (origin, direction, UMean) and superimposes elevation and velocity; created on demand from constant/waveProperties. | eta = sum_i eta_i; U = UMean + sum_i u_i, optionally multiplied by 'scale' along and 'crossScale' across the direction; heightAboveWave switch changes the vertical coordinate |

---

## finiteVolume discretisation schemes, surface interpolation and fvMatrix

> **Subsystem notes**
>
> STRUCTURE 1. Scheme selection is Istream-based, not dictionary-based. Every scheme in finiteVolume/{ddt,d2dt2,div,grad,laplacian,snGrad,convection}Schemes registers into an `addIstreamConstructorToTable` keyed on the first word of the fvSchemes entry, and then consumes the rest of the stream itself.
> That is why schemes nest textually: `div(phi,U) Gauss limitedLinearV 1;` selects gaussConvectionScheme, which then selects `limitedLinearV` from the limitedSurfaceInterpolationScheme MeshFlux table, which then reads the coefficient `1`.
> Same for `laplacian(nu,U) Gauss linear corrected;` (interpolation scheme + snGrad scheme) and `grad(U) cellLimited<cubic> 1.5 Gauss linear 1;`. 2. Two runtime-selection tables for interpolation. surfaceInterpolationScheme has a `Mesh` table (geometry only) and a `MeshFlux` table (flux-aware).
> limitedSurfaceInterpolationScheme mirrors both. `makeSurfaceInterpolationScheme(X)` registers into both for all 5 field types; `makelimitedSurfaceInterpolationScheme(X)` registers a limited scheme into both the limited and plain tables.
> Schemes that need the flux (upwind, all TVD/NVD limiters, linearUpwind, LUST, deferred, CoBlended, outletStabilised) exist only meaningfully in the MeshFlux table; harmonic is registered for scalar only. 3. The limiter family is generated by macro combinatorics, not by hand.
> Four macros in LimitedScheme.H produce every user-visible name from one Limiter class: - makeLimitedSurfaceInterpolationScheme(Name, Limiter) -> scalar+all types via NVDTVD/magSqr, giving e.g.
> `vanLeer` - makeLimitedVSurfaceInterpolationScheme(NameV, Limiter) -> vector-specific via NVDVTVDV, giving e.g. `vanLeerV` - makeLLimitedSurfaceInterpolationTypeScheme(..., LimitedLimiter, ...) -> explicitly bounded, giving e.g.
> `limitedVanLeer <lo> <hi>` - makeLLimitedSurfaceInterpolationTypeScheme(..., Limited01Limiter, ...) -> [0,1]-bounded, giving e.g.
> `vanLeer01` Only Gamma, vanLeer, limitedLinear, limitedCubic and MUSCL have the full four-way expansion; Minmod, SuperBee, OSPRE, UMIST, QUICK, SFCD, vanAlbada have only the plain and V forms; filteredLinear has no V form; Phi is vector-only. 4.
> Scheme keyword collisions are resolved by table, not by name.
> "Gauss" is registered separately as a div scheme, a convection scheme, a multivariate convection scheme, a grad scheme and a laplacian scheme; "Euler"/"steadyState" appear in both the ddt and d2dt2 tables; "bounded" in both ddt and convection; "upwind" in both the limited and multivariate tables; "linearFit"/"quadraticFit"/"phaseStabilised" in both the interpolation and snGrad tables.
> The dictionary section (ddtSchemes vs divSchemes vs ...) disambiguates. 5. cellLimited uses angle-bracket sub-selection.
> cellLimitedGrads.C registers three distinct type names via defineTemplateTypeNameAndDebugWithName: "cellLimited" (minmod, kept unbracketed for backward compatibility), "cellLimited<Venkatakrishnan>" and "cellLimited<cubic>".
> The limiter classes are plain policy classes constructed from the Istream, not runtime-selectable objects; adding a limiter means adding a makeFvLimitedGradScheme line.
> Note the source's own advice: the Venkatakrishnan limiter is clipped at 1 (breaking differentiability) and cubic is recommended in preference. 6. Polynomial-fit schemes are a three-way template product: Scheme x Polynomial x Stencil.
> CentredFitScheme/UpwindFitScheme/PureUpwindFitScheme x {linear, biLinear, quadratic, quadraticLinear, cubicUpwind, quadraticUpwind, quadraticLinearUpwind}FitPolynomial x {centredCFC, centredFEC, upwindCFC, upwindFEC, pureUpwindCFC}CellToFaceStencilObject.
> Only 10 combinations are actually instantiated (8 interpolation + 2 snGrad). The FitData base does a weighted SVD pseudo-inverse per face and falls back to plain linear/upwind if the fit deviates by more than linearLimitFactor. 7.
> This subsystem contains NO turbulence models, wall functions, thermophysical packages, mesh generators/movers or topo-changers, and no linear solvers/preconditioners/smoothers.
> Those live elsewhere: RAS/LES/wall functions in src/MomentumTransportModels, thermo in src/thermophysicalModels, mesh motion/topo-change in src/fvMeshMovers, src/fvMeshTopoChangers, src/mesh, and the lduMatrix solver/preconditioner/smoother families in src/OpenFOAM/matrices/lduMatrix.
> The only solver-adjacent things here are fvMatrix (which delegates to lduMatrix::solver, selecting `solver`/`preconditioner`/`smoother` keywords from the fvSolution `solvers` dictionary), faceAreaPairGAMGAgglomeration (the FV-specific GAMG agglomeration, keyword `faceAreaPair`), and the MULES/CMULES bounded explicit FCT solvers used for phase-fraction transport.
> 8. fvMatrix is not a scheme-selectable object. It is the assembly target: schemes call fvm::* which return fvMatrix<Type>, and the matrix carries internalCoeffs/boundaryCoeffs harvested from each fvPatchField's valueInternalCoeffs/valueBoundaryCoeffs/gradientInternalCoeffs/gradientBoundaryCoeffs.
> That interface is the contract that makes every boundary condition in fields/fvPatchFields pluggable. 9. Boundary conditions split into three tiers with different registration semantics.
> basic/ conditions (calculated, fixedValue, fixedGradient, zeroGradient, mixed, directionMixed, transform, sliced, coupled, basicSymmetry, extrapolatedCalculated) are mostly base classes.
> constraint/ conditions take their type name from the corresponding fvPatch (TypeName(cyclicFvPatch::typeName_()) etc.), so they are chosen automatically by the mesh's boundary file, not by the user.
> derived/ conditions are the ~95 user-selectable BCs; they register via makePatchFields (all field types), makePatchTypeField (one field type) or makeNullConstructablePatchTypeField (noSlip only, so it can be a default). 10. PrghPressure is a BC-generating template, not a BC.
> makePrghPatchScalarField(base, name) wraps any static-pressure BC into its p_rgh equivalent, producing five extra selectable keywords (prghPressure, prghTotalPressure, prghUniformTotalPressure, prghEntrainmentPressure, prghUniformDensityHydrostaticPressure) from classes defined elsewhere in derived/.
> 11. fvFieldSources is the source-term analogue of fvPatchFields. Where an fvModel injects material, each field needs a `sources` entry saying what value the injected material carries; only four conditions exist (internal, uniformFixedValue, uniformInletOutlet, turbulentIntensityKineticEnergy).
> CAVEAT ABOUT THIS CHECKOUT 12. One component could not be read because of a Windows case-collision. src/finiteVolume/finiteVolume/gradSchemes/ contains, upstream, BOTH `leastSquaresGrad/` (concrete, keyword `leastSquares`) and `LeastSquaresGrad/` (templated on an extended cell-to-cell stencil).
> On this case-insensitive filesystem git merged them into a single `LeastSquaresGrad/` directory and the capitalised files (LeastSquaresGrad.H, LeastSquaresGrad.C, LeastSquaresGrads.C, LeastSquaresVectors.H, LeastSquaresVectors.C) were overwritten by their lowercase counterparts.
> Make/files still lists `$(gradSchemes)/LeastSquaresGrad/LeastSquaresGrads.C` as a compiled unit, and the stencil objects it needs (centredCPCCellToCellStencilObject, centredCECCellToCellStencilObject, centredCFCCellToCellStencilObject) are all present under fvMesh/extendedStencil/cellToCell/MeshObjects/.
> So the stencil-based least-squares gradient scheme exists in OpenFOAM-12 but its registered scheme keywords cannot be verified from this checkout - I have deliberately left its `selectable` field empty rather than guess.
> Anyone needing those keywords should re-clone on a case-sensitive filesystem (or use `git config core.ignorecase false` plus a fresh checkout).
> The same class of collision should be assumed for any other capitalised/lowercase directory pair in the tree; I checked the rest of this subsystem's Make/files entries and found no other affected unit, though the check itself is weakened by the filesystem being case-insensitive. 13.
> Two least-squares vector implementations in that directory (unweightedLeastSquaresVectors.C, invDistLeastSquaresVectors.C) are alternative definitions of the same class, not separate selectable models - only leastSquaresVectors.C is in Make/files.
> They are drop-in replacements changing the weighting from 1/|d|^2 to 1 or 1/|d| respectively.

### GAMG agglomeration  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `faceAreaPairGAMGAgglomeration` | `faceAreaPair` | `[Foundation-12] src/finiteVolume/fvMatrices/solvers/GAMGSymSolver/GAMGAgglomerations/faceAreaPairGAMGAgglomeration` | GAMG coarse-level agglomeration using the pair algorithm with face areas (mesh.magSf) as the pairing weights instead of matrix coefficients; the standard choice for FV meshes. Registered in both the lduMatrix and geometry GAMG agglomeration tables. | pairwise agglomeration maximising sum of \|Sf\| between merged cells |

### NVD limiter  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `GammaLimiter (Gamma / GammaV / limitedGamma / Gamma01)` | `Gamma, GammaV, limitedGamma, Gamma01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/Gamma` | Jasak's Gamma NVD scheme; blending coefficient k in (0,1] specified after the scheme name. V variant uses NVDVTVDV for vectors; limitedGamma takes explicit bounds; Gamma01 is bounded to [0,1]. | limiter = min(max(phict/k, 0), 1) |
| `SFCDLimiter (SFCD / SFCDV)` | `SFCD, SFCDV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/SFCD` | Self-Filtered Central Differencing NVD scheme. | limitPhict = min(max(phict, 0), 0.5) ; limiter = limitPhict/(1 - limitPhict) |

### TVD limiter  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MUSCLLimiter (MUSCL / MUSCLV / limitedMUSCL / MUSCL01)` | `MUSCL, MUSCLV, limitedMUSCL, MUSCL01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/MUSCL` | van Leer MUSCL TVD limiter. | limiter(r) = max(min(min(2r, 0.5r + 0.5), 2), 0) |
| `MinmodLimiter (Minmod / MinmodV)` | `Minmod, MinmodV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/Minmod` | Minmod TVD limiter - the most diffusive of the classical TVD family. | limiter(r) = max(min(r, 1), 0) |
| `OSPRELimiter (OSPRE / OSPREV)` | `OSPRE, OSPREV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/OSPRE` | OSPRE smooth TVD limiter (r clipped at 0 before evaluation). | rrp1 = r*(r+1) ; limiter = 1.5*rrp1/(rrp1 + 1) |
| `SuperBeeLimiter (SuperBee / SuperBeeV)` | `SuperBee, SuperBeeV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/SuperBee` | Roe's SuperBee TVD limiter - the most compressive/least diffusive of the classical TVD family. | limiter(r) = max(max(min(2r, 1), min(r, 2)), 0) |
| `UMISTLimiter (UMIST / UMISTV)` | `UMIST, UMISTV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/UMIST` | UMIST TVD limiter (Lien & Leschziner) - a piecewise-linear approximation to QUICK. | limiter(r) = max(min(min(min(2r, 0.75r + 0.25), 0.25r + 0.75), 2), 0) |
| `limitedCubicLimiter / limitedCubicVLimiter (limitedCubic / limitedCubicV / limitedLimitedCubic / limitedCubic01)` | `limitedCubic, limitedCubicV, limitedLimitedCubic, limitedCubic01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedCubic` | TVD-limited centred-cubic scheme with strength coefficient k; higher accuracy than limitedLinear at the cost of a wider stencil dependence. | cubicLimiter from the cubic interpolation ratio; limiter = max(min(min(2r, cubicLimiter), 2), 0) |
| `limitedLinearLimiter (limitedLinear / limitedLinearV / limitedLimitedLinear / limitedLinear01)` | `limitedLinear, limitedLinearV, limitedLimitedLinear, limitedLinear01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedLinear` | TVD-limited linear (central) scheme with a strength coefficient k in (0,1]; k=1 is the most limited/stable, small k approaches pure linear. | twoByk = 2/max(k, small) ; limiter(r) = max(min(twoByk*r, 1), 0) |
| `vanAlbadaLimiter (vanAlbada / vanAlbadaV)` | `vanAlbada, vanAlbadaV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/vanAlbada` | van Albada smooth TVD limiter (r clipped at 0 before evaluation). | limiter(r) = r*(r + 1)/(r^2 + 1) |
| `vanLeerLimiter (vanLeer / vanLeerV / limitedVanLeer / vanLeer01)` | `vanLeer, vanLeerV, limitedVanLeer, vanLeer01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/vanLeer` | van Leer TVD limiter, smooth and symmetric; the most commonly used general-purpose bounded scheme. | limiter(r) = (r + \|r\|)/(1 + \|r\|) |

### basic BC  <sub>(11)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `basicSymmetryFvPatchField` |  | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/basicSymmetry` | Generic symmetry patch behaviour shared by symmetry, symmetryPlane and slip; reflects the internal field in the patch normal. | psi_b = 0.5*(psi_P + transform(I - 2 n n, psi_P)) |
| `calculatedFvPatchField` | `calculated` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/calculated` | Not evaluated; the value is assigned by the code that owns the field. Default for derived (calculated) fields. |  |
| `coupledFvPatchField` |  | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/coupled` | Abstract base for all coupled patches (processor, cyclic, non-conformal); declares patchNeighbourField() and the coupled matrix coefficient interface. |  |
| `directionMixedFvPatchField` | `directionMixed` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/directionMixed` | Base for direction-dependent mixed conditions: valueFraction is a symmTensor so different components/directions can be fixed value or fixed gradient. | psi_b = (I - V)&(psi_P + refGrad/deltaCoeffs) + V&refValue, V a symmTensor fraction |
| `extrapolatedCalculatedFvPatchField` | `extrapolatedCalculated` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/extrapolatedCalculated` | Applies a zero-gradient extrapolation from the internal field on evaluate() but may also simply be assigned; snGrad is the gradient of the current values. | psi_b = psi_P on evaluate |
| `fixedGradientFvPatchField` | `fixedGradient` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/fixedGradient` | Neumann constraint with a user-supplied normal gradient field. | psi_b = psi_P + gradient/deltaCoeffs |
| `fixedValueFvPatchField` | `fixedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/fixedValue` | Dirichlet constraint; base class for many derived conditions. | psi_b = value ; valueInternalCoeffs = 0 |
| `mixedFvPatchField` | `mixed` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/mixed` | Base for Robin-type conditions blending fixed value and fixed gradient via valueFraction. | psi_b = f*refValue + (1-f)*(psi_P + refGrad/deltaCoeffs) |
| `slicedFvPatchField` | `sliced` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/sliced` | Creates the patch field as a non-owning slice of a complete external field; the destructor is wrapped to avoid deallocating the borrowed storage. |  |
| `transformFvPatchField` | `transform` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/transform` | Base for conditions whose value is a transformation (rotation/reflection) of the internal value; supplies snGradTransformDiag. | psi_b = T & psi_P |
| `zeroGradientFvPatchField` | `zeroGradient` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/basic/zeroGradient` | Homogeneous Neumann condition. | psi_b = psi_P ; snGrad = 0 |

### boundary condition base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvPatchField<Type>` | `fvPatchField` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/fvPatchField` | Abstract base with a fat interface for all boundary conditions; four runtime-selection tables (patch, patchMapper, dictionary, and the constraint/patch-type table). Supplies valueInternalCoeffs/valueBoundaryCoeffs/gradientInternalCoeffs/gradientBoundaryCoeffs used to assemble the matrix boundary contributions. | psi_b = valueInternalCoeffs*psi_P + valueBoundaryCoeffs |

### bounded explicit solver  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MULES` | `MULES::explicitSolve / MULES::limit / MULES::limiter` | `[Foundation-12] src/finiteVolume/fvMatrices/solvers/MULES` | Multidimensional Universal Limiter for Explicit Solution. Solves a convective-only transport equation explicitly with a multi-dimensional flux limiter that guarantees boundedness between psiMin and psiMax; supports multiple fields/fluxes (UPtrList form) for multiphase. Controls: nLimiterIter, smoothLimiter, extremaCoeff, psiMax, psiMin. | psi^n = psi^o + rDeltaT^-1*(Su - Sp*psi - div(phiBD + lambda*phiCorr)); lambda from iterative Zalesak-style flux-corrected transport limiting |

### bounded semi-implicit solver  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CMULES` | `MULES::correct` | `[Foundation-12] src/finiteVolume/fvMatrices/solvers/MULES/CMULES.H` | Corrected MULES: applies the explicit multi-dimensional limited correction on top of an implicit, rigorously bounded solution (Euler-implicit in time, upwind in space), giving boundedness without an explicit Courant limit. | psi := psi + rDeltaT^-1 * (-div(lambda*phiCorr)) after the bounded implicit solve |

### constraint BC  <sub>(13)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicFvPatchField` | `cyclic (matches cyclicFvPatch::typeName)` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/cyclic` | Enforces a cyclic (periodic) coupling between a pair of boundaries, with rotational/translational transform support. | psi_b coupled to the transformed neighbour-patch internal values |
| `cyclicSlipFvPatchField` | `cyclicSlip` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/cyclicSlip` | Light wrapper on cyclicFvPatchField providing no new functionality (used where the patch is registered as cyclicSlip). |  |
| `emptyFvPatchField` | `empty` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/empty` | Removes the patch from the solution for reduced-dimension (1-D/2-D) cases; the patch field has zero size. |  |
| `internalFvPatchField` | `internal` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/internal` | Holds values for internal faces exposed by mesh sub-setting so the sub-mesh remains solvable. |  |
| `jumpCyclicFvPatchField` | `jumpCyclic` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/jumpCyclic` | Base for cyclic conditions with a specified jump (offset) between the two sides; used by fixedJump, uniformJump, fanPressureJump, prghCyclicPressure. | psi_neighbourSide = psi_ownerSide + jump |
| `nonConformalCyclicFvPatchField` | `nonConformalCyclic` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/nonConformalCyclic` | Cyclic coupling between non-conformal (arbitrarily meshed, e.g. sliding/AMI-like) boundary pairs. |  |
| `nonConformalErrorFvPatchField` | `nonConformalError` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/nonConformalError` | Holds the residual (unmatched) area of a non-conformal coupling so the geometry remains closed. |  |
| `nonConformalProcessorCyclicFvPatchField` | `nonConformalProcessorCyclic` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/nonConformalProcessorCyclic` | Processor communication across a non-conformal cyclic coupling split between processors. |  |
| `processorCyclicFvPatchField` | `processorCyclic` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/processorCyclic` | Processor communication across a cyclic patch pair that has been split between processors. |  |
| `processorFvPatchField` | `processor` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/processor` | Inter-processor coupling: sends/receives the patch internal field and supplies the coupled matrix interface for parallel solution. |  |
| `symmetryFvPatchField` | `symmetry` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/symmetry` | General (possibly curved) symmetry constraint derived from basicSymmetry. |  |
| `symmetryPlaneFvPatchField` | `symmetryPlane` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/symmetryPlane` | Planar symmetry constraint using the single plane normal (more accurate/cheaper than symmetry for genuinely flat patches). |  |
| `wedgeFvPatchField` | `wedge` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/wedge` | Axisymmetric (wedge) constraint - like cyclic but for 2-D axisymmetric geometries, applying the wedge rotation transform. | psi_b = cellT & psi_P (wedge rotation tensor) |

### constraint BC specialisation  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `processorFvPatchScalarField` |  | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/constraint/processor/processorFvPatchScalarField.H` | Scalar specialisation of processorFvPatchField (specialised transform handling for scalars). |  |

### convection scheme  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::boundedConvectionScheme<Type>` | `bounded` | `[Foundation-12] src/finiteVolume/finiteVolume/convectionSchemes/boundedConvectionScheme` | Bounded wrapper on a runtime-selected convection scheme; subtracts the divergence of the flux to improve stability of bounded scalars in steady solvers. | div_bounded(phi,vf) = div(phi,vf) - Sp(div(phi), vf) |
| `fv::gaussConvectionScheme<Type>` | `Gauss` | `[Foundation-12] src/finiteVolume/finiteVolume/convectionSchemes/gaussConvectionScheme` | Standard Gauss convection; takes a runtime-selected limitedSurfaceInterpolationScheme (face flux aware) to build the face weights and the explicit correction. | fvmDiv: a_N = (1-w)*phi_f, a_P = w*phi_f ; plus explicit correction phi_f*corr(vf) |

### convection scheme (multivariate)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::multivariateGaussConvectionScheme<Type>` | `Gauss (in the Multivariate table)` | `[Foundation-12] src/finiteVolume/finiteVolume/convectionSchemes/multivariateGaussConvectionScheme` | Gauss convection for a set of coupled fields sharing one limiter; constructed from a multivariateSurfaceInterpolationScheme over a field table. | as gaussConvectionScheme, with weights from the shared multivariate limiter |

### convection scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::convectionScheme<Type>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/convectionSchemes/convectionScheme/convectionScheme.H` | Abstract base for div(phi,vf) convection schemes; Istream and Multivariate runtime-selection tables; supplies interpolate/flux/fvmDiv/fvcDiv. | div(phi, vf) = (1/V) sum_f phi_f * vf_f |

### d2dt2 scheme  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::EulerD2dt2Scheme<Type>` | `Euler` | `[Foundation-12] src/finiteVolume/finiteVolume/d2dt2Schemes/EulerD2dt2Scheme` | First-order Euler implicit second time derivative using current and two previous time-step values. | d2dt2(vf) = (vf^n - (1+dt/dt0)*vf^o + (dt/dt0)*vf^oo)/(dt*deltaT0_) |
| `fv::steadyStateD2dt2Scheme<Type>` | `steadyState` | `[Foundation-12] src/finiteVolume/finiteVolume/d2dt2Schemes/steadyStateD2dt2Scheme` | Steady-state second time derivative returning zero. | d2dt2(vf) = 0 |

### d2dt2 scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::d2dt2Scheme<Type>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/d2dt2Schemes/d2dt2Scheme/d2dt2Scheme.H` | Abstract base for second time-derivative schemes with Istream runtime selection. | d2/dt2(rho*vf) |

### ddt scheme  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::CoEulerDdtScheme<Type>` | `CoEuler` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/CoEulerDdtScheme` | Courant-number-limited first-order Euler ddt; the local time step is adjusted so the local Courant number does not exceed the specified maxCo. Steady-state use with transient codes. | rDeltaT = max(1/deltaT, (2*Co_cell)/(maxCo*deltaT)) then Euler ddt with that local rDeltaT |
| `fv::CrankNicolsonDdtScheme<Type>` | `CrankNicolson` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/CrankNicolsonDdtScheme` | Second-order Crank-Nicolson implicit ddt using current/previous fields and previous ddt, with mandatory off-centring coefficient ocCoeff in [0,1] (1 = fully centred, 0 = Euler); ocCoeff may be a Function1 (e.g. scale/linearRamp) to ramp from Euler. | ddt stored as DDt0 field; ddt = (1+ocCoeff)*(vf-vf^o)/dt - ocCoeff*ddt0 |
| `fv::EulerDdtScheme<Type>` | `Euler` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme` | First-order implicit/explicit Euler time derivative using only current and previous time values. | ddt(vf) = (vf^n - vf^o)/deltaT ; moving-mesh form uses (V^n vf^n - V^o vf^o)/(V^n deltaT) |
| `fv::SLTSDdtScheme<Type>` | `SLTS` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/SLTSDdtScheme` | Stabilised local time-step first-order Euler ddt; local time step chosen so the advective equation remains diagonally dominant (alphaTemp smoothing coefficient). | rDeltaT from sumPhi/V scaled to keep diagonal dominance, then Euler ddt |
| `fv::backwardDdtScheme<Type>` | `backward` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/backwardDdtScheme` | Second-order backward-differencing ddt using current and two previous time-step values (variable time step supported). | coefft = 1 + dt/(dt+dt0); coefft00 = dt^2/(dt0*(dt+dt0)); coefft0 = coefft + coefft00; ddt = (coefft*vf - coefft0*vf^o + coefft00*vf^oo)/dt |
| `fv::boundedDdtScheme<Type>` | `bounded` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/boundedDdtScheme` | Wrapper applying a bounded form of a runtime-selected base ddt scheme; improves stability for bounded scalars when the flux field is temporarily divergent. | ddt_bounded(rho,vf) = ddt(rho,vf) - Sp(ddt(rho), vf) (conservative when ddt(rho)=0) |
| `fv::localEulerDdtScheme<Type>` | `localEuler` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/localEulerDdtScheme` | Local time-step first-order Euler ddt; the reciprocal local time-step field rDeltaT is looked up from the object registry. For steady-state runs with transient codes using local time stepping. | ddt(vf) = rDeltaT*(vf^n - vf^o) |
| `fv::steadyStateDdtScheme<Type>` | `steadyState` | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/steadyStateDdtScheme` | Steady-state ddt returning zero; setting it as the default marks the fvSchemes as steady. | ddt(vf) = 0 |

### ddt scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::ddtScheme<Type>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/ddtScheme/ddtScheme.H` | Abstract base for all time-derivative schemes; Istream runtime-selection table, provides fvcDdt/fvmDdt/fvcDdtPhiCorr and the mesh/ddt-phi-corr machinery. | d/dt(rho*vf) discretised implicitly (fvmDdt) or explicitly (fvcDdt) |

### ddt scheme helper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::localEulerDdt` |  | `[Foundation-12] src/finiteVolume/finiteVolume/ddtSchemes/localEulerDdtScheme/localEulerDdt.C` | Static helper providing rDeltaTName, enabled() test and localRDeltaT()/localRSubDeltaT() lookup of the local reciprocal time-step field used by localEulerDdtScheme. |  |

### derived BC  <sub>(16)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fixedInternalValueFvPatchField` | `fixedInternalValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedInternalValue` | Sets boundary-adjacent cell (internal) values directly in the matrix as a constraint; behaves as zeroGradient for the patch face value. | matrix setValues on the boundary cells |
| `fixedMeanFvPatchField` | `fixedMean` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedMean` | Extrapolates the field to the patch and rescales/offsets it so the area-weighted mean matches a specified (optionally time-varying) value. | psi_b = psi_extrap * meanValue/areaAverage(psi_extrap) |
| `fixedMeanOutletInletFvPatchField` | `fixedMeanOutletInlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedMeanOutletInlet` | fixedMean applied as the inlet value of an outletInlet condition, so the mean is only imposed on reverse flow. |  |
| `fixedNormalSlipFvPatchField` | `fixedNormalSlip` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedNormalSlip` | Fixes the patch-normal component to a specified value while leaving the tangential components free (slip). | psi_b = (I - n n)&psi_P + n*fixedValue |
| `fixedProfileFvPatchField` | `fixedProfile` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedProfile` | Fixed value taken from a Function1 profile evaluated against the coordinate along a given direction, scaled between user-supplied bounds. | psi_b = profile((x&dir - min)/(max - min)) |
| `fixedValueInletOutletFvPatchField` | `fixedValueInletOutlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedValueInletOutlet` | Acts as fixedValue on inflow; on outflow approximates the fixed value in a way that keeps the matrix diagonally dominant (avoids the destabilising downwind coupling). |  |
| `freestreamFvPatchField` | `freestream` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/freestream` | Generic free-stream mixed condition derived from inletOutlet, switching between the free-stream value and zero gradient by flow direction. |  |
| `inletOutletFvPatchField` | `inletOutlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/inletOutlet` | Generic outflow condition (zeroGradient) that switches to a specified inletValue when flow reverses. | valueFraction = 1 - pos0(phi_b) |
| `outletInletFvPatchField` | `outletInlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/outletInlet` | Generic inflow condition (zeroGradient) that switches to a specified outletValue on reverse (outward) flow. | valueFraction = pos0(phi_b) |
| `outletMappedUniformInletFvPatchField` | `outletMappedUniformInlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/outletMappedUniformInlet` | Averages the field over a named outlet patch and applies that single value uniformly over this (inlet) patch - a simple recycling inlet. | psi_b = areaAverage(psi over outletPatch) |
| `partialSlipFvPatchField` | `partialSlip` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/partialSlip` | Blends slip and no-slip using a user-supplied valueFraction field. | psi_b = f*0 + (1-f)*slip value |
| `slipFvPatchField` | `slip` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/slip` | Free-slip constraint (zero normal component, zero normal gradient of tangential components); derived from basicSymmetry. | U_n = 0, dU_t/dn = 0 |
| `uniformFixedGradientFvPatchField` | `uniformFixedGradient` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/uniformFixedGradient` | Fixed normal gradient uniform over the patch, supplied as a Function1 of time. |  |
| `uniformFixedValueFvPatchField` | `uniformFixedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/uniformFixedValue` | Fixed value uniform over the patch, supplied as a Function1 of time. |  |
| `uniformInletOutletFvPatchField` | `uniformInletOutlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/uniformInletOutlet` | inletOutlet with the inlet value supplied as a uniform time-varying Function1. |  |
| `zeroInletOutletFvPatchField` | `zeroInletOutlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/zeroInletOutlet` | inletOutlet with a zero inlet value: zeroGradient on outflow, zero on reverse flow. |  |

### derived BC (coded)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `codedFixedValueFvPatchField` | `codedFixedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/codedFixedValue` | Compiles user C++ code on the fly into a new fixedValue-derived BC which is then used to evaluate the patch. |  |
| `codedMixedFvPatchField` | `codedMixed` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/codedMixed` | Compiles user C++ code on the fly into a new mixed-derived BC which is then used to evaluate the patch. |  |

### derived BC (coupling)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `externalCoupledMixedFvPatchField` | `externalCoupled` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/externalCoupledMixed` | Mixed condition exchanging plain-text data files with an external application through a lock-file protocol. |  |

### derived BC (cyclic jump)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fanPressureJumpFvPatchScalarField` | `fanPressureJump` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fanPressureJump` | Fan pressure jump across a cyclic pair; the jump is a Function1 of the total volumetric flow rate through the patch, with optional reverse-flow handling and rpm scaling. | deltaP = fanCurve(Q) |
| `fixedJumpFvPatchField` | `fixedJump` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedJump` | Cyclic condition with a user-specified spatially varying jump (neighbour minus owner). | psi_neighbour - psi_owner = jump |
| `uniformJumpFvPatchField` | `uniformJump` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/uniformJump` | fixedJump with the jump supplied as a uniform time-varying Function1. |  |

### derived BC (cyclic)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `prghCyclicPressureFvPatchScalarField` | `prghCyclicPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/prghCyclicPressure` | Cyclic condition for p_rgh that corrects value and gradient on both sides to account for the non-cyclicity of the gravitational force. | jump = -rho*g&(patchTransform offset) |

### derived BC (density)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fixedPressureCompressibleDensityFvPatchScalarField` | `fixedPressureCompressibleDensity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedPressureCompressibleDensity` | Computes a (liquid) compressible density on the patch as a function of the specified pressure and the fluid properties. | rho = rho(p) from the thermo package evaluated at the patch pressure |

### derived BC (mapped)  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `mappedFlowRateVelocityFvPatchVectorField` | `mappedFlowRateVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/mappedFlowRateVelocity` | Maps the flow rate from a neighbouring patch and imposes the corresponding patch-normal velocity here. | U_b = -n * Q_mapped/(rho*A) |
| `mappedInternalValueFvPatchField` | `mappedInternalValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/mappedInternalValue` | Maps values from internal cells (an arbitrary sample location) onto this patch. |  |
| `mappedValueFvPatchField` | `mappedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/mappedValue` | Maps values from a neighbouring patch (possibly in another region) onto this patch, with optional offset/average control. |  |
| `mappedVelocityFluxFvPatchField` | `mappedVelocityFlux` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/mappedVelocityFlux` | Maps both velocity and the corresponding face flux from a neighbouring patch, keeping U and phi consistent (recycling inlet). |  |
| `timeVaryingMappedFixedValueFvPatchField` | `timeVaryingMappedFixedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/timeVaryingMappedFixedValue` | Interpolates a fixed value from a set of supplied points and time directories (constant/boundaryData), in space (nearest/planar interpolation) and linearly in time; supports setAverage/offset. |  |

### derived BC (outflow)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `advectiveFvPatchField` | `advective` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/advective` | Advective outflow condition solving DDt(W, field)=0 at the boundary with wave speed W; optional lInf/fieldInf relaxation towards a far-field value. | psi_b^n = (psi_b^o + k*psi_P)/(1 + k), k = W*deltaT*deltaCoeffs (+ far-field relaxation term) |
| `waveTransmissiveFvPatchField` | `waveTransmissive` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/waveTransmissive` | Wave-transmissive outflow derived from advective, with the advection speed set to U_n + c (acoustic speed from the thermo psi field and gamma). | W = phi/(rho*magSf) + sqrt(gamma/psi) |

### derived BC (phase fraction)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `interfaceCompressionFvPatchScalarField` | `interfaceCompression` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/interfaceCompression` | Applies interface compression to the phase-fraction distribution at the patch by sharpening alpha to 0 or 1 about 0.5. | alpha_b = pos0(alpha_P - 0.5) |
| `variableHeightFlowRateFvPatchScalarField` | `variableHeightFlowRate` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/variableHeightFlowRate` | Phase-fraction condition driven by local flow direction and constrained between user-specified lowerBound and upperBound. | zeroGradient on outflow, clipped to [lowerBound, upperBound]; fixed on inflow |

### derived BC (pressure)  <sub>(22)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dynamicPressureFvPatchScalarField` | `dynamicPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/dynamicPressure` | Base class for total/entrainment pressure conditions: subtracts a kinetic-energy term from a reference pressure to obtain the fixed boundary value; handles incompressible, compressible and transonic (gamma) variants. | p = p0 - 0.5*rho*\|U\|^2 (or the compressible/transonic isentropic form) |
| `entrainmentPressureFvPatchScalarField` | `entrainmentPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/entrainmentPressure` | Pressure condition for boundaries with uncertain flow direction: switches by flow direction, applying static pressure for outflow and total pressure (subtracting the dynamic head) for entrained inflow. |  |
| `fanPressureFvPatchScalarField` | `fanPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fanPressure` | Assigns either an inlet or outlet total-pressure condition for a fan, with the fan curve p(Q) supplied as a Function1 and a direction (in/out) keyword. | p0 = p_ambient +/- fanCurve(Q) |
| `fixedFluxExtrapolatedPressureFvPatchScalarField` | `fixedFluxExtrapolatedPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedFluxExtrapolatedPressure` | fixedFluxPressure that additionally extrapolates the pressure value from the interior (used where the boundary pressure itself must be second-order accurate). |  |
| `fixedFluxPressureFvPatchScalarField` | `fixedFluxPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedFluxPressure` | Sets the pressure gradient so the boundary flux matches exactly that implied by the velocity boundary condition; the standard p/p_rgh condition on walls and velocity-specified inlets. | snGrad(p) = (phiHbyA - phi_specified)/(\|Sf\|*Dp) |
| `freestreamPressureFvPatchScalarField` | `freestreamPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/freestreamPressure` | Free-stream pressure: outlet-inlet condition blending zero gradient at normal inlet with fixed value at normal outlet, based on the velocity orientation (partner of freestreamVelocity). |  |
| `phaseHydrostaticPressureFvPatchScalarField` | `phaseHydrostaticPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/phaseHydrostaticPressure` | Phase-based hydrostatic pressure: applies a hydrostatic distribution weighted by the phase fraction, as a mixed condition switched by the flux. | p = pRefValue + rho*g&(Cf - pRefPoint), blended by alpha |
| `plenumPressureFvPatchScalarField` | `plenumPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/plenumPressure` | Zero-dimensional model of an enclosed upstream plenum volume; the boundary pressure follows from mass and energy balances of the plenum given supply mass flow and temperature. | d(rho*V)/dt = mdot_supply - mdot_out ; ideal-gas energy balance for T and p |
| `pressureFvPatchScalarField` | `pressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressure` | Static pressure condition supplied as a Function1; base class for static-pressure conditions on alternative pressure variables such as p_rgh. | p_b = p(t) |
| `prghEntrainmentPressure` | `prghEntrainmentPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/PrghPressure/prghPressureFvPatchScalarFields.C` | p_rgh form of the entrainmentPressure condition - the usual open boundary for VoF cases. |  |
| `prghPressure` | `prghPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/PrghPressure/prghPressureFvPatchScalarFields.C` | p_rgh form of the static `pressure` condition. | p_rgh = p - rho*g&(h-hRef) |
| `prghTotalHydrostaticPressureFvPatchScalarField` | `prghTotalHydrostaticPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/prghTotalHydrostaticPressure` | Static p_rgh condition computed from the total pressure and the pre-computed hydrostatic field ph_rgh. | p_rgh = ph_rgh - 0.5*rho*\|U\|^2 on inflow, ph_rgh on outflow |
| `prghTotalPressure` | `prghTotalPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/PrghPressure/prghPressureFvPatchScalarFields.C` | p_rgh form of the totalPressure condition. |  |
| `prghUniformDensityHydrostaticPressure` | `prghUniformDensityHydrostaticPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/PrghPressure/prghPressureFvPatchScalarFields.C` | p_rgh form of the uniformDensityHydrostaticPressure condition. |  |
| `prghUniformTotalPressure` | `prghUniformTotalPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/PrghPressure/prghPressureFvPatchScalarFields.C` | p_rgh form of the uniformTotalPressure condition. |  |
| `rotatingTotalPressureFvPatchScalarField` | `rotatingTotalPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/rotatingTotalPressure` | Total pressure for patches in a rotating frame; the reference total pressure is adjusted by the rotational kinetic energy at each face radius. | p0_eff = p0 + 0.5*\|omega x r\|^2 (per unit rho) |
| `syringePressureFvPatchScalarField` | `syringePressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/syringePressure` | Zero-dimensional syringe-cylinder model: pressure from the initial volume, piston area, piston speed, compressibility and the accumulated boundary flux. | p = p0 + (V0 - Ap*x - integral(phi dt))/(psi*V) type compressibility relation |
| `totalPressureFvPatchScalarField` | `totalPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/totalPressure` | Inflow/outflow/entrainment pressure from a constant total pressure assumption: on outflow the static pressure equals the external p0; on inflow the dynamic head is subtracted. | p = p0 for outflow; p = p0 - 0.5*\|U\|^2 (per unit rho) for inflow |
| `transonicEntrainmentPressureFvPatchScalarField` | `transonicEntrainmentPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/transonicEntrainmentPressure` | Entrainment pressure extended to support supersonic jets leaving the domain (transonic isentropic relation with gamma). | p = p0/(1 + 0.5*(gamma-1)*M^2)^(gamma/(gamma-1)) form |
| `uniformDensityHydrostaticPressureFvPatchScalarField` | `uniformDensityHydrostaticPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/uniformDensityHydrostaticPressure` | Hydrostatic pressure computed with a single uniform reference density. | p = pRefValue + rhoRef*g&(Cf - pRefPoint) |
| `uniformTotalPressureFvPatchScalarField` | `uniformTotalPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/uniformTotalPressure` | totalPressure with the total pressure p0 supplied as a uniform Function1 of time. |  |
| `waveSurfacePressureFvPatchScalarField` | `waveSurfacePressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/waveSurfacePressure` | Free-surface pressure computed as the hydrostatic pressure of a surface displacement that is itself integrated in time from the boundary flux (selectable ddtScheme: Euler or CrankNicolson). | p = -rho*g&(zeta) with dzeta/dt = phi/(rho*\|Sf\|) |

### derived BC (pressure) template  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PrghPressureFvPatchScalarField<PressureField>` | `PrghPressure` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/PrghPressure` | Template converting any static-pressure BC into its p_rgh equivalent by subtracting the hydrostatic head. | p_rgh = p - rho*g&(h - hRef) |

### derived BC (temperature)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `inletOutletTotalTemperatureFvPatchScalarField` | `inletOutletTotalTemperature` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/inletOutletTotalTemperature` | Outflow condition for total temperature (supersonic cases) applying a user-specified value on reverse flow. |  |
| `totalTemperatureFvPatchScalarField` | `totalTemperature` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/totalTemperature` | Total (stagnation) temperature condition, converting T0 to static T using the local velocity and gamma. | T = T0/(1 + 0.5*(gamma-1)*\|U\|^2/(gamma*R*T)) |

### derived BC (turbulence)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `turbulentInletFvPatchField` | `turbulentInlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/turbulentInlet` | Generates a fluctuating inlet by adding a scaled random component to a reference (mean) field, with optional temporal correlation alpha. | psi_b = (1-alpha)*psi_b^o + alpha*(refValue + fluctuationScale*refValue*rand) |
| `turbulentIntensityKineticEnergyInletFvPatchScalarField` | `turbulentIntensityKineticEnergyInlet` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/turbulentIntensityKineticEnergyInlet` | Turbulent kinetic energy inlet from a user-supplied turbulence intensity expressed as a fraction of the mean velocity; inletOutlet-based. | k = 1.5*(I*\|U\|)^2 |

### derived BC (velocity inlet)  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `flowRateInletVelocityFvPatchVectorField` | `flowRateInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity` | Patch-normal inlet velocity scaled to match a specified massFlowRate, volumetricFlowRate or meanVelocity (Function1 of time); an optional Function1 profile (of normalised wall distance) shapes the velocity distribution. | U_b = -n * Q * profile(yHat) / integral(profile*dA) (mass form divides by rho) |
| `interstitialInletVelocityFvPatchVectorField` | `interstitialInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/interstitialInletVelocity` | Multiphase inlet where the actual interstitial phase velocity is the specified superficial inletVelocity divided by the local phase fraction. | U_b = inletVelocity/alpha |
| `pressureDirectedInletVelocityFvPatchVectorField` | `pressureDirectedInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureDirectedInletVelocity` | Inflow velocity from the flux directed along a user-specified inletDirection rather than the patch normal. | U_b = d * phi/(rho*\|Sf\|*(n & d)) |
| `pressureInletUniformVelocityFvPatchVectorField` | `pressureInletUniformVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureInletUniformVelocity` | As pressureInletVelocity but the inflow speed is area-averaged over the patch and applied uniformly along the average normal. |  |
| `pressureInletVelocityFvPatchVectorField` | `pressureInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureInletVelocity` | Inflow velocity obtained from the flux, directed normal to the patch faces; used where pressure is specified. | U_b = n * phi/(rho*\|Sf\|) |
| `swirlFlowRateInletVelocityFvPatchVectorField` | `swirlFlowRateInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/swirlFlowRateInletVelocity` | Normal velocity set to match a mass/volumetric flow rate, with radial and tangential components specified as Function1s of time and radius or by an rpm. | U = -n*Un + r_hat*Ur(r,t) + theta_hat*Ut(r,t) |
| `swirlInletVelocityFvPatchVectorField` | `swirlInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/swirlInletVelocity` | Axial, radial and tangential inlet velocity components each specified directly as Function1s of time and radius, or via an angular speed. |  |
| `variableHeightFlowRateInletVelocityFvPatchVectorField` | `variableHeightFlowRateInletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/variableHeightFlowRateInletVelocity` | Multiphase inlet velocity from a specified volumetric flow rate, distributed in proportion to the phase fraction alpha over the patch. | U_b = -n * Q * alpha_f / sum(alpha_f*\|Sf\|) |

### derived BC (velocity outlet)  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `flowRateOutletVelocityFvPatchVectorField` | `flowRateOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/flowRateOutletVelocity` | Corrects the extrapolated outlet velocity so the mass or volumetric flow rate matches a specified value. | U_b = U_extrap * Q_target/Q_extrap (normal-component scaling) |
| `fluxCorrectedVelocityFvPatchVectorField` | `fluxCorrectedVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fluxCorrectedVelocity` | Velocity outlet for pressure-specified patches: zeroGradient then the normal component corrected from the flux. | U_b = U_extrap - n*(n & U_extrap) + n*phi/(rho*\|Sf\|) |
| `matchedFlowRateOutletVelocityFvPatchVectorField` | `matchedFlowRateOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/matchedFlowRateOutletVelocity` | Corrects the extrapolated outlet velocity to match the flow rate of a named corresponding inlet patch. |  |
| `outletPhaseMeanVelocityFvPatchVectorField` | `outletPhaseMeanVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/outletPhaseMeanVelocity` | Adjusts the outlet velocity of a given phase to achieve a specified mean, so the phase fraction adapts to the mass flow rate (wave/free-surface outlets). |  |

### derived BC (velocity)  <sub>(10)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fixedNormalInletOutletVelocityFvPatchVectorField` | `fixedNormalInletOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/fixedNormalInletOutletVelocity` | Combines a fixed normal component from a supplied "normalVelocity" patchField with a fixed or zero-gradient tangential component chosen by the flow direction. |  |
| `freestreamVelocityFvPatchVectorField` | `freestreamVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/freestreamVelocity` | Free-stream velocity: inlet-outlet condition that uses the velocity orientation to blend continuously between fixed value at normal inlet and zero gradient at normal outlet. | valueFraction from the cosine of the angle between U and n |
| `pressureDirectedInletOutletVelocityFvPatchVectorField` | `pressureDirectedInletOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureDirectedInletOutletVelocity` | zeroGradient on outflow; on inflow the velocity is obtained from the flux along a specified inletDirection. |  |
| `pressureInletOutletParSlipVelocityFvPatchVectorField` | `pressureInletOutletParSlipVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureInletOutletParSlipVelocity` | As pressureInletOutletVelocity but the tangential (parallel) component on inflow is taken as slip (from the internal field) rather than zero. |  |
| `pressureInletOutletVelocityFvPatchVectorField` | `pressureInletOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureInletOutletVelocity` | zeroGradient on outflow; on inflow the velocity comes from the flux normal to the patch. Optional tangentialVelocity. Standard partner of fixedValue/totalPressure/entrainmentPressure p BCs. |  |
| `pressureNormalInletOutletVelocityFvPatchVectorField` | `pressureNormalInletOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/pressureNormalInletOutletVelocity` | zeroGradient on outflow; on inflow the velocity is obtained from the flux in the patch-normal direction only. |  |
| `rotatingPressureInletOutletVelocityFvPatchVectorField` | `rotatingPressureInletOutletVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/rotatingPressureInletOutletVelocity` | pressureInletOutletVelocity in a rotating frame: the tangential velocity is computed from the angular speed (Function1) and rotation axis/origin. | tangentialVelocity = omega x (Cf - origin) |
| `supersonicFreestreamFvPatchVectorField` | `supersonicFreestream` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/supersonicFreestream` | Supersonic free-stream: supersonic outflow is vented, supersonic inflow is treated with a Prandtl-Meyer expansion from the specified free-stream conditions (UInf, pInf, TInf, gamma). | Prandtl-Meyer turning to match the local flow direction; isentropic relations for p, T |
| `surfaceNormalFixedValueFvPatchVectorField` | `surfaceNormalFixedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/surfaceNormalFixedValue` | Vector fixed value obtained by scaling the patch normals by a supplied scalar field (negative = into the domain). | U_b = refValue * n |
| `surfaceNormalUniformFixedValueFvPatchVectorField` | `surfaceNormalUniformFixedValue` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/surfaceNormalUniformFixedValue` | As surfaceNormalFixedValue but the scalar magnitude comes from a Function1 of time. | U_b = uniformValue(t) * n |

### derived BC (wall velocity)  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `movingMappedWallVelocityFvPatchVectorField` | `movingMappedWallVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/movingMappedWallVelocity` | No-slip velocity for mapped walls where the wall velocity is taken to be the mesh velocity of the neighbouring (e.g. solid) region. |  |
| `movingWallSlipVelocityFvPatchVectorField` | `movingWallSlipVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/movingWallSlipVelocity` | Slip velocity condition on moving walls: tangential velocity free, normal component follows the mesh motion. |  |
| `movingWallVelocityFvPatchVectorField` | `movingWallVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/movingWallVelocity` | No-slip on rigid or flexible moving walls whose mesh vertices move with the surface; removes the normal mesh-motion component so the flux is consistent. | U_b = Uwall - n*(n & Uwall) + n*(phi_mesh/\|Sf\|) |
| `noSlipFvPatchVectorField` | `noSlip` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/noSlip` | Zero velocity at stationary walls; null-constructable so it can be used as a default. For walls whose vertices slide, the normal component is still handled correctly. | U_b = 0 |
| `rotatingWallVelocityFvPatchVectorField` | `rotatingWallVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/rotatingWallVelocity` | Velocity on a rotating solid of revolution: tangential velocity from angular speed (Function1) and axis/origin, with the normal component removed. | U_b = omega x (Cf - origin), then U_b -= n*(n & U_b) |
| `translatingWallVelocityFvPatchVectorField` | `translatingWallVelocity` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/translatingWallVelocity` | Uniform translational wall velocity supplied as a Function1 of time, with the normal component removed. | U_b = U(t) - n*(n & U(t)) |

### discretisation operator set  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvc (explicit operator namespace)` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvc.H` | Explicit finite-volume calculus operators evaluated on existing fields (no matrix). |  |
| `fvm (implicit operator namespace)` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvm/fvm.H` | Implicit (matrix-generating) operators: fvm::ddt, fvm::d2dt2, fvm::div, fvm::laplacian, fvm::Sp, fvm::SuSp, fvm::S. Each dispatches through the corresponding runtime-selected scheme from fvSchemes. | assembles fvMatrix<Type> diag/upper/lower/source for each term |

### div scheme  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::gaussDivScheme<Type>` | `Gauss` | `[Foundation-12] src/finiteVolume/finiteVolume/divSchemes/gaussDivScheme` | Second-order Gauss divergence using a runtime-selected face interpolation of the field and Gauss' theorem. | div(vf) = (1/V) sum_f Sf & interp(vf)_f |

### div scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::divScheme<Type>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/divSchemes/divScheme/divScheme.H` | Abstract base for explicit divergence schemes (div of a vol tensor/vector field, not flux-convection); holds the surfaceInterpolationScheme used for face values. | div(vf) = (1/V) * sum_f Sf & vf_f |

### explicit field utility  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvc::smooth / fvc::spread / fvc::sweep` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcSmooth/fvcSmooth.H` | FvFaceCellWave-based field operations: smooth (ensures neighbour values are at least coeff*cell value), spread (spreads a field into a region using an alpha field), sweep (propagates values a number of layers). |  |

### explicit field utility support  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `smoothData / sweepData` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcSmooth/smoothData.H, sweepData.H` | FvFaceCellWave transport types carrying the value and the update rule for fvc::smooth and fvc::sweep respectively. |  |

### explicit operator  <sub>(16)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvc::Sp / fvc::SuSp / fvc::S` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcSup.H` | Explicit evaluation of the implicit/explicit source terms matching the fvm equivalents. |  |
| `fvc::average` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcAverage.H` | Area-weighted average of a surface field back onto cells. | average = surfaceSum(\|Sf\|*ssf)/surfaceSum(\|Sf\|) |
| `fvc::cellReduce` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcCellReduce.H` | Build a vol field from a surface field with an arbitrary combine operator (e.g. maxEqOp, minEqOp) over each cell's faces - used by localMax/localMin type reductions. |  |
| `fvc::curl` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcCurl.H` | Curl of a vol field as the Hodge dual of the skew part of its gradient. | curl(U) = 2*(*skew(grad(U))) |
| `fvc::ddt / fvc::d2dt2 / fvc::ddtCorr` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcDdt.H, fvcD2dt2.H` | Explicit time derivatives and the ddtCorr flux corrections used for Rhie-Chow-style transient consistency. |  |
| `fvc::div` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcDiv.H` | Explicit divergence of surface fluxes and of volume fields (routed through divScheme or convectionScheme). | div(ssf) = (1/V) sum_f ssf_f |
| `fvc::flux` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcFlux.H, fvcFluxTemplates.C` | Face flux of a field: flux(vvf) = Sf & interp(vvf), and flux(phi, vf, scheme) for convective fluxes through the selected convection scheme. | phi_f = Sf & interp(U)_f |
| `fvc::grad` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcGrad.H` | Explicit gradient of a vol or surface field through the selected gradScheme. |  |
| `fvc::interpolate / fvc::dotInterpolate` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/surfaceInterpolation/surfaceInterpolate.H` | Cell-to-face interpolation entry point; looks the scheme up in the interpolationSchemes dictionary of fvSchemes (or accepts an explicit scheme name/weights). | vf_f = w_f*vf_P + (1-w_f)*vf_N + correction |
| `fvc::laplacian` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcLaplacian.H` | Explicit laplacian through the selected laplacianScheme. |  |
| `fvc::magSqrGradGrad` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcMagSqrGradGrad.H` | Magnitude squared of the gradient of the gradient - used as a mesh-refinement/error indicator. | sum_i magSqr(grad(grad(vf)_i)) |
| `fvc::meshPhi / makeRelative / makeAbsolute / absolute / relative` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcMeshPhi.H, fvcMeshPhiTemplates.C` | Compute the mesh-motion flux and convert face fluxes between absolute and relative (moving-mesh) frames. | phi_rel = phi_abs - meshPhi |
| `fvc::reconstruct / fvc::simpleReconstruct / fvc::reconstructMag` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcReconstruct.H, fvcSimpleReconstruct.C, fvcReconstructMag.C` | Reconstruct a cell-centred vector field from a face flux field (inverse of Sf&U); reconstructMag returns the magnitude form used for the p_rgh/interface pressure work. | reconstruct(ssf) = inv(sum_f Sf (Sf/\|Sf\|)) & sum_f Sf*(ssf_f/\|Sf\|) |
| `fvc::snGrad` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcSnGrad.H` | Explicit surface-normal gradient through the selected snGradScheme. |  |
| `fvc::surfaceIntegrate / fvc::surfaceSum` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcSurfaceIntegrate.H` | Sum a surface field over the faces of each cell, divided by cell volume (surfaceIntegrate) or not (surfaceSum). | (1/V_P) sum_f ssf_f |
| `fvc::volumeIntegrate / fvc::domainIntegrate` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvc/fvcVolumeIntegrate.H` | Multiply a vol field by cell volume, or integrate it over the whole (parallel-reduced) domain returning a dimensioned<Type>. | sum_cells vf_P * V_P |

### field mapping  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MapFvFields / MapFvVolField / MapFvSurfaceField` |  | `[Foundation-12] src/finiteVolume/interpolation/mapping/fvFieldMappers` | Templated mapping functors that remap vol and surface fields (and their boundary fields) when the mesh changes topology or is redistributed. |  |

### field source condition  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `internalFvFieldSource` | `internal` | `[Foundation-12] src/finiteVolume/fields/fvFieldSources/derived/internal` | Source value equals the local internal (cell) value - i.e. the source carries the field it finds. | psi_source = psi_cell |
| `turbulentIntensityKineticEnergyFvScalarFieldSource` | `turbulentIntensityKineticEnergy` | `[Foundation-12] src/finiteVolume/fields/fvFieldSources/derived/turbulentIntensityKineticEnergy` | Turbulent kinetic energy of injected material from a specified turbulence intensity as a fraction of the mean velocity. | k = 1.5*(I*\|U\|)^2 |
| `uniformFixedValueFvFieldSource` | `uniformFixedValue` | `[Foundation-12] src/finiteVolume/fields/fvFieldSources/derived/uniformFixedValue` | Source value is a uniform Function1 of time. |  |
| `uniformInletOutletFvFieldSource` | `uniformInletOutlet` | `[Foundation-12] src/finiteVolume/fields/fvFieldSources/derived/uniformInletOutlet` | Uniform fixed value when the source is positive (injection) and the internal value when negative (a sink). |  |

### field source condition base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvFieldSource<Type>` | `fvFieldSource` | `[Foundation-12] src/finiteVolume/fields/fvFieldSources/fvFieldSource` | Base class for the per-field "sources" entries that specify what value a field takes where an fvModel injects or removes mass/energy; runtime selected by name per field type. |  |

### filtered limiter  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `filteredLinear2Limiter / filteredLinear2VLimiter (filteredLinear2 / filteredLinear2V)` | `filteredLinear2, filteredLinear2V` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/filteredLinear2` | Filtered linear scheme with two user coefficients (k strength, l twist/scaling of the cell deltas); clipped to [0,1]. | limiter = 1 - k*min(max(df - tdcP, 0), max(df - tdcN, 0))/(max(\|df\|, max(\|tdcP\|,\|tdcN\|)) + small) |
| `filteredLinear3Limiter / filteredLinear3VLimiter (filteredLinear3 / filteredLinear3V)` | `filteredLinear3, filteredLinear3V` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/filteredLinear3` | Filtered linear scheme variant 3 with coefficients k, l; the filter is a product form that detects local extrema. | limiter = 1 - k*(dN - df)*(dP - df)/max((dN + dP)^2, small) ; clipped to [0,1] |
| `filteredLinearLimiter (filteredLinear)` | `filteredLinear` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/filteredLinear` | Linear scheme with a mild filter that never drops the limiter below 0.8; no user coefficients. | limiter = 1 - 0.5*min(\|df-dcP\|, \|df-dcN\|)/(max(\|dcP\|,\|dcN\|) + small) ; clipped to [0.8, 1] |

### fit polynomial  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `biLinearFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/biLinearFit/biLinearFitPolynomial.H` | Bi-linear polynomial basis for centred fit schemes. | P = {1, x, y, z, xy, xz} |
| `cubicUpwindFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/cubicUpwindFit/cubicUpwindFitPolynomial.H` | Cubic polynomial basis for upwind-biased fit interpolation. | P = {1, x, x^2, x^3, y, ...} |
| `linearFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearFit/linearFitPolynomial.H` | Linear polynomial basis (nTerms = 3 in 3-D, 2 in 2-D, 1 in 1-D) for centred fit schemes. | P = {1, x, y, z} |
| `quadraticFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticFit/quadraticFitPolynomial.H` | Full quadratic polynomial basis for centred fit interpolation and centred fit snGrad. | P = {1, x, y, z, x^2, y^2, z^2, xy, xz, yz} |
| `quadraticLinearFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticLinearFit/quadraticLinearFitPolynomial.H` | Quadratic normal to the face, linear in the plane of the face. | P = {1, x, y, z, x^2, xy, xz} |
| `quadraticLinearUpwindFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticLinearUpwindFit/quadraticLinearUpwindFitPolynomial.H` | Quadratic-normal / linear-tangential polynomial basis for upwind-biased fit interpolation. |  |
| `quadraticUpwindFitPolynomial` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticUpwindFit/quadraticUpwindFitPolynomial.H` | Quadratic polynomial basis for upwind-biased fit interpolation. |  |

### flux-based limiter  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PhiLimiter (Phi)` | `Phi` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/Phi` | Phi scheme (registered for vector only through makePhiSurfaceInterpolationScheme); limiter computed from the face flux and face area rather than the NVD ratio, with coefficient k in [0,1]. | limiter based on phiU = faceFlux/magSf relative to the CD weight and k |

### grad limiter  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::gradientLimiters::Venkatakrishnan` | `Venkatakrishnan (as cellLimited<Venkatakrishnan>)` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad/gradientLimiters/VenkatakrishnanGradientLimiter.H` | Differentiable Venkatakrishnan (1993) limiter; formally can exceed 1 so it is clipped, which breaks differentiability (cubic is recommended instead). | limiter(r) = (r^2 + 2r)/(r^2 + r + 2) |
| `fv::gradientLimiters::cubic` | `cubic (as cellLimited<cubic> <rt>)` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad/gradientLimiters/cubicGradientLimiter.H` | Michalak & Ollivier-Gooch (2008) cubic limiter fitted to obey value and gradient constraints; takes a transition point rt (>1, typically 1.5) at which the limiter reaches exactly 1. | a = 1/rt^2 - 2/rt^3 ; b = -(3/2) a rt - (1/2)/rt ; limiter(r) = ((a r + b) r + 1) r for r<rt, else 1 |
| `fv::gradientLimiters::minmod` | `minmod (implicit default of cellLimited)` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad/gradientLimiters/minmodGradientLimiter.H` | Default cellLimited limiter that simply clips the gradient to remove cell-to-face extrapolation unboundedness. | limiter(r) = min(r, 1) |

### grad scheme  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::fourthGrad<Type>` | `fourth` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/fourthGrad` | Fourth-order gradient: combines a leastSquares gradient with a Gauss linear gradient of the same field to add a higher-order correction. | grad = lsGrad + correction from surface-integrated (interp(vf) - 1/2(vf_P+vf_N) - 1/8 d&(gradP-gradN)) |
| `fv::gaussGrad<Type>` | `Gauss` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/gaussGrad` | Second-order Gauss gradient from face interpolation (runtime-selected surfaceInterpolationScheme) and Gauss' theorem; applies correctBoundaryConditions to remove the normal component on non-fixedValue patches. | grad(vf)_P = (1/V_P) sum_f Sf * interp(vf)_f |
| `fv::leastSquaresGrad<Type>` | `leastSquares` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/leastSquaresGrad.H` | Second-order gradient by least-squares fit over face neighbours using precomputed leastSquaresVectors. | grad(vf)_P = sum_f lsP_f * (vf_N - vf_P) ; grad(vf)_N = sum_f lsN_f * (vf_N - vf_P) |

### grad scheme (extended stencil)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::LeastSquaresGrad<Type,Stencil>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/ (LeastSquaresGrad.H, LeastSquaresGrads.C, LeastSquaresVectors.*)` | Stencil-templated least-squares gradient compiled via $(gradSchemes)/LeastSquaresGrad/LeastSquaresGrads.C in Make/files, using centred cell-to-cell stencil objects (centredCPCCellToCellStencilObject / centredCECCellToCellStencilObject / centredCFCCellToCellStencilObject, all present under fvMesh/extendedStencil/cellToCell). NOTE: its source files are NOT readable in this checkout - the case-insensitive Windows filesystem merged directory LeastSquaresGrad/ with leastSquaresGrad/ and the capitalised files were overwritten, so its registered scheme keywords could not be verified from source. | grad(vf)_P = sum_{c in stencil} lsv_c * (vf_c - vf_P) |

### grad scheme (limited)  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::cellLimitedGrad<Type,Limiter>` | `cellLimited (minmod default), cellLimited<Venkatakrishnan>, cellLimited<cubic>` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad` | Applies a scalar cell limiter to a runtime-selected base gradient scheme so extrapolated face values stay between the min/max of the cell and its neighbours; the same limiter scales all gradient components. Coefficient k in [0,1]. | maxDelta/minDelta from neighbour values; r = delta/(extrapolated face delta); limiter = Limiter::limiter(r); grad := limiter*grad |
| `fv::cellMDLimitedGrad<Type>` | `cellMDLimited` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellMDLimitedGrad` | Multi-directional cell limiter on a runtime-selected base gradient: limits the extrapolated face values between the cell/neighbour min and max, applied in each face direction separately rather than isotropically. | per-face directional clipping of g via limitFace(g, maxDelta, minDelta, extrapolate) |
| `fv::faceLimitedGrad<Type>` | `faceLimited` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/faceLimitedGrad` | Limits extrapolated face values between the two face-neighbour cell values only; single scalar limiter applied to all gradient components. | maxDelta = k*(vfN - vfP); limiter from min/max of (d & grad) vs those bounds |
| `fv::faceMDLimitedGrad<Type>` | `faceMDLimited` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/faceMDLimitedGrad` | Multi-directional variant of faceLimitedGrad; limiter applied to the gradient in each face direction separately. | per-face directional clipping between face-neighbour values |

### grad scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::gradScheme<Type>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/gradScheme/gradScheme.H` | Abstract base for gradient schemes with Istream runtime selection and grad-field caching (fvSolution cache "grad(...)"). | grad(vf) |

### grad scheme support  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `invDistLeastSquaresVectors (alternative implementation)` |  | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/invDistLeastSquaresVectors.C` | Alternative calcLeastSquaresVectors() implementation using inverse-distance (1/\|d\|) weighting (not listed in Make/files). | dd_P = sum_f (1/\|d\|) d d |
| `unweightedLeastSquaresVectors (alternative implementation)` |  | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/unweightedLeastSquaresVectors.C` | Alternative calcLeastSquaresVectors() implementation with unit weights (not listed in Make/files; swap-in replacement for leastSquaresVectors.C). | dd_P = sum_f d d (no 1/\|d\|^2 weight) |

### grad scheme support (MeshObject)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `leastSquaresVectors` | `leastSquaresVectors` | `[Foundation-12] src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/leastSquaresVectors.H` | DemandDrivenMeshObject caching the per-face least-squares vectors pVectors_/nVectors_ obtained by inverting the symmetric dd tensor; default weighting is 1/\|d\|^2. | dd_P = sum_f w_f d d ; lsP_f = w_f * (inv(dd_P) & d) |

### implicit operator  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvm::Sp / fvm::SuSp / fvm::S` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvm/fvmSup.H` | Implicit source terms: Sp (always implicit on the diagonal), SuSp (implicit if the coefficient is negative, otherwise explicit), S (explicit/implicit split of a general source). | Sp: diag += sp*V ; SuSp: diag += max(sp,0)*V, source -= min(sp,0)*V*vf |
| `fvm::ddt / fvm::d2dt2` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvm/fvmDdt.H, fvmD2dt2.H` | Implicit time-derivative operators; overloads for ddt(vf), ddt(rho,vf), ddt(alpha,rho,vf) and the ddtCorr flux corrections. |  |
| `fvm::div` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvm/fvmDiv.H` | Implicit convection operator div(phi, vf) routed through the selected convectionScheme. |  |
| `fvm::laplacian` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fvm/fvmLaplacian.H` | Implicit diffusion operator laplacian(gamma, vf) for scalar, vol and surface gamma, routed through the selected laplacianScheme. |  |

### inlet velocity profile (Function1)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Function1s::laminarBL` | `laminarBL` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity/laminarBL` | Laminar boundary-layer shape function of the normalised wall distance, for use as the flowRateInletVelocity profile. | f(yHat) = yHat*(2 - yHat) |
| `Function1s::turbulentBL` | `turbulentBL` | `[Foundation-12] src/finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity/turbulentBL` | Turbulent power-law boundary-layer shape function of the normalised wall distance; exponent defaults to 1/7. | f(yHat) = yHat^exponent |

### laplacian scheme  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::gaussLaplacianScheme<Type,GType>` | `Gauss` | `[Foundation-12] src/finiteVolume/finiteVolume/laplacianSchemes/gaussLaplacianScheme` | Second-order Gauss laplacian: implicit orthogonal part from deltaCoeffs plus explicit non-orthogonal correction from the selected snGrad scheme; sets flux if fluxRequired. | fvm: upper = deltaCoeffs*gammaMagSf, diag = -sum(upper); source += gammaMagSf*snGradCorrection |

### laplacian scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::laplacianScheme<Type,GType>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/laplacianSchemes/laplacianScheme/laplacianScheme.H` | Abstract base for laplacian schemes; owns both the gamma interpolation scheme and the snGrad scheme selected from the stream. | laplacian(gamma, vf) = (1/V) sum_f gamma_f * \|Sf\| * snGrad(vf)_f |

### limited surface interpolation base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `limitedSurfaceInterpolationScheme<Type>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSurfaceInterpolationScheme` | Base for all flux-aware limited (NVD/TVD) schemes; converts a limiter field into interpolation weights blending central differencing with upwind. Defines Mesh and MeshFlux tables for scalar, vector, sphericalTensor, symmTensor and tensor. | w_f = limiter*w_CD + (1 - limiter)*pos0(phi_f) |

### limited surface interpolation scheme  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `blended<Type>` | `blended` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/blended` | Fixed linear/upwind blend expressed as a limitedSurfaceInterpolationScheme with a constant limiter (blending factor). | w = blend*w_linear + (1-blend)*pos0(phi) ; limiter = blend |
| `linearUpwind<Type>` | `linearUpwind` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind` | Second-order linear-upwind: upwind weights plus a gradient-based explicit correction using a named gradScheme (e.g. `linearUpwind grad(U)`). | corr_f = (Cf - C_upwind) & grad(vf)_upwind |
| `linearUpwindV<Type>` | `linearUpwindV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind/linearUpwindV.H` | Vector-specific linear-upwind that limits the correction by the direction of the change in the vector field (registered for vector only). | corr scaled by max(min((vfN-vfP)&corr/magSqr, 1), 0) style directional limiting |
| `upwind<Type>` | `upwind` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/upwind` | First-order upwind; the base limitedSurfaceInterpolationScheme with a limiter of zero. Registered in both limited and plain surfaceInterpolation tables. | w_f = pos0(phi_f) ; limiter = 0 |

### limited surface interpolation template  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `LimitedScheme<Type,Limiter,LimitFunc>` | `LimitedScheme` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme` | Template generating NVD/TVD limited weights: computes the limit function of the field, its gradient, then calls Limiter::limiter(cdWeight, faceFlux, phiP, phiN, gradcP, gradcN, d) per face; caches the limiter field if fvSolution cache "limiter" is on. |  |
| `PhiScheme<Type,PhiLimiter>` | `PhiScheme` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/PhiScheme` | Template generating weighting factors from the face flux and face area rather than the NVD/TVD ratio; converts mass flux to volumetric flux using rho when needed. |  |

### limiter (quadratic upwind)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `QUICKLimiter / QUICKVLimiter (QUICK / QUICKV)` | `QUICK, QUICKV` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/QUICK` | Quadratic-upwind interpolation expressed through the limiter framework; weights are not bounded between upwind and central so some downwind contribution occurs, but the interpolate is clipped between the upwind and downwind cell values. | limiter based on the QUICK 3/8-6/8-(-1/8) weighting expressed via r |

### limiter function base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NVDTVD` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/NVDTVD.H` | Scalar NVD/TVD normalised-variable helper providing phict() and r() from the face and upwind-cell gradients. | gradf = phiN - phiP ; gradcf = d & grad_upwind ; phict = 1 - 0.5*gradf/gradcf ; r = 2*(gradcf/gradf) - 1 |

### limiter function base (vector)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NVDVTVDV` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/NVDVTVDV.H` | Vector form of the NVD/TVD helper, projecting the tensor gradients onto the face vector difference; used by all the *V scheme variants. | gradfV = phiN - phiP ; gradf = gradfV & gradfV ; gradcf = gradfV & (d & gradc_upwind) |

### limiter limit-function  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `limitFuncs::magSqr<Type>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/LimitFuncs.H` | Default limit function mapping a general field to the scalar used for limiting: magSqr for general types, identity for scalar, trace for symmTensor/tensor. |  |
| `limitFuncs::rhoMagSqr<Type>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/LimitFuncs.H` | Limit function based on the density-normalised field magSqr(phi/rho) (rho looked up from the registry); available but the MUSCL registrations that used it are commented out. |  |

### limiter wrapper  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Limited01Limiter<LimitedScheme>` | `used as <Scheme>01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/Limited01/Limited01.H` | LimitedLimiter specialised to the range [0,1] - used for phase fractions and other bounded scalars. | LimitedLimiter(0, 1, is) |
| `LimitedLimiter<LimitedScheme>` | `used as limited<Scheme> <lower> <upper>` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/Limited/Limited.H` | Wraps any NVD/TVD limiter and reverts to pure upwind (limiter = 0) when either cell value falls outside a user-supplied [lowerBound, upperBound] range. | limiter = 0 if out of bounds else base limiter |

### matrix  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMatrix<Type>` |  | `[Foundation-12] src/finiteVolume/fvMatrices/fvMatrix` | The core FV matrix: an lduMatrix plus a source field, per-patch internal/boundary coefficients, face-flux storage and the psi field reference. Supplies relax(), solve(), A(), H(), H1(), flux(), DD(), setValues/setReference, residual(), operator arithmetic (+,-,*,==) and dimension checking. Face addressing keeps assembly and solution loops vectorised. | [A]{psi} = {b}; A() = diag/V ; H() = (b - sum_N a_N psi_N)/V ; flux_f = a_f*(psi_N - psi_P) |
| `fvScalarMatrix (fvMatrix<scalar> specialisation)` |  | `[Foundation-12] src/finiteVolume/fvMatrices/fvScalarMatrix` | Scalar specialisation providing setComponentReference, the direct solver() and solveSegregated() overloads, and the residual for scalar equations. |  |

### matrix solver wrapper  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMatrix<Type>::fvSolver` |  | `[Foundation-12] src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C` | Holds a cached lduMatrix::solver so a matrix can be solved repeatedly (e.g. per component) without re-selecting and re-initialising the linear solver; solver/preconditioner/smoother keywords come from the fvSolution solvers dictionary. |  |

### mesh interpolation data  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `surfaceInterpolation` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/surfaceInterpolation/surfaceInterpolation.H` | fvMesh base class holding the geometric central-differencing weights, deltaCoeffs, nonOrthDeltaCoeffs and nonOrthCorrectionVectors used by all schemes. | w_f = \|Sf&(Cf-CN)\| / (\|Sf&(Cf-CP)\| + \|Sf&(Cf-CN)\|) |

### multivariate interpolation base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `multivariateSurfaceInterpolationScheme<Type>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/multivariateSurfaceInterpolationScheme` | Abstract base for schemes that apply one shared limiter to a table of fields (used by multivariateGaussConvectionScheme); registered for scalar. |  |

### multivariate interpolation scheme  <sub>(10)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `multivariate Gamma` | `Gamma` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/Gamma/multivariateGamma.C` | Gamma limiter shared across the field table via multivariateScheme. |  |
| `multivariate MUSCL` | `MUSCL` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/MUSCL/multivariateMUSCL.C` | MUSCL limiter shared across the field table. |  |
| `multivariate Minmod` | `Minmod` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/Minmod/multivariateMinmod.C` | Minmod limiter shared across the field table. |  |
| `multivariate SuperBee` | `SuperBee` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/SuperBee/multivariateSuperBee.C` | SuperBee limiter shared across the field table. |  |
| `multivariate limitedCubic` | `limitedCubic` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/limitedCubic/multivariateLimitedCubic.C` | limitedCubic limiter shared across the field table. |  |
| `multivariate limitedLinear (+ limitedLimitedLinear, limitedLinear01)` | `limitedLinear, limitedLimitedLinear, limitedLinear01` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/limitedLinear/multivariateLimitedLinear.C` | limitedLinear limiter shared across the field table, plus the explicitly bounded and [0,1]-bounded variants. |  |
| `multivariate vanLeer` | `vanLeer` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/vanLeer/multivariateVanLeer.C` | vanLeer limiter shared across the field table. |  |
| `multivariateIndependentScheme<Type>` | `multivariateIndependent` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/multivariateIndependentScheme` | Applies each field's own limited scheme independently (no coupling of limiters between fields). |  |
| `multivariateSelectionScheme<Type>` | `multivariateSelection` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/multivariateSelectionScheme` | Allows a different limited scheme per field but applies the minimum resulting limiter to all of them, keeping the set consistent. | limiter = min over fields of their individual scheme limiters |
| `multivariateUpwind<Type>` | `upwind (Multivariate table)` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/upwind/multivariateUpwind.H` | Pure upwind applied uniformly to all fields in the multivariate table. | w = pos0(phi) |

### multivariate interpolation template  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `multivariateScheme<Type,Scheme>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/multivariateSchemes/multivariateScheme` | Template that wraps a single limited scheme and applies the minimum of that scheme's limiters across all fields in the table. | limiter = min over fields of limiter_i |

### namespace  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv` |  | `[Foundation-12] src/finiteVolume/finiteVolume/fv/fv.H` | Namespace holder for all finite-volume scheme classes; declares NamespaceName("fv") used for the shared debug switch. |  |

### point interpolation  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `interpolationCell<Type>` | `cell` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationCell` | Zeroth-order: returns the owning cell value for any point in the cell. | psi(x) = psi_cell |
| `interpolationCellPatchConstrained<Type>` | `cellPatchConstrained` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationCellPatchConstrained` | Cell value everywhere except on a boundary face, where the boundary value is used directly. Does not work on empty patches. |  |
| `interpolationCellPoint<Type>` | `cellPoint` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationCellPoint` | Decomposes the cell into tetrahedra using cell-centre and point values and linearly interpolates within the containing tet. | psi(x) = sum_i w_i psi_i over the tet vertices (barycentric weights) |
| `interpolationCellPointFace<Type>` | `cellPointFace` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationCellPointFace` | Interpolation using cell, point and face values; searches for the containing tet and falls back to the closest tet if the position lies outside all of them. |  |
| `interpolationCellPointWallModified<Type>` | `cellPointWallModified` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationCellPointWallModified` | As cellPoint but the point field is modified on wall faces (vectors only): extrapolated to the wall then rotated towards the reverse point normal so vectors never point out of the domain, scaled to zero beyond 90 degrees. Prevents unresolvable drag-rebound couplings in Lagrangian tracking. |  |
| `interpolationPointMVC<Type>` | `pointMVC` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationPointMVC` | Mean Value Coordinates interpolation directly from the cell's vertices (no tet decomposition). | spherical barycentric / mean-value coordinate weights (Langer, Belyaev & Seidel 2006) |

### point interpolation base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `interpolation<Type>` | `interpolation` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolation` | Abstract base for interpolating a vol field to an arbitrary position inside a cell (used mainly by Lagrangian tracking and sampling); runtime selected by name for all field types. |  |
| `interpolationVolPointInterpolation<Type>` | `interpolationVolPointInterpolation` | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationVolPointInterpolation` | Intermediate base for interpolations that first need a vol-to-point interpolated field; owns the point field and its lifetime. |  |

### point interpolation support  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cellPointWeight` |  | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationCellPoint/cellPointWeight` | Finds the tetrahedron containing a position and computes its barycentric weights and the indices of the contributing cell/face/points. |  |
| `pointMVCWeight` |  | `[Foundation-12] src/finiteVolume/interpolation/interpolation/interpolationPointMVC/pointMVCWeight.H` | Container computing the Mean Value Coordinates weights for a position within a cell, following VTK's vtkMeanValueCoordinatesInterpolator. |  |

### scheme framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvSchemes` | `system/fvSchemes` | `[Foundation-12] src/finiteVolume/finiteVolume/fvSchemes/fvSchemes.H` | IOdictionary selector that fvMesh derives from; holds ddtSchemes, d2dt2Schemes, interpolationSchemes, divSchemes, gradSchemes, snGradSchemes, laplacianSchemes and fluxRequired sub-dictionaries, each with an optional `default` entry, wildcard lookup and field-group extension. Sets steady_ when default ddt is steadyState. |  |

### snGrad scheme  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::correctedSnGrad<Type>` | `corrected` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/correctedSnGrad` | Central-difference snGrad with full explicit non-orthogonal correction built from the mesh nonOrthCorrectionVectors dotted with the runtime-selected cell gradient. | snGrad = nonOrthDeltaCoeffs*(vfN-vfP) + linear-interp(k_f & grad(vf)) |
| `fv::faceCorrectedSnGrad<Type>` | `faceCorrected` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/faceCorrectedSnGrad` | Central-difference snGrad whose non-orthogonal correction is evaluated from point-interpolated face values (volPointInterpolation) rather than a cell gradient. | correction built from face-point values and face area vectors |
| `fv::limitedSnGrad<Type>` | `limited` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/limitedSnGrad` | Wraps a runtime-selected corrected snGrad scheme and limits its non-orthogonal correction with a coefficient in [0,1]: 0 = uncorrected, 1 = full correction, 0.5 = non-orthogonal part not exceeding the orthogonal part. Accepts `limited <scheme> <coeff>` or legacy `limited <coeff>`. | limiter = min(k*\|snGrad\| / ((1-k)*\|corr\| + small), 1); correction := limiter*corr |
| `fv::orthogonalSnGrad<Type>` | `orthogonal` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/orthogonalSnGrad` | Central-difference snGrad using the plain cell-centre-to-cell-centre delta coefficients (valid on orthogonal meshes). | snGrad = deltaCoeffs*(vfN - vfP), deltaCoeffs = 1/\|d\| |
| `fv::phaseStabilisedSnGrad<Type>` | `phaseStabilised` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/phaseStabilisedSnGrad` | Wraps a runtime-selected snGrad scheme and zeroes the non-orthogonal correction where the phase fraction is below 1e-3. | correction := pos0(alpha_f - 1e-3)*corr |
| `fv::uncorrectedSnGrad<Type>` | `uncorrected` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/uncorrectedSnGrad` | Central-difference snGrad using the non-orthogonal mesh delta-coefficients but with no non-orthogonal correction. | snGrad = nonOrthDeltaCoeffs*(vfN - vfP) |
| `linearFit snGrad (CentredFitSnGradScheme<linearFitPolynomial, centredFECCellToFaceStencilObject>)` | `linearFit` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/linearFitSnGrad/linearFitSnGrads.C` | Linear polynomial fit snGrad correction over the face-edge-cell (FEC) centred stencil. |  |
| `quadraticFit snGrad (CentredFitSnGradScheme<quadraticFitPolynomial, centredCFCCellToFaceStencilObject>)` | `quadraticFit` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/quadraticFitSnGrad/quadraticFitSnGrads.C` | Quadratic polynomial fit snGrad correction over the cell-face-cell (CFC) centred stencil. |  |

### snGrad scheme (polynomial fit)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::CentredFitSnGradScheme<Type,Polynomial,Stencil>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/CentredFitSnGrad/CentredFitSnGradScheme.H` | Template applying an explicit polynomial-fit correction to corrected snGrad over an extended centred cell-to-face stencil; linearLimitFactor bounds the deviation from linear, centralWeight biases the fit toward the owner/neighbour. | snGradCorr_f = sum_{c in stencil} coeff_c * vf_c (fit coefficients from CentredFitSnGradData) |

### snGrad scheme base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fv::snGradScheme<Type>` |  | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/snGradScheme/snGradScheme.H` | Abstract base for surface-normal-gradient schemes; supplies snGrad(vf, deltaCoeffs, name), corrected() flag and correction(). | snGrad(vf)_f = deltaCoeffs_f*(vf_N - vf_P) + correction |

### snGrad scheme support  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CentredFitSnGradData<Polynomial>` | `CentredFitSnGradData` | `[Foundation-12] src/finiteVolume/finiteVolume/snGradSchemes/CentredFitSnGrad/CentredFitSnGradData.H` | MeshObject computing and caching the least-squares fit coefficients for the centred-fit snGrad schemes (per-face weighted SVD inversion of the polynomial Vandermonde matrix). |  |

### solution framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvSolution` | `system/fvSolution` | `[Foundation-12] src/finiteVolume/finiteVolume/fvSolution/fvSolution.H` | Thin wrapper on Foam::solution read from system/fvSolution; fvMesh derives from it so every field can reach solvers/relaxationFactors/cache controls. |  |

### surface BC (basic)  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `calculatedFvsPatchField` | `calculated` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/basic/calculated` | Default surface-field patch type; values assigned rather than evaluated. |  |
| `coupledFvsPatchField` | `coupled` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/basic/coupled` | Abstract base for coupled surface-field patches. |  |
| `fixedValueFvsPatchField` | `fixedValue` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/basic/fixedValue` | Fixed-value patch type for surface fields. |  |
| `slicedFvsPatchField` | `sliced` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/basic/sliced` | Surface patch field created as a non-owning slice of a complete field (used for mesh Sf/magSf/C). |  |

### surface BC (constraint)  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicFvsPatchField / cyclicSlipFvsPatchField` | `cyclic, cyclicSlip` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/constraint/cyclic, cyclicSlip` | Cyclic (and cyclicSlip) constraint types for surface fields. |  |
| `emptyFvsPatchField` | `empty` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/constraint/empty` | Empty (reduced-dimension) constraint for surface fields; zero size. |  |
| `internalFvsPatchField` | `internal` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/constraint/internal` | Holds surface-field values for internal faces exposed by sub-setting. |  |
| `nonConformalCyclicFvsPatchField / nonConformalErrorFvsPatchField / nonConformalProcessorCyclicFvsPatchField` | `nonConformalCyclic, nonConformalError, nonConformalProcessorCyclic` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/constraint/nonConformalCyclic, nonConformalError, nonConformalProcessorCyclic` | Non-conformal coupling constraint types for surface fields. |  |
| `processorFvsPatchField / processorCyclicFvsPatchField` | `processor, processorCyclic` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/constraint/processor, processorCyclic` | Inter-processor (and processor-split cyclic) constraint types for surface fields. |  |
| `symmetryFvsPatchField / symmetryPlaneFvsPatchField / wedgeFvsPatchField` | `symmetry, symmetryPlane, wedge` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/constraint/symmetry, symmetryPlane, wedge` | Symmetry, symmetry-plane and wedge constraint types for surface fields. |  |

### surface BC (derived)  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `nonConformalMappedPolyFacesFvsPatchLabelField` | `nonConformalMappedPolyFaces` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/derived/nonConformalMappedPolyFaces` | polyFaces variant for non-conformal mapped (inter-region) couplings. |  |
| `nonConformalPolyFacesFvsPatchLabelField` | `nonConformalPolyFaces` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/derived/nonConformalPolyFaces` | polyFaces variant carrying the original (owner) patch information for non-conformal couplings. |  |
| `polyFacesFvsPatchLabelField` | `polyFaces` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/derived/polyFaces` | Stores the poly-mesh face indices corresponding to each fv patch face; the addressing backbone for the fvMesh-to-polyMesh mapping. |  |

### surface BC base  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvsPatchField<Type>` | `fvsPatchField` | `[Foundation-12] src/finiteVolume/fields/fvsPatchFields/fvsPatchField` | Abstract base for boundary values of surface (face) fields; runtime-selectable by patch, patchMapper and dictionary. |  |

### surface interpolation base  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `blendedSchemeBase<Type>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/blendedSchemeBase/blendedSchemeBase.H` | Interface giving access to the blending-factor surface field of any blended scheme (used e.g. by post-processing function objects). |  |
| `surfaceInterpolationScheme<Type>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/surfaceInterpolationScheme/surfaceInterpolationScheme.H` | Abstract base for all cell-to-face interpolation schemes; two runtime-selection tables (Mesh and MeshFlux) so schemes can be flux-aware; provides interpolate()/dotInterpolate()/correction(). | vf_f = w_f vf_P + (1-w_f) vf_N (+ correction) |

### surface interpolation scheme  <sub>(16)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `clippedLinear<Type>` | `clippedLinear` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/clippedLinear` | Central interpolation with weights clipped to a user-supplied minimum cell/face weight ratio, to stabilise meshes with rapid cell-size variation. | w = min(max(w_linear, wfLimit), 1-wfLimit) |
| `cubic<Type>` | `cubic` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/cubic` | Cubic scheme derived from linear: linear weights plus an explicit correction built from the cell gradients. | corr = kScheme*(vf_N - vf_P) type term using lambda(1-lambda) and grad(vf) interpolation |
| `deferred<Type>` | `deferred` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/deferred` | Deferred-correction wrapper: returns upwind weights (guaranteeing a diagonally equal matrix, so no matrix relaxation is needed) with the difference to the specified scheme applied explicitly. | w = w_upwind ; corr = interp_scheme(vf) - upwind(vf) |
| `downwind<Type>` | `downwind` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/downwind` | Downwind (anti-upwind) interpolation based on the sign of the face flux. | w_f = 1 - pos0(phi_f) |
| `harmonic` | `harmonic` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/harmonic` | Harmonic-mean interpolation for scalars only: interpolates 1/field with a runtime-selected scheme and returns the reciprocal. Registered for scalar in both the Mesh and MeshFlux tables. | vf_f = 1 / interp(1/vf)_f |
| `limitWith<Type>` | `limitWith` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitWith` | Limits an arbitrary scheme with the limiter of a separately specified limitedSurfaceInterpolationScheme. | w = limiter(limitScheme)*w_scheme + (1-limiter)*pos0(phi) |
| `linear<Type>` | `linear` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/linear` | Central-differencing: returns the geometric mesh weights, no correction. | w_f = mesh central-differencing weights |
| `localMax<Type>` | `localMax` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/localMax` | Face value set to the maximum of the two neighbouring cell values. | vf_f = max(vf_P, vf_N) |
| `localMin<Type>` | `localMin` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/localMin` | Face value set to the minimum of the two neighbouring cell values. | vf_f = min(vf_P, vf_N) |
| `midPoint<Type>` | `midPoint` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/midPoint` | Mid-point interpolation with uniform weights of 0.5 regardless of mesh geometry. | w_f = 0.5 |
| `outletStabilised<Type>` | `outletStabilised` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/outletStabilised` | Applies upwind interpolation to all faces of cells adjacent to outlets (outflow boundary faces), otherwise the selected scheme; stabilises entrainment boundaries in LES with centred schemes. |  |
| `phaseStabilised<Type>` | `phaseStabilised` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/phaseStabilised` | Switches to upwind on faces where the upwind phase fraction is below 1e-3, otherwise uses the selected scheme. |  |
| `pointLinear<Type>` | `pointLinear` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/pointLinear` | Face-point interpolation: linear weights plus an explicit correction from volPointInterpolation face-point values. | corr_f = average of point values at the face - linear face value |
| `reverseLinear<Type>` | `reverseLinear` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/reverseLinear` | Inverse-weight central interpolation (weights swapped); building block for inverse-weighted and harmonic interpolations. | w_f = 1 - w_linear |
| `skewCorrected<Type>` | `skewCorrected` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/skewCorrected` | Applies an explicit skewness correction to a runtime-selected base scheme using the skew-correction vectors. | corr += skewCorrVec_f & interp(grad(vf))_f |
| `weighted<Type>` | `weighted` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/weighted` | Uses an arbitrary surfaceScalarField of weights looked up from the objectRegistry by name. | vf_f = w_f vf_P + (1-w_f) vf_N with w from the registry |

### surface interpolation scheme (blended)  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CoBlended<Type>` | `CoBlended` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/CoBlended` | Two-scheme blend driven by the face Courant number computed directly from the flux, with user-supplied lower (Co1) and upper (Co2) limits. | weight = 1 - max(min((Co - Co1)/(Co2 - Co1), 1), 0) ; Co = \|phi\|*deltaT*deltaCoeffs/(magSf*rho) |
| `LUST<Type>` | `LUST` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/LUST` | Linear-Upwind Stabilised Transport - derived from linearUpwind, blends 75% linear with 25% linear-upwind weights and applies the linearUpwind gradient correction. Registered for scalar and vector. | w = 0.75*w_linear + 0.25*w_linearUpwind ; corr = 0.75*linearCorr + 0.25*linearUpwindCorr |
| `cellCoBlended<Type>` | `cellCoBlended` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/cellCoBlended` | As CoBlended but the Courant number is evaluated per cell exactly as in the solvers and then interpolated to faces using a runtime-selected interpolation for "Co" (localMax is suggested). | Co_cell = 0.5*sumPhi/V*deltaT ; weight = 1 - max(min((Co_f - Co1)/(Co2 - Co1), 1), 0) |
| `fixedBlended<Type>` | `fixedBlended` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/fixedBlended` | Blends two arbitrary runtime-selected schemes with a single global constant factor (factor applies to the first scheme, 1-factor to the second). | w = f*w1 + (1-f)*w2 ; corr = f*corr1 + (1-f)*corr2 |
| `limiterBlended<Type>` | `limiterBlended` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/limiterBlended` | Blends two schemes using the limiter function supplied by a limitedSurfaceInterpolationScheme, e.g. `Gauss limiterBlended vanLeer linear linearUpwind grad(U)`. | w = limiter*w1 + (1-limiter)*w2 |
| `localBlended<Type>` | `localBlended` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/localBlended` | Blends two schemes using a face-varying blending-factor surfaceScalarField looked up from the registry by name. | w = bf*w1 + (1-bf)*w2 |

### surface interpolation scheme (fit)  <sub>(12)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CentredFitScheme<Type,Polynomial,Stencil>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/CentredFitScheme/CentredFitScheme.H` | Template for centred polynomial-fit schemes: linear weights plus an explicit high-order correction from an extended centred stencil. Parameters: linearLimitFactor, centralWeight. | vf_f = linear(vf)_f + sum_{c in stencil} coeff_c vf_c |
| `PureUpwindFitScheme<Type,Polynomial,Stencil>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/PureUpwindFitScheme/PureUpwindFitScheme.H` | Upwind-biased polynomial fit applying an explicit correction to pure upwind (rather than to linear); derived from upwind so the implicit part stays fully upwind. |  |
| `UpwindFitScheme<Type,Polynomial,Stencil>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/UpwindFitScheme/UpwindFitScheme.H` | Upwind-biased polynomial fit applying an explicit correction to linear interpolation, using an upwind-biased extended stencil selected by the sign of the face flux. |  |
| `biLinearFit (CentredFitScheme<biLinearFitPolynomial, centredCFCCellToFaceStencilObject>)` | `biLinearFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/biLinearFit/biLinearFit.C` | Bi-linear polynomial fit correction to linear interpolation on the CFC centred stencil. |  |
| `cubicUpwindFit (UpwindFitScheme<cubicUpwindFitPolynomial, upwindCFCCellToFaceStencilObject>)` | `cubicUpwindFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/cubicUpwindFit/cubicUpwindFit.C` | Cubic upwind-biased polynomial fit correction on the upwind CFC stencil. |  |
| `linearFit (CentredFitScheme<linearFitPolynomial, centredCFCCellToFaceStencilObject>)` | `linearFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearFit/linearFit.C` | Linear polynomial fit correction to linear interpolation on the CFC centred stencil. |  |
| `linearPureUpwindFit (PureUpwindFitScheme<linearFitPolynomial, pureUpwindCFCCellToFaceStencilObject>)` | `linearPureUpwindFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearPureUpwindFit/linearPureUpwindFit.C` | Linear polynomial fit correction to upwind on the pure-upwind CFC stencil. |  |
| `quadraticFit (CentredFitScheme<quadraticFitPolynomial, centredCFCCellToFaceStencilObject>)` | `quadraticFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticFit/quadraticFit.C` | Quadratic polynomial fit correction to linear interpolation on the CFC centred stencil. |  |
| `quadraticLinearFit (CentredFitScheme<quadraticLinearFitPolynomial, centredCFCCellToFaceStencilObject>)` | `quadraticLinearFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticLinearFit/quadraticLinearFit.C` | Fit that is quadratic normal to the face and linear in the plane of the face, for consistency with 2nd-order Gauss. |  |
| `quadraticLinearPureUpwindFit (PureUpwindFitScheme<quadraticLinearUpwindFitPolynomial, upwindCFCCellToFaceStencilObject>)` | `quadraticLinearPureUpwindFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticLinearPureUpwindFit/quadraticLinearPureUpwindFit.C` | Quadratic/linear polynomial fit correction to upwind on the upwind CFC stencil. |  |
| `quadraticLinearUpwindFit (UpwindFitScheme<quadraticLinearUpwindFitPolynomial, upwindFECCellToFaceStencilObject>)` | `quadraticLinearUpwindFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticLinearUpwindFit/quadraticLinearUpwindFit.C` | Quadratic/linear upwind-biased polynomial fit correction on the upwind FEC stencil. |  |
| `quadraticUpwindFit (UpwindFitScheme<quadraticUpwindFitPolynomial, upwindCFCCellToFaceStencilObject>)` | `quadraticUpwindFit` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/quadraticUpwindFit/quadraticUpwindFit.C` | Quadratic upwind-biased polynomial fit correction on the upwind CFC stencil. |  |

### surface interpolation support  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CentredFitData<Polynomial>` | `CentredFitData` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/CentredFitScheme/CentredFitData.H` | MeshObject caching the fit coefficients for all centred-fit interpolation schemes. |  |
| `FitData<Form,ExtendedStencil,Polynomial>` |  | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/FitData/FitData.H` | Common base for centred and upwinded polynomial-fit interpolation data; linearCorrection_ selects whether the fit corrects a linear scheme (owner+neighbour corrections) or a pure-upwind scheme (owner correction only). Uses weighted SVD pseudo-inverse with linearLimitFactor fallback to linear. | minimise sum_c w_c (P(x_c)a - vf_c)^2 ; coefficients = pseudo-inverse of weighted Vandermonde |
| `UpwindFitData<Polynomial>` | `UpwindFitData` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/UpwindFitScheme/UpwindFitData.H` | MeshObject caching upwind-biased fit coefficients; linearCorrection true = fit corrects linear, false = fit corrects upwind. |  |

### surface interpolation support (MeshObject)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `skewCorrectionVectors` | `skewCorrectionVectors` | `[Foundation-12] src/finiteVolume/interpolation/surfaceInterpolation/schemes/skewCorrected/skewCorrectionVectors.H` | DemandDrivenMeshObject computing the vector from the face centre to the intersection of the P-N line with the face plane; also reports whether the mesh is skew at all. | Cf - (w*Cp + (1-w)*Cn) projected along the face normal |

### vol-to-point interpolation (MeshObject)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `volPointInterpolation` | `volPointInterpolation` | `[Foundation-12] src/finiteVolume/interpolation/volPointInterpolation` | Interpolates cell-centred values to mesh points (vertices) using inverse-distance weighting; handles boundary-value blending and applies pointConstraints. Used by pointLinear, faceCorrectedSnGrad, cellPoint interpolations and by post-processing. | psi_point = sum_cells (1/\|x_p - C_c\|) psi_c / sum_cells (1/\|x_p - C_c\|) |

### vol-to-point interpolation support (MeshObject)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pointConstraints` | `pointConstraints` | `[Foundation-12] src/finiteVolume/interpolation/volPointInterpolation/pointConstraints.H` | Applies the geometric constraints (symmetry planes, wedges, empty, cyclics, processor) to point fields after volPointInterpolation, using rank-reduced constraint tensors per point. | psi_point := constraintTensor & psi_point |

---

## Mesh: Mesh generation and manipulation

> **Subsystem notes**
>
> Scope actually present in OpenFOAM-12 differs from the task brief in four places: **boundaryCutter, tetDecomposer, faceCollapser and polyTopoChanger do not exist anywhere in this source tree** (verified with `find src -iname "*<name>*"` — zero hits).
> polyTopoChanger and its polyMeshModifier hierarchy were removed in the Foundation v11/v12 rework in favour of direct polyTopoChange calls plus the new fvMeshTopoChanger/fvMeshDistributor framework in src/dynamicFvMesh (outside this part).
> tetDecomposer's role is now covered by polyMeshTetDecomposition in src/OpenFOAM. faceCollapser's role is covered by edgeCollapser::collapseToEdge / collapseToPoint.
> Runtime-selection tables actually declared in this part (verified by grepping declareRunTimeSelectionTable/defineRunTimeSelectionTable): blockVertex (Istream), blockEdge (Istream), blockFace (Istream), block (Istream), extrudeModel (dictionary), externalDisplacementMeshMover (dictionary).
> cellLooper has TypeName but **no** declareRunTimeSelectionTable in v12 — geomCellLooper/hexCellLooper include addToRunTimeSelectionTable.H but register nothing, so they are chosen in code (e.g. by multiDirRefinement's `useHexTopology` switch), not by dictionary keyword.
> Likewise perfectInterface includes the header but adds no table entry. badQualityToCell/badQualityToFace register into the *topoSetSource* table (owned by src/meshTools) and are the only dictionary-selectable entries this library adds there.
> zeroFixedValuePointPatchField registers into the pointPatchField tables.
> Note two distinct keyword collisions that are legitimate because the tables differ: "project" is the keyword for blockVertices::projectVertex, blockEdges::projectEdge AND blockFaces::projectFace; "name" is the keyword for both namedVertex and namedBlock.
> Key file paths: blockMesh library C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh (Make/files lists all 30 compiled sources); extrudeModel C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel; snappyHexMesh C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh; polyTopoChange C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange.
> The three snappyHexMesh phases map to snappyRefineDriver.H (castellation), snappySnapDriver.H + snappySnapDriverFeature.C (snapping), snappyLayerDriver.H + snappyLayerDriverShrink.C (layer addition), all under src/mesh/snappyHexMesh/snappyHexMeshDriver.
> Each phase reads its own parameter container (refinementParameters, snapParameters, layerParameters) which is where the exact dictionary keywords live.
> The mesh-quality checks that motionSmoother/badQualityToCell/badQualityToFace use are NOT in this part — motionSmootherAlgo.C:903 calls `meshCheck::checkMesh`, which lives in src/meshTools.

### blockMesh / block  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `block` | `hex (base class of the block RTS table; keyword read as the block shape name)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blocks/block/block.H` | Creates the points and cells of a single hexahedral block from corner points, division counts and expansion ratios. | Trilinear (transfinite/Coons) blend of the three edge-weight families, then curved-edge correction vectors added, then curved-face (projectFace) correction applied face by face |
| `namedBlock` | `name` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blocks/namedBlock/namedBlock.H` | Wraps a block so that it can be given a user name in blockMeshDict. |  |

### blockMesh / core  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `blockDescriptor` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockDescriptor/blockDescriptor.H` | Holds one block's vertex labels, cell counts and grading, and builds the 12 edge point distributions. | Maps the standard hex vertex ordering 0-7 to faces f0..f5 (x-min/x-max, y-min/y-max, z-min/z-max); builds edgePoints/edgeWeights per edge via lineDivide |
| `blockMesh` | `no (top-level class; dict blockMeshDict, optional 'fastMerge yes' switch selects calcMergeInfoFast)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockMesh/blockMesh.H` | Multi-block structured mesh generator driven by blockMeshDict; demand-driven vertices, cells and patches. | Per-block transfinite (trilinear Coons) point generation, then geometric merge of coincident block-face points within sqrMergeTol = (min squared edge length)/10 |
| `blockMesh::calcMergeInfo / calcMergeInfoFast` | `dict switch 'fastMerge' (default false) in blockMeshDict` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockMesh/blockMeshMerge.C` | Two alternative algorithms to glue coincident points on shared block faces into one global point list. | Pairwise squared-distance test against a per-face tolerance derived from the shortest face edge (/10), then iterative transitive closure of the merge map until no change |

### blockMesh / edge  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BSplineEdge` | `BSpline` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/BSplineEdge/BSplineEdge.H` | blockEdge interface for a uniform cubic B-spline through the given interior control points. | P(t) = 1/6 [t^3 t^2 t 1] * [[-1 3 -3 1],[3 -6 3 0],[-3 0 3 0],[1 4 1 0]] * [P-1 P0 P1 P2]^T |
| `arcEdge` | `arc` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.H` | Circular arc through a third point, or defined by sector angle plus axis (giving a helical segment if the axis is not perpendicular to the chord). | p(lambda) = c + r1*cos(t) + (axis^r1)*sin(t) + axis*length*lambda, with t = lambda*angle; centre from the three-point circumcentre, or c = pM - l*axis/2 - rM*\|chord\|/(2 tan(theta/2)) |
| `blockEdge` | `base class of the blockEdge Istream RTS table (entries in the 'edges' list)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/blockEdge/blockEdge.H` | Abstract base for a curved block edge parameterised by lambda in [0,1] between start and end vertex. | position(lambda) and length() interface; lambda is arc-length-normalised parameter |
| `lineEdge` | `line` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/lineEdge/lineEdge.H` | Straight edge between the two end vertices. | p(lambda) = p0 + lambda*(p1 - p0) |
| `polyLineEdge` | `polyLine` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/polyLineEdge/polyLineEdge.H` | Edge defined as a chain of straight segments through a list of interior control points. | Piecewise-linear interpolation with chord-length (cumulative arc-length) parameterisation |
| `projectCurveEdge` | `projectCurve` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/projectCurveEdge/projectCurveEdge.H` | Edge defined from projection onto a surface, or from the intersection curve of two surfaces, following an extendedEdgeMesh feature curve. | Nearest-point queries against a searchableSurface feature curve, with pointConstraint accumulation to keep the point on the curve |
| `projectEdge` | `project` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/projectEdge/projectEdge.H` | Edge obtained by projecting the straight edge onto a single searchable surface (or the intersection of two). | Fixed-point iteration (maxIter=10, relTol=0.1, absTol=1e-4): project current points by findNearest with search span magSqr(end-start), re-space along the projected polyline, repeat; ends pinned |
| `splineEdge` | `spline` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/splineEdge/splineEdge.H` | blockEdge interface for a Catmull-Rom (Overhauser) spline through the given interior points. | P(t) = 1/2 [t^3 t^2 t 1] * [[-1 3 -3 1],[2 -5 4 -1],[-1 0 1 0],[0 2 0 0]] * [P-1 P0 P1 P2]^T |

### blockMesh / edge helper  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `BSpline` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/BSplineEdge/BSpline.H` | Cubic B-spline implementation with automatic end tangents by reflection. | Local basis-matrix form above; approximating (not interpolating) spline, chord-length discretisation |
| `CatmullRomSpline` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/splineEdge/CatmullRomSpline.H` | Catmull-Rom spline implementation with automatic end tangents by reflection. | Interpolating cubic on t=[0,1] per segment; end control points created by reflecting the neighbouring point; discretised by segment chord length |
| `polyLine` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/polyLineEdge/polyLine.H` | Reusable series of straight line segments, also used as the control polygon for splines. | Cumulative chord lengths param_[i] = sum \|p_j+1 - p_j\| normalised by total length |

### blockMesh / face  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `blockFace` | `base class of the blockFace Istream RTS table (entries in the 'faces' list)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockFaces/blockFace/blockFace.H` | Abstract base for a curved block face used to correct interior block points. |  |
| `projectFace` | `project` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockFaces/projectFace/projectFace.H` | Projects the structured grid of face points onto a named searchableSurface of the geometry sub-dictionary. | Alternating i-then-j prediction/projection sweeps: normalised edge lengths give (u,v); findNearest projection; then re-interpolation along constant-j and constant-i lines to keep the parameterisation smooth; iterated to relTol/absTol |

### blockMesh / grading  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `gradingDescriptor` | `read as a single scalar (expansion ratio) or as a ( blockFraction nDivFraction expansionRatio ) triple` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/gradingDescriptor/gradingDescriptor.H` | One grading section: block length fraction, division-count fraction and expansion ratio. | Triple (blockFraction, nDivFraction, expansionRatio); inverse() returns the same section with 1/expansionRatio |
| `gradingDescriptors` | `blockMeshDict keywords 'simpleGrading' (3 entries, one per direction) and 'edgeGrading' (12 entries, one per block edge)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/gradingDescriptor/gradingDescriptors.H` | Multi-section grading for one block direction (multi-grading), with IO and inverse(). | List of gradingDescriptor whose blockFraction and nDivFraction each sum to 1 |
| `lineDivide` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockEdges/lineDivide/lineDivide.H` | Divides one blockEdge into nDiv segments according to a gradingDescriptors specification. | Geometric expansion factor g = expRatio^(1/(n-1)); division i at s + blockFrac*(1 - g^(i+1))/(1 - g^n); uniform s + blockFrac*(i+1)/n when expRatio == 1; divisions per section rounded from nDivFraction*nDiv with the remainder given to the largest section |

### blockMesh / utility  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `blockMeshTools` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockMeshTools/blockMeshTools.H` | Read/write helpers that resolve blockMeshDict vertex labels given either as integers or as vertex names. |  |

### blockMesh / vertex  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `blockVertex` | `base class of the blockVertex Istream RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockVertices/blockVertex/blockVertex.H` | Abstract base for a blockMesh vertex definition; runtime-selected from Istream. |  |
| `namedVertex` | `name` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockVertices/namedVertex/namedVertex.H` | Gives a symbolic name to a vertex so blocks/patches can refer to it by word. |  |
| `pointVertex` | `point` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockVertices/pointVertex/pointVertex.H` | Plain vertex given directly by its (x y z) coordinates. |  |
| `projectVertex` | `project` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/blockMesh/blockVertices/projectVertex/projectVertex.H` | Vertex snapped onto one or more searchableSurfaces of the geometry sub-dictionary. | searchableSurfacesQueries::findNearest of the raw point onto the named surface set (intersection of constraints for >1 surface) |

### extrudeModel  <sub>(11)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cyclicSector` | `cyclicSector` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/cyclicSector/cyclicSector.H` | Sector extrusion whose front/back patches are created as a cyclic pair. | Same rotation as sector; front/back given cyclic patch types with rotational transform |
| `cylindricalRadial` | `cylindricalRadial (axisPt, axis, R as Function1<scalar>)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/cylindricalRadial/cylindricalRadial.H` | Extrudes in the cylindrical-radial direction about a given axis, layer radii set by a Function1 of the layer index. | Decompose p about (axisPt, axis); replace the radial magnitude by R(layer) from the Function1; axial component retained |
| `extrudeModel` | `base class; keyword 'extrudeModel' in extrudeMeshDict / extrusion sub-dicts, with nLayers and expansionRatio` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/extrudeModel/extrudeModel.H` | Top-level runtime-selectable extrusion model returning the new point for a surface point on a given layer. | sumThickness(layer) = (1 - r^layer)/(1 - r^nLayers) for expansionRatio r != 1, else layer/nLayers (geometric series 1+r+...+r^(n-1)) |
| `linearDirection` | `linearDirection (direction, thickness)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/linearDirection/linearDirection.H` | Extrudes along a fixed user-specified direction by a given thickness. | p = surfacePoint + dHat*thickness*sumThickness(layer) |
| `linearNormal` | `linearNormal (thickness, firstCellThickness, layerPoints)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/linearNormal/linearNormal.H` | Extrudes along the local surface normal by a given total thickness, optionally with a specified first-cell thickness or explicit layer point distribution. | p = surfacePoint + n*thickness*sumThickness(layer), or p = surfacePoint + n*thickness*layerPoints[layer] |
| `linearRadial` | `linearRadial (R, Rsurface)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/linearRadial/linearRadial.H` | Extrudes radially outward from the origin to a target radius R. | rs = \|p\| (or Rsurface if set); r = rs + (R - rs)*sumThickness(layer); p_new = r*p/\|p\| |
| `plane (planeExtrusion)` | `plane` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/planeExtrusion/planeExtrusion.H` | Single-layer normal extrusion producing a 2-D mesh with empty front/back patches. | linearNormal with nLayers = 1 |
| `sector` | `sector (axisPt, axis, angle in degrees)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/sector/sector.H` | Extrudes by rotating the source patch about an axis through a total sector angle; extrusion is opposite the patch normal. | p_new = axisPt + Ra(theta)&(p - axisPt) with theta = -angle/2 + angle*sumThickness(layer); Ra = rotation tensor about the normalised axis |
| `sigmaRadial` | `sigmaRadial (RTbyg, pRef, pStrat)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/sigmaRadial/sigmaRadial.H` | Radial extrusion on a hydrostatic (sigma-pressure) coordinate, for atmospheric/geophysical meshes. | p_lay = pRef - layer*(pRef - pStrat)/nLayers; r = \|x\| - (RT/g)*ln(p_lay/pRef); p_new = r*xHat |
| `sphericalRadial` | `sphericalRadial (R as Function1<scalar>)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/sphericalRadial/sphericalRadial.H` | Extrudes in the spherical-radial direction, layer radii set by a Function1 of the layer index; surface normal unused. | p_new = R(layer) * p/\|p\| |
| `wedge` | `wedge` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/extrudeModel/wedge/wedge.H` | One-layer symmetric rotation about an axis producing an OpenFOAM axisymmetric wedge mesh. | sector with nLayers = 1 and rotation +/- angle/2 about the axis; front/back patches of type wedge |

### meshCut / cutting  <sub>(13)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cellCuts` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/cellCuts/cellCuts.H` | Describes cuts across cells as cut vertices plus weighted cut edges, and derives non-overlapping cell circumference loops. | Constructs cellLoops by walking the cell circumference; each face may be cut only once (split into two); orients each loop so its normal points towards the cellAnchorPoints; 2x2x2 refinement needs three passes because cuts may not overlap |
| `cellLooper` | `base class; derived types are named geomCellLooper and hexCellLooper (chosen in code / by multiDirRefinement's useHexTopology switch)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/cellLooper/cellLooper.H` | Abstract base for algorithms that determine a cut loop around the circumference of one cell. | cut(refDir or cutPlane, celli, current cuts) -> loop of edgeVertex labels plus weights in [0,1] |
| `directionInfo` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/directions/directionInfo/directionInfo.H` | FaceCellWave transport of the local cut direction: a normal vector plus a topological edge/face-point label. | If the label is set (>= -1) the topological information is used (exact for hexes), otherwise the vector is used (geometric cut for other shapes) |
| `directions` | `coordinateSystem: global \| patchLocal \| fieldBased; directions: e1 e2 e3; useHexTopology switch` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/directions/directions.H` | Provides the per-cell refinement direction vectors, uniform or locally varying. | 'global' uses fixed e1/e2 vectors; 'patchLocal' propagates a patch-local coordinate system into the mesh (topologically for hexes via directionInfo, geometrically otherwise); 'fieldBased' reads named vector fields |
| `edgeVertex` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/edgeVertex/edgeVertex.H` | Encodes an edge or a vertex in a single label so a cut loop can mix both. | vertex v -> v; edge e -> -(e+1); helpers isEdge/getEdge/getVertex/coord(weight) |
| `geomCellLooper` | `geomCellLooper` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/cellLooper/geomCellLooper.H` | Purely geometric cell cut with a plane through the cell centre normal to a given direction; handles all cell shapes. | Intersect every cell edge with the plane; snap cuts within snapTol*minEdgeLen of an endpoint to that vertex; sort resulting cuts by angle about the plane normal to form the loop; edges parallel to the plane handled by endpoint-distance test |
| `hexCellLooper` | `hexCellLooper` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/cellLooper/hexCellLooper.H` | Topological cut of hexahedra (edges always cut at their midpoint); falls back to geomCellLooper for any non-hex. | Walk: cross an edge to the opposite face, then cross that face edge-point-edge to reach the other side; cut weight fixed at 0.5 |
| `meshCutter` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/meshModifiers/meshCutter/meshCutter.H` | Turns a cellCuts description into the actual add-point/add-face/add-cell topo actions, splitting each cut cell in two. | Insert the split face at the anchor-point side of the cell, then sweep it up to the cut position; side faces are split into two; anchor side is the master cell and cell edges get the split as a duplicate of the anchor point |
| `multiDirRefinement` | `refineMeshDict: directions, coordinateSystem, useHexTopology, geometricCut` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/meshModifiers/multiDirRefinement/multiDirRefinement.H` | Refines selected cells in several directions in succession (directional/anisotropic refinement). | One pass per direction vectorField; each pass cuts every wanted cell with a plane normal to that direction, then the added cells are added to the wanted set for the next direction |
| `refineCell` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/refineCell/refineCell.H` | Simple container pairing a cell label with the single direction in which it should be refined. |  |
| `refinementIterator` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/meshModifiers/refinementIterator.H` | Repeatedly invokes meshCutter until all requested cell cuts have been satisfied. | Because a cell can only be cut once per pass, iterate: build cellCuts for the still-unrefined requests, cut, remap, repeat; termination decided on a reduced (parallel) count |
| `splitCell` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/splitCell/splitCell.H` | Node of the undo tree: original cell label plus parent and two child pointers. | Leaf (both children null) = live cell; internal node = already-split cell |
| `undoableMeshCutter` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/meshCut/meshModifiers/undoableMeshCutter/undoableMeshCutter.H` | Wrapper around meshCutter that maintains a refinement tree so cuts can be undone by removing the split faces. | Binary splitCell tree; liveSplitCells are the leaves; undo collects the faces between visible siblings and hands them to removeFaces |

### polyTopoChange / engine  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `polyTopoChange` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/polyTopoChange.H` | Direct mesh-change engine holding the evolving points/faces/cells and producing the new polyMesh plus a polyTopoChangeMap. | Immediate-application model with compaction/renumbering; removed cell = 0 faces, removed face = 0 vertices, removed point = vector::max; reorderCoupledFaces re-sorts cyclic/processor patch faces |

### polyTopoChange / mesh creation  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `createShellMesh` | `no (used by extrudeMesh / extrudeToRegionMesh with an extrudeModel)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/createShellMesh/createShellMesh.H` | Creates a new (shell/region) mesh by extruding an indirect primitive patch layer by layer. | Per patch point a displacement per layer; adds points/faces/cells for each layer and generates the side patches from the patch outline |

### polyTopoChange / mesh filtering  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `polyMeshFilter` | `no (driven by collapseDict; used by collapseEdges)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyMeshFilter/polyMeshFilter.H` | Removes short edges and small faces of a polyMesh subject to mesh-quality constraints, working on a copy. | Iterative loop (maxIterations): shrink target edge/face length by edgeReductionFactor/faceReductionFactor each pass, run edgeCollapser, accept only collapses that do not create points failing the quality checks (up to maxPointErrorCount) |
| `polyMeshFilterSettings` | `collapseDict: controlMeshQuality, minimumEdgeLength (minLen), maximumCosAngle (maxCos), edgeReductionFactor, maximumIterations, maximumSmoothingIterations, initialFaceLengthFactor, faceReductionFactor, maximumPointErrorCount` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyMeshFilter/polyMeshFilterSettings.H` | Reads and stores the polyMeshFilter control parameters. |  |

### polyTopoChange / mesh joining  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `faceCoupleInfo` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyMeshAdder/faceCoupleInfo.H` | Builds the point and face correspondence between two meshes to be coupled. | Either exact matching (identical face/point sets) or a cut-face construction where master and slave patches are intersected to a common 'cut' patch, with per-point mapping derived from it |
| `mergePatchPairs` | `blockMeshDict 'mergePatchPairs' list; stitchMesh utility` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/mergePatchPairs/mergePatchPairs.H` | Stitches a mesh by merging pairs of patches, including non-conformal (cut-and-stitch) pairs. | Per pair, either a perfectInterface match or a slidingInterface-style projection with a merge tolerance relative to the local face size |
| `perfectInterface` | `perfectInterface (polyMeshModifier-style type name)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/perfectInterface/perfectInterface.H` | Couples two perfectly aligned patches into internal faces (used by stitchMesh); does not decouple. | Geometric face-centre matching within a tolerance derived from the local face size; slave faces removed and master faces turned into internal faces |
| `polyMeshAdder` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyMeshAdder/polyMeshAdder.H` | Adds two polyMeshes into one without morphing, given a faceCoupleInfo describing the shared faces. | Concatenates points/faces/cells, then merges the coupled face pairs into internal faces and returns the two maps from old to new addressing |

### polyTopoChange / mesh manipulation  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshAdder` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/fvMeshAdder/fvMeshAdder.H` | Adds two fvMeshes (mesh plus all vol/surface fields) using polyMeshAdder. | Field values mapped through the two polyTopoChangeMaps returned by polyMeshAdder |
| `fvMeshDistribute` | `no (driven by decomposition method; used by snappyHexMesh balancing and redistributePar)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/fvMeshDistribute/fvMeshDistribute.H` | Sends and receives mesh and field parts between processors for load balancing / redistribution. | Input is a per-cell destination processor; sub-meshes are extracted with fvMeshSubset, exchanged, then merged with fvMeshAdder; faces matched topologically |
| `fvMeshSubset` | `fvMeshSubset` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/fvMeshSubset/fvMeshSubset.H` | Builds a mesh consisting only of a selected set of cells, with point/face/cell maps and field interpolation. | Exposed internal faces go into a user-supplied patch or a new 'oldInternalFaces' patch; setCellSubset uses Maps (small subsets), setLargeCellSubset uses labelLists and handles coupled patches losing a neighbour |
| `fvMeshTools` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/fvMeshTools/fvMeshTools.H` | Collection of helpers for adding, removing, reordering and zero-sizing fvMesh patches and for writing meshes. |  |

### polyTopoChange / refinement  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `hexRef8` | `no (used by snappyHexMesh and by dynamicRefineFvMesh)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/hexRef8/hexRef8.H` | Isotropic 2x2x2 refinement (and matching unrefinement) of split hexes using polyTopoChange. | Each selected cell splits into 8 by adding the cell centre point, 6 face-centre points and 12 edge-mid points; per-cell/point refinement level maintained; unrefinement removes the 8 children back to the parent via refinementHistory |
| `hexRef8::consistentRefinement / consistentSlowRefinement` | `nCellsBetweenLevels in castellatedMeshControls` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/hexRef8/hexRef8.C` | Extends the requested refinement set so the mesh keeps a 2:1 (or slower) level transition across faces. | consistentRefinement: iterate marking neighbours until \|cellLevel_own - cellLevel_nei\| <= 1; consistentSlowRefinement uses a FaceCellWave of refinementData to enforce nBufferLayers cells per level jump; consistentSlowRefinement2 uses refinementDistanceData for a distance-based version |
| `hexRef8Data` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/hexRef8/hexRef8Data.H` | Reads/decomposes/reconstructs/distributes the cellLevel, pointLevel and refinementHistory data that accompany a hexRef8 mesh. |  |
| `refinementHistory` | `refinementHistory (registered IOobject type in constant/polyMesh)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/hexRef8/refinementHistory.H` | Octree history of all refinements, allowing unrefinement back to parent cells. | splitCells_ is a tree of (parent index, 8 child indices); visibleCells_ maps each live cell to its splitCells index or -1 if never refined; a cell is unrefinable when all 8 siblings are visible |

### polyTopoChange / repatching  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `repatchMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/repatchMesh/repatchMesh.H` | Surface-of-mesh addressing (from a polyMesh or a triSurface) used to drive repatching decisions. | Feature-angle based edge classification of the boundary surface to define candidate patch splits |
| `repatchPatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/repatchMesh/repatchPatch.H` | Mesh-free stand-in for polyPatch holding name, type, size and start for repatchMesh. |  |
| `repatcher` | `no (used by createPatch, autoPatch, splitMeshRegions)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/repatcher/repatcher.H` | Allows boundary faces to be moved between patches via changePatchID, then rebuilds the mesh. | Face-to-patch reassignment followed by a polyTopoChange that reorders boundary faces into contiguous per-patch ranges |

### polyTopoChange / topo change  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `addPatchCellLayer` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/addPatchCellLayer.H` | Adds layers of prismatic cells outside a polyPatch, optionally creating a stand-alone extruded mesh. | Per patch point an offset vector and per point/face a layer count; new points at p + offset*(cumulative fraction); differing face layer counts are terminated at the original patch face side; added faces inherit the source face patch, side faces take the neighbouring patch/zone |
| `combineFaces` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/combineFaces.H` | Merges sets of boundary faces of the same cell into a single face taking the master face's patch. | getMergeSets groups patch faces of a cell whose normals agree within a feature cosine; the merged outline is the boundary walk of the set, with interior points removed; supports undo (setUnrefinement) |
| `duplicatePoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/duplicatePoints.H` | Duplicates points along baffle/zone boundaries so that faces on either side no longer share vertices. | For each localPointRegion region of a point, add a new point at the same position and re-index the faces of that region |
| `edgeCollapser` | `no (settings supplied via polyMeshFilterSettings/collapseEdgesCoeffs)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/edgeCollapser.H` | Collapses short edges and sliver faces, removing faces (but never cells) and unused points. | markSmallEdges (\|e\| < minimum length), markSmallSliverFaces via eigenvector analysis of the face giving a collapse axis and aspect ratio, markMergeEdges, markFaceZoneEdges; collapse strings broken by breakStringsAtEdges and made parallel-consistent by syncCollapse |
| `edgeCollapser::collapseToEdge / collapseToPoint` | `collapseEdgesCoeffs: maxCollapseFaceToPointSideLengthCoeff, allowEarlyCollapseToPoint, allowEarlyCollapseCoeff` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/edgeCollapser.C` | The two face-collapse modes: reduce a sliver face to an edge, or a small face to a single point. | faceCollapseAxisAndAspectRatio does an eigen-decomposition of the face's second-moment tensor; the minor eigenvector is the collapse axis and the eigenvalue ratio the aspect ratio; collapse to point when aspect ratio < allowEarlyCollapseToPoint threshold |
| `pointEdgeCollapse` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/pointEdgeCollapse/pointEdgeCollapse.H` | PointEdgeWave transport type that determines the length of the string of edges walked to a point during collapse. | Transports (collapsePoint, collapseIndex, priority); a point adopts a neighbour's collapse target when the neighbour has higher priority or a lower index |
| `removeCells` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/removeCells.H` | Inserts all topology changes needed to delete a list of cells, exposing their faces into given patches. | Two passes: (1) find faces that become boundary faces, (2) convert those faces (flipping owner/neighbour as needed) into the supplied exposedPatchIDs and remove orphan points/faces |
| `removeFaces` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/removeFaces.H` | Removes internal faces, merging the two cells either side, with a helper to grow a consistent removal set. | compatibleRemoves closes the removal set so that merged cell regions stay valid; remaining faces of merged cells are stitched and coplanar faces combined |
| `removePoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/removePoints.H` | Removes selected (typically collinear edge-mid) points and updates every face using them. | A point is removable when it is used by exactly two edges of every face and the two edges are collinear within a tolerance; supports undo via savedPoints |

### snappyHexMesh / castellation  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `baffleAndSplitMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementBaffles.C` | Creates baffles (duplicated boundary faces) on all intersected faces and splits off unreachable mesh regions. | createBaffles duplicates each internal face flagged with ownPatch/neiPatch; regionSplit then discards unreachable regions |
| `refinementParameters::cellSelectionPoints` | `castellatedMeshControls: insidePoints / outsidePoints` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/refinementParameters.H` | Holds the inside/outside seed points that select which mesh regions are kept. |  |
| `removeInsideCells` | `castellatedMeshControls/insidePoints (and outsidePoints), nCellsBetweenLevels buffer` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyRefineDriver.C` | Removes all cells not reachable from the insidePoints (or reachable from outsidePoints), leaving a buffer of layers. | Flood-fill of cell regions bounded by intersected faces; keep the region containing each locationInMesh, drop the rest |
| `snappyRefineDriver` | `no (invoked by snappyHexMesh with castellatedMeshControls)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyRefineDriver.H` | Drives the whole castellated-mesh phase: feature, surface, gap, dangling-cell and shell refinement, inside-cell removal, baffling and zoning. | Loop of hexRef8 2:1-consistent octree splits until maxGlobalCells/maxLocalCells reached or fewer than minRefineCells are marked; load-balances via decompositionMethod when maxLoadUnbalance exceeded |
| `splitAndMergeBaffles` | `castellatedMeshControls/allowFreeStandingZoneFaces` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementBaffles.C` | Re-merges baffle face pairs that no longer need to be separate, and keeps free-standing zone faces if allowed. |  |
| `surfaceZonesInfo` | `cellZoneInside: inside \| outside \| insidePoint \| none; faceType: internal \| baffle \| boundary` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/refinementSurfaces/surfaceZonesInfo.H` | Per-surface faceZone/cellZone naming and the rule used to decide which side is 'inside'. |  |
| `zonify` | `refinementSurfaces entries: faceZone, cellZone, cellZoneInside, insidePoint` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementBaffles.C` | Assigns cells to cellZones and faces to faceZones according to the named surfaces. | findCellZoneInsideWalk / findCellZoneGeometric: topological walk with zone faces blocked, or inside-test against the closed surface |

### snappyHexMesh / core  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `meshRefinement` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinement.H` | Central engine maintaining mesh/surface intersections, refinement, baffling, zoning, redistribution and IO for snappyHexMesh. | Stores per-face the surface index hit by the owner-neighbour cc-cc segment; updates it incrementally after every topology change |
| `meshRefinement debug/write/output flags` | `snappyHexMeshDict: debugFlags, writeFlags, outputFlags word lists` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinement.H` | Bit-mask word lists controlling intermediate debug/write output during meshing. | debugFlags: mesh, intersections, featureSeeds, attraction, layerInfo; writeFlags: mesh, noRefinement, scalarLevels, layerSets, layerFields; outputFlags: layerInfo |
| `meshRefinement::mapType` | `no (internal enum)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinement.H` | Controls how user face data is mapped when a face is refined. | MASTERONLY = 1 (master keeps value), KEEPALL = 2 (children inherit), REMOVE = 4 (set -1 on any refined face); combined as a bit mask |

### snappyHexMesh / feature snapping  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `calcNearestSurfaceFeature` | `snapControls/nFeatureSnapIter` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriverFeature.C` | Top-level feature-attraction routine combining surface displacement with feature edge/point attraction and constraints. | Blends the plain surface displacement with a feature attraction vector; pointConstraint rank 1/2/3 fixes the point to a plane, a line or a point |
| `determineBaffleFeatures / multiRegionFeatureSnap` | `snapControls/multiRegionFeatureSnap` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriverFeature.C` | Derives features from baffle edges and from points where more than two surface regions meet. | Multi-patch point detection: count distinct patch/region ids around a point; >2 gives a corner constraint |
| `featureAttractionUsingFeatureEdges` | `snapControls/explicitFeatureSnap` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriverFeature.C` | Explicit feature snapping: attracts points to the supplied eMesh feature edges and feature points. | findNearFeatureEdge / findNearFeaturePoint within snapDistance; nearest point on the edge segment gives the attraction vector |
| `featureAttractionUsingReconstruction` | `snapControls/implicitFeatureSnap` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriverFeature.C` | Implicit feature snapping: reconstructs features from the surface normals sampled by the surrounding faces. | Collect distinct surface normals around a point; least-squares/constraint combination gives a plane (rank 1), edge (rank 2) or corner (rank 3) attraction |
| `stringFeatureEdges / avoidDiagonalAttraction / preventFaceSqueeze` | `no (internal)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriverFeature.C` | Corrective filters that string attractions along edges, suppress attraction across a face diagonal, and stop faces collapsing. | Diagonal detection on quad faces; if both diagonal points attract to the same feature, one attraction is released |

### snappyHexMesh / geometry  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `patchFaceOrientation` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/patchFaceOrientation.H` | Transport type for PatchEdgeFaceWave that propagates consistent face orientation across a patch. | Flips the orientation flag when crossing an edge whose two faces disagree in walk direction |
| `pointData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/pointData/pointData.H` | pointEdgePoint variant carrying an extra scalar and vector, used by the medial-axis and layer waves. | PointEdgeWave update on squared distance to origin, additionally transporting s and v |
| `refinementFeatures` | `castellatedMeshControls/features { file "x.eMesh"; level n; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/refinementFeatures/refinementFeatures.H` | Wraps a set of extendedEdgeMesh feature files with per-feature refinement level and distance queries. | findNearestEdge / findNearestPoint on the edge-mesh octree; level from the per-feature (distance, level) table |
| `refinementRegions` | `castellatedMeshControls/refinementRegions` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/refinementRegions/refinementRegions.H` | Volume-refinement shells; answers 'what level should this point have'. | findLevel by inside/outside test or by nearest-distance interpolation of the (distance -> level) table; span modes interpolate a surface-closeness field |
| `refinementSurfaces` | `castellatedMeshControls/refinementSurfaces sub-dict; per-region keywords level, gapLevel, patchInfo, perpendicularAngle` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/refinementSurfaces/refinementSurfaces.H` | Container mapping every (surface, surface region) to its min/max refinement level, gap level, perpendicular angle and patch type. | findNearestIntersection / findAllIntersections against the searchableSurfaces set; per-global-region level lookup |
| `trackedParticle` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/trackedParticle/trackedParticle.H` | Particle that walks a feature edge through the mesh, marking every cell it passes through with the feature level. | Standard barycentric particle tracking from edge start to edge end, writing max(level) into the visited cell |

### snappyHexMesh / layers  <sub>(18)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `calculateLayerThickness` | `addLayersControls/relativeSizes plus exactly two of firstLayerThickness, finalLayerThickness, thickness, expansionRatio` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Computes the wanted and minimum total layer thickness at every patch point from the layerParameters specification. | Geometric series: total = first*(r^n - 1)/(r - 1); expansion ratio solved from total/first by Newton iteration in layerExpansionRatio; sizes scaled by local cell size when relativeSizes is on |
| `checkAndUnmark / countExtrusion` | `addLayersControls/nLayerIter, nRelaxedIter; meshQualityControls/relaxed` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | After trial layer insertion, checks the new cells against the quality criteria and unmarks the points that produced bad cells; loop repeats. | nLayerIter outer iterations, the last nRelaxedIter of which use the relaxed meshQualityControls sub-dict |
| `determineSidePatches` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Decides which patch the side faces of a layer stack at the edge of the extruded region belong to. | Side faces inherit the patch/zone of the single other patch connected across the boundary edge, creating a new patch if none exists |
| `handleFeatureAngle` | `addLayersControls/featureAngle, concaveAngle (default 90 deg)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Stops extrusion across sharp convex/concave feature edges of the patch. | Unmark where the angle between adjacent face normals exceeds featureAngle (convex) or falls below concaveAngle |
| `handleFeatureAngleLayerTerminations / findIsolatedRegions` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Stops layer growth where the mesh wraps around a sharp edge, and removes isolated single-point/edge/face layer islands. | isMaxEdge test on edge length ratio; connected-component analysis of the extrude mask |
| `handleNonManifolds / checkManifold` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Disables extrusion at non-manifold patch points and edges where a consistent layer cannot be built. | Edge-face count on the extrusion patch: any edge with more than two faces, or a point with more than one face fan, is unmarked |
| `handleWarpedFaces / detectWarpedFaces` | `addLayersControls/maxFaceThicknessRatio` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Disables extrusion on badly warped patch faces where the layer thickness would exceed a face-size ratio. | Unmark where layer thickness / face size > maxFaceThicknessRatio |
| `layerParameters` | `addLayersControls: nSurfaceLayers, relativeSizes, minThickness, featureAngle, nGrow, maxFaceThicknessRatio, nBufferCellsNoExtrude, nLayerIter, nRelaxedIter, meshShrinker, additionalReporting` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/layerParameters/layerParameters.H` | Container for all addLayersControls settings, per patch and global. |  |
| `layerParameters::layerSpecification` | `combinations of firstLayerThickness / finalLayerThickness / thickness / expansionRatio` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/layerParameters/layerParameters.H` | Enumerates the five valid ways of specifying layer thickness (exactly two parameters must be given). | FIRST_AND_TOTAL, FIRST_AND_EXPANSION, FINAL_AND_TOTAL, FINAL_AND_EXPANSION, TOTAL_AND_EXPANSION |
| `layerParameters::mergeFace` | `per-patch 'mergeFaces' entry, read as a boolean` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/layerParameters/layerParameters.H` | Whether the boundary faces of a layer cell on a patch are merged into one face. | no \| ifOnMeshedPatch (default) \| yes |
| `medialAxisSmoothingInfo` | `addLayersControls/nSmoothNormals, nSmoothSurfaceNormals, minMedialAxisAngle, nMedialAxisIter, nSmoothThickness` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriverShrink.C` | Builds the medial-axis field: nearest wall point/normal, medial-axis distance and ratio for every mesh point. | Two PointEdgeWaves: one from the moving walls, one seeded on medial-axis points detected where the wall normal reverses (n_i & n_j < medialAxisAngleCos) or at cusps of displacement |
| `printLayerData / writeLayerData` | `addLayersControls/additionalReporting; writeFlags layerSets, layerFields` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Reports achieved layer coverage per patch and optionally writes layer cell/face sets and fields. | Coverage = extruded faces / total patch faces; overall thickness reported both absolutely and as fraction of desired |
| `setNumLayers / growNoExtrusion` | `addLayersControls/nGrow, layers { <patch> { nSurfaceLayers n; } }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Converts per-patch layer counts to per-point counts and grows the no-extrusion zone by nGrow layers. | pointNLayers = max over surrounding faces; nGrow sweeps of dilation of the NOEXTRUDE set |
| `shrinkMeshMedialDistance` | `addLayersControls/meshShrinker (default displacementMedialAxis)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriverShrink.C` | Main routine that shrinks the mesh away from the boundary before layers are inserted, using medial-axis distances. | Displacement of an interior point scaled by the medial-axis ratio d_wall/(d_wall + d_medial), so motion decays to zero at the medial axis; iterated with quality-controlled scale-back |
| `smoothField / smoothPatchNormals / smoothNormals / averageNeighbours` | `addLayersControls/nSmoothThickness, nSmoothSurfaceNormals, nSmoothNormals` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriverShrink.C` | Edge-weighted smoothing of layer thickness and of patch/interior normals prior to shrinking. | x_p <- (sum_e w_e x_nbr)/(sum_e w_e) with w_e = 1 (unit edge weights); coupled-patch synchronised, applied nSmoothDisp times |
| `snappyLayerDriver` | `addLayersControls sub-dict` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.H` | Drives the layer-addition phase: decide extrusion, shrink the mesh, then insert the prismatic layer cells. | Shrink-and-insert strategy: existing mesh is shrunk away from the patch by the total layer thickness, then addPatchCellLayer extrudes nLayers cells into the freed space |
| `snappyLayerDriver::extrudeMode` | `no (internal enum)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.H` | Per point/face state of whether a layer is extruded there. | NOEXTRUDE (no layers), EXTRUDE, EXTRUDEREMOVE (extrude then locally remove the added faces) |
| `truncateDisplacement / setupLayerInfoTruncation` | `addLayersControls/nBufferCellsNoExtrude` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyLayerDriver.C` | Tapers the number of layers down towards the no-extrusion region so layer counts change gradually. | Reduce point layer count so that adjacent faces differ by at most one layer over nBufferCellsNoExtrude cells |

### snappyHexMesh / mesh motion  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `displacementMeshMoverMotionSolver` | `displacementMeshMover (dynamicMeshDict solver; sub-key 'meshMover')` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/externalDisplacementMeshMover/displacementMeshMoverMotionSolver.H` | Adapter making any externalDisplacementMeshMover usable as an fvMesh motionSolver. | Solves the cell-centre Laplacian for the motion displacement, then delegates the move to the wrapped mesh mover |
| `externalDisplacementMeshMover` | `base class of the externalDisplacementMeshMover dictionary RTS table; selected by addLayersControls/meshShrinker` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/externalDisplacementMeshMover/externalDisplacementMeshMover.H` | Abstract base for mesh movers whose boundary conditions come from an externally supplied displacement field; moves the mesh rather than returning new points. |  |
| `medialAxisMeshMover` | `displacementMedialAxis` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/externalDisplacementMeshMover/medialAxisMeshMover.H` | Shrinks the mesh by scaling motion with a medial-axis fraction between the moving surface and the nearest fixed surface. | scale = dispVec-weighted ratio medialRatio = d_wall/(d_wall + d_medial); requires fixedValue on moving patches, zeroFixedValue on stationary and slip on sliding patches |
| `zeroFixedValuePointPatchField` | `zeroFixedValue (pointPatchField type)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/externalDisplacementMeshMover/zeroFixedValue/zeroFixedValuePointPatchField.H` | Point patch field fixing the value to zero; marks stationary patches for the medial-axis mover. | value = 0 on the patch |

### snappyHexMesh / quality  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `badQualityToCell` | `badQualityToCell (topoSetDict source; takes a meshQualityControls sub-dict)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/motionSmoother/badQualityToCell/badQualityToCell.H` | topoSetSource that selects cells failing the snappyHexMesh mesh-quality criteria. | Runs the meshQualityControls checks (non-orthogonality, skewness, face weight, volume ratio, determinant, tet quality, twist, flatness, concavity) and selects failing cells |
| `badQualityToFace` | `badQualityToFace (topoSetDict source)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/motionSmoother/badQualityToFace/badQualityToFace.H` | topoSetSource that selects faces failing the snappyHexMesh mesh-quality criteria. | As badQualityToCell but returns the offending faces |
| `motionSmoother` | `no (constructed with meshQualityControls dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/motionSmoother/motionSmoother.H` | Moves the mesh by a given displacement, repeatedly scaling the displacement back until no mesh-quality errors remain. | scaleMesh: multiply the pointDisplacement by a per-point factor, halved around error faces, and re-check; supports baffles as single internal faces |
| `motionSmootherAlgo` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/motionSmoother/motionSmootherAlgo.H` | Algorithmic core of motionSmoother, separated from the field storage. | Laplacian smoothing of the point displacement field with 1/\|edge\| diffusivity, plus quality-driven scale-back using meshCheck::checkMesh |
| `motionSmootherData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/motionSmoother/motionSmootherData.H` | Holds the pointDisplacement field and scaling factor used by motionSmootherAlgo. |  |

### snappyHexMesh / refinement criterion  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `danglingCellRefine` | `no (internal, called with nFaces from the driver)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyRefineDriver.C` | Refines cells that have nearly all of their faces already refined, to avoid single unrefined cells hanging in a refined region. | Refine cells whose count of faces belonging to a finer neighbour exceeds nFaces |
| `featureEdgeRefine` | `castellatedMeshControls/features { file; level; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyRefineDriver.C` | Refines every cell pierced by an explicit feature edge up to that feature's level. | trackedParticle tracking along each feature edge marks cells; markFeatureCellLevel gives per-cell max feature level; refine where cellLevel < featureLevel |
| `markInternalDistanceToFeatureRefinement` | `features entry with distance-based levels` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementRefine.C` | Refines cells within a distance band of a feature edge mesh. | Nearest-feature-edge distance query against refinementFeatures; level from the (distance, level) table |
| `markInternalRefinement / shellRefine` | `castellatedMeshControls/refinementRegions { <shell> { mode ...; levels ((dist lvl)); } }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementRefine.C` | Volume ('shell') refinement: refines cells inside/outside a searchable region or within distance bands of it. | refinementRegions::findHigherLevel by inside/outside test or by nearest-distance lookup in the (distance, level) table |
| `markProximityRefinement / gapOnlyRefine` | `castellatedMeshControls/planarAngle, per-surface 'gapLevel'` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementRefine.C` | Refines cells that sit in a narrow gap between two nearly parallel, oppositely-facing surface sheets. | checkProximity: for a cell, compare each surface hit against the stored max-level hit; treat as a gap when normals are anti-parallel within planarCos = cos(planarAngle) and the separation is under the cell span |
| `markSurfaceCurvatureRefinement` | `castellatedMeshControls/resolveFeatureAngle (curvature = cos of it)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementRefine.C` | Refines cells where the surface normal varies strongly across the cell or where different surface regions meet. | Refine if (n_own & n_nei) < curvature, i.e. cos(angle between intersected surface normals) below the 'resolveFeatureAngle'-derived curvature threshold; also markDifferingRegions |
| `markSurfaceRefinement / surfaceOnlyRefine` | `castellatedMeshControls/refinementSurfaces { <surf> { level (min max); } }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/meshRefinement/meshRefinementRefine.C` | Refines cells whose cell-centre-to-cell-centre segment intersects a refinement surface, up to the surface's min/max level. | Segment/surface intersection of the owner-neighbour cc-cc vector; refine while cellLevel < surfaceMinLevel |
| `refinementRegions::refineMode` | `mode: inside \| outside \| distance \| insideSpan \| outsideSpan` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/refinementRegions/refinementRegions.H` | Enumerates the five volume-refinement modes available for a refinementRegion shell. | insideSpan/outsideSpan use an interpolated surface-closeness (local span) field and refine to give nCellsAcrossSpan cells across the span |

### snappyHexMesh / refinement support  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `refinementData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/refinementData.H` | FaceCellWave transport type that spreads refinement level so the transition between levels is gradual. | Wave-propagated (refinementCount, count) pair; cell accepts a neighbour's level when its own count is lower, guaranteeing a buffer of cells per level jump |
| `refinementDistanceData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/polyTopoChange/polyTopoChange/refinementDistanceData.H` | FaceCellWave transport of the nearest high-level origin point plus level-0 buffer size, giving distance-based level smoothing. | wantedLevel(pt) from origin level and \|pt - origin\|: level decreases by one per level0Size*nBufferLayers of distance walked out from the origin |

### snappyHexMesh / snapping  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `calcNearestSurface` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Per patch point, finds the nearest point on the surfaces and sets that as the raw displacement. | disp(p) = nearestPoint(p) - p, searched within calcSnapDistance(p) |
| `calcSnapDistance` | `snapControls/tolerance (snapTol)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Per patch point search span used when looking for the nearest surface/feature. | snapDistance(p) = snapTol * max(length of edges attached to p) |
| `detectNearSurfaces` | `snapControls/detectNearSurfacesSnap` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Overrides the displacement of points sitting inside a thin gap so both sides of the gap get captured. | Ray-casts along the point normal; if a second, oppositely-oriented surface hit is found within the local span, redirect the attraction |
| `preSmoothPatch / smoothPatchDisplacement` | `snapControls/nSmoothPatch` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Smooths the castellated boundary patch before matching it to the surface, to improve visibility of the surface from the patch. | nSmoothPatch sweeps of edge-weighted Laplacian averaging of patch point positions (weights 1/\|edge\|), constrained on feature/multi-patch points |
| `repatchToSurface` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Reassigns boundary faces to the patch of the surface region nearest their face centre after snapping. | Nearest-surface-region query at the face centre; face moved to the corresponding meshed patch |
| `scaleMesh / snapping relaxation` | `snapControls/nSolveIter, nRelaxIter; meshQualityControls sub-dict` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Moves the mesh by the computed displacement, backing off where mesh-quality checks fail. | Iterative scale-back: displacement multiplied by a per-point factor reduced around faces failing meshQualityControls until no errors beyond nInitErrors remain |
| `smoothDisplacement` | `snapControls/nSmoothDispl` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.C` | Diffuses the patch displacement into the interior point field before moving the mesh. | nSmoothDispl sweeps of edge-weighted Laplacian smoothing of the pointDisplacement field with 1/\|edge\| weights |
| `snapParameters` | `snapControls: nSmoothPatch, tolerance, nSolveIter, nRelaxIter, nFeatureSnapIter, explicitFeatureSnap, implicitFeatureSnap, multiRegionFeatureSnap, detectNearSurfacesSnap` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snapParameters/snapParameters.H` | Container for all snapControls settings. |  |
| `snappySnapDriver` | `snapControls sub-dict` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/mesh/snappyHexMesh/snappyHexMeshDriver/snappySnapDriver.H` | Drives the snapping phase: pre-smooth the patch, compute displacement to the surface, smooth into the interior, then move the mesh under quality control. | Outer loop of nSnap iterations; each iteration computes patch displacement, smooths it into the volume and scales it back with motionSmoother until quality checks pass |

---

## Mesh: Mesh tools and searching

> **Subsystem notes**
>
> Scope covered: src/meshTools (indexedOctree, searchableSurfaces, sets/topoSets + all topoSetSources, regionSplit, momentOfInertia, algorithms/{FaceCellWave,PointEdgeWave,PatchEdgeFaceWave}, cellClassification, edgeMesh, patchDist, pointDist, meshSearch, triSurface tools, plus the general tools), src/meshCheck, src/surfMesh, src/triSurface, and the two OpenFOAM-core directories the assignment implies (OpenFOAM/algorithms/indexedOctree and OpenFOAM/meshes/primitiveMesh/PatchTools).
> Two items named in the task brief do not exist under these names in OpenFOAM-12 and I verified this by grep rather than assuming: - There is no `src/meshTools/cellDist` directory and no `cellDistFuncs`/`wallDist` there.
> Wall/patch distance now lives in `src/meshTools/patchDist` (patchDistWave function namespace, WallInfo, WallLocationData, wallPoint, wallFace) and `src/meshTools/pointDist` (pointDist, pointEdgeDist). `cellDistance` survives only as a local variable name in patchDistWave.C/H.
> - There is no `MeshWave` class. The wave algorithms are `FaceCellWave` (cell/face), `PointEdgeWave` (point/edge) and `PatchEdgeFaceWave` (patch edge/face), all in src/meshTools/algorithms. I catalogued all three plus every transported-datum type used with them.
> - `PatchTools` is not under meshTools; it is at src/OpenFOAM/meshes/primitiveMesh/PatchTools, split across PatchToolsCheck/EdgeOwner/GatherAndMerge/Match/Normals/Search/SortEdges/SortPoints .C files. I listed it as one entry with the method list in the equations field.
> Runtime-selection keywords were taken from the actual addToRunTimeSelectionTable / addNamedToRunTimeSelectionTable / addNamedTemplatedToRunTimeSelectionTable calls, not from memory. Four distinct table families appear in this subsystem: 1.
> `topoSetSource` — table `word`, constructed from (polyMesh, dictionary). 49 concrete sources; the keyword is what a user writes as `source` in system/topoSetDict. 2.
> `topoSet` — three tables (`word`, `size`, `set`); cellSet/faceSet/pointSet/cellZoneSet/faceZoneSet/pointZoneSet each register in all three. 3. `searchableSurface` — table `dict`, constructed from (IOobject, dictionary); the keyword is `type` in the snappyHexMeshDict `geometry` sub-dictionary.
> 11 concrete surfaces. 4.
> File-format tables keyed on `fileExtension`, not on a type word: edgeMesh (read + write), extendedEdgeMesh (read), and the templated MeshedSurface / UnsortedMeshedSurface (read) and MeshedSurfaceProxy / MeshedSurface / UnsortedMeshedSurface (write) tables, each registered separately for Face = `face` and Face = `triFace`.
> Where a format registers only a write function I said so explicitly (SMESH, WRL, X3D are write-only; NAS is read-only).
> Note the asymmetry between src/surfMesh and src/triSurface: surfMesh formats are genuinely runtime-selected by extension, whereas `triSurface::read`/`triSurface::write` (src/triSurface/triSurface/triSurface.C, lines ~289-420) dispatch on the extension with a hard-coded if/else chain over its own readers in triSurface/triSurface/interfaces/{AC3D,GTS,NAS,OBJ,OFF,SMESH,STL,TRI,VTK}.
> Those are compiled-in, not registered, so a user cannot extend them without editing that chain. Read extensions: ftr, stl, stlb, gts, obj, off, tri, ac, nas, vtk (and transparent .gz). Write extensions: ftr, stl, stlb, gts, obj, off, vtk, tri, ac, smesh.
> `polyCellSet` is a fifth, lighter selection mechanism: not a RTS table but a NamedEnum on a `select` keyword (all | cellSet | cellZone | points), used pervasively by fvModels and fvConstraints.
> meshCheck is entirely function namespaces (Foam::meshCheck) rather than classes — I listed the four headers as namespace entries and put the actual quality metric formulae (orthogonality, skewness, pyramid volume, closedness/openness, face weight, volume ratio, determinant, flatness) into their equations fields, since those are the numbers checkMesh reports and the ones snappyHexMesh's meshQualityControls thresholds act on.
> Entries total 152.
> Overlap risk with other catalogue parts: meshToMesh, patchToPatch, cellsToCells, mappedPatches, nonConformal, cutPoly, cutTriTet, coordinateSystems, layerInfo, patchIntersection and triIntersect are also under src/meshTools but read as separate subsystems (mapping / non-conformal coupling / cutting), so I deliberately left them out.

### Cell selection  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `polyCellSet` | `yes - `select` keyword: all \| cellSet \| cellZone \| points` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/polyCellSet/polyCellSet.H` | Run-time selected cell-set selection used by fvModels/fvConstraints - by points, cellSet, cellZone, or all cells. | `points` uses meshSearch::findCell for each supplied location |

### Distance calculation  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `patchDistWave` | `no (function namespace: getChangedFaces, wave, calculate)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/patchDist/patchDistWave/patchDistWave.H` | Namespace of functions that FaceCellWave the distance (and optional data) from a chosen set of patches into the mesh. | getChangedFaces seeds the patch faces; wave() returns d = \|C_cell - x_nearestWallFace\|; replaces the old cellDistFuncs/cellDist directory |
| `patchPatchDist` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PatchEdgeFaceWave/patchPatchDist.H` | Computes, on one patch, the distance to the nearest neighbouring patches. | PatchEdgeFaceWave of patchEdgeFacePoint seeded on the shared edges; d = \|x - x_nearest\| |
| `pointDist` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/pointDist/pointDist.H` | Point field of distance to a specified set of patch points and pointZone points (or to all points). | PointEdgeWave of pointEdgeDist seeded on the selected points; d = \|p - p_nearest\| |

### Distance transport data  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `WallInfo` | `no (template)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/patchDist/WallInfo/WallInfo.H` | Templated wrapper adding a transported payload to a wall-location type for FaceCellWave. |  |
| `WallLocationData` | `no (template)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/patchDist/WallLocation/WallLocationData.H` | Templated wall-location holding origin, squared distance and an arbitrary datum. | distSqr = magSqr(x - x_origin); the datum of the winning origin is carried along |
| `pointEdgeDist` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/pointDist/pointEdgeDist.H` | PointEdgeWave datum holding the nearest origin point and squared distance for pointDist. | distSqr = magSqr(p - p_origin) |
| `wallFace` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/patchDist/WallLocation/wallFace.H` | Wall-location carrying the whole wall face so distance is measured to the face polygon, not just its centre. | d = nearest distance from the cell centre to the wall face polygon (per-triangle projection) |
| `wallPoint` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/patchDist/WallLocation/wallPoint.H` | Wall-location using the nearest wall face centre as the origin (point-to-point distance). | d = \|C - Cf_wall\| |

### General mesh tools  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PrimitiveOldTimePatch` | `no (template)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/PrimitiveOldTimePatch/PrimitiveOldTimePatch.H` | PrimitivePatch that additionally carries the old-time point positions, with primitiveOldTimePatch and uindirectPrimitiveOldTimePatch typedefs. |  |
| `cellFeatures` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/cellFeatures/cellFeatures.H` | Analyses a single cell for feature edges and points above a given angle, and merges coplanar faces into superFaces. | edge is a feature if the angle between its two adjacent face normals exceeds the specified feature angle; superFaces are the connected sets of non-feature-separated faces |
| `edgeFaceCirculator` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeFaceCirculator/edgeFaceCirculator.H` | STL-style iterator that walks around a mesh edge from face to face through the cells sharing it. | implicit edge = (face, index in face, walk direction); ++ steps face -> cell -> opposite face containing the same edge |
| `meshStructure` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshStructure/meshStructure.H` | Detects whether a mesh is an extrusion of a given set of patch faces and returns the layer/column addressing. | FaceCellWave of topoDistanceData and PointEdgeWave of pointTopoDistanceData from the patch; layer index = topological distance, column index = originating patch face |
| `meshTools (namespace)` | `no (function namespace)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshTools/meshTools.H` | Static helpers for simple mesh queries and OBJ debug output (face normals, edge/face relations, visNormal, writeOBJ). | normEdgeVec, visNormal (a normal is visible if it has positive dot product with every adjacent face normal), getEdgeFaces, walking helpers |
| `polyMeshZipUpCells` | `no (global function bool polyMeshZipUpCells(polyMesh&))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/polyMeshZipUpCells/polyMeshZipUpCells.H` | Global function that inserts extra face vertices so that every cell becomes topologically closed. | detects open edge loops in a cell and inserts the missing points into the offending faces |
| `surfaceSets` | `no (static function collection)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/surfaceSets/surfaceSets.H` | Utilities relating mesh cell/point sets to a surface (getSurfaceSets, getHangingCells) used by the meshing tools. | nearest-surface distance and inside/outside tests to split cells into inside/outside/cut sets |
| `tetOverlapVolume` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/tetOverlapVolume/tetOverlapVolume.H` | Computes the overlap volume of two cells by tetrahedral decomposition (used by mesh-to-mesh mapping and AMI-type checks). | decompose both cells into tets, clip tet A successively by the four planes of tet B (Sutherland-Hodgman style), sum the resulting tet volumes |
| `twoDPointCorrector` | `no (uses the empty patch direction to detect the 2D plane)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/twoDPointCorrector/twoDPointCorrector.H` | Corrects a mesh-motion point displacement field so a 2D mesh does not acquire a third dimension. | for every edge approximately normal to the empty-direction plane, force the edge onto the plane normal by averaging and re-imposing the end points' out-of-plane coordinates |

### Geometric integration  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `momentOfInertia` | `no (static function collection; reference C code in volumeIntegration/volInt.c)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/momentOfInertia/momentOfInertia.H` | Computes mass, centre of mass and inertia tensor of polyhedra, mesh cells or triSurfaces, as solid body or thin shell. | Mirtich (1996) polyhedral volume integrals - divergence theorem reduces the ten volume integrals to face and then edge (projection) integrals; applyParallelAxisTheorem gives J' = J + m(\|d\|^2 I - d d) |

### Mesh checking  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `mergeAndWrite` | `no (function namespace; the writer itself is a runtime-selected surfaceWriter/setWriter)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshCheck/mergeAndWrite/mergeAndWrite.H` | Gathers faceSet/cellSet/pointSet representations onto the master and writes them as surface or set files under postProcessing/. | PatchTools::gatherAndMerge collocated-point merge before writing |
| `meshCheck (namespace)` | `no (function namespace)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshCheck/meshCheck.H` | Top-level topology and geometry checking entry points used by checkMesh and by the meshing utilities. | findOppositeWedge, checkWedges (opposite normal and common axis), checkCoupledPoints (0th vertex agreement on coupled faces), checkTopology, checkGeometry, checkMesh against a meshQualityDict |
| `polyMeshCheck` | `no (function namespace Foam::meshCheck)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshCheck/polyMeshCheck/polyMeshCheck.H` | Coupled-aware polyMesh checks including the mesh-quality criteria used by snappyHexMesh. | checkFaceOrthogonality, checkFaceSkewness, checkFaceWeight (min(\|d_own\|,\|d_nei\|)/\|d\|), checkVolRatio (min(V_own,V_nei)/max), checkFaceTets (tet decomposition quality), checkFaceTwist, checkTriangleTwist, checkFaceFlatness, checkFaceArea, checkCellDeterminant, checkEdgeAlignment |
| `primitiveMeshCheck` | `no (function namespace Foam::meshCheck)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshCheck/primitiveMeshCheck/primitiveMeshCheck.H` | Topology and geometry checks on a primitiveMesh, plus the underlying quality metric fields. | faceOrthogonality = (d & Sf)/(\|d\|\|Sf\|); faceSkewness = \|Cf - intersection of d with the face plane\| / \|d\|; facePyramidVolume = (1/3)(Cf-C)&Sf; cellClosedness openness = \|sum(Sf)\|/V^(2/3) and aspect ratio; faceConcavity, faceFlatness = \|Sf\| / sum(triangle areas); edgeAlignment; cellDeterminant from sum(Sf Sf) |

### Mesh search  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `meshSearch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshSearch/meshSearch.H` | Local (non-parallel) searches on a polyMesh - find cell/face/point containing or nearest to a location. | demand-driven octrees on cells, boundary faces and points; findCell by tet-decomposition walk or octree containment |
| `meshSearchFACE_CENTRE_TRISMeshObject` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshSearch/meshSearchFACE_CENTRE_TRISMeshObject.H` | Cached meshSearch using the FACE_CENTRE_TRIS cell tet-decomposition. | tet decomposition about face centres rather than face diagonals |
| `meshSearchMeshObject` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshSearch/meshSearchMeshObject.H` | DemandDrivenMeshObject wrapper so a meshSearch is cached on the mesh registry. |  |

### Mesh/surface classification  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cellClassification` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/cellClassification/cellClassification.H` | Cuts a mesh with a surface and marks every cell as CUT, OUTSIDE or INSIDE; provides feature (hanging cell, region edge/point) utilities. | cut detection by edge-surface pierce and surface-edge/face pierce; OUTSIDE by FaceCellWave/cellInfo flood from supplied outside points, blocked by CUT cells; INSIDE is the remainder |

### Octree data wrapper  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dynamicTreeDataPoint` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/algorithms/dynamicIndexedOctree/dynamicTreeDataPoint.H` | Wraps a DynamicList of points for nearest-point searches in a dynamicIndexedOctree. | nearest = argmin \|p - p_i\|; no bounding boxes around points |
| `treeDataCell` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/algorithms/indexedOctree/treeDataCell.H` | Encapsulates polyMesh cells for octree search, used to find the cell containing a point. | cell bounding-box overlap test then tet-decomposition point-in-cell test |
| `treeDataEdge` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/indexedOctree/treeDataEdge.H` | Holds a (subset of an) edgeList so an octree can search on edges. | nearest point on line segment: t = clamp((p-a)&(b-a)/\|b-a\|^2, 0, 1) |
| `treeDataFace` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/indexedOctree/treeDataFace.H` | Encapsulates polyMesh faces (optionally a face subset) for octree face searches. | face bounding-box overlap plus triangle-fan ray/nearest tests about the face centre |
| `treeDataPoint` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/indexedOctree/treeDataPoint.H` | Holds a reference to a pointField (optionally a subset) for nearest-point octree searches. | nearest = argmin \|p - p_i\|; only overlaps() and findNearest() are meaningful |
| `treeDataPrimitivePatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/indexedOctree/treeDataPrimitivePatch.H` | Encapsulates any PrimitivePatch so an octree can search its faces (templated on patch type). | per-face triangle decomposition for nearest/intersection; inside/outside from patch point normals |
| `treeDataTriSurface` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/indexedOctree/treeDataTriSurface.H` | Encapsulates a triSurface for indexedOctree searches (nearest, intersection, inside/outside). | triangle::nearestPointClassify barycentric projection; Moller-Trumbore style ray-triangle pierce |

### Patch tools  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PatchTools` | `no (static function collection)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PatchTools/PatchTools.H` | Static collection of searching, sorting, matching and normal-calculation tools for any PrimitivePatch. | checkOrientation, markZone/markZones (edge-connected flood fill), subsetMap, calcBounds, sortedEdgeFaces (faces sorted by angle about the edge), sortedPointEdges, edgeOwner, matchPoints/matchEdges, parallel-consistent pointNormals/edgeNormals (area-weighted normal sum synchronised over coupled points), gatherAndMerge |

### Region splitting  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `localPointRegion` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/regionSplit/localPointRegion.H` | Determines, for points on baffle boundary faces, which point region each face-point belongs to, so points can be duplicated when splitting baffles. | cell-face-cell walk around each boundary point; regions numbered -1..nRegions-1 (not consecutive per processor) |
| `regionSplit` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/regionSplit/regionSplit.H` | Separates the mesh into distinct unconnected regions and labels each cell with a globally numbered region. | local cell-face-cell flood fill, offset into a global numbering, then FaceCellWave/minData merge of regions across processor boundaries and a final compaction pass |

### Search structure  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dynamicIndexedOctree` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/algorithms/dynamicIndexedOctree/dynamicIndexedOctree.H` | Octree with dynamic storage so that elements can be inserted and deleted after construction. | same octant subdivision as indexedOctree, with DynamicList content lists allowing insert/remove |
| `indexedOctree` | `no (template, TemplateName(indexedOctree))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/algorithms/indexedOctree/indexedOctree.H` | Non-pointer-based hierarchical recursive octree, templated on the tree-data type it searches. | recursive 8-way subdivision of a treeBoundBox until maxLevel or minSize/maxLeafRatio; node/leaf indices packed in labelBits |
| `labelBits` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/algorithms/indexedOctree/labelBits.H` | A 29-bit label plus 3-bit direction packed into a single label, used for octree node/octant addressing. | bit packing: value = (index<<3) \| octant |
| `volumeType` | `no (NamedEnum: unknown, mixed, inside, outside)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/algorithms/indexedOctree/volumeType.H` | Enumeration unknown/mixed/inside/outside returned by inside-outside octree and surface queries. |  |

### Searchable surface  <sub>(11)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `closedTriSurfaceMesh` | `yes - `type closedTriSurfaceMesh;` (entries: file, scale, minQuality)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/closedTriSurfaceMesh/closedTriSurfaceMesh.H` | triSurfaceMesh that is asserted closed despite imperfections such as small holes or multiple parts. | as triSurfaceMesh but hasVolumeType() forced true so inside/outside queries are permitted |
| `searchableBox` | `yes - `type searchableBox;` (entries: min, max)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableBox/searchableBox.H` | Axis-aligned rectangular box surface geometry for snappyHexMesh, defined by min and max points. | treeBoundBox slab intersection; nearest point by per-component clamping to [min,max] |
| `searchableCylinder` | `yes - `type searchableCylinder;` (entries: point1, point2, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableCylinder/searchableCylinder.H` | Cylinder surface geometry defined by two axis end points and a radius. | radial distance \|(p-p1) - ((p-p1)&a)a\| = r with axial clamp; quadratic solve for line intersection |
| `searchableDisk` | `yes - `type searchableDisk;` (entries: origin, normal, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableDisk/searchableDisk.H` | Flat circular disk surface geometry defined by origin, normal and radius. | project onto plane ((p-o)&n)=0 then clamp radial offset to r; ray-plane intersection with radius test |
| `searchableExtrudedCircle` | `yes - `type searchableExtrudedCircle;` (entries: file, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableExtrudedCircle/searchableExtrudedCircle.H` | Tube geometry formed by sweeping a circle of given radius along a line read from an .eMesh edgeMesh. | distance to the edge-mesh polyline (via edge octree) offset by radius; local frame by quaternion rotation along the spine |
| `searchablePlane` | `yes - `type searchablePlane;` (planeType pointAndNormal\|planeEquation\|embeddedPoints)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchablePlane/searchablePlane.H` | Infinite plane surface geometry, constructed from any Foam::plane specification. | signed distance d = (p - p0)&n; line hit at t = -d/(n&(e-s)) |
| `searchablePlate` | `yes - `type searchablePlate;` (entries: origin, span)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchablePlate/searchablePlate.H` | Finite axis-aligned rectangular plate; the span must have exactly one zero component giving the normal. | plane projection then clamp the two in-plane components to [origin, origin+span] |
| `searchableSphere` | `yes - `type searchableSphere;` (entries: centre, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableSphere/searchableSphere.H` | Sphere surface geometry defined by centre and radius. | nearest = c + r*(p-c)/\|p-c\|; ray hit from quadratic \|s+t*d-c\|^2 = r^2 |
| `searchableSurfaceCollection` | `yes - `type searchableSurfaceCollection;` (sub-entries: surface, scale, transform, mergeSubRegions)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableSurfaceCollection/searchableSurfaceCollection.H` | Builds copies of an existing surface geometry, each scaled and transformed; no boolean intersection is done. | per-instance affine map x' = R*(S*x) + origin using coordinateSystem/coordinateRotation; queries take the nearest hit over instances |
| `searchableSurfaceWithGaps` | `yes - `type searchableSurfaceWithGaps;` (entry: gap)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableSurfaceWithGaps/searchableSurfaceWithGaps.H` | Wraps another surface and fires offset test rays so that pierces do not slip through small gaps. | test vector shifted by +/- gap perpendicular to the ray; a hit is registered only if both offset rays hit |
| `triSurfaceMesh` | `yes - `type triSurfaceMesh;` (entries: file, scale, minQuality)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/triSurfaceMesh/triSurfaceMesh.H` | Faceted surface geometry read from OBJ/STL etc., searched through an indexedOctree of treeDataTriSurface. | octree nearest/pierce on triangles; normals ignore triangles with quality < minQuality; inside/outside from triangle normal orientation |

### Searchable surface (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `searchableSurface` | `runtime-selectable base, declareRunTimeSelectionTable(autoPtr, searchableSurface, dict, (io, dict)); dictionary keyword is `type`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableSurface/searchableSurface.H` | Abstract base for analytical or triangulated surfaces encapsulating all search routines. | queries return pointIndexHit {hit?, point, surface index}; findNearest, findLine, findLineAny, findLineAll, getVolumeType |

### Searchable surface container  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `searchableSurfaces` | `no (container; reads the snappyHexMeshDict `geometry` dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableSurfaces/searchableSurfaces.H` | PtrList container of named searchableSurfaces built from a geometry sub-dictionary, with name/region lookup. |  |

### Searchable surface utility  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `searchableSurfacesQueries` | `no (static function collection)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/searchableSurfaces/searchableSurfacesQueries/searchableSurfacesQueries.H` | Static tools that query several searchableSurfaces at once (nearest, all intersections, inside/outside, signed distance). | findAllIntersections sorted along the ray; findNearest picks min over surfaces; signedDistance from getVolumeType |

### Surface boolean ops  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `booleanSurface` | `no (booleanOpType enum: OR/union, AND/intersection, MINUS/difference)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/booleanOps/booleanSurface/booleanSurface.H` | Combines two triSurfaces with a boolean operation (union, intersection, difference). | 1) surfaceIntersection edge-surface cuts, 2) intersectedSurface retriangulation, 3) subset each side by flood fill from a user-supplied face, 4) merge sharing only the intersection points |
| `edgeIntersections` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/booleanOps/surfaceIntersection/edgeIntersections.H` | Holds the intersections of one surface's edges with another surface, and can perturb points to resolve degenerate hits. | iterative random perturbation of surface points to remove edge-hits-edge and edge-hits-point degeneracies |
| `edgeSurface` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/booleanOps/intersectedSurface/edgeSurface.H` | Cloud-of-edges description (points, edges, faceEdges, parentEdge) used to insert cuts and split faces. | surface local points first then intersection points; split surface edges first then intersection edges |
| `intersectedSurface` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/booleanOps/intersectedSurface/intersectedSurface.H` | Builds the properly triangulated surface that results from inserting an intersection into a triSurface. | split the surface edges at the cut points, build face-edge addressing, right-handed edge-point-edge walk to reconstruct faces (splitFace), then retriangulate; floating interior cut edges handled by resplitFace |
| `surfaceIntersection` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/booleanOps/surfaceIntersection/surfaceIntersection.H` | Computes the intersection line(s) between two triSurfaces (or a surface with itself), with full addressing back to the originals. | intersect every edge of one surface with the other surface; a face pair seen twice yields an intersection edge; point-touching vs edge-piercing decided with intersection::planarTol() as a fraction of the edge length |

### Surface file format  <sub>(15)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `AC3DsurfaceFormat` | `yes - file extension `ac` (MeshedSurface read; MeshedSurfaceProxy and UnsortedMeshedSurface write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/ac3d/AC3DsurfaceFormat.H` | Reads and writes AC3D format; input is already zone-organised and output is always zone sorted. |  |
| `GTSsurfaceFormat` | `yes - file extension `gts` (UnsortedMeshedSurface read; MeshedSurface and UnsortedMeshedSurface write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/gts/GTSsurfaceFormat.H` | Reads and writes GTS format; written only for all-triangle surfaces and never zone sorted. |  |
| `NASsurfaceFormat` | `yes - file extensions `bdf` and `nas` (MeshedSurface read only)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/nas/NASsurfaceFormat.H` | Nastran surface reader handling short and long formats, compact floating point and ANSA/Hypermesh zone names. |  |
| `OBJsurfaceFormat` | `yes - file extension `obj` (MeshedSurface read; MeshedSurfaceProxy write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/obj/OBJsurfaceFormat.H` | Reads and writes Alias/Wavefront OBJ surfaces (negative face indices unsupported). |  |
| `OFFsurfaceFormat` | `yes - file extension `off` (MeshedSurface read; MeshedSurfaceProxy write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/off/OFFsurfaceFormat.H` | Reads and writes Geomview OFF polyList format; the colorspec is ignored on read and set to the zone number on write. |  |
| `OFSsurfaceFormat` | `yes - file extension `ofs` (MeshedSurface read; MeshedSurfaceProxy write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/ofs/OFSsurfaceFormat.H` | Reads and writes the native single-file OpenFOAM surface format (points, faces and zones in one file). |  |
| `SMESHsurfaceFormat` | `yes - file extension `smesh` (MeshedSurfaceProxy write only)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/smesh/SMESHsurfaceFormat.H` | Writes the tetgen .smesh piecewise-linear-complex format. |  |
| `STARCDsurfaceFormat` | `yes - file extension `inp` (MeshedSurface read; MeshedSurfaceProxy write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/starcd/STARCDsurfaceFormat.H` | Reads and writes surface shells from pro-STAR vrt/cel file pairs. |  |
| `STLpoint / STLtriangle` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/stl/STLtriangle.H` | Single-precision point and 50-byte triangle records matching the binary STL layout. | binary record = normal + 3 vertices (12 floats) + 2-byte attribute/region count |
| `STLsurfaceFormat` | `yes - file extensions `stl` (ASCII) and `stlb` (binary), on MeshedSurface read and on MeshedSurfaceProxy / UnsortedMeshedSurface write` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/stl/STLsurfaceFormat.H` | Reads and writes ASCII and binary STL; zones are sorted before the faces are created. | ASCII reader is a flex lexer (STLsurfaceFormatASCII.L); binary reader uses the 50-byte STLtriangle record |
| `TRIsurfaceFormat` | `yes - file extension `tri` (MeshedSurface read; MeshedSurfaceProxy and UnsortedMeshedSurface write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/tri/TRIsurfaceFormat.H` | Reads and writes the .tri triangle format, sorting zones before creating faces. |  |
| `VTKsurfaceFormat` | `yes - file extension `vtk` (MeshedSurface read; MeshedSurfaceProxy and UnsortedMeshedSurface write)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/vtk/VTKsurfaceFormat.H` | Reads and writes VTK legacy polydata surfaces; output is never zone sorted. |  |
| `WRLsurfaceFormat` | `yes - file extension `wrl` (MeshedSurfaceProxy write only)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/wrl/WRLsurfaceFormat.H` | Writes VRML97 (.wrl) surfaces. |  |
| `X3DsurfaceFormat` | `yes - file extension `x3d` (MeshedSurfaceProxy write only)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/x3d/X3DsurfaceFormat.H` | Writes X3D surfaces. |  |
| `surfaceFormatsCore` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceFormats/surfaceFormatsCore.H` | Shared helpers (zone naming, one-zone construction, file checking) for all surface readers and writers. |  |

### Surface interpolation  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `geompack` | `no (C library)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/triSurfaceTools/geompack/geompack.H` | Third-party 2D Delaunay triangulation routines used by pointToPointPlanarInterpolation. | dtris2 divide-and-conquer Delaunay with Lawson edge swapping (swapec, diaedg in-circle test) |
| `pointToPointPlanarInterpolation` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/triSurfaceTools/pointToPointPlanarInterpolation.H` | Interpolates between two unstructured point sets by 2D Delaunay triangulation in a fitted plane; used by timeVaryingMapped boundary conditions. | least-squares plane fit, project to 2D, Delaunay triangulate (geompack dtris2), then barycentric weights of the three enclosing vertices (nearest-point fallback outside the hull) |

### Wave algorithm  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FaceCellWave` | `no (template, TemplateName(FaceCellWave))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/FaceCellWave/FaceCellWave.H` | Propagates templated information through the mesh face-to-cell-to-face, one cell layer per iteration. | iterative face->cell then cell->face updates; propagation continues while Type::updateCell/updateFace report a change larger than propagationTol (default 0.01); handles processor and cyclic couplings |
| `PatchEdgeFaceWave` | `no (template, TemplateName(PatchEdgeFaceWave))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PatchEdgeFaceWave/PatchEdgeFaceWave.H` | Propagates templated information along a patch surface, one face layer per iteration. | alternating edge->face and face->edge updates restricted to the patch, with global edge/point synchronisation across processor boundaries |
| `PointEdgeWave` | `no (template, TemplateName(PointEdgeWave))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PointEdgeWave/PointEdgeWave.H` | Propagates templated information through the mesh point-to-edge-to-point, one edge layer per iteration. | alternating point->edge and edge->point updates; parallel exchange in two steps - patch points in (patchFace, index) offset notation, then a reduce over globally shared points |

### Wave transport data  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PatchEdgeFacePointData` | `no (template)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PatchEdgeFaceWave/PatchEdgeFacePointData.H` | Transports the nearest point location plus an arbitrary payload datum for PatchEdgeFaceWave. | as patchEdgeFacePoint with the payload carried alongside the winning origin |
| `cellInfo` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/cellClassification/cellInfo.H` | FaceCellWave datum holding a cell type flag, used by cellClassification for inside/outside determination. | type propagates unless the receiving cell is CUT (which blocks the wave) |
| `minData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/regionSplit/minData.H` | FaceCellWave datum that transports the minimum of a passive label (used by regionSplit). | update: data = min(data, neighbourData) |
| `patchEdgeFacePoint` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PatchEdgeFaceWave/patchEdgeFacePoint.H` | Transports the nearest origin point location for PatchEdgeFaceWave. | update keeps the origin with the smaller squared distance \|x - x0\|^2 (tolerance-guarded) |
| `patchEdgeFaceRegion` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PatchEdgeFaceWave/patchEdgeFaceRegion.H` | Transports a single region label along a patch; -2 marks blocked elements. | update takes min(region) over the incoming values |
| `patchEdgeFaceRegions` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PatchEdgeFaceWave/patchEdgeFaceRegions.H` | Transports a list of region labels along a patch; -1 marks blocked. | element-wise min over the region lists |
| `pointEdgePoint` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/algorithms/PointEdgeWave/pointEdgePoint.H` | Holds the nearest wall point origin and squared distance for PointEdgeWave wall-distance calculation. | distSqr = magSqr(x - x_origin); update accepts the candidate when it reduces distSqr by more than the tolerance |
| `pointTopoDistanceData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshStructure/pointTopoDistanceData.H` | PointEdgeWave datum recording topological distance from starting points. | distance incremented by one for each point->edge->point step |
| `topoDistanceData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/meshStructure/topoDistanceData.H` | FaceCellWave datum recording topological (layer-count) distance from starting faces. | distance incremented by one for each face->cell->face step |

### edgeMesh  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `edgeMesh` | `runtime-selectable base, declareRunTimeSelectionTable(autoPtr, edgeMesh, fileExtension, (name)) plus a `write` member-function table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMesh.H` | Points connected by edges, readable from file with the reader selected by file extension. |  |
| `extendedEdgeMesh` | `runtime-selectable base, declareRunTimeSelectionTable(autoPtr, extendedEdgeMesh, fileExtension, (name))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/extendedEdgeMesh/extendedEdgeMesh.H` | Feature-edge and feature-point description with adjacent normals and sorted classification bands. | points sorted convex \| concave \| mixed \| non-feature; edges sorted external(convex) \| internal(concave) \| flat \| open \| multiply-connected, delimited by concaveStart_, mixedStart_, nonFeatureStart_, internalStart_, flatStart_, openStart_, multipleStart_ |
| `extendedFeatureEdgeMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/extendedEdgeMesh/extendedFeatureEdgeMesh/extendedFeatureEdgeMesh.H` | extendedEdgeMesh with regIOobject IO (the .extendedFeatureEdgeMesh files written by surfaceFeatures). |  |
| `featureEdgeMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/featureEdgeMesh/featureEdgeMesh.H` | edgeMesh with regIOobject IO, stored as constant/triSurface/*.eMesh. |  |

### edgeMesh format  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `NASedgeFormat` | `yes - file extensions `bdf` and `nas` (edgeMesh read table)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/nas/NASedgeFormat.H` | Reads edges from Nastran bulk-data files. |  |
| `OBJedgeFormat` | `yes - file extension `obj` (edgeMesh read and write tables)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/obj/OBJedgeFormat.H` | Reads and writes Alias/Wavefront OBJ line geometry (negative vertex indices unsupported). |  |
| `STARCDedgeFormat` | `yes - file extension `inp` (edgeMesh read and write tables)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/starcd/STARCDedgeFormat.H` | Reads and writes lines from pro-STAR vrt/cel files. |  |
| `VTKedgeFormat` | `yes - file extension `vtk` (edgeMesh read and write tables)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/vtk/VTKedgeFormat.H` | Reads and writes VTK legacy polydata line format. |  |
| `edgeMeshFormat` | `yes - file extension `eMesh` (edgeMesh read and write tables)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/edgeMesh/edgeMeshFormat.H` | Reads and writes the native single-file OpenFOAM edge format. |  |
| `edgeMeshFormatsCore` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/edgeMeshFormatsCore.H` | Helper functions shared by the edge-mesh readers and writers. |  |
| `extendedEdgeMeshFormat` | `yes - file extension `extendedFeatureEdgeMesh` (extendedEdgeMesh read table)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/extendedEdgeMesh/extendedEdgeMeshFormats/extendedEdgeMeshFormat/extendedEdgeMeshFormat.H` | Reads and writes the native single-file OpenFOAM extendedEdgeMesh format. |  |
| `extendedFeatureEdgeMeshFormat` | `yes - file extension `featureEdgeMesh` (edgeMesh read table)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/edgeMesh/edgeMeshFormats/extendedFeatureEdgeMesh/extendedFeatureEdgeMeshFormat.H` | Reads an extendedFeatureEdgeMesh file back as a plain edgeMesh. |  |

### surfMesh  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MeshedSurface` | `runtime-selectable base, declareRunTimeSelectionTable on fileExtension per Face type; New(name) or New(name, ext)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/MeshedSurface/MeshedSurface.H` | Surface geometry mesh with contiguous surface-zone information, templated on the face type (face or triFace). |  |
| `MeshedSurfaceIOAllocator` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/MeshedSurfaceAllocator/MeshedSurfaceIOAllocator.H` | Helper storing points, faces and zones as registered IOobjects for surfMesh. |  |
| `MeshedSurfaceProxy` | `holds the `write` member-function selection table keyed on fileExtension` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/MeshedSurfaceProxy/MeshedSurfaceProxy.H` | Write-only proxy that lets MeshedSurface, UnsortedMeshedSurface and surfMesh be written to any registered format. |  |
| `UnsortedMeshedSurface` | `runtime-selectable base, declareRunTimeSelectionTable on fileExtension per Face type` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/UnsortedMeshedSurface/UnsortedMeshedSurface.H` | Surface mesh whose zone information is carried per face as a zoneId, so faces need not be zone-contiguous. |  |
| `surfMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfMesh/surfMesh.H` | A registered, IO-capable surface mesh of general polygon faces with zones. |  |
| `surfZone` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfZone/surfZone/surfZone.H` | A contiguous zone of faces on a MeshedSurface (size plus start index). |  |
| `surfZoneIOList` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfZone/surfZone/surfZoneIOList.H` | IOobject wrapper for a list of surfZones. |  |
| `surfZoneIdentifier` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfZone/surfZoneIdentifier/surfZoneIdentifier.H` | Name, index and geometricType identifier for a surface zone. |  |
| `surfaceRegistry` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfaceRegistry/surfaceRegistry.H` | Wraps an objectRegistry with a local instance directory for surfaces. |  |

### surfMesh fields  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `surfFields / surfPointFields` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfFields/surfFields/surfFields.H` | DimensionedField typedefs (surfScalarField, surfVectorField, ... and the point equivalents) for surfMesh. |  |
| `surfGeoMesh / surfPointGeoMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/surfMesh/surfFields/surfFields/surfGeoMesh.H` | GeoMesh wrappers sizing fields by surfMesh nFaces and nPoints respectively. |  |

### topoSet  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cellSet` | `yes - `cellSet` (topoSet word/size/set tables; topoSetDict `type cellSet;`)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/cellSet.H` | A collection of cell labels, stored under constant/polyMesh/sets. |  |
| `cellZoneSet` | `yes - `cellZoneSet`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/cellZoneSet.H` | Like cellSet but reads from and updates a cellZone on write. |  |
| `faceSet` | `yes - `faceSet`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/faceSet.H` | A list of face labels. |  |
| `faceZoneSet` | `yes - `faceZoneSet`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/faceZoneSet.H` | Like faceSet but reads from and updates a faceZone (including its flipMap) on write. |  |
| `pointSet` | `yes - `pointSet`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/pointSet.H` | A set of point labels. |  |
| `pointZoneSet` | `yes - `pointZoneSet`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/pointZoneSet.H` | Like pointSet but reads from and updates a pointZone on write. |  |

### topoSet (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `topoSet` | `runtime-selectable base with three tables: word, size and set (declareRunTimeSelectionTable(autoPtr, topoSet, word\|size\|set, ...))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSets/topoSet.H` | General labelHashSet of mesh entities (points, cells, faces) with IO and mesh-update mapping. |  |

### topoSetSource (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `topoSetSource` | `runtime-selectable base, declareRunTimeSelectionTable(autoPtr, topoSetSource, word, (mesh, dict)); keyword written as `source` in topoSetDict` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/topoSetSource/topoSetSource.H` | Abstract base for anything that modifies a topoSet according to a setAction (new/add/delete/subset/clear/invert/remove). |  |

### topoSetSource: cell  <sub>(20)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `boxToCell` | `yes - `boxToCell` (entries: box / boxes)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/boxToCell/boxToCell.H` | Selects cells whose cell centres lie inside one or more axis-aligned boxes. | treeBoundBox::contains(C[celli]) over a list of boxes |
| `cellToCell` | `yes - `cellToCell` (entry: set)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/cellToCell/cellToCell.H` | Selects the cells contained in another named cellSet. |  |
| `cylinderAnnulusToCell` | `yes - `cylinderAnnulusToCell` (entries: point1, point2, outerRadius, innerRadius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/cylinderAnnulusToCell/cylinderAnnulusToCell.H` | Selects cells whose centres lie inside a cylindrical annulus. | innerRadius <= radial offset from the axis <= outerRadius, within the axial extent |
| `cylinderToCell` | `yes - `cylinderToCell` (entries: point1, point2, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/cylinderToCell/cylinderToCell.H` | Selects cells whose centres lie inside a finite cylinder. | 0 <= (C-p1)&a <= \|p2-p1\| and \|(C-p1) - ((C-p1)&a)a\| <= radius |
| `faceToCell` | `yes - `faceToCell` (entries: set, option neighbour\|owner\|any\|all)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/faceToCell/faceToCell.H` | Selects cells based on their use of the faces in a faceSet (neighbour/owner/any/all). |  |
| `faceZoneToCell` | `yes - `faceZoneToCell` (entries: zone, option master\|slave)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/faceZoneToCell/faceZoneToCell.H` | Selects the cells on a chosen side (master/slave) of a faceZone. | side chosen from the faceZone flipMap orientation |
| `hemisphereToCell` | `yes - `hemisphereToCell` (entries: centre, radius, axis)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/hemisphereToCell/hemisphereToCell.H` | Selects cells whose centres lie inside a hemisphere. | \|C - centre\| <= radius and (C - centre)&axis >= 0 |
| `labelToCell` | `yes - `labelToCell` (entry: value)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/labelToCell/labelToCell.H` | Selects cells from an explicitly given list of cell labels. |  |
| `nbrToCell` | `yes - `nbrToCell` (entry: neighbours)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/nbrToCell/nbrToCell.H` | Selects cells by their number of neighbouring cells (internal or coupled faces). | count of internal + coupled faces per cell <= minNbrs |
| `nearestToCell` | `yes - `nearestToCell` (entry: points)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/nearestToCell/nearestToCell.H` | Selects the cells whose centres are nearest to a supplied list of points. | for each probe point, argmin \|C_cell - p\| with a parallel reduction on the minimum distance |
| `patchDistanceToCell` | `yes - `patchDistanceToCell` (entries: patches, distance)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/patchDistanceToCell/patchDistanceToCell.H` | Selects cells whose wall distance to a set of patches is below a threshold. | FaceCellWave patchDistWave distance field compared against the given distance |
| `pointToCell` | `yes - `pointToCell` (entries: set, option any\|all\|edge)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/pointToCell/pointToCell.H` | Selects cells that use points from a pointSet (any or all of the cell points). |  |
| `regionToCell` | `yes - `regionToCell` (entries: insidePoints, set, nErode)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/regionToCell/regionToCell.H` | Selects the topologically connected cell region(s) containing given seed points. | regionSplit cell-face-cell flood fill, optionally after nErode layers of erosion to expose holes |
| `rotatedBoxToCell` | `yes - `rotatedBoxToCell` (entries: origin,i,j,k or box,centre,n1,n2)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/rotatedBoxToCell/rotatedBoxToCell.H` | Selects cells whose centres lie inside a rotated and/or skewed parallelepiped. | local coordinates from origin and edge vectors i,j,k, all in [0,1]; alternative form rotates a bounding box by the rotation taking n1 onto n2 |
| `shapeToCell` | `yes - `shapeToCell` (entry: type, e.g. hex, prism, tet, splitHex)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/shapeToCell/shapeToCell.H` | Selects cells by cell shape recognised by the cellModeller, plus the special splitHex shape. | cellFeatures superface detection with a 10 degree feature angle for splitHex |
| `sphereToCell` | `yes - `sphereToCell` (entries: centre, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/sphereToCell/sphereToCell.H` | Selects cells whose centres lie inside a sphere. | \|C - centre\| <= radius |
| `surfaceToCell` | `yes - `surfaceToCell` (entries: file, outsidePoints, includeCut, includeInside, includeOutside, useSurfaceOrientation, nearDistance, curvature)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/surfaceToCell/surfaceToCell.H` | Selects cells by their relation to a triSurface: inside/outside/cut, near the surface, or at points of high curvature. | triSurfaceSearch octree; inside test by nearest-triangle normal or ray count; curvature test compares surface normals at cell centre and cell corners against curvature |
| `targetVolumeToCell` | `yes - `targetVolumeToCell` (entries: volume, normal, set)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/targetVolumeToCell/targetVolumeToCell.H` | Selects cells on one side of a plane, sweeping the plane until the selected volume reaches a target. | bisection on the plane offset so that sum(V_cell \| (C-p0)&n < 0) = targetVolume |
| `truncatedConeToCell` | `yes - `truncatedConeToCell` (entries: point1, point2, radius1, radius2, innerRadius1/2)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/truncatedConeToCell/truncatedConeToCell.H` | Selects cells whose centres lie inside a truncated cone (frustum). | radial offset <= linearly interpolated radius r1 + (r2-r1)*axialFraction |
| `zoneToCell` | `yes - `zoneToCell` (entry: zone)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellSources/zoneToCell/zoneToCell.H` | Selects cells belonging to a named cellZone (wildcards allowed). |  |

### topoSetSource: face  <sub>(13)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `boundaryToFace` | `yes - `boundaryToFace`` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/boundaryToFace/boundaryToFace.H` | Selects all external (boundary) faces of the mesh. |  |
| `boxToFace` | `yes - `boxToFace` (entries: box / boxes)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/boxToFace/boxToFace.H` | Selects faces whose face centres lie inside one or more boxes. | treeBoundBox::contains(Cf[facei]) |
| `cellToFace` | `yes - `cellToFace` (entries: set, option all\|both)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/cellToFace/cellToFace.H` | Builds a faceSet from a cellSet - all faces of the cells, or only the faces on the set boundary. | `all` takes every cell face; `both` keeps faces whose owner and neighbour are both in the set (coupled faces resolved across processors) |
| `cylinderAnnulusToFace` | `yes - `cylinderAnnulusToFace` (entries: point1, point2, outerRadius, innerRadius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/cylinderAnnulusToFace/cylinderAnnulusToFace.H` | Selects faces whose centres lie inside a cylindrical annulus. | innerRadius <= radial offset <= outerRadius within the axial extent |
| `cylinderToFace` | `yes - `cylinderToFace` (entries: point1, point2, radius)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/cylinderToFace/cylinderToFace.H` | Selects faces whose centres lie inside a finite cylinder. | axial and radial bounds on Cf relative to the axis point1->point2 |
| `faceToFace` | `yes - `faceToFace` (entry: set)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/faceToFace/faceToFace.H` | Selects faces contained in another named faceSet. |  |
| `labelToFace` | `yes - `labelToFace` (entry: value)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/labelToFace/labelToFace.H` | Selects faces from an explicitly given list of face labels. |  |
| `normalToFace` | `yes - `normalToFace` (entries: normal, cos)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/normalToFace/normalToFace.H` | Selects faces whose unit normal is within a tolerance of a given direction. | \|n_face & n_given\| > 1 - tol (cosine test on the normalised face area vector) |
| `patchToFace` | `yes - `patchToFace` (entry: patch / patches)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/patchToFace/patchToFace.H` | Selects all faces belonging to named patches (wildcards allowed). |  |
| `pointToFace` | `yes - `pointToFace` (entries: set, option any\|all\|edge)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/pointToFace/pointToFace.H` | Selects faces that use points from a pointSet (any, all, or edge-wise). |  |
| `regionToFace` | `yes - `regionToFace` (entries: set, nearPoint)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/regionToFace/regionToFace.H` | Selects the topologically connected face region containing a given point. | PatchEdgeFaceWave region propagation over the patch of candidate faces, seeded at the face nearest the given point |
| `rotatedBoxToFace` | `yes - `rotatedBoxToFace` (entries: origin,i,j,k or box,centre,n1,n2)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/rotatedBoxToFace/rotatedBoxToFace.H` | Selects faces whose centres lie inside a rotated and/or skewed parallelepiped. | local box coordinates from origin and i,j,k edge vectors, all in [0,1] |
| `zoneToFace` | `yes - `zoneToFace` (entry: zone)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceSources/zoneToFace/zoneToFace.H` | Selects faces belonging to a named faceZone. |  |

### topoSetSource: point  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `boxToPoint` | `yes - `boxToPoint` (entries: box / boxes)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/boxToPoint/boxToPoint.H` | Selects mesh points inside one or more boxes. | treeBoundBox::contains(p) |
| `cellToPoint` | `yes - `cellToPoint` (entries: set, option all)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/cellToPoint/cellToPoint.H` | Selects points used by the cells of a cellSet. |  |
| `faceToPoint` | `yes - `faceToPoint` (entries: set, option all)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/faceToPoint/faceToPoint.H` | Selects points used by the faces of a faceSet. |  |
| `labelToPoint` | `yes - `labelToPoint` (entry: value)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/labelToPoint/labelToPoint.H` | Selects points from an explicitly given list of point labels. |  |
| `nearestToPoint` | `yes - `nearestToPoint` (entry: points)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/nearestToPoint/nearestToPoint.H` | Selects the mesh points nearest to a supplied list of probe points. | for each probe point, argmin \|p_mesh - p_probe\| with parallel min reduction |
| `pointToPoint` | `yes - `pointToPoint` (entry: set)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/pointToPoint/pointToPoint.H` | Selects points contained in another named pointSet. |  |
| `surfaceToPoint` | `yes - `surfaceToPoint` (entries: file, nearDistance, includeInside, includeOutside)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/surfaceToPoint/surfaceToPoint.H` | Selects points by distance to, or inside/outside status relative to, a triSurface. | octree nearest distance <= nearDistance; inside/outside from the normal of the nearest surface triangle |
| `zoneToPoint` | `yes - `zoneToPoint` (entry: zone)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointSources/zoneToPoint/zoneToPoint.H` | Selects points belonging to a named pointZone. |  |

### topoSetSource: zone  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `faceZoneToFaceZone` | `yes - `faceZoneToFaceZone` (entry: zone)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceZoneSources/faceZoneToFaceZone/faceZoneToFaceZone.H` | Copies faces and flipMap from an existing faceZone into another faceZoneSet. |  |
| `planeToFaceZone` | `yes - `planeToFaceZone` (entries: point, normal, include all\|closest)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceZoneSources/planeToFaceZone/planeToFaceZone.H` | Selects internal faces whose adjacent cell centres straddle a given plane, optionally keeping only the region closest to the plane point. | sign change of (C_own - p0)&n versus (C_nei - p0)&n; `closest` restricts to the contiguous region containing the plane point |
| `searchableSurfaceToFaceZone` | `yes - `searchableSurfaceToFaceZone` (entry: surface, plus that surface's own dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceZoneSources/searchableSurfaceToFaceZone/searchableSurfaceToFaceZone.H` | Selects faces whose owner-to-neighbour cell-centre vector pierces a searchable surface. | findLine(C_own, C_nei) hit on the surface; flipMap from the sign of Sf & surface normal |
| `setAndNormalToFaceZone` | `yes - `setAndNormalToFaceZone` (entries: faceSet, normal)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceZoneSources/setAndNormalToFaceZone/setAndNormalToFaceZone.H` | Populates a faceZoneSet from a faceSet, orienting the faces with a supplied normal vector. | flipMap[i] = ((Sf[i] & normal) < 0) |
| `setToCellZone` | `yes - `setToCellZone` (entry: set)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/cellZoneSources/setToCellZone/setToCellZone.H` | Populates a cellZoneSet from a named cellSet. |  |
| `setToFaceZone` | `yes - `setToFaceZone` (entry: faceSet)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceZoneSources/setToFaceZone/setToFaceZone.H` | Populates a faceZoneSet from a faceSet, setting the flipMap entries to true. |  |
| `setToPointZone` | `yes - `setToPointZone` (entry: set)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/pointZoneSources/setToPointZone/setToPointZone.H` | Populates a pointZoneSet from a named pointSet. |  |
| `setsToFaceZone` | `yes - `setsToFaceZone` (entries: faceSet, cellSet, flip)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/sets/faceZoneSources/setsToFaceZone/setsToFaceZone.H` | Populates a faceZoneSet from a faceSet, orienting each face using an accompanying cellSet. | flipMap set so that the zone normal points out of the cellSet cells |

### triSurface  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `geometricSurfacePatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/triSurface/geometricSurfacePatch/geometricSurfacePatch.H` | Surface equivalent of patchIdentifier - holds geometric type, name and index. |  |
| `labelledTri` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/tools/labelledTri/labelledTri.H` | triFace carrying an additional region (patch) number. |  |
| `meshTriangulation` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/meshTriangulation/meshTriangulation.H` | Triangulates polyMesh faces into a (possibly multiply connected) triSurface, keeping patchIDs as regions. | face triangulation about the face centre or by fan decomposition; internal faces get the supplied region number |
| `sortLabelledTri` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/tools/labelledTri/sortLabelledTri.H` | Builds the mapping that sorts triSurface faces by region number. | stable sort of face indices by region key |
| `surfacePatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/triSurface/surfacePatch/surfacePatch.H` | A contiguous patch on a triSurface - geometricSurfacePatch plus size and start index. |  |
| `surfacePatchIOList` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/triSurface/surfacePatch/surfacePatchIOList.H` | IOobject wrapper for a list of surfacePatches. |  |
| `triSurface` | `no (hard-coded extension dispatch inside triSurface::read/write, not a runtime selection table)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/triSurface/triSurface.H` | Triangulated surface (PrimitivePatch of labelledTri) with patch information and its own file readers and writers. | read dispatch by extension: ftr, stl, stlb, gts, obj, off, tri, ac, nas, vtk (and .gz); write dispatch: ftr, stl, stlb, gts, obj, off, vtk, tri, ac, smesh |

### triSurface algorithm  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `orientedSurface` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/orientedSurface/orientedSurface.H` | Flips triangles so that all normals point consistently, given an outside point. | ray count from the outside point fixes one triangle's orientation, then edge-connected flood fill propagates consistent winding |
| `surfaceFeatures` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/surfaceFeatures/surfaceFeatures.H` | Extracts and stores feature edges and feature points of a triSurface, sorted into region/external/internal bands. | included angle between the two adjacent triangle planes compared with the specified angle (180 deg = coplanar, 90 deg = right angle; use 91 deg to pick up straight edges only) |
| `surfaceLocation` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/surfaceLocation/surfaceLocation.H` | pointIndexHit extended with the element type (triangle/edge/point) and the last known triangle, for walking over a surface. | elementType from triPointRef::proxType |
| `triSurfaceTools` | `no (static function collection)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/triSurfaceTools/triSurfaceTools.H` | Large static collection of triSurface operations - edge collapse/split, greenRefine, redGreen refinement, curvature, surfaceSide, tracking along the surface. | red-green triangle refinement; vertex normal by area/angle weighting; curvature from the normal variation between adjacent triangles; surfaceSide by nearest-element classification (triangle/edge/point) |
| `triangleFuncs` | `no (static function collection)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/triangleFuncs/triangleFuncs.H` | Low-level triangle intersection and bounding-box utilities. | triangle-line intersection by barycentric solve; triangle-triangle intersection segment; triangle vs treeBoundBox overlap (separating-axis style) |

### triSurface fields  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `triSurfaceFields` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/triSurfaceFields/triSurfaceFields.H` | DimensionedField typedefs (triSurfaceScalarField, triSurfacePointVectorField, ...) for triSurfaces. |  |
| `triSurfaceGeoMesh / triSurfacePointGeoMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/triSurfaceFields/triSurfaceGeoMesh.H` | GeoMesh wrappers sizing fields by triSurface triangle count and point count respectively. |  |

### triSurface search  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `triSurfaceRegionSearch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/triSurfaceSearch/triSurfaceRegionSearch.H` | Builds one octree per surface region so searches can be restricted to selected regions. | per-region indexedOctree<treeDataTriSurface> |
| `triSurfaceSearch` | `no (constructible from a dictionary with tolerance and maxTreeDepth)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/meshTools/triSurface/triSurfaceSearch/triSurfaceSearch.H` | Demand-driven octree helper for nearest-point, inside/outside and line-intersection searches on a triSurface. | indexedOctree<treeDataTriSurface>; inside/outside by nearest-triangle normal sign; findLineAll returns sorted pierces |

### triSurface tools  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `hashSignedLabel` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/tools/hashSignedLabel/hashSignedLabel.H` | Hash functor for signed labels, since Hash<label> assumes unsigned. | h(key) = mag(key) % tableSize |
| `labelPairLookup` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/triSurface/tools/labelPair/labelPairLookup.H` | HashTable from an ordered label pair to a label, e.g. (face1,face2) -> shared edge. | keyed on FixedList<label,2> so hashing is non-commutative, unlike edge |

---

## Mesh: Mesh motion, run-time topology change, decomposition

> **Subsystem notes**
>
> Coverage and cross-cutting observations.
> RUNTIME-SELECTION TABLES FOUND (base -> dictionary entry the user writes): - motionSolver (dictionary) -> dynamicMeshDict "motionSolver"; 6 entries in libmotionSolvers (displacementLayeredMotion, displacementLinearMotion, solidBody, interpolatingSolidBody, multiSolidBodyMotionSolver, motionSolverList), 5 in libfvMotionSolvers (displacementLaplacian, velocityLaplacian, displacementComponentLaplacian, velocityComponentLaplacian, displacementSBRStress), 2 in librigidBodyMeshMotion (rigidBodyMotion, rigidBodyMotionSolver) and 1 in libsixDoFRigidBodyMotion (sixDoFRigidBodyMotion).
> - solidBodyMotionFunction (dictionary) -> 9 functions. - motionDiffusivity (Istream, not dictionary) -> the "diffusivity" entry is parsed as a stream, so the syntax is e.g. `diffusivity inverseDistance (movingWall);` or `diffusivity quadratic inverseDistance (movingWall);` for the manipulators.
> 10 concrete models.
> - fvMeshMover / fvMeshTopoChanger / fvMeshStitcher / fvMeshDistributor: the abstract bases and their none/list/stationary implementations live in src/finiteVolume/fvMesh/..., not in the src/fvMesh* directories I was asked to read; I included them because they define the keywords and complete the picture.
> The plug-in libraries are src/fvMeshMovers (4), src/fvMeshTopoChangers (2), src/fvMeshStitchers (1), src/fvMeshDistributors (2). - RBD::rigidBody (5 selectable bodies + 2 internal merge bodies), RBD::joint (20 joints), RBD::restraint (5), RBD::rigidBodySolver (3).
> - sixDoFRigidBodyMotionConstraint (5), sixDoFRigidBodyMotionRestraint (6), sixDoFSolver (3). - renumberMethod (6 in librenumberMethods + Sloan in a separate library). - decompositionMethod has TWO tables: "decomposer" (initial decomposition, decomposePar) and "distributor" (run-time redistribution).
> simple, hierarchical, manual, multiLevel, none, random, structured and scotch register in BOTH; metis registers only as a decomposer; parMetis, ptscotch and zoltan register only as distributors. This decomposer/distributor split is the main thing a user gets wrong in decomposeParDict.
> - decompositionConstraint (5 named constraints).
> NOT RUNTIME-SELECTABLE but load-bearing: pointEdgeStructuredWalk, pistonPointEdgeData, meshPhiPreCorrectInfo/meshPhiCorrectInfo (all FaceCellWave/PointEdgeWave transported data types), OppositeFaceCellWave, dynamicMeshPointInterpolator, rigidBodyInertia, rigidBodyModel/forwardDynamics, the domainDecomposition family and pairPatchAgglomeration.
> DUPLICATION IN UPSTREAM: there are two parallel implementations of 6-DoF rigid-body dynamics under src/rigidBodyMotion — the Featherstone multi-body librigidBodyDynamics (RBD:: namespace, joints/bodies/restraints, rigidBodySolvers) and the older single-body libsixDoFRigidBodyMotion (sixDoFRigidBodyMotion with its own constraints/restraints/sixDoFSolvers).
> Restraint names collide across the two (linearSpring, linearDamper, linearAxialAngularSpring, sphericalAngularDamper) and so do the solver names (symplectic, Newmark, CrankNicolson); which set applies depends on whether the motion solver is rigidBodyMotion/rigidBodyMotionSolver or sixDoFRigidBodyMotion.
> axialAngularSpring and sphericalAngularSpring exist only in the sixDoF set; externalForce only in the RBD set.
> SOURCES READ: every .H under src/motionSolvers, src/fvMotionSolver, src/fvMeshMovers, src/fvMeshTopoChangers, src/fvMeshStitchers, src/fvMeshDistributors, src/rigidBodyMotion, src/renumber, src/parallel, src/fvAgglomerationMethods, plus the .C files where the equation was only visible in the implementation (all motionDiffusivity operator()(), the fvMotionSolver solve() equations, RBD restraint restrain() functions, crankConnectingRodMotionI.H) and all Make/files for library names and completeness.
> src/parallel contains no "reconstruct" directory — reconstruction lives in src/parallel/parallel/domainDecompositionReconstruct.C.

### 6-DoF constraint  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFRigidBodyMotionConstraints::axis` | `type axis;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/constraints/axis/sixDoFRigidBodyMotionAxisConstraint.H` | Restricts rotation to a single fixed axis. | pi := (pi . a) a with a the allowed rotation axis |
| `sixDoFRigidBodyMotionConstraints::line` | `type line;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/constraints/line/sixDoFRigidBodyMotionLineConstraint.H` | Restricts translation of the centre of rotation to a line. | v := (v . d) d with d the line direction |
| `sixDoFRigidBodyMotionConstraints::orientation` | `type orientation;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/constraints/orientation/sixDoFRigidBodyMotionOrientationConstraint.H` | Fixes the body orientation in global space (no rotation at all). | pi := 0, Q := const |
| `sixDoFRigidBodyMotionConstraints::plane` | `type plane;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/constraints/plane/sixDoFRigidBodyMotionPlaneConstraint.H` | Restricts translation of the centre of rotation to a plane. | v := v - (v . n) n with n the plane normal |
| `sixDoFRigidBodyMotionConstraints::point` | `type point;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/constraints/point/sixDoFRigidBodyMotionPointConstraint.H` | Fixes the centre of rotation in space (all three translations removed). | v := 0 for the centre of rotation |

### 6-DoF constraint (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFRigidBodyMotionConstraint` | `base of RTS table (constraints sub-dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/constraints/sixDoFRigidBodyMotionConstraint/sixDoFRigidBodyMotionConstraint.H` | Abstract base for kinematic constraints removing DoF from a 6-DoF body. | declareRunTimeSelectionTable(autoPtr, sixDoFRigidBodyMotionConstraint, dictionary); setCentreOfRotation/constrainTranslation/constrainRotation project out the constrained directions |

### 6-DoF core  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFRigidBodyMotion` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion.H` | Standalone six-degree-of-freedom motion of a single rigid body with restraints and constraints. | m a = sum F; body-frame angular momentum pi with dpi/dt = tau, orientation advanced by successive rotations about the principal axes |
| `sixDoFRigidBodyMotionState` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotionState.H` | Compact motion state (centre of rotation, orientation Q, velocity, angular momentum) kept identical on all processors. |  |

### 6-DoF restraint  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFRigidBodyMotionRestraints::axialAngularSpring` | `type axialAngularSpring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/axialAngularSpring/axialAngularSpring.H` | Torsional spring about an axis whose moment is a general Function1 of angle, plus linear damping. | M = -(moment(theta) + damping*(axis . omega)) axis |
| `sixDoFRigidBodyMotionRestraints::linearAxialAngularSpring` | `type linearAxialAngularSpring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/linearAxialAngularSpring/linearAxialAngularSpring.H` | Linear torsional spring about a fixed axis with axial angular damping. | M = -(stiffness*theta + damping*(axis . omega)) axis, theta measured from refQ about axis |
| `sixDoFRigidBodyMotionRestraints::linearDamper` | `type linearDamper;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/linearDamper/linearDamper.H` | Damping force proportional to the body's linear velocity. | F = -coeff * v |
| `sixDoFRigidBodyMotionRestraints::linearSpring` | `type linearSpring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/linearSpring/linearSpring.H` | Linear spring between a fixed anchor and a body attachment point, with damping. | F = (-stiffness(\|r\| - restLength) - damping (r_hat . v)) r_hat |
| `sixDoFRigidBodyMotionRestraints::sphericalAngularDamper` | `type sphericalAngularDamper;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/sphericalAngularDamper/sphericalAngularDamper.H` | Isotropic angular damper acting on all rotational DoF. | M = -coeff * omega |
| `sixDoFRigidBodyMotionRestraints::sphericalAngularSpring` | `type sphericalAngularSpring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/sphericalAngularSpring/sphericalAngularSpring.H` | Isotropic torsional spring restoring the body towards a reference orientation. | M = -stiffness * rotation-vector(Q refQ^T) - damping * omega |

### 6-DoF restraint (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFRigidBodyMotionRestraint` | `base of RTS table (restraints sub-dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotion/restraints/sixDoFRigidBodyMotionRestraint/sixDoFRigidBodyMotionRestraint.H` | Abstract base for force/moment restraints on a 6-DoF body. | declareRunTimeSelectionTable(autoPtr, sixDoFRigidBodyMotionRestraint, dictionary); restrain() returns restraintPosition, restraintForce, restraintMoment |

### 6-DoF time integrator  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFSolvers::CrankNicolson` | `solver { type CrankNicolson; aoc 0.5; voc 0.5; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFSolvers/CrankNicolson/CrankNicolson.H` | Implicit off-centred Crank-Nicolson 2nd-order integrator for 6-DoF motion. | acceleration off-centring aoc and velocity off-centring voc, both defaulting to 0.5 |
| `sixDoFSolvers::Newmark` | `solver { type Newmark; gamma 0.5; beta 0.25; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFSolvers/Newmark/Newmark.H` | Implicit Newmark 2nd-order integrator for 6-DoF motion, usable with outer correctors. | v^{n+1} = v^n + dt((1-gamma)a^n + gamma a^{n+1}); x^{n+1} = x^n + dt v^n + dt^2((0.5-beta)a^n + beta a^{n+1}) |
| `sixDoFSolvers::symplectic` | `solver { type symplectic; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFSolvers/symplectic/symplectic.H` | Explicit 2nd-order symplectic integrator for 6-DoF motion; single call per time step. | velocity/angular-momentum half steps around a full position/orientation step (Dullweber splitting with successive principal-axis rotations) |

### 6-DoF time integrator (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `sixDoFSolver` | `base of RTS table 'sixDoFSolver' ('solver' sub-dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFSolvers/sixDoFSolver/sixDoFSolver.H` | Abstract base for the time integrators of a sixDoFRigidBodyMotion. | declareRunTimeSelectionTable(autoPtr, sixDoFSolver, dictionary) |

### Decomposition (abstract)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `decompositionMethods::geometric` | `no (base of simple and hierarchical)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/geometric/geometric.H` | Base for purely geometrical decompositions that work on cell-centre coordinates only. | reads the n = (nx ny nz) subdivision and an optional coordinate transformation/delta |

### Decomposition (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `decompositionMethod` | `base of decomposeParDict 'decomposer' and 'distributor' entries` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionMethod/decompositionMethod.H` | Abstract base for domain decomposition, with two separate RTS tables for initial decomposition and for run-time redistribution. | declareRunTimeSelectionTable(..., decomposer) and (..., distributor); decompose() returns the cell-to-processor labelList |

### Decomposition constraint  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `decompositionConstraints::preserveBafflesConstraint` | `type preserveBaffles;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionConstraints/preserveBaffles/preserveBafflesConstraint.H` | Detects coincident (baffle) face pairs and keeps their owners on the same processor. | geometric detection of duplicate faces, then connectivity added between the two owners |
| `decompositionConstraints::preserveFaceZonesConstraint` | `type preserveFaceZones;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionConstraints/preserveFaceZones/preserveFaceZonesConstraint.H` | Keeps owner and neighbour cells of the named faceZones on the same processor. |  |
| `decompositionConstraints::preservePatchesConstraint` | `type preservePatches;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionConstraints/preservePatches/preservePatchesConstraint.H` | Keeps owner and neighbour of a (cyclic) patch on the same processor. | cyclic patch face pairs added as explicit connections in the decomposition graph |
| `decompositionConstraints::singleProcessorFaceSetsConstraint` | `type singleProcessorFaceSets; ((faceSetName procNo) pairs)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionConstraints/singleProcessorFaceSets/singleProcessorFaceSetsConstraint.H` | Forces all cells connected to a faceSet (by face or point) onto one named processor. |  |
| `refinementHistoryConstraint` | `type refinementHistory;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionConstraints/refinementHistory/refinementHistoryConstraint.H` | Keeps all cells descended from the same originally refined cell on one processor, reading polyMesh/refinementHistory. | octree sibling groups from the hexRef8 refinement history become single decomposition units |

### Decomposition constraint (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `decompositionConstraint` | `base of the decomposeParDict 'constraints' RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/decompositionConstraints/decompositionConstraint/decompositionConstraint.H` | Abstract base for constraints that force sets of cells onto the same processor. | declareRunTimeSelectionTable(autoPtr, decompositionConstraint, dictionary); add() supplies blocked faces and specified processor groups, apply() enforces them after decomposition |

### Decomposition method  <sub>(12)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `decompositionMethods::hierarchical` | `decomposer hierarchical; (n (nx ny nz), order xyz)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/hierarchical/hierarchical.H` | Geometric decomposition performing the directional splits in a user-specified nested order. | recursive equal-size binning in the given order (e.g. xyz): each x-bin independently split in y, each of those in z; finalDecomp[i,j,k] = i*n0*n1 + j*n1 + k |
| `decompositionMethods::manual` | `decomposer manual; (dataFile entry)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/manual/manual.H` | Decomposition read from a user-supplied cell-to-processor file. |  |
| `decompositionMethods::metis` | `decomposer metis; (libmetisDecomp.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/metis/metis.H` | Serial Metis graph partitioning of the cell adjacency graph. | multilevel k-way / recursive-bisection graph partitioning minimising edge cut subject to balance; optional cell weights |
| `decompositionMethods::multiLevel` | `decomposer multiLevel; (per-level sub-dictionaries)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/multiLevel/multiLevel.H` | Applies several decomposition methods consecutively, e.g. across nodes then within a node. | level_i decomposes each domain produced by level_{i-1}; total nProcs = product of the per-level counts |
| `decompositionMethods::none` | `decomposer none; / distributor none;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/none/none.H` | Dummy decomposition that leaves everything on the current processor. |  |
| `decompositionMethods::parMetis` | `distributor parMetis; (libparMetisDecomp.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/parMetis/parMetis.H` | Parallel Metis redistribution (parallel-only, no serial mode). | method kWay \| geomKway \| adaptiveRepart; itr sets the ratio of communication to redistribution cost (default 1000) |
| `decompositionMethods::ptscotch` | `distributor ptscotch; (libptscotchDecomp.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/ptscotch/ptscotch.H` | Fully distributed PTScotch graph partitioning for parallel redistribution. | distributed multilevel graph partitioning; writeGraph=true dumps .dgr files for dgpart |
| `decompositionMethods::random` | `decomposer random; / distributor random;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/random/random.H` | Random assignment of cells to processors; for testing only. | proc = uniform random in [0, nProcs) |
| `decompositionMethods::scotch` | `decomposer scotch; / distributor scotch; (libscotchDecomp.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/scotch/scotch.H` | Scotch graph partitioning; collects the whole graph on the master when run in parallel. | user-definable mapping strategy string (recursive bipartitioning 'b' with Fiduccia-Mattheyses 'f' separators, multilevel framework); optional processorWeights and cell weights |
| `decompositionMethods::simple` | `decomposer simple; (n (nx ny nz))` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/simple/simple.H` | Geometric decomposition into nx*ny*nz equal-count slabs, split independently in each direction. | sort by x into nx equal bins, then y into ny, then z into nz; proc = i*ny*nz + j*nz + k |
| `decompositionMethods::structured` | `decomposer structured; (patches, nested method)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/decompositionMethods/structured/structured.H` | Decomposes named patch cells with a nested method and extrudes that decomposition along mesh layers. | nested method decomposes the patch-adjacent cell layer; FaceCellWave walk assigns every cell in a column the processor of its patch cell |
| `decompositionMethods::zoltan` | `distributor zoltan; (libzoltanDecomp.so, zoltanCoeffs)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/decompose/zoltan/zoltan.H` | Zoltan parallel (re)partitioning with a wide choice of geometric and graph/hypergraph algorithms. | lb_method block \| random \| rcb \| rib \| hsfc \| reftree \| graph \| hypergraph; lb_approach partition \| repartition \| refine; defaults graph/repartition, imbalance_tol 1.05 |

### Distributed geometry  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `distributedTriSurfaceMesh` | `type distributedTriSurfaceMesh; (libdistributed.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/distributed/distributedTriSurfaceMesh/distributedTriSurfaceMesh.H` | searchableSurface holding a triSurface split across processors, with queries routed to the owning processor. | distribution modes follow (each processor holds all triangles in its mesh bounding box), independent (split by triangle centres), frozen (no change); queries mapped by bounding-box overlap and answers returned |

### FV boundary condition  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cellMotionFvPatchField` | `type cellMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvPatchFields/derived/cellMotion/cellMotionFvPatchField.H` | Sets the cell-centre motion field on a patch from the corresponding point motion field (scalar and vector variants). | patch face value = area-weighted average of the point field values around the face |
| `surfaceSlipDisplacementFvPatchField` | `type surfaceSlipDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvPatchFields/derived/surfaceSlipDisplacement/surfaceSlipDisplacementFvPatchField.H` | Slip-type cell-displacement counterpart of the surfaceSlipDisplacement point BC (projection is done on the points). | zero normal gradient / slip; no projection performed here |

### FV motion solver  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `displacementComponentLaplacianFvMotionSolver` | `motionSolver displacementComponentLaplacian;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvMotionSolvers/componentDisplacement/componentLaplacian/displacementComponentLaplacianFvMotionSolver.H` | Laplacian mesh-displacement solve for one Cartesian component only (2D/extruded meshes). | fvm::laplacian(gamma, cellDisplacement_cmpt) = 0 for the named component x\|y\|z |
| `displacementLaplacianFvMotionSolver` | `motionSolver displacementLaplacian;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/displacementLaplacianFvMotionSolver.H` | Solves a cell-centre Laplacian for the mesh displacement, then interpolates to points. | fvm::laplacian(gamma, cellDisplacement) = 0 with gamma from a run-time selected motionDiffusivity; points x = points0 + interp(cellDisplacement) |
| `displacementSBRStressFvMotionSolver` | `motionSolver displacementSBRStress;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvMotionSolvers/displacement/SBRStress/displacementSBRStressFvMotionSolver.H` | Solves the solid-body-rotation stress (pseudo-elasticity) equations for mesh displacement, better preserving cell shape. | fvm::laplacian(2*gamma, D) + fvc::div(gamma*(gradD.T() - I*tr(gradD))) - fvc::laplacian(gamma, D) = 0, iterated with explicit non-orthogonal/deviatoric terms |
| `velocityComponentLaplacianFvMotionSolver` | `motionSolver velocityComponentLaplacian;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvMotionSolvers/componentVelocity/componentLaplacian/velocityComponentLaplacianFvMotionSolver.H` | Laplacian mesh-velocity solve for one Cartesian component only. | fvm::laplacian(gamma, cellMotionU_cmpt) = 0 |
| `velocityLaplacianFvMotionSolver` | `motionSolver velocityLaplacian;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvMotionSolvers/velocity/laplacian/velocityLaplacianFvMotionSolver.H` | Solves a cell-centre Laplacian for the mesh point velocity. | fvm::laplacian(gamma, cellMotionU) = 0; x^{n+1} = x^n + interp(cellMotionU)*deltaT |

### FV motion solver framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMotionSolver` | `no (ClassName displacementMotionSolver for debug switch)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/fvMotionSolvers/fvMotionSolver/fvMotionSolver.H` | Base helper for fvMesh-based motion solvers; interpolates cell-centre motion to points and sets up the fv solution. | point value = inverse-distance-weighted average of surrounding cell-centre values, with boundary patch values applied directly |

### Function object  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `functionObjects::meshToMeshAdjustTimeStepFunctionObject` | `type meshToMeshAdjustTimeStep;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshTopoChangers/meshToMesh/meshToMeshAdjustTimeStep/meshToMeshAdjustTimeStepFunctionObject.H` | Clips the time step so that the solver lands exactly on the meshToMesh mapping times. | deltaT = min(deltaT, t_map - t) |
| `functionObjects::multiValveEngineState` | `type multiValveEngineState;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/multiValveEngineState/multiValveEngineState.H` | Writes the position and speed of the piston and each valve of a multiValveEngine mover. |  |
| `functionObjects::rigidBodyForces` | `type rigidBodyForces; (librigidBodyForces.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyForces/rigidBodyForces.H` | Integrates pressure and viscous forces/moments over patches of a moving rigid body about its centre of rotation. | F = sum_f (Sf p + Sf.tau); M = sum_f (x_f - CofR) ^ F_f, with optional binning |
| `functionObjects::rigidBodyPoints` | `type rigidBodyPoints; (librigidBodyState.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyState/rigidBodyPoints/rigidBodyPoints.H` | Writes position, linear/angular velocity and acceleration of named body-local points. | x = X0(body) & p_local; v = v_body + omega ^ r; a includes the omega ^ (omega ^ r) centripetal term |
| `functionObjects::rigidBodyState` | `type rigidBodyState; (librigidBodyState.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyState/rigidBodyState/rigidBodyState.H` | Writes the position, orientation and velocities of every body of a rigidBodyMotion, in selectable angle units. |  |
| `functionObjects::sixDoFRigidBodyControl` | `type sixDoFRigidBodyControl; (libsixDoFRigidBodyState.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyState/sixDoFRigidBodyControl/sixDoFRigidBodyControl.H` | Stops the run once window-averaged body linear and angular velocities fall below convergence criteria. | running mean of v and omega over 'window'; stop when both are below convergedVelocity / convergedAngularVelocity |
| `functionObjects::sixDoFRigidBodyState` | `type sixDoFRigidBodyState; (libsixDoFRigidBodyState.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyState/sixDoFRigidBodyState/sixDoFRigidBodyState.H` | Writes the 6-DoF motion state (position, angles, velocities) in selectable units. |  |

### Function1 (engine motion)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Function1s::crankConnectingRodMotion` | `type crankConnectingRodMotion; (scalar Function1)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/crankConnectingRodMotion/crankConnectingRodMotion.H` | Standard crank-and-connecting-rod piston position as a function of crank angle in degrees. | x(theta) = (L + S/2) - (S/2)cos(theta) - sqrt(L^2 - (S sin(theta)/2)^2), L = conRodLength, S = stroke |

### GAMG agglomeration  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MGridGenGAMGAgglomeration` | `agglomerator MGridGen; (libMGridGenGAMGAgglomeration.so, in fvSolution GAMG)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvAgglomerationMethods/MGridGenGamgAgglomeration/MGridGenGAMGAgglomeration.H` | GAMG coarse-level agglomeration using the external MGridGen library on the cell volume/face-area graph. | MGridGen coarsening driven by cell volumes and face areas, targeting nCellsInCoarsestLevel with min/max coarse cell size limits |

### Joint  <sub>(20)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::joints::Pa` | `type Pa; (with 'axis')` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Pa/Pa.H` | 1-DoF prismatic joint translating along an arbitrary specified axis. | S = (0 \| axis); X_J = translation(q*axis) |
| `RBD::joints::Px` | `type Px;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Px/Px.H` | 1-DoF prismatic joint translating along the x-axis. | S = (0 0 0 \| 1 0 0); X_J = translation(q ex) |
| `RBD::joints::Pxyz` | `type Pxyz;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Pxyz/Pxyz.H` | 3-DoF prismatic joint translating freely in x, y and z. | S spans the three linear directions |
| `RBD::joints::Py` | `type Py;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Py/Py.H` | 1-DoF prismatic joint translating along the y-axis. | S = (0 0 0 \| 0 1 0) |
| `RBD::joints::Pz` | `type Pz;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Pz/Pz.H` | 1-DoF prismatic joint translating along the z-axis. | S = (0 0 0 \| 0 0 1) |
| `RBD::joints::Ra` | `type Ra; (with 'axis')` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Ra/Ra.H` | 1-DoF revolute joint rotating about an arbitrary specified axis. | S = (axis \| 0); X_J = rotation(axis, q) |
| `RBD::joints::Rs` | `type Rs;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Rs/Rs.H` | 3-DoF spherical joint parameterised by a quaternion to avoid gimbal lock. | Euler-parameter (quaternion) state, S spans the three angular directions |
| `RBD::joints::Rx` | `type Rx;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Rx/Rx.H` | 1-DoF revolute joint rotating about the x-axis. | S = (1 0 0 \| 0 0 0); X_J = Rx(q) |
| `RBD::joints::Rxyz` | `type Rxyz;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Rxyz/Rxyz.H` | 3-DoF spherical joint using Euler angles applied in the order x, y, z. | X_J = Rz(q3) Ry(q2) Rx(q1) |
| `RBD::joints::Ry` | `type Ry;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Ry/Ry.H` | 1-DoF revolute joint rotating about the y-axis. | S = (0 1 0 \| 0 0 0); X_J = Ry(q) |
| `RBD::joints::Ryxz` | `type Ryxz;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Ryxz/Ryxz.H` | 3-DoF spherical joint using Euler angles applied in the order y, x, z. | X_J = Rz(q3) Rx(q2) Ry(q1) |
| `RBD::joints::Rz` | `type Rz;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Rz/Rz.H` | 1-DoF revolute joint rotating about the z-axis. | S = (0 0 1 \| 0 0 0); X_J = Rz(q) |
| `RBD::joints::Rzyx` | `type Rzyx;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/Rzyx/Rzyx.H` | 3-DoF spherical joint using Euler angles applied in the order z, y, x. | X_J = Rx(q3) Ry(q2) Rz(q1) |
| `RBD::joints::composite` | `type composite;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/composite/compositeJoint.H` | Joint formed by chaining several joints in series through massless bodies. | X_J = product of the component joint transforms; S is the block concatenation of component subspaces |
| `RBD::joints::floating` | `type floating;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/floating/floatingJoint.H` | Full 6-DoF free joint (three translations plus a spherical rotation) for a floating body. | composite of Pxyz and Rs; S = I6 |
| `RBD::joints::function` | `type function;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/function/function.H` | Joint whose position is a prescribed Function1 of the parent joint's position (e.g. cam/gear coupling). | q_child = f(q_parent) |
| `RBD::joints::functionDot` | `type functionDot;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/functionDot/functionDot.H` | Joint whose position is a prescribed function of the parent joint's velocity. | q_child = f(qDot_parent) |
| `RBD::joints::null` | `type null;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/null/nullJoint.H` | Zero-DoF joint used for the root body. | S = 0, X_J = I |
| `RBD::joints::rigid` | `type rigid;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/rigid/rigid.H` | Zero-DoF joint welding a body rigidly to its parent with a fixed transform. | X_J = const, S = 0 |
| `RBD::joints::rotating` | `type rotating;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/rotating/rotating.H` | Prescribed-motion joint rotating at a specified angular speed. | q(t) = omega*t, qDot = omega imposed rather than solved |

### Joint (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::joint` | `base of RTS table 'joint'` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/joints/joint/joint.H` | Abstract base for all rigid-body joints; supplies the motion subspace and joint transform. | declareRunTimeSelectionTable(autoPtr, joint, dictionary); jcalc gives X_J(q) and the motion subspace S so v_J = S qDot |

### Mesh distributor  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshDistributors::distributor` | `distributor distributor; (libfvMeshDistributors.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshDistributors/distributor/fvMeshDistributorsDistributor.H` | Periodically redistributes the mesh using the 'distributor' method named in decomposeParDict. | triggered every redistributionInterval when (maxCells - meanCells)/meanCells > maxImbalance; new cell-to-processor map from the selected decompositionMethod distributor |
| `fvMeshDistributors::loadBalancer` | `distributor loadBalancer; (libfvMeshDistributors.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshDistributors/loadBalancer/fvMeshDistributorsLoadBalancer.H` | Redistributes based on measured per-processor CPU load rather than cell count alone. | per-cell weights from measured processor solve times; rebalanced when the fractional imbalance exceeds maxImbalance |
| `fvMeshDistributors::none` | `distributor none;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshDistributors/none/fvMeshDistributorsNone.H` | Null distributor; the parallel decomposition is never changed. |  |

### Mesh distributor framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshDistributor` | `base of the dynamicMeshDict 'distributor' RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshDistributors/fvMeshDistributor/fvMeshDistributor.H` | Abstract base for run-time parallel redistribution of the mesh and all its fields. | declareRunTimeSelectionTable(fvMeshDistributor, fvMesh); update() returns a polyDistributionMap |

### Mesh mover  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshMovers::inkJet` | `mover inkJet; (libfvMeshMoversInkJet.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/inkJet/fvMeshMoversInkJet.H` | Sinusoidal compression/expansion of an ink-jet 'pumping' region to impose a sinusoidal nozzle-exit flow. | points in [0, pistonLength] scaled by (1 + amplitude*sin(omega*t)*(1 - z/pistonLength)) |
| `fvMeshMovers::interpolator` | `mover interpolator; (libfvMeshMoversInterpolator.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/interpolator/fvMeshMoversInterpolator.H` | Replays pre-computed motion supplied as a set of pointVectorFields (displacement or absolute position) in the time directories. | time interpolation with the given interpolationScheme between bracketing stored point fields |
| `fvMeshMovers::motionSolver` | `mover motionSolver; (libfvMeshMoversMotionSolver.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/motionSolver/fvMeshMoversMotionSolver.H` | Mesh mover that delegates the point motion to a run-time selected motionSolver. | x = motionSolver::curPoints(); mesh fluxes from the swept volumes |
| `fvMeshMovers::multiValveEngine` | `mover multiValveEngine; (libfvMeshMoversMultiValveEngine.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/multiValveEngine.H` | Explicit node-translation engine mover supporting a piston and any number of valves with per-object scaled distance weighting. | per moving object: d_point = w(dist)*motion(t) with w a linear or cosine scaling of patch-to-object distance, clipped by maxMotionDistance and by moving/static frozen layer thicknesses; sliding patches keep nodes on the liner/NCC surfaces |
| `fvMeshMovers::none` | `mover none;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshMovers/none/fvMeshMoversNone.H` | Null mover; the mesh points do not move. |  |

### Mesh mover framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshMover` | `base of the dynamicMeshDict 'mover' RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshMovers/fvMeshMover/fvMeshMover.H` | Abstract base for fvMesh movers: moves points, updates volumes and generates mesh fluxes without topology change. | declareRunTimeSelectionTable(fvMeshMover, fvMesh); meshPhi from swept volumes satisfying the space-conservation law dV/dt - sum(phi_mesh) = 0 |

### Mesh mover support  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `multiValveEngine::movingObject` | `no (sub-dictionary per object name)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/multiValveEngine.H` | Base description of one moving engine object (patches, motion Function1, moving/frozen zones, scaling weights). | caches the per-point motion scale field, recomputed every travelInterval of the stroke |
| `multiValveEngine::pistonObject` | `piston { ... } sub-dictionary` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/multiValveEngine.H` | Piston specialisation of movingObject, with cylinder-liner sliding and clearance reporting. | axial translation from the piston Function1 (often crankConnectingRodMotion) |
| `multiValveEngine::valveObject / valveList` | `valves { <name> { ... } } sub-dictionaries` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/multiValveEngine.H` | Valve specialisation (lift Function1, minLift closure detection) and the container of all valves. | valve translation = lift(t) along the valve axis; considered closed below minLift |
| `pistonPointEdgeData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshMovers/multiValveEngine/pistonPointEdgeData/pistonPointEdgeData.H` | PointEdgeWave datum carrying nearest-wall-point information for the piston motion-distance calculation. | PointEdgeWave propagation of nearest wall point and squared distance |

### Mesh stitcher  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshStitchers::moving` | `stitcher moving; (libfvMeshStitchers.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshStitchers/moving/fvMeshStitchersMoving.H` | Stitcher for moving meshes; re-intersects sliding interfaces and repairs the mesh fluxes each time step. | re-computes the non-conformal face partitioning, then corrects meshPhi so the discrete space-conservation law dV/dt = sum(meshPhi) is satisfied across the interface |
| `fvMeshStitchers::stationary` | `stitcher stationary;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/stationary/fvMeshStitchersStationary.H` | Stitcher for meshes whose non-conformal interfaces do not move. | intersection computed once; no mesh-flux correction needed |

### Mesh stitcher framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshStitcher` | `base of the 'stitcher' RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/fvMeshStitcher/fvMeshStitcher.H` | Abstract base that turns cyclic non-conformal poly patches into non-conformal finite-volume interfaces. | declareRunTimeSelectionTable(fvMeshStitcher, fvMesh); conformal <-> non-conformal face-area partitioning by patch-to-patch intersection |

### Mesh stitcher support  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `conformedFvPatchField / conformedFvsPatchField` | `type conformed; (internal, written during un-conforming)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/fvMeshStitcher/conformedFvPatchField.H` | Placeholder patch fields that store the original boundary values while a non-conformal interface is un-stitched. |  |
| `meshPhiCorrectInfo` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshStitchers/moving/meshPhiCorrectInfo.H` | FaceCellWave datum that distributes the mesh-flux volume error away from the non-conformal interface. | conservative redistribution of the cell volume-conservation error along the wave, face flux corrections accumulated per layer |
| `meshPhiPreCorrectInfo` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshStitchers/moving/meshPhiPreCorrectInfo.H` | FaceCellWave datum that propagates the layer/weight information used before the mesh-flux correction. | wave of (layer index, weight) outward from the non-conformal interface |

### Motion diffusivity  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `directionalDiffusivity` | `diffusivity directional (Dx Dy Dz);` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/directional/directionalDiffusivity.H` | Anisotropic diffusivity given as a fixed diagonal vector in global directions. | gamma_f = n & cmptMultiply(D, n) with n = Sf/\|Sf\| |
| `fileDiffusivity` | `diffusivity file <fieldName>;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/file/fileDiffusivity.H` | Face diffusivity read from a named surfaceScalarField file. | gamma_f read from disk, held fixed |
| `inverseDistanceDiffusivity` | `diffusivity inverseDistance (patch1 patch2 ...);` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/inverseDistance/inverseDistanceDiffusivity.H` | Diffusivity inversely proportional to cell-centre distance from the listed patches, stiffening the near-wall mesh. | gamma_f = 1/interpolate(y), y = cell-centre wall distance to the given patches |
| `inverseFaceDistanceDiffusivity` | `diffusivity inverseFaceDistance (patch1 patch2 ...);` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/inverseFaceDistance/inverseFaceDistanceDiffusivity.H` | Diffusivity from the face-based (patch-wave) distance to the listed patches. | fvPatchDistWave on faces; gamma_f = 1/y_f |
| `inversePointDistanceDiffusivity` | `diffusivity inversePointDistance (patch1 patch2 ...);` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/inversePointDistance/inversePointDistanceDiffusivity.H` | Diffusivity from a PointEdgeWave point-to-patch distance, giving smoother near-wall stiffening. | PointEdgeWave point distance averaged to faces; gamma_f = 1/dist |
| `inverseVolumeDiffusivity` | `diffusivity inverseVolume;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/inverseVolume/inverseVolumeDiffusivity.H` | Diffusivity inversely proportional to cell volume, stiffening small cells. | gamma_f = 1/interpolate(V) |
| `motionDirectionalDiffusivity` | `diffusivity motionDirectional (Dx Dy Dz);` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/motionDirectional/motionDirectionalDiffusivity.H` | Anisotropic diffusivity aligned with the local mesh-motion direction rather than a fixed axis. | D = Dy*I + (Dx - Dy)*cellMotionU/(\|cellMotionU\| + small); gamma_f = n & cmptMultiply(interp(D), n) |
| `uniformDiffusivity` | `diffusivity uniform;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/uniform/uniformDiffusivity.H` | Constant unity face diffusivity everywhere. | gamma_f = 1 |

### Motion diffusivity (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `motionDiffusivity` | `base of RTS table 'motionDiffusivity' (read from the 'diffusivity' entry as an Istream)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/motionDiffusivity/motionDiffusivity.H` | Abstract base returning the face diffusivity field gamma used by the Laplacian/SBRStress motion solvers. | declareRunTimeSelectionTable(autoPtr, motionDiffusivity, Istream); operator()() -> surfaceScalarField faceDiffusivity |

### Motion diffusivity (manipulator)  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `exponentialDiffusivity` | `diffusivity exponential <alpha> <baseDiffusivity ...>;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/manipulators/exponential/exponentialDiffusivity.H` | Manipulator returning an exponential function of an underlying diffusivity. | gamma_f = exp(-alpha/gamma_base) |
| `quadraticDiffusivity` | `diffusivity quadratic <baseDiffusivity ...>;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/motionDiffusivity/manipulators/quadratic/quadraticDiffusivity.H` | Manipulator that squares an underlying diffusivity to sharpen its variation. | gamma_f = sqr(gamma_base) |

### Motion solver  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `displacementLayeredMotionMotionSolver` | `motionSolver displacementLayeredMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/layeredSolver/displacementLayeredMotionMotionSolver.H` | Interpolates displacement through a layered (extruded) cellZone between two opposing faceZones by topological walking. | topological PointEdgeWave along layer edges; linear blend of the two faceZone boundary displacements by accumulated edge-walk distance; faceZone modes follow / uniformFollow / fixedValue / timeVaryingUniformFixedValue / slip (normal component removed) |
| `displacementLinearMotionMotionSolver` | `motionSolver displacementLinearMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/linearSolver/displacementLinearMotionMotionSolver.H` | Linear expansion/contraction of a mesh slab between a fixed and a moving station along a given axis. | scale = clip((axis&x - xFixed)/(xMoving - xFixed), 0, 1); d = scale*displacement(t)*axis |
| `interpolatingSolidBodyMotionSolver` | `motionSolver interpolatingSolidBody;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/interpolatingSolidBodyMotionSolver/interpolatingSolidBodyMotionSolver.H` | Solid-body motion blended into the far field by a distance-based weight so the mesh deforms smoothly. | x = points0 + w(y)*(T(t)&points0 - points0) with septernion SLERP of T and w a function of wall distance y (innerDistance/outerDistance) |
| `motionSolverList` | `motionSolver motionSolverList;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/motionSolverList/motionSolverList.H` | Executes a list of motion solvers in order and accumulates their point displacements. | x = x0 + sum_i d_i, displacements accumulated sequentially |
| `multiSolidBodyMotionSolver` | `motionSolver multiSolidBodyMotionSolver;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/multiSolidBodyMotionSolver/multiSolidBodyMotionSolver.H` | Applies a different solidBodyMotionFunction to each of several cellZones. | per-zone x = transform(T_zone(t), points0) |
| `rigidBodyMeshMotion` | `motionSolver rigidBodyMotion; (librigidBodyMeshMotion.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyMeshMotion/rigidBodyMeshMotion/rigidBodyMeshMotion.H` | Couples a full multi-body rigidBodyMotion to the mesh, moving points directly by distance-weighted interpolation. | septernion SLERP interpolation of each body's transform weighted by wall distance between innerDistance and outerDistance |
| `rigidBodyMeshMotionSolver` | `motionSolver rigidBodyMotionSolver; (librigidBodyMeshMotion.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyMeshMotion/rigidBodyMeshMotionSolver/rigidBodyMeshMotionSolver.H` | As rigidBodyMotion, but delegates the interior mesh deformation to a nested motion solver (e.g. displacementLaplacian). | body transforms applied as pointDisplacement boundary values, interior solved by the nested motionSolver |
| `sixDoFRigidBodyMotionSolver` | `motionSolver sixDoFRigidBodyMotion; (libsixDoFRigidBodyMotion.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/sixDoFRigidBodyMotion/sixDoFRigidBodyMotionSolver/sixDoFRigidBodyMotionSolver.H` | Moves an fvMesh with a single 6-DoF body, blending the rigid motion into the far field. | x = points0 + w(y)*(septernion SLERP transform - identity) applied to points0, w from innerDistance/outerDistance wall distance |
| `solidBodyMotionSolver` | `motionSolver solidBody;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionSolver/solidBodyMotionSolver.H` | Rigid solid-body displacement of all (or a cellZone's) points from a run-time selected solidBodyMotionFunction. | x = transform(septernion T(t), points0); optional restriction to a cellZone/pointset |

### Motion solver (abstract)  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `componentDisplacementMotionSolver` | `no (abstract, typeName componentDisplacementMotionSolver)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/componentDisplacement/componentDisplacementMotionSolver.H` | Base for single-component displacement solvers using pointDisplacementX/Y/Z pointScalarField. | x_cmpt = points0_cmpt + d_cmpt for one Cartesian component only |
| `componentVelocityMotionSolver` | `no (abstract, typeName componentVelocityMotionSolver)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/componentVelocity/componentVelocityMotionSolver.H` | Base for single-component velocity solvers using pointMotionUX/Y/Z pointScalarField. | x_cmpt^{n+1} = x_cmpt^n + U_cmpt*deltaT |
| `displacementMotionSolver` | `no (abstract, typeName displacementMotionSolver)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/displacement/displacementMotionSolver.H` | Base for solvers whose boundary condition is set on the pointDisplacement pointVectorField. | x = points0 + pointDisplacement |
| `points0MotionSolver` | `no (abstract, typeName points0MotionSolver)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/points0/points0MotionSolver.H` | Base for displacement-type solvers; stores and maps the undeformed reference point field points0. | x = points0 + d; points0 remapped on topology change |
| `velocityMotionSolver` | `no (abstract, typeName velocityMotionSolver)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/velocity/velocityMotionSolver.H` | Base for solvers whose boundary condition is set on the pointMotionU pointVectorField. | x^{n+1} = x^n + pointMotionU*deltaT |

### Motion solver framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `motionSolver` | `base class of RTS table 'motionSolver' (dictionary constructor)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/motionSolver/motionSolver.H` | Abstract base class for all mesh motion solvers; owns the RTS table selected by the dynamicMeshDict 'motionSolver' entry. | declareRunTimeSelectionTable(autoPtr, motionSolver, dictionary); returns curPoints() and applies newPoints() to the polyMesh |

### Motion solver support  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `dynamicMeshPointInterpolator` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/pointInterpolator/dynamicMeshPointInterpolator.H` | Interpolates a set of stored pointVectorFields in time to give the current displacement. | time interpolation with a selectable interpolationScheme (e.g. linear) over available time directories |
| `pointEdgeStructuredWalk` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/layeredSolver/pointEdgeStructuredWalk.H` | PointEdgeWave transported datum recording the length of the string of edges walked to a point. | accumulates \|edge\| along the walk to give the layer coordinate used for blending |

### Parallel infrastructure  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `domainDecomposition` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/parallel/domainDecomposition.H` | Automatic finite-volume domain decomposition and reconstruction driver used by decomposePar/reconstructPar. | builds per-processor meshes with processor patch addressing (procFaceAddressing, procCellAddressing, procPointAddressing) and inverts them for reconstruction |
| `domainDecompositionNonConformal` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/parallel/domainDecompositionNonConformal.C` | Handles non-conformal cyclic patches when decomposing and reconstructing. | per-processor non-conformal coupled patch generation from the original interface addressing |
| `domainDecompositionReconstruct` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/parallel/domainDecompositionReconstruct.C` | Rebuilds the complete mesh and fields from the per-processor meshes using the stored addressing. | inverse of the decompose addressing; processor patch faces merged back into internal faces |
| `multiDomainDecomposition` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/parallel/multiDomainDecomposition.H` | Manages the decomposition of a multi-region case as a set of domainDecomposition objects. |  |
| `processorRunTimes` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/parallel/parallel/processorRunTimes.H` | Holds the complete-case Time plus one Time object per processorN directory for decompose/reconstruct tools. |  |

### Patch agglomeration  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `pairPatchAgglomeration` | `no (constructed directly, libpairPatchAgglomeration.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvAgglomerationMethods/pairPatchAgglomeration/pairPatchAgglomeration.H` | Pairwise face agglomeration of a primitive patch, used to build coarse levels for view-factor/radiation exchange. | greedy pairing of neighbouring faces by largest shared edge weight, repeated for maxLevels or until nFacesInCoarsestLevel is reached; restrict/prolong operators built per level |

### Point boundary condition  <sub>(10)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `angularOscillatingDisplacementPointPatchVectorField` | `type angularOscillatingDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/angularOscillatingDisplacement/angularOscillatingDisplacementPointPatchVectorField.H` | Oscillating rigid rotation of a patch expressed as point displacement. | angle = amplitude*sin(omega*t); d = (rotationTensor(axis, angle) - I) & (p0 - origin) |
| `angularOscillatingVelocityPointPatchVectorField` | `type angularOscillatingVelocity;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/angularOscillatingVelocity/angularOscillatingVelocityPointPatchVectorField.H` | Oscillating rigid rotation of a patch expressed as point velocity. | U = ((rotationTensor(axis, amplitude*sin(omega*t)) & (p0 - origin)) + origin - x)/deltaT |
| `oscillatingDisplacementPointPatchVectorField` | `type oscillatingDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/oscillatingDisplacement/oscillatingDisplacementPointPatchVectorField.H` | Sinusoidally oscillating point displacement on a patch. | d(t) = amplitude*sin(omega*t) |
| `oscillatingVelocityPointPatchVectorField` | `type oscillatingVelocity;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/oscillatingVelocity/oscillatingVelocityPointPatchVectorField.H` | Sinusoidally oscillating point velocity on a patch. | x = p0 + amplitude*sin(omega*t); U = (x - x_old)/deltaT |
| `solidBodyMotionDisplacementPointPatchVectorField` | `type solidBodyMotionDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/pointPatchFields/derived/solidBodyMotionDisplacement/solidBodyMotionDisplacementPointPatchVectorField.H` | Fixed-value pointDisplacement patch driven by a solidBodyMotionFunction. | d_p = transform(T(t), points0_p) - points0_p |
| `surfaceDisplacementPointPatchVectorField` | `type surfaceDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/surfaceDisplacement/surfaceDisplacementPointPatchVectorField.H` | Point displacement fixed by projection onto a searchableSurface (triSurface), velocity-clipped per time step. | projectMode NEAREST \| POINTNORMAL \| FIXEDNORMAL; d = clip(x_proj - x, velocity*deltaT); optional wedgePlane component removal and frozenPointsZone |
| `surfaceSlipDisplacementPointPatchVectorField` | `type surfaceSlipDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/surfaceSlipDisplacement/surfaceSlipDisplacementPointPatchVectorField.H` | Point displacement that slides along a triSurface (following, not fixing, the surface). | projectMode NEAREST \| POINTNORMAL \| FIXEDNORMAL; displacement re-evaluated each call so points remain on the surface |
| `timeVaryingMappedFixedValuePointPatchField` | `type timeVaryingMappedFixedValue;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/timeVaryingMappedFixedValue/timeVaryingMappedFixedValuePointPatchField.H` | Fixed-value point BC mapped from external time-stamped boundaryData point clouds. | spatial mapping from an external point cloud + linear interpolation in time |
| `uniformInterpolatedDisplacementPointPatchVectorField` | `type uniformInterpolatedDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/uniformInterpolatedDisplacement/uniformInterpolatedDisplacementPointPatchVectorField.H` | Prescribed motion read from stored pointVectorFields in the time directories and interpolated in time. | time interpolation (interpolationScheme, e.g. linear) between the two bracketing stored fields |
| `waveDisplacementPointPatchVectorField` | `type waveDisplacement;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMotionSolver/pointPatchFields/derived/waveDisplacement/waveDisplacementPointPatchVectorField.H` | Travelling cosine surface wave imposed as boundary point displacement, with optional spatial and temporal ramps. | d = amplitude*cos(omega*t - k&x) * startRamp * endRamp * timeRamp |

### Renumbering (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `renumberMethod` | `base of the renumberMeshDict 'method' RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/renumberMethod/renumberMethod.H` | Abstract base for cell renumbering methods used by renumberMesh to reduce matrix bandwidth. | declareRunTimeSelectionTable(autoPtr, renumberMethod, dictionary); returns the ordered-to-original cell map |

### Renumbering method  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CuthillMcKeeRenumber` | `method CuthillMcKee; (with optional 'reverse')` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/CuthillMcKeeRenumber/CuthillMcKeeRenumber.H` | Cuthill-McKee bandwidth-reducing renumbering, optionally reversed (RCM). | breadth-first level-set ordering from a pseudo-peripheral node, neighbours sorted by degree; optional reverse of the resulting order |
| `SloanRenumber` | `method Sloan; (libSloanRenumber.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/../SloanRenumber/SloanRenumber.H` | Sloan profile/wavefront-reducing renumbering (Boost graph implementation). | priority = W1*distance-to-end-node - W2*degree; greedy front advance minimising profile and wavefront; optional reverse |
| `manualRenumber` | `method manual; (dataFile entry)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/manualRenumber/manualRenumber.H` | Renumbering read from a user-supplied ordered-to-original cell list file. |  |
| `randomRenumber` | `method random;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/randomRenumber/randomRenumber.H` | Random cell permutation; diagnostic only. | random shuffle of the cell order |
| `springRenumber` | `method spring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/springRenumber/springRenumber.H` | Spring-analogy renumbering that iteratively pulls neighbouring cells to nearby indices. | iterative relaxation of cell indices toward the mean neighbour index, displacement limited by maxCo*nCells and decayed by freezeFraction each of maxIter iterations |
| `structuredRenumber` | `method structured; (with 'patches', 'depthFirst', nested 'method')` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/structuredRenumber/structuredRenumber.H` | Renumbers by walking mesh layers away from named patches, either column-first or layer-first. | OppositeFaceCellWave layer numbering from the patches; depthFirst = true numbers each column 0..nLayers-1, false numbers layer by layer |

### Renumbering support  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `OppositeFaceCellWave` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/renumber/renumberMethods/structuredRenumber/OppositeFaceCellWave.H` | FaceCellWave variant that walks only through prismatic cells to the single opposite face, used to detect mesh structure. | face -> cell -> unique opposite face propagation; cells with split faces are marked but not traversed |

### Rigid body  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::compositeBody` | `no (created by merge)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/compositeBody/compositeBody.H` | Holds the original body after it has been merged into a parent body. | parent inertia += X^T I_sub X (spatial inertia transformed and summed) |
| `RBD::cuboid` | `type cuboid;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/cuboid/cuboid.H` | Rigid body constructed from a mass and the lengths of the sides. | I = (m/12) diag(Ly^2 + Lz^2, Lz^2 + Lx^2, Lx^2 + Ly^2) |
| `RBD::jointBody` | `type jointBody;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/jointBody/jointBody.H` | Massless body used purely as an attachment point for a joint. |  |
| `RBD::masslessBody` | `type masslessBody;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/masslessBody/masslessBody.H` | Body with zero mass and inertia, used to carry frames or joints. | I = 0, m = 0 |
| `RBD::rigidBody` | `type rigidBody;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/rigidBody/rigidBody.H` | General rigid body specified by mass, centre of mass and inertia tensor; base of the body RTS table. | declareRunTimeSelectionTable(autoPtr, rigidBody, dictionary) |
| `RBD::sphere` | `type sphere;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/sphere/sphere.H` | Rigid body constructed from a mass and radius. | I = (2/5) m r^2 * I3 |
| `RBD::subBody` | `no (created by merge)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/bodies/subBody/subBody.H` | Holds the original body after it has been merged into a master body. | stores the constant spatial transform X from master frame to sub-body frame |

### Rigid-body dynamics algorithm  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::forwardDynamics` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodyModel/rigidBodyModel.H` | Computes joint accelerations from joint forces using Featherstone's articulated-body algorithm. | three-pass articulated-body algorithm: outward velocity/bias pass, inward articulated-inertia pass, outward acceleration pass |

### Rigid-body dynamics core  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::rigidBodyInertia` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodyInertia/rigidBodyInertia.H` | Spatial inertia of a body: mass, centre of mass and inertia tensor about the CoM. | I_spatial = [[Ic + m c^x c^xT, m c^x], [m c^xT, m*I3]] (parallel-axis / spatial-inertia form) |
| `RBD::rigidBodyModel` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodyModel/rigidBodyModel.H` | System of rigid bodies connected by 1-6 DoF joints, holding the kinematic and forward-dynamics state (Featherstone). | Featherstone articulated-body / composite-rigid-body forward dynamics: tau = H(q) qdd + C(q, qd); spatial transforms X, motion subspace S per joint |
| `RBD::rigidBodyModelState` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodyModelState/rigidBodyModelState.H` | Holds q, qDot and qDdot for the whole rigid-body model, plus time-step data. |  |
| `RBD::rigidBodyMotion` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodyMotion/rigidBodyMotion.H` | Driver that adds restraints and a run-time selected time integrator to a rigidBodyModel. | integrates the forward-dynamics accelerations with symplectic / CrankNicolson / Newmark schemes |

### Rigid-body restraint  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::restraints::externalForce` | `type externalForce;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/restraints/externalForce/externalForce.H` | Time-dependent external force applied at a body-local location via a Function1. | F = force(t) at 'location'; M = location ^ F |
| `RBD::restraints::linearAxialAngularSpring` | `type linearAxialAngularSpring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/restraints/linearAxialAngularSpring/linearAxialAngularSpring.H` | Linear torsional spring plus damper resisting rotation about a specified axis. | theta = acos(oldDir . newDir) about axis; M = -(k*theta*a + c*(axis.omega)*axis) |
| `RBD::restraints::linearDamper` | `type linearDamper;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/restraints/linearDamper/linearDamper.H` | Linear velocity-proportional damper acting in the body local frame. | F = -c * v_body |
| `RBD::restraints::linearSpring` | `type linearSpring;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/restraints/linearSpring/linearSpring.H` | Linear tension/compression spring with damping between an anchor and a body attachment point, with optional slack. | F = (-k(\|r\| - L0) - c(r_hat.v)) r_hat, M = attachmentPt ^ F; with allowSlack and \|r\| < L0 only the damping term acts |
| `RBD::restraints::sphericalAngularDamper` | `type sphericalAngularDamper;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/restraints/sphericalAngularDamper/sphericalAngularDamper.H` | Isotropic angular damper acting on all three rotational DoF in the body local frame. | M = -c * omega_body |

### Rigid-body restraint (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::restraint` | `base of RTS table 'restraint' (restraints sub-dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/restraints/restraint/rigidBodyRestraint.H` | Abstract base for external force/moment restraints applied to bodies of a rigidBodyModel. | declareRunTimeSelectionTable(autoPtr, restraint, dictionary); restrain() accumulates spatialVector(moment, force) into fx |

### Rigid-body time integrator  <sub>(3)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::rigidBodySolvers::CrankNicolson` | `solver { type CrankNicolson; aoc 0.5; voc 0.5; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodySolvers/CrankNicolson/CrankNicolson.H` | Implicit off-centred Crank-Nicolson 2nd-order integrator usable with outer correctors. | qDot^{n+1} = qDot^n + dt((1-aoc)a^n + aoc a^{n+1}); q^{n+1} = q^n + dt((1-voc)qDot^n + voc qDot^{n+1}); defaults aoc = voc = 0.5 |
| `RBD::rigidBodySolvers::Newmark` | `solver { type Newmark; gamma 0.5; beta 0.25; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodySolvers/Newmark/Newmark.H` | Implicit Newmark 2nd-order integrator usable with outer correctors. | qDot^{n+1} = qDot^n + dt((1-gamma)a^n + gamma a^{n+1}); q^{n+1} = q^n + dt qDot^n + dt^2((0.5-beta)a^n + beta a^{n+1}); defaults gamma = 0.5, beta = 0.25 |
| `RBD::rigidBodySolvers::symplectic` | `solver { type symplectic; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodySolvers/symplectic/symplectic.H` | Explicit 2nd-order symplectic (leapfrog/Verlet splitting) integrator; one call per time step only. | half-kick qDot += a dt/2, drift q += qDot dt, half-kick qDot += a dt/2 (Dullweber-Leimkuhler-McLachlan splitting) |

### Rigid-body time integrator (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `RBD::rigidBodySolver` | `base of RTS table 'rigidBodySolver' ('solver' sub-dictionary)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/rigidBodyMotion/rigidBodyDynamics/rigidBodySolvers/rigidBodySolver/rigidBodySolver.H` | Abstract base for the time integrators of a rigidBodyMotion. | declareRunTimeSelectionTable(autoPtr, rigidBodySolver, dictionary) |

### Solid-body motion function  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `SDA` | `solidBodyMotionFunction SDA;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/SDA/SDA.H` | Ship-design-analysis 3-DoF motion: coupled sinusoidal roll, heave and sway with time-varying amplitude and phase. | roll about x, heave in z, sway in y as damped sinusoids with Tp period, Tpn natural period and dTi/dTp amplitude/phase modulation |
| `axisRotationMotion` | `solidBodyMotionFunction axisRotationMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/axisRotationMotion/axisRotationMotion.H` | Constant-angular-velocity rotation about the CoG specified as an angular-velocity vector. | Euler angles = radialVelocity*t; quaternion from (rollPitchYaw) about CoG |
| `linearMotion` | `solidBodyMotionFunction linearMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/linearMotion/linearMotion.H` | Constant-velocity rigid translation. | d(t) = velocity*t; septernion(-d, quaternion::I) |
| `multiMotion` | `solidBodyMotionFunction multiMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/multiMotion/multiMotion.H` | Composition of several solid-body motion functions applied in sequence. | T(t) = T_n * ... * T_2 * T_1 (septernion product) |
| `oscillatingLinearMotion` | `solidBodyMotionFunction oscillatingLinearMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/oscillatingLinearMotion/oscillatingLinearMotion.H` | Sinusoidally oscillating rigid translation. | d(t) = amplitude*sin(omega*t) |
| `oscillatingRotatingMotion` | `solidBodyMotionFunction oscillatingRotatingMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/oscillatingRotatingMotion/oscillatingRotatingMotion.H` | Sinusoidally oscillating rigid rotation about a CoG. | eulerAngles(t) = amplitude*sin(omega*t); quaternion(rollPitchYaw) |
| `rotatingMotion` | `solidBodyMotionFunction rotatingMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/rotatingMotion/rotatingMotion.H` | Rotation about a given origin and axis at a Function1 angular speed. | theta(t) = integral omega dt; septernion(origin, quaternion(axis, theta)) |
| `sixDoFMotion` | `solidBodyMotionFunction sixDoFMotion;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/sixDoFMotion/sixDoFMotion.H` | Tabulated 6-DoF motion: surge, sway, heave, roll, pitch, yaw interpolated in time. | linear interpolation of tabulated (t, translation, rotation) pairs into a septernion |

### Solid-body motion function (base)  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `solidBodyMotionFunction` | `base of RTS table 'solidBodyMotionFunction'` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/motionSolvers/displacement/solidBody/solidBodyMotionFunctions/solidBodyMotionFunction/solidBodyMotionFunction.H` | Abstract base returning the septernion transformation of a rigid body as a function of time. | declareRunTimeSelectionTable(autoPtr, solidBodyMotionFunction, dictionary); returns septernion transformation() |

### Topology changer  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshTopoChangers::list` | `topoChanger list;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshTopoChangers/list/fvMeshTopoChangersList.H` | Applies a sequence of topology changers in the order listed. | composition of the individual polyTopoChangeMaps |
| `fvMeshTopoChangers::meshToMesh` | `topoChanger meshToMesh; (libmeshToMeshTopoChanger.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshTopoChangers/meshToMesh/fvMeshTopoChangersMeshToMesh.H` | Maps all fields onto a new mesh (or a repeating/cycling sequence of meshes) at prescribed times. | conservative cell-to-cell meshToMesh volume-overlap interpolation at the listed times, with optional repeat/cycle periods and timeDelta tolerance |
| `fvMeshTopoChangers::none` | `topoChanger none;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshTopoChangers/none/fvMeshTopoChangersNone.H` | Null topology changer; the mesh topology is fixed. |  |
| `fvMeshTopoChangers::refiner` | `topoChanger refiner; (libfvMeshTopoChangers.so)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshTopoChangers/refiner/fvMeshTopoChangersRefiner.H` | Dynamic hexahedral refinement/unrefinement driven by a volScalarField, optionally per cellZone region. | hexRef8 2:1-limited octree split of hex cells where lowerRefineLevel < field < upperRefineLevel, up to maxRefinement/maxCells with nBufferLayers; fluxes on changed faces reconstructed from the named correctFluxes velocity |

### Topology changer framework  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvMeshTopoChanger` | `base of the dynamicMeshDict 'topoChanger' RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshTopoChangers/fvMeshTopoChanger/fvMeshTopoChanger.H` | Abstract base for run-time topology changes (refinement, layering, mesh-to-mesh mapping). | declareRunTimeSelectionTable(fvMeshTopoChanger, fvMesh); update() returns a polyTopoChangeMap used to map all registered fields |

### Topology changer support  <sub>(1)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MeshToMeshMapGeometricFields` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/fvMeshTopoChangers/meshToMesh/MeshToMeshMapGeometricFields.H` | Template machinery that maps every registered vol/surface/point GeometricField through a meshToMesh change. | per-field src->tgt interpolation using the shared meshToMesh addressing and weights |

---

## Mesh: Core mesh data structures

> **Subsystem notes**
>
> Everything below was read from the source tree at C:/Users/sdd32/Documents/GitHub/open_cfd/OpenFOAM-Foundation-12, not from memory. Runtime-selection tables found in this part (from declareRunTimeSelectionTable): 1.
> polyPatch - two tables, "word" and "dictionary", keyed on the `type` entry in constant/polyMesh/boundary. Registered here: patch, coupled(abstract), cyclic, cyclicSlip, empty, internal, processor (dictionary only), processorCyclic (dictionary only), symmetry, symmetryPlane, wedge, wall.
> Further polyPatch types (nonConformal, nonConformalCoupled, nonConformalCyclic, nonConformalError, nonConformalProcessorCyclic, mapped, mappedWall, mappedInternal, mappedExtrudedWall, nonConformalMappedWall) are registered in src/meshTools and are outside this part, but their fvPatch counterparts are inside it.
> 2. facePointPatch - table "polyPatch", selected automatically from the polyPatch type; registered: patch(base), cyclic, cyclicSlip, empty, internal, processor, processorCyclic, symmetry, symmetryPlane, wedge, wall. 3.
> fvPatch - table "polyPatch", selected automatically from the polyPatch type; registered: patch(base), cyclic, cyclicSlip, empty, internal, processor, processorCyclic, symmetry, symmetryPlane, wedge, wall, nonConformalCyclic, nonConformalError, nonConformalProcessorCyclic, mapped, mappedWall, mappedInternal, mappedExtrudedWall, nonConformalMappedWall.
> 4. Zone<ZoneType,ZonesType> - "dictionary" table; registered types cellZone, faceZone, pointZone (the `type` entry in constant/polyMesh/{cellZones,faceZones,pointZones}). 5. patchDistMethod - "dictionary" table; keyword is `method` inside the `wallDist` sub-dictionary of fvSchemes.
> Registered: meshWave, Poisson, advectionDiffusion. 6. fvMeshStitcher / fvMeshTopoChanger / fvMeshMover / fvMeshDistributor - "fvMesh" tables read from dynamicMeshDict.
> Only the bases plus `stationary`, `none`, `list` live inside this part; the concrete movers/topo-changers/distributors are catalogued in the mesh-motion and topology-change parts.
> Notable checkout artefact: `src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatch.H` is absent from this Windows checkout because it collides case-insensitively with `primitivePatch.H` in the same directory; only the lower-case typedef header survived.
> The templated PrimitivePatch class implementation files (PrimitivePatch.C and its ten companions) are all present, so the class is catalogued from those.
> Geometry conventions confirmed by reading the code, not assumed: - face area/centre (face::areaAndCentre, faceTemplates.C): triangles handled directly; polygons decomposed into a triangle fan about the point average, with each triangle weighted by its area projected onto the face unit normal, which makes the centre independent of the initial estimate.
> - cell centre/volume (primitiveMeshCellCentresAndVols.C): pyramid decomposition about the cell-centre estimate cEst = mean of face centres, 3V_pyr = Sf&(Cf-cEst), pyramid centroid 3/4 Cf + 1/4 cEst.
> - fvPatch::delta() is the patch-normal delta n(n&(Cf-Cn)) for all non-coupled patches, whereas coupledFvPatch::delta() is the raw Cf-Cn; coupled weights use the projected normal distances on both sides.
> Boundary between this part and neighbouring parts: polyTopoChange/polyTopoChanger engines, meshTools search trees (indexedOctree, treeData*), AMI/patch-to-patch intersection, mesh generation and mesh quality checking (primitiveMeshCheck/polyMeshCheck) are not under the directories assigned here and are excluded, though several classes above (polyMesh::cellTree, nonConformal patches, mapped patches) reference them.

### Identifiers  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `coupleGroupIdentifier` | `coupleGroup <groupName>; (patch dictionary entry)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/Identifiers/patch/coupleGroupIdentifier.H` | Resolves the neighbour of a coupled patch by patchGroup name instead of an explicit patch name, possibly in another region. |  |
| `patchIdentifier` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/Identifiers/patch/patchIdentifier.H` | Identifies a patch by name, index, physical type and the groups it belongs to. |  |

### bounding volumes  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `boundBox` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/boundBox/boundBox.H` | Axis-aligned bounding box from a point field, with span, midpoint, containment, overlap, inflation and parallel reduction. | min/max componentwise over points (optionally reduced across processors); overlaps by per-component interval test; sphere overlap via nearest-corner distance |
| `treeBoundBox` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/treeBoundBox/treeBoundBox.H` | boundBox extended for octree use: octant subdivision, corner/edge/face numbering, ray clipping and nearest/farthest corner queries. | subBbox(octant) halves each direction about the midpoint; octant index built from the three coordinate sign bits; intersects(ray) by slab clipping |

### extendedStencil  <sub>(18)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `CECCellToCellStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToCell/globalIndexStencils/CECCellToCellStencil.H` | Cell-Edge-Cell stencil: all cells connected to the cell through any of its edges. | stencil(c) = union over edges e of c of edgeCells(e), globally numbered and synchronised across coupled patches |
| `CECCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/globalIndexStencils/CECCellToFaceStencil.H` | Face stencil from the combined CEC cell stencils of owner and neighbour. | stencil(f) = CEC(own(f)) union CEC(nei(f)) |
| `CFCCellToCellStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToCell/globalIndexStencils/CFCCellToCellStencil.H` | Cell-Face-Cell stencil: all cells sharing a face with the cell. | stencil(c) = {c} union {cells adjacent across each face of c} |
| `CFCCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/globalIndexStencils/CFCCellToFaceStencil.H` | Face stencil formed by merging the CFC cell stencils of the owner and neighbour cells. | stencil(f) = CFC(own(f)) union CFC(nei(f)) with own/nei first |
| `CPCCellToCellStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToCell/globalIndexStencils/CPCCellToCellStencil.H` | Cell-Point-Cell stencil: all cells sharing a point with the cell. | stencil(c) = union over points p of c of pointCells(p) |
| `CPCCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/globalIndexStencils/CPCCellToFaceStencil.H` | Face stencil from the combined CPC cell stencils of owner and neighbour. | stencil(f) = CPC(own(f)) union CPC(nei(f)) |
| `FECCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/globalIndexStencils/FECCellToFaceStencil.H` | Face-Edge-Cell stencil: all cells connected via an edge of the face. | stencil(f) = union over edges e of f of edgeCells(e) |
| `cellToCellStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToCell/globalIndexStencils/cellToCellStencil.H` | Base class for extended cell-centred addressing: per cell, the list of neighbouring cells and boundary faces in global numbering. | global indices from globalIndex over cells then non-empty boundary faces |
| `cellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/globalIndexStencils/cellToFaceStencil.H` | Base for extended cell-to-face stencils supplying face values from surrounding cells. | element 0 = owner, element 1 = neighbour, then the extended entries |
| `centredCFC/CEC/CPC/FEC CellToFaceStencilObjects` | `registered names centredCFCCellToFaceStencil, centredCECCellToFaceStencil, centredCPCCellToFaceStencil, centredFECCellToFaceStencil` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/MeshObjects/centredCFCCellToFaceStencilObject.H` | Cached MeshObjects providing the centred CFC, CEC, CPC and FEC cell-to-face stencils. |  |
| `centredCFCCellToCellStencilObject / centredCECCellToCellStencilObject / centredCPCCellToCellStencilObject` | `registered names centredCFCCellToCellStencil, centredCECCellToCellStencil, centredCPCCellToCellStencil` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToCell/MeshObjects/centredCFCCellToCellStencilObject.H` | Cached MeshObjects giving the centred CFC, CEC and CPC cell-to-cell stencils for a mesh. |  |
| `extendedCellToCellStencil / extendedCentredCellToCellStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToCell/extendedCellToCellStencil.H` | Compacted, parallel-distributable form of a cell-to-cell stencil with weighted summation over the collected values. | result(c) = sum_j w_j(c) * f[stencil_j(c)] after distributing the compact field |
| `extendedCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/extendedCellToFaceStencil.H` | Compact extended cell-to-face stencil: collects cell and non-empty boundary values, distributes them in parallel and sums with weights. | f_face = sum_j w_j * collectedField[compactIndex_j]; index layout is all cells then all non-empty boundary faces |
| `extendedCentredCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/extendedCentredCellToFaceStencil.H` | Centred (owner and neighbour symmetric) form of the extended cell-to-face stencil. |  |
| `extendedFaceToCellStencil / extendedCentredFaceToCellStencil / centredCFCFaceToCellStencilObject` | `registered name centredCFCFaceToCellStencil` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/faceToCell/extendedFaceToCellStencil.H` | Compact, distributable face-to-cell stencil and its cached MeshObject; transforms on coupled patches are not supported. | cell value = sum_j w_j * surfaceField[stencil_j(c)] |
| `extendedUpwindCellToFaceStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/extendedUpwindCellToFaceStencil.H` | Builds separate owner and neighbour (upwind/downwind) stencils by shifting a centred stencil, optionally keeping only pure-upwind faces. | a candidate face is included when (upwindArea & myArea)/magSqr(myArea) > minOpposedness |
| `faceToCellStencil / CFCFaceToCellStencil` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/faceToCell/globalIndexStencils/faceToCellStencil.H` | Base and CFC implementation of extended face-to-cell addressing: per cell, the neighbouring faces in global numbering. | stencil(c) = faces of the cells in CFC(c) |
| `upwindCFC/CEC/CPC/FEC CellToFaceStencilObjects and pureUpwindCFCCellToFaceStencilObject` | `registered names upwindCFCCellToFaceStencil, upwindCECCellToFaceStencil, upwindCPCCellToFaceStencil, upwindFECCellToFaceStencil, pureUpwindCFCCellToFaceStencil` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/extendedStencil/cellToFace/MeshObjects/upwindCFCCellToFaceStencilObject.H` | Cached MeshObjects providing the upwind-biased (and pure-upwind) cell-to-face stencils used by high-order upwind schemes. | minOpposedness parameter (default 0.5, 1 for pureUpwind) controls which shifted faces are combined |

### fvMesh core  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `fvBoundaryMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvBoundaryMesh/fvBoundaryMesh.H` | PtrList of fvPatches built one-for-one from the polyBoundaryMesh, with group/name lookup and the interfaces list. |  |
| `fvCellSet` | `select all \| cellSet \| cellZone \| points;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvCellSet/fvCellSet.H` | Runtime-selected cell selection for fvMesh (used by fvModels, fvConstraints, functionObjects), also reporting the selected volume. | V = sum of cell volumes in the selection, reduced in parallel |
| `fvMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMesh.H` | polyMesh plus all finite-volume geometry (V, V0, V00, Sf, magSf, C, Cf, phi), lduAddressing, and the stitcher/topo-changer/mover/distributor function classes. | V and Sf/C from primitiveMesh; V0/V00 retained old-time volumes; phi = mesh motion flux = swept volume/deltaT; non-conformal geometry set by unconform()/conform() |
| `fvMeshLduAddressing` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshLduAddressing.H` | lduAddressing built from the fvMesh owner/neighbour lists, including the losort ordering for matrix assembly. | lowerAddr = faceOwner, upperAddr = faceNeighbour on internal faces; losort permutation groups faces by neighbour cell |
| `fvMeshMapper / fvPatchMapper / fvSurfaceMapper / fvBoundaryMeshMapper` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshMapper/fvMeshMapper.H` | Bundles the volume (cellMapper), surface and patch mappers needed to map fvMesh fields across a topology change. | direct addressing for surviving faces/cells; uniform 1/n weights for inserted entities |
| `singleCellFvMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/singleCellFvMesh/singleCellFvMesh.H` | An fvMesh of one cell carrying all original boundary faces, for manipulating and writing boundary data alone. | optional agglomeration of patch faces with area-weighted averaging of the mapped boundary values |
| `zeroDimensionalFvMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/zeroDimensionalFvMesh/zeroDimensionalFvMesh.H` | Free function constructing a zero-dimensional unit-cube single-cell fvMesh for 0-D (reactor) solvers. | unit cube with all six faces empty; V = 1 |

### fvMesh function classes  <sub>(10)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `conformedFvPatchField / conformedFvsPatchField` | `type conformed;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/fvMeshStitcher/conformedFvPatchField.H` | Temporary patch-field types holding both original and non-conformal face data between the un-stitch and stitch steps so nothing is lost during mapping. | non-conformal values area-averaged onto their originating conformal face |
| `fvMeshDistributor` | `distributor { type <name>; } - none defined here, other distributors elsewhere` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshDistributors/fvMeshDistributor/fvMeshDistributor.H` | Abstract runtime-selectable base for in-run mesh redistribution (load balancing). |  |
| `fvMeshDistributors::none` | `none` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshDistributors/none/fvMeshDistributorsNone.H` | Null distributor; the mesh is never redistributed. |  |
| `fvMeshMover` | `mover { type <name>; } - none defined here, other movers elsewhere` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshMovers/fvMeshMover/fvMeshMover.H` | Abstract runtime-selectable base for mesh movers: move points, update volumes and generate the mesh motion fluxes without topology change. | phi from the volume swept by each face; V00/V0 retained for the space-conservation law |
| `fvMeshMovers::none` | `none` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshMovers/none/fvMeshMoversNone.H` | Null mesh mover for static meshes. |  |
| `fvMeshStitcher` | `declares fvMeshStitcher RTS table (fvMesh); concrete: stationary` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/fvMeshStitcher/fvMeshStitcher.H` | Abstract runtime-selectable manipulator that turns the cyclic non-conformal poly-patch intersection into non-conformal finite-volume interfaces. | conform(): fv faces collapse back to poly faces; unconform(): fv faces become the intersection polygons with their own Sf, Cf and motion flux |
| `fvMeshStitcherTools` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/fvMeshStitcher/fvMeshStitcherTools.H` | Free functions used by the stitching process (field synchronisation, area-weighted sums over the non-conformal faces). | sum over the non-conformal faces of a poly face weighted by magSf |
| `fvMeshStitchers::stationary` | `stationary` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshStitchers/stationary/fvMeshStitchersStationary.H` | Mesh stitcher for stationary (non-moving) meshes. |  |
| `fvMeshTopoChanger` | `topoChanger { type <name>; } - none and list defined here, other models elsewhere` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshTopoChangers/fvMeshTopoChanger/fvMeshTopoChanger.H` | Abstract runtime-selectable base for fvMesh topology changers (refinement, layering, mesh-to-mesh); held by fvMesh and read from dynamicMeshDict. |  |
| `fvMeshTopoChangers::none / fvMeshTopoChangers::list` | `none, list` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvMeshTopoChangers/none/fvMeshTopoChangersNone.H` | Null topo-changer, and a composite that applies a list of topo-changers in sequence. |  |

### fvPatch (runtime-selectable)  <sub>(26)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `coupledFvPatch` | `abstract base (coupled)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/basic/coupled/coupledFvPatch.H` | Abstract base for fv patches coupling two regions; supplies delta and the two-sided interpolation weights. | delta = Cf - Cn; w = (nf&delta_nbr) / ((nf&delta_own) + (nf_nbr&delta_nbr)) guarded against degenerate deltas |
| `cyclicFvPatch` | `cyclic` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/cyclic/cyclicFvPatch.H` | Cyclic-plane fv patch providing the neighbour patch field and the coupled matrix interface. | neighbour values transformed by the cyclic transform before use |
| `cyclicSlipFvPatch` | `cyclicSlip` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/cyclicSlip/cyclicSlipFvPatch.H` | Cyclic fv patch counterpart of cyclicSlipPolyPatch. |  |
| `emptyFvPatch` | `empty` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/empty/emptyFvPatch.H` | Zero-sized fv patch: the faces exist in the polyMesh but carry no finite-volume discretisation. | size() = 0; faceCells empty |
| `fvPatch` | `type patch;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/fvPatch/fvPatch.H` | Finite-volume patch wrapping a polyPatch: faceCells, Cf, Cn, Sf, magSf, nf, delta, weights and deltaCoeffs. | nf = Sf/magSf; delta = n (n & (Cf - Cn)) for non-coupled patches; weights = 1; deltaCoeffs from the mesh surface interpolation |
| `fvPatchList` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/fvPatch/fvPatchList.H` | PtrList typedef of fvPatch used by fvBoundaryMesh. |  |
| `internalFvPatch` | `internal` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/internal/internalFvPatch.H` | fv patch holding internal faces exposed by sub-setting. |  |
| `mappedExtrudedWallFvPatch` | `mappedExtrudedWall` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/mapped/mappedExtrudedWallFvPatch.H` | Mapped wall patch for an extruded (film/baffle) region, mapping between the extruded patch and its source. |  |
| `mappedFvPatch` | `mapped` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/mapped/mappedFvPatch.H` | Generic fv patch mapping values from another globally conforming fv patch. |  |
| `mappedFvPatchBase` | `mappedFvPatchBase` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/mapped/mappedFvPatchBase.H` | Base for fv patches performing interpolative mapping between two globally conforming fv patches. | distribution via a distributionMap built from the patch-to-patch address/weight pairs |
| `mappedFvPatchBaseBase` | `mappedFvPatchBaseBase` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/mapped/mappedFvPatchBaseBase.H` | Base providing the neighbour region/patch lookup and field transfer for all mapped fv patches. |  |
| `mappedInternalFvPatch` | `mappedInternal` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/mapped/mappedInternalFvPatch.H` | fv patch that maps values from internal cells located at an offset from the patch. | sampling at Cf + offset (or normal distance) into the interior |
| `mappedWallFvPatch` | `mappedWall` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/mapped/mappedWallFvPatch.H` | Wall fv patch that maps values from another globally conforming wall patch (region coupling). |  |
| `nonConformalCoupledFvPatch` | `nonConformalCoupled (base)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/nonConformalCoupled/nonConformalCoupledFvPatch.H` | Non-conformal fv patch that is also coupled to another non-conformal patch. |  |
| `nonConformalCyclicFvPatch` | `nonConformalCyclic` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/nonConformalCyclic/nonConformalCyclicFvPatch.H` | Non-conformal cyclic interface whose neighbour patch is local, built from the cyclic patch intersection. | coupling weights from the polygon intersection areas of the two patch face sets |
| `nonConformalErrorFvPatch` | `nonConformalError` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/nonConformalError/nonConformalErrorFvPatch.H` | Uncoupled patch collecting the area left over where the two sides of a non-conformal coupling do not intersect. | error area = original patch area minus the total intersected area |
| `nonConformalFvPatch` | `nonConformal (base)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/nonConformal/nonConformalFvPatch.H` | Base non-conformal fv patch giving access to the stitched geometry/topology generated by the fvMeshStitcher. | maps between the non-conformal fv faces and the underlying original poly faces |
| `nonConformalMappedFvPatchBase` | `nonConformalMappedFvPatchBase` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/nonConformalMapped/nonConformalMappedFvPatchBase.H` | Base providing non-conformal (intersection-weighted) mapping between two potentially non-conforming fv patches. | patch-to-patch intersection weights (AMI-style area overlap) |
| `nonConformalMappedWallFvPatch` | `nonConformalMappedWall` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/nonConformalMapped/nonConformalMappedWallFvPatch.H` | Wall fv patch mapping non-conformally from another wall patch, typically in another region. |  |
| `nonConformalProcessorCyclicFvPatch` | `nonConformalProcessorCyclic` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/nonConformalProcessorCyclic/nonConformalProcessorCyclicFvPatch.H` | Non-conformal cyclic interface whose neighbour side lives on another processor. |  |
| `processorCyclicFvPatch` | `processorCyclic` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/processorCyclic/processorCyclicFvPatch.H` | Processor fv patch that also applies the referenced cyclic transform. |  |
| `processorFvPatch` | `processor` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/processor/processorFvPatch.H` | Inter-processor fv patch implementing the lduInterface send/receive of neighbour cell values. | initInternalFieldTransfer/internalFieldTransfer of the adjacent cell values across the processor boundary |
| `symmetryFvPatch` | `symmetry` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/symmetry/symmetryFvPatch.H` | Symmetry fv patch for non-planar or multi-plane symmetry surfaces. | per-face reflection I - 2 nf nf |
| `symmetryPlaneFvPatch` | `symmetryPlane` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/symmetryPlane/symmetryPlaneFvPatch.H` | Symmetry fv patch for a single flat plane, using one cached normal. | reflection I - 2 n n with the single patch normal n |
| `wallFvPatch` | `wall` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/derived/wall/wallFvPatch.H` | fv patch marking a solid wall for wall-function and wall-distance models. |  |
| `wedgeFvPatch` | `wedge` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/fvPatches/constraint/wedge/wedgeFvPatch.H` | Wedge fv patch exposing the face transform (faceT) and the cell-to-cell wedge rotation (cellT). | cellT = rotation by the full wedge angle; faceT = rotation onto the wedge centre plane |

### lduMesh  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `lduMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/lduMesh/lduMesh.H` | Abstract base for meshes that supply LDU addressing and interfaces for lduMatrix assembly and solution. | lower/upper/diagonal matrix addressing plus lduInterfacePtrsList for coupled patches |
| `lduPrimitiveMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/lduMesh/lduPrimitiveMesh.H` | Simplest concrete lduMesh storing owner/neighbour addressing directly; also used to build combined (agglomerated/merged) ldu meshes. | upper-triangular ordering of the lower/upper addressing with optional reordering and interface merging |

### mesh mapping  <sub>(9)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `cellMapper` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/cellMapper/cellMapper.H` | Provides direct or weighted mapping and fill-in for cell data across a topology change. | uniform weights 1/n over the master cells of an inserted cell |
| `faceMapper` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/faceMapper/faceMapper.H` | Provides direct or weighted mapping and fill-in for face data across a topology change, split into internal and patch faces. | uniform weights 1/n over the master faces of an inserted face |
| `mapAddedPolyMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/mapAddedPolyMesh.H` | Mapping data after adding one mesh to another (old-to-new and added-to-new point/face/cell/patch maps). |  |
| `mapPatchChange` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/mapPatchChange/mapPatchChange.H` | Mapping data after patches are added, removed or reordered. |  |
| `mapSubsetMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/mapSubsetMesh/mapSubsetMesh.H` | Mapping data after a mesh subset operation (subset->base point/face/cell maps and patch starts). |  |
| `morphFieldMapper` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/morphFieldMapper.H` | Abstract base holding the Field mapping (direct addressing or interpolation weights) for a mesh morph. | interpolated value = sum_j w_j * f_old[addr_j] |
| `objectMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/objectMap/objectMap.H` | Pair of an object index and the list of master indices it is mapped from. |  |
| `polyMeshMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshMap/polyMeshMap.H` | Lightweight marker map signalling a complete mesh-to-mesh replacement (fields must be re-interpolated, not renumbered). |  |
| `polyTopoChangeMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyTopoChangeMap/polyTopoChangeMap.H` | Complete mesh-to-mesh mapping data after a polyMesh topology change (point/face/cell maps, reverse maps, inserted/merged lists). | forward maps pull old->new (map[i] = old index, -1 if inserted); reverse maps push new->old; masterObjects give multi-parent weights |

### mesh objects  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `DemandDrivenMeshObject` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshObjects/DemandDrivenMeshObject.H` | Templated base for demand-driven mesh data with update policies DeletableMeshObject, MoveableMeshObject, DistributeableMeshObject, TopoChangeableMeshObject and RepatchableMeshObject. | movePoints/distribute/topoChange/mapMesh callbacks decide whether cached data is updated or deleted |
| `GeoMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/GeoMesh/GeoMesh.H` | Generic mesh wrapper giving a size() and boundary type to field templates; base of volMesh, surfaceMesh and pointMesh. |  |
| `MeshObject / meshObjects` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshObjects/meshObjects.H` | Registry and templated base automating allocation of optional mesh-derived data and its participation in the mesh-modifier event loop. |  |
| `Residuals` | `registered name residuals` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/meshObjects/Residuals/Residuals.H` | DemandDrivenMeshObject storing the solver performance residual history of all fields of a given type. |  |
| `cpuLoad / optionalCpuLoad` | `registered name cpuLoad` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/meshObjects/cpuLoad/cpuLoad.H` | Per-cell CPU time field used for load balancing; optionalCpuLoad::New returns a dummy unless loadBalancing is enabled. | accumulates measured cpuTime per cell between resets |

### mesh utilities  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MultiRegionRefs / MultiRegionList / regionName` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/multiRegion/MultiRegionRefs.H` | Wraps a UPtrList of region-associated objects so accessing one sets the Info output prefix to that region's name. |  |
| `bandCompression` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/bandCompression/bandCompression.H` | Renumbers cell addressing to reduce matrix bandwidth using the Cuthill-McKee algorithm. | breadth-first traversal of the cell-cell graph from a low-degree seed, visiting neighbours in order of connectivity |
| `matchPoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshTools/matchPoints.H` | Determines the one-to-one correspondence between two point fields within a per-point matching distance. | sort by mag(pt - origin), then compare candidates with equal magnitude within matchDistance |
| `mergePoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshTools/mergePoints.H` | Sorts and merges coincident points within a tolerance, returning the old-to-new map and the unique point count. | sort by mag(pt - origin); merge all points with separation <= mergeTol |

### meshShapes  <sub>(19)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `EdgeMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/edge/EdgeMap.H` | HashTable keyed on an edge (its two endpoints) used throughout edge-based mesh algorithms. |  |
| `cell` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cell/cell.H` | A cell as a list of face labels, with labels(), points(), edges(), centre() and mag() helpers. | centre/mag by pyramid decomposition about the average face centre |
| `cell::opposingFaceLabel / opposingFace` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cell/oppositeCellFace.C` | For a prismatic cell, finds the face opposite a given face and returns it oriented consistently. | opposite face = the one face sharing no point with the master face; vertex correspondence via shared side faces |
| `cellIOList / cellShapeIOList / faceIOList / edgeIOList / pointIndexHitIOList` | `IO class names: cellList, cellCompactList, shapeList, faceList, edgeList, edgeCompactList` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cell/cellIOList.H` | Registered IO (and CompactIO) list wrappers written to constant/polyMesh as cellList, shapeList, faceList and edgeList. |  |
| `cellMatcher` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cellMatcher/cellMatcher.H` | Abstract shape-recognition engine that orients a cell against a canonical model and builds cell-vertex and cell-face mappings. | local renumbering, edge-to-face addressing, then walk of the model's face/vertex ordering to find a consistent permutation |
| `cellModel` | `named models in etc/cellModels: unknown, hex, wedge, prism, pyr, tet, tetWedge, splitHex, sammTrim1-5, hexagonalPrism` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cellModel/cellModel.H` | Maps a named canonical cell geometry to its faces/edges so geometric quantities follow without primitive mesh access. | centre/mag by pyramid decomposition of the model faces about the cell-centre estimate |
| `cellModeller` | `lookup("hex"), lookup("tet"), ... (names from etc/cellModels)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cellModeller/cellModeller.H` | Static registry that reads etc/cellModels and looks a cellModel up by name or index. |  |
| `cellShape` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cellShape/cellShape.H` | An analytical cell: a cellModel plus its vertex labels, able to emit faces/edges and collapse to a degenerate model. | collapse removes duplicate points and re-selects the matching cellModel |
| `degenerateMatcher` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cellMatcher/degenerateMatcher.H` | Tries all hex-degenerate matchers in turn (hex, wedge, prism, pyr, tetWedge, tet) and returns the matching cellShape. |  |
| `edge` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/edge/edge.H` | A pair of point labels with commonVertex, otherVertex, vec, mag and direction-insensitive comparison. | vec = p[end] - p[start]; equality is order-independent |
| `face` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/face/face.H` | A face as an ordered list of point labels, with area, centre, normal, edges, triangulation and comparison operations. | areaAndCentre triangle fan about the point average; areaAndCentreStabilised additionally tracks round-off; sweptVol between two point fields |
| `face::areaInContact` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/face/faceAreaInContact.C` | Fraction of a face area in contact given a signed vertex displacement field (negative = in contact). | per-triangle area weighted by the linearly interpolated in-contact sub-area from the vertex sign changes |
| `face::contactSphereDiameter` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/face/faceContactSphere.C` | Diameter of the sphere touching the face from a given point along a given direction. | d = 2 * ((p - Cf) & n_hat) projected against the face normal, scaled by mag(n) |
| `face::ray / intersection (faceIntersection)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/face/faceIntersection.C` | Ray/line intersection of a polygonal face by decomposing it into triangles. | per-triangle ray intersection with intersection::algorithm and direction settings; nearest hit returned as a pointHit |
| `hexMatcher / prismMatcher / pyrMatcher / tetMatcher / tetWedgeMatcher / wedgeMatcher` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cellMatcher/hexMatcher.H` | Concrete cellMatchers recognising hex, prism, pyramid, tet, tet-wedge and wedge cells. | per-shape face-count/vertex-count screening followed by cellMatcher's permutation walk |
| `oppositeFace` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/face/oppositeFace.H` | Holds the opposite face of a prismatic cell together with its master face index and a validity flag. |  |
| `pyramidPointFaceRef` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/cell/pyramidPointFaceRef.H` | Typedef of pyramid<point, const point&, const face&> used for the face-pyramid decomposition of a cell. |  |
| `tetCell` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/tetCell/tetCell.H` | Four-label tetrahedral cell primitive with fixed face and edge ordering matching tetrahedron and the tet cellModel. |  |
| `triFace` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/meshShapes/triFace/triFace.H` | Fixed-size three-label face with direct area/centre/normal and triangle interoperation. | Sf = 1/2 (b-a)^(c-a); Cf = (a+b+c)/3 |

### parallel decomposition  <sub>(5)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `IOdistributionMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyDistributionMap/IOdistributionMap.H` | distributionMap with registry IO so a decomposition map can be written and re-read. |  |
| `distributionMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyDistributionMap/distributionMap.H` | distributionMapBase extended with transform handling for cyclic/coupled elements. | transformed elements distributed via globalIndexAndTransform transform indices |
| `distributionMapBase` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyDistributionMap/distributionMapBase.H` | Processor-to-processor element exchange map built from global or compact numbering, with a deadlock-free communication schedule. | compact layout: own localSize() elements first, then used-only remote elements sorted by processor; schedule from commSchedule |
| `lagrangianDistributionMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyDistributionMap/lagrangianDistributionMap.H` | Mesh-to-mesh mapping information for particle clouds during redistribution. |  |
| `polyDistributionMap` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyDistributionMap/polyDistributionMap.H` | Mesh-to-mesh mapping after redistribution: sub-maps of the parts sent and construct-maps of the parts received, for points, faces, cells and patches. |  |

### parallel/global addressing  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `ProcessorTopology` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/globalMeshData/processorTopology.H` | Builds the processor-to-processor connection table from processor patches and the resulting patch-swap schedule. |  |
| `commSchedule` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/globalMeshData/commSchedule.H` | Determines a deadlock-free, maximum-overlap ordering of processor-pair communications. | greedy per-iteration scheduling of non-conflicting processor pairs assuming equal cost per exchange |
| `dummyTransform` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/syncTools/dummyTransform.H` | No-op transform functor for syncTools when the synchronised data is transform-invariant. |  |
| `globalIndex` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/globalMeshData/globalIndex.H` | Maps (processor, local index) to a unique global label and back, with gather/scatter helpers. | globalIndex = offset[proci] + localIndex; offset[p+1] = offset[p] + nLocal[p] |
| `globalMeshData` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/globalMeshData/globalMeshData.H` | Parallel-run mesh information: shared/coupled point and edge addressing, global point/face/cell numbering and the master-slave exchange maps for coupled patches. | master election = lowest processor holding the point; slave data gathered to master via distributionMap, combined, then scattered back with transforms |
| `globalPoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/globalMeshData/globalPoints.H` | Topologically determines points shared by more than two processor or cyclic patches, purely by local communication. | iterative merging of point equivalence lists in globalIndexAndTransform numbering until no change on any processor |
| `syncTools` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/syncTools/syncTools.H` | Static helpers to synchronise point, edge and face lists across coupled (processor/cyclic) patches with a combine operator and transform. | value = combineOp over all coupled duplicates, then broadcast back; positions transformed by the patch transformer |

### pointMesh  <sub>(4)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `MapPointField` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointMeshMapper/MapPointField.H` | Field-mapping functor specialisation for pointFields under a polyTopoChangeMap. |  |
| `pointBoundaryMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointBoundaryMesh/pointBoundaryMesh.H` | PtrList of pointPatches constructed one-for-one from the polyBoundaryMesh. |  |
| `pointMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointMesh.H` | Point-based GeoMesh derived on demand from a polyMesh; the mesh for pointFields. | size() = nPoints; RepatchableMeshObject of polyMesh |
| `pointMeshMapper / pointMapper / pointPatchMapper / pointBoundaryMeshMapper` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointMeshMapper/pointMeshMapper.H` | Mapping objects that carry point and point-patch field mapping information across a topology change. | direct addressing where points survive; uniform 1/n weights over master points for inserted points |

### pointPatch (runtime-selectable)  <sub>(12)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `coupledPointPatch / coupledFacePointPatch` | `coupled` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/basic/coupled/coupledPointPatch.H` | Base classes for coupled (processor/cyclic) point patches, providing separation and transform data. |  |
| `cyclicPointPatch` | `cyclic` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/cyclic/cyclicPointPatch.H` | Cyclic point patch with owner/neighbour point ordering derived from the cyclic polyPatch. |  |
| `cyclicSlipPointPatch` | `cyclicSlip` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/cyclicSlip/cyclicSlipPointPatch.H` | Cyclic point patch that additionally applies a slip constraint to point motion. | constraint tensor I - n n applied to point displacement |
| `emptyPointPatch` | `empty` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/empty/emptyPointPatch.H` | Empty-plane point patch constraining motion out of the 2-D plane. |  |
| `facePointPatch` | `patch (selected automatically from the polyPatch type)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/facePointPatch/facePointPatch.H` | pointPatch derived from a polyPatch; holds the runtime-selection table keyed on the polyPatch type. |  |
| `internalPointPatch` | `internal` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/internal/internalPointPatch.H` | Point patch for internal faces exposed by sub-setting. |  |
| `pointPatch` | `basePatch` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/pointPatch/pointPatch.H` | Abstract base representing a set of mesh points with meshPoints(), pointNormals() and constraint application. | applyConstraint reduces the point degrees of freedom by the patch constraint tensor |
| `processorCyclicPointPatch` | `processorCyclic` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/processorCyclic/processorCyclicPointPatch.H` | Processor point patch carrying the referenced cyclic transform. |  |
| `processorPointPatch` | `processor` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/processor/processorPointPatch.H` | Processor point patch with identical point ordering on both sides for parallel point-field swaps. | slave point order reconstructed by reversing all owner-side patch faces |
| `symmetryPointPatch / symmetryPlanePointPatch` | `symmetry, symmetryPlane` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/symmetry/symmetryPointPatch.H` | Symmetry point patches constraining point motion to the symmetry surface/plane. | constraint tensor I - n n per point (patch normal for symmetryPlane, local point normal for symmetry) |
| `wallPointPatch` | `wall` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/derived/wall/wallPointPatch.H` | Point patch corresponding to a wall polyPatch. |  |
| `wedgePointPatch` | `wedge` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/pointMesh/pointPatches/constraint/wedge/wedgePointPatch.H` | Wedge point patch constraining points to the wedge plane (and axis points to the axis). | constraint tensor I - n n with n the wedge patch normal |

### polyMesh core  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `polyBoundaryMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyBoundaryMesh/polyBoundaryMesh.H` | Registered PtrList of polyPatches read from constant/polyMesh/boundary, with group lookup, patch-index-by-name and whichPatch(faceI). |  |
| `polyBoundaryMeshEntries` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyBoundaryMesh/polyBoundaryMeshEntries.H` | Reads the boundary file as raw dictionary entries without constructing patches (used by pre-processing utilities). |  |
| `polyMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMesh.H` | Mesh consisting of general polyhedral cells; the top-level registered, IO-capable mesh object holding points/faces/owner/neighbour, boundary, zones and parallel data. | upper-triangular face ordering (owner<neighbour); demand-driven boundBox, geometricD/solutionD direction vectors, tetBasePtIs, cell search octree |
| `polyMesh from cellShapes (polyMeshFromShapeMesh)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshFromShapeMesh.C` | Constructor helper that builds face/owner/neighbour lists from a list of cellShapes plus boundary face lists. | face matching by sorted point labels; internal faces detected where two cells share a shape face |
| `polyMesh::initMesh (polyMeshInitMesh)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshInitMesh.C` | Initialises cell list and internal-face count from owner/neighbour addressing when constructing without cells. |  |
| `polyMesh::readUpdate / polyMeshIO` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshIO.C` | Re-reads mesh files at a new time instance and reports UNCHANGED/POINTS_MOVED/TOPO_CHANGE/TOPO_PATCH_CHANGE. |  |
| `polyMesh::topoChange / mapMesh / distribute (polyMeshUpdate)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshUpdate.C` | Applies a polyTopoChangeMap / polyMeshMap / polyDistributionMap to the mesh primitives, boundary and zones. |  |
| `preservePatchTypes` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/preservePatchTypes/preservePatchTypes.H` | Reads an existing boundary file (or boundaryPatches dict) so mesh conversion utilities preserve user patch types. |  |

### polyPatch (runtime-selectable)  <sub>(14)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `coupledPolyPatch` | `abstract base (type coupled)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/basic/coupled/coupledPolyPatch.H` | Abstract base for patches coupling regions of the domain (cyclic, processor); provides face-ordering and per-face geometric tolerance. | calcFaceTol: per-face tolerance from face size; matchTolerance-scaled geometric ordering of coupled faces |
| `cyclicPolyPatch` | `type cyclic;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/cyclic/cyclicPolyPatch.H` | Cyclic plane patch coupled to a named neighbourPatch, with an optional rotational or translational transform. | geometric face matching between owner/neighbour halves; transform applied via cyclicTransform |
| `cyclicSlipPolyPatch` | `type cyclicSlip;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/cyclicSlip/cyclicSlipPolyPatch.H` | Copy of cyclic used so a cyclicSlip pointPatch (cyclic plus slip constraint) can be instantiated. |  |
| `cyclicTransform` | `transformType unspecified\|none\|rotational\|translational;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/cyclic/cyclicTransform.H` | Holds and infers the cyclic plane transformation between a coupled patch pair. | rotational: rotation tensor about rotationAxis through rotationCentre by inferred angle; translational: separationVector = Cf_nbr - Cf_own |
| `emptyPolyPatch` | `type empty;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/empty/emptyPolyPatch.H` | Front/back plane patch used to reduce a 3-D mesh to 2-D or 1-D; faces carry no discretisation. | removes the patch-normal direction from geometricD/solutionD |
| `internalPolyPatch` | `type internal;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/internal/internalPolyPatch.H` | Constraint patch holding internal faces exposed by mesh sub-setting. |  |
| `polyPatch` | `type patch;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/polyPatch/polyPatch.H` | Base patch: a contiguous slice of the global face list with its own edge/point addressing; base of two runtime-selection tables (word and dictionary). |  |
| `polyPatchList` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/polyPatch/polyPatchList.H` | PtrList typedef of polyPatch used by polyBoundaryMesh. |  |
| `processorCyclicPolyPatch` | `type processorCyclic; (dictionary table only)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/processorCyclic/processorCyclicPolyPatch.H` | Processor patch that carries the transform of a referenced cyclic patch across a processor split. | applies the referPatch cyclicTransform to transferred data |
| `processorPolyPatch` | `type processor; (dictionary table only)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/processor/processorPolyPatch.H` | Inter-processor boundary patch between myProcNo and neighbProcNo, with matched face and point ordering. | slave-side face ordering recovered by reversing owner-side faces; geometric ordering fallback |
| `symmetryPlanePolyPatch` | `type symmetryPlane;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/symmetryPlane/symmetryPlanePolyPatch.H` | Symmetry constraint for a single flat plane; caches one plane normal for the whole patch. | n = normalised sum(Sf); reflection tensor I - 2 n n; checks planarity within tolerance |
| `symmetryPolyPatch` | `type symmetry;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/symmetry/symmetryPolyPatch.H` | Symmetry constraint for non-planar or multi-plane patches. | per-face reflection tensor I - 2 n n |
| `wallPolyPatch` | `type wall;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/derived/wall/wallPolyPatch.H` | Marks a patch as a solid wall so wall-dependent models (wall functions, wall distance) can find it. |  |
| `wedgePolyPatch` | `type wedge;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/wedge/wedgePolyPatch.H` | Front/back plane pair of an axisymmetric wedge sector; supplies the rotation between the two wedge faces. | faceT = rotation from patch normal to the centre plane; cellT = rotation by the full wedge angle 2*theta between the wedge faces |

### primitive patches  <sub>(17)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `PatchTools` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PatchTools/PatchTools.H` | Static toolbox for patches: normals, edge owners, edge/point sorting, matching, search and parallel gather-and-merge. |  |
| `PatchTools::gatherAndMerge` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PatchTools/PatchToolsGatherAndMerge.C` | Gathers a distributed patch onto the master and merges duplicated points within a tolerance. | mergePoints with mergeDist relative to the patch bounding box |
| `PatchTools::markZones / subsetMap / checkOrientation / matchEdges` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PatchTools/PatchToolsSearch.C` | Flood-fills connected patch zones stopping at marked edges, extracts subset maps and checks face orientation consistency. |  |
| `PatchTools::pointNormals / edgeNormals (PatchToolsNormals)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PatchTools/PatchToolsNormals.C` | Computes parallel-consistent point and edge normals on a coupled patch. | area-weighted sum of adjoining face normals, synchronised across coupled patches then normalised |
| `PatchTools::sortedEdgeFaces / sortedPointEdges` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PatchTools/PatchToolsSortEdges.C` | Orders the faces around an edge and the edges around a point in a consistent rotational sense. | faces sorted by angle about the edge vector using a reference plane |
| `PrimitivePatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatch.C` | Templated surface patch over a face container and point field; computes local addressing, edges, normals and centres for any face/point list combination. |  |
| `PrimitivePatch addressing (edges, faceFaces, edgeFaces, faceEdges, pointEdges, pointFaces)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchAddressing.C` | Builds the patch-local connectivity with internal edges (2 faces) sorted before boundary edges (1 face). | nInternalEdges = count of edges shared by exactly two patch faces |
| `PrimitivePatch meshData / meshEdges / localPoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchMeshData.C` | Produces meshPoints, meshPointMap, localFaces and the mapping of patch edges onto global mesh edges. |  |
| `PrimitivePatch normals and centres` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchPointAddressing.C` | Computes faceCentres, faceAreas, faceNormals and area-weighted pointNormals of the patch. | pointNormal_p = normalised sum over faces f containing p of Sf_f |
| `PrimitivePatch::boundaryPoints` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchBdryPoints.C` | Returns the local points that lie on a boundary (single-face) edge of the patch. |  |
| `PrimitivePatch::checkTopology / checkPointManifold` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchCheck.C` | Detects non-manifold edges (>2 faces) and non-manifold points on a surface patch. |  |
| `PrimitivePatch::edgeLoops` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchEdgeLoops.C` | Traces the boundary edges of a patch into closed point loops. | walk from each unvisited boundary edge following pointEdges until the start point returns |
| `PrimitivePatch::localPointOrder` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchLocalPointOrder.C` | Returns a bandwidth-reducing visit order for patch points and faces. | breadth-first face-to-face walk (front expansion) from an unvisited seed face |
| `PrimitivePatch::projectPoints / projectFaceCentres` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchProjectPoints.C` | Projects the points (or face centres) of this patch onto a target patch along a projection direction, returning pointIndexHits. | ray-triangle intersection with intersection::algorithm (fullRay/halfRay/visible) and direction (vector/contactSphere), with neighbour-face walking on miss |
| `patchZones` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/patchZones.H` | Assigns a zone number to every patch face, splitting where a feature edge or a normal-angle criterion is crossed. | face-to-face flood fill blocked by borderEdge or by cos(angle) between face normals below the given tolerance |
| `primitivePatch / primitiveFacePatch / indirectPrimitivePatch / uindirectPrimitivePatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/primitivePatch.H` | Typedefs of PrimitivePatch over SubList<face>, List<face>, IndirectList<face> and UIndirectList<face> with a referenced pointField. |  |
| `walkPatch` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/walkPatch.H` | Static face-to-face walking utilities that propagate an orientation/index across a patch. |  |

### primitiveMesh  <sub>(7)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `primitiveMesh` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMesh.H` | Cell-face mesh analysis engine: demand-driven topological addressing and primitive geometry for any face/owner/neighbour mesh. | stores nPoints/nEdges/nFaces/nCells; lazily builds all connectivity and geometry, with clearOut/clearGeom invalidation |
| `primitiveMesh cell centres and volumes` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCellCentresAndVols.C` | Computes cellCentres and cellVolumes from face centres and areas. | pyramid decomposition about the cell-centre estimate cEst = mean(Cf): 3V_pyr = Sf&(Cf-cEst), C_pyr = 3/4 Cf + 1/4 cEst; V = sum 3V_pyr/3, C = sum 3V_pyr C_pyr / sum 3V_pyr |
| `primitiveMesh edge addressing (edges, faceEdges, edgeFaces, edgeCells, cellEdges)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMeshEdges.C` | Builds the unique mesh edge list plus faceEdges/edgeFaces/edgeCells/cellEdges, ordered internal edges first. | edges identified by sorted point-pair via pointEdges hashing; nInternalEdges counted by number of boundary points on the edge (0, 0-1, 0-2) |
| `primitiveMesh face centres and areas` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMeshFaceCentresAndAreas.C` | Computes faceCentres, faceAreas (Sf) and magFaceAreas for every face. | per face.areaAndCentre: triangle fan about the point average; Sf = 1/2 sum_i (p_{i+1}-p_i)^(pAvg-p_i); Cf = (1/3) sum_i a_n,i (p_i+p_{i+1}+pAvg) / sum_i a_n,i with a_n = a & unit(Sf) |
| `primitiveMesh topological addressing (cellCells, cellPoints, pointCells, pointPoints, pointFaces, cellFaces)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCellCells.C` | Demand-driven inversion of the face/owner/neighbour lists into cell-cell, cell-point, point-cell, point-point, point-face and cell-face addressing. | list inversion with per-entity size estimates (nFacesPerCell, nPointsPerCell, ...) to pre-size the ragged arrays |
| `primitiveMesh::calcCellShapes` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCalcCellShapes.C` | Recognises each cell as a standard cellShape using degenerateMatcher. |  |
| `primitiveMesh::findCell / findNearestCell / pointInCell` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveMesh/primitiveMeshFindCell.C` | Locates the cell containing or nearest to a point. | pointInCell: sign test of (p - Cf) & Sf against the outward normal for every face of the cell; findNearestCell minimises \|p - C\| |

### primitiveShapes  <sub>(10)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `intersection` | `direction: vector \| contactSphere; algorithm: fullRay \| halfRay \| visible` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/triangle/intersection.H` | Static intersection settings: projection direction, ray algorithm and the relative planar tolerance. |  |
| `line` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/line/line.H` | Templated line segment primitive with centre, vec, mag and nearest-point-between-lines operations. | nearest points from the mutual perpendicular: solve the 2x2 system for the parameters on each segment |
| `objectHit / PointHit (pointHit) / PointIndexHit (pointIndexHit)` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/objectHit/objectHit.H` | Result containers for geometric queries: hit flag, hit point, distance and target index. |  |
| `plane` | `dictionary: point/normal, or planeType with pointAndNormal / embeddedPoints / planeEquation coefficients` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/plane/plane.H` | Geometric plane with distance, side, mirror, line-cut and plane-plane/plane-plane-plane intersection. | ax+by+cz+d=0; signedDistance = n&(p - basePoint); lineIntersect at t = (n&(base - p0))/(n&d) |
| `point / pointField / pointIOField` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/point/point.H` | Typedefs of vector as a spatial point plus the point Field and registered IO field types. |  |
| `point2D / point2DField` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/point2D/point2D.H` | Two-dimensional point typedefs used by 2-D geometric operations. |  |
| `pointHitSort` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/objectHit/pointHitSort.H` | Sortable wrapper pairing a pointHit with an index so multiple intersections can be ordered by distance. | ordered by hit distance |
| `pyramid` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/pyramid/pyramid.H` | Parametric pyramid built from an n-sided base polygon and an apex point. | V = (1/3) \|Sf_base & (apex - Cf_base)\|; centre = 3/4 Cf_base + 1/4 apex |
| `tetrahedron / tetPointRef` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/tetrahedron/tetrahedron.H` | Tetrahedron primitive with volume, centroid, circum-sphere, quality, barycentric mapping, containment and random sampling. | V = (1/6) ((b-a)^(c-a)) & (d-a); quality = V / (8/(9*sqrt(3)) * pi * R_circum^3); barycentric solve of the 3x3 edge matrix |
| `triangle / triPointRef` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/primitiveShapes/triangle/triangle.H` | Triangle primitive: area, centre, normal, circum-circle, quality, swept volume, ray intersection, nearest point and barycentric coordinates. | Sf = 1/2 (b-a)^(c-a); quality = mag(Sf)/(pi R_circum^2 + rootVSmall); classify nearest point as inside/edge/vertex |

### tet decomposition  <sub>(2)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `polyMeshTetDecomposition` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshTetDecomposition/polyMeshTetDecomposition.H` | Finds per-face base points and produces the minimum triangle/tet decomposition of faces and cells with non-degenerate tets. | tet = (cell centre, face base point, facePtA, facePtB); base point chosen to maximise the minimum tet quality above minTetQuality |
| `tetIndices` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/polyMesh/polyMeshTetDecomposition/tetIndices.H` | Named storage of (cell, face, tetPt) indices identifying one tet of a cell decomposition, with tet() and faceTri() accessors. | triangle fan about faceBasePt: facePtA = fcIndex^tetPt(base), facePtB = next |

### wallDist  <sub>(8)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `FvWallInfo / WallLocationData / wallPoint` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/FvWallInfo/FvWallInfo.H` | Wave-transported information holding the nearest wall point (and optional payload) with the update rules used by FvFaceCellWave. | update accepts the new origin if \|C - p_new\|^2 < \|C - p_old\|^2 - tol |
| `fvPatchDistWave` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/fvPatchDistWave/fvPatchDistWave.H` | Namespace of functions that run FvFaceCellWave from a set of patches to compute distance and optionally transported data at cells and patches. | iterative face-cell wave with optional nCorrections of the near-wall values |
| `nearWallDist` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/nearWallDist/nearWallDist.H` | Distance from wall-adjacent cell centres to the wall, evaluated exactly per wall face (used by wall functions). | y = min over the point-neighbouring wall faces of the point-to-triangle nearest distance from the cell centre |
| `patchDistMethod` | `declares patchDistMethod dictionary RTS table; keyword method` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/patchDistMethods/patchDistMethod/patchDistMethod.H` | Abstract runtime-selectable base for computing distance (and normal) to a given set of patches. |  |
| `patchDistMethods::Poisson` | `method Poisson;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/patchDistMethods/Poisson/PoissonPatchDistMethod.H` | Approximate wall distance from the solution of a Poisson equation (Spalding/Fares-Schroder/Tucker). | solve laplacian(yPsi) = -1; y = sqrt(\|grad(yPsi)\|^2 + 2 yPsi) - \|grad(yPsi)\|; n = -grad(yPsi)/\|grad(yPsi)\| |
| `patchDistMethods::advectionDiffusion` | `method advectionDiffusion; advectionDiffusionCoeffs { method Poisson; epsilon 0.1; tolerance 1e-3; maxIter 10; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/patchDistMethods/advectionDiffusion/advectionDiffusionPatchDistMethod.H` | Wall distance from the Eikonal equation in advection form with diffusion smoothing, predicted by a simpler method (Tucker et al.). | div(yPhi,y) - Sp(div(yPhi),y) - epsilon*y*laplacian(y) = 1 with yPhi = interpolate(grad(y))&Sf; epsilon default 0.1; iterated to tolerance/maxIter |
| `patchDistMethods::meshWave` | `method meshWave; (nCorrectors, nRequired)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/patchDistMethods/meshWave/meshWavePatchDistMethod.H` | Fast topological mesh-wave distance to the nearest patch for all cells and boundary faces. | FvFaceCellWave propagation of the nearest wall point; y = \|C - p_wall\|; optional nCorrectors near-wall corrections for mesh distortion |
| `wallDist` | `fvSchemes: wallDist { method meshWave\|Poisson\|advectionDiffusion; nRequired false; }` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/finiteVolume/fvMesh/wallDist/wallDist/wallDist.H` | MeshObject giving the distance-to-wall field y (and optionally the normal-to-wall n) through a runtime-selected patchDistMethod. |  |

### zones  <sub>(6)</sub>

| Name | Keyword | Path | What it computes | Equations |
|---|---|---|---|---|
| `Zone` | `declares the ZoneType dictionary RTS table` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/zones/Zone/Zone.H` | Templated base for mesh zones: a named labelList with a demand-driven inverse lookup map and topo-change renumbering. | lookupMap: index -> zone-local position; topoChange renumbers via map/reverseMap |
| `ZoneList` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/zones/ZoneList/ZoneList.H` | Templated registered PtrList of zones with a whichZone() reverse map and IO/mapping support. |  |
| `cellZone` | `type cellZone; (in constant/polyMesh/cellZones)` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/zones/cellZones/cellZone.H` | A named subset of mesh cells. |  |
| `cellZoneList / faceZoneList / pointZoneList` | `no` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/zones/cellZones/cellZoneList.H` | Concrete registered zone containers held by polyMesh, providing whichZone lookups and zone IO. |  |
| `faceZone` | `type faceZone;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/zones/faceZones/faceZone.H` | A named subset of mesh faces organised as a primitivePatch, with an optional per-face flip map and orientation. | oriented normals: Sf flipped where flipMap is true; builds a primitiveFacePatch over the zone faces |
| `pointZone` | `type pointZone;` | `C:/Users/sdd32/Documents/GitHub/open_cfd/[Foundation-12] src/OpenFOAM/meshes/zones/pointZones/pointZone.H` | A named subset of mesh points. |  |

