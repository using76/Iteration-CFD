# 13 — The data-centre axis: a corpus, a typed ontology and a reasoner over `ofgpu-datacentre`

**Status:** adopted 2026-09-15. Part of the ontology programme; the machinery it extends is `docs/12`.
Researched by five readers, judged and integrated by Fable 5.1, reviewed and adopted by Opus 5; coded by
GLM-5.3-Flash one unit at a time, each unit verified and committed by a supervisor. Branch `feat/ontology`.

The goal is a basis for automatic thermal-fluid simulation, analysis and design of a data centre. The
user asked for it to be built with GraphRAG, an ontology and KAG together; §2 below decides exactly what
each of those three names becomes here, and the one rule that keeps them from becoming one mush.

The research behind every number is in the five fact sheets kept beside the briefs
(`facts-solver-dc.md`, `facts-graphrag-kag.md`, `facts-standards.md`, `facts-physics.md`,
`facts-workflow-goal.md`). What follows is the integrating judgement, adopted verbatim.

**One decision is the user's, and is marked in §6:** unit D3 changes when a data-centre run *stops*.
The house rule is that solver numerics are never changed for convergence, and a stop rule is control
rather than numerics — but it changes what a run reports, so it is not cut until the user says so.

---

Written 2026-09-15 by the integrating planner after reading all five fact sheets in full, `docs/12`,
`AIP ontology/DOMAIN-MODEL.md` and `AIP ontology/RESEARCH.md` §0–§2, plus seven in-repo checks run
for this document (listed where used). This decides; it does not survey. Marks follow the sheets:
VERIFIED / PARTIAL / UNVERIFIED / ANALYSIS / IN-REPO.

## 0. Where the tree actually is (IN-REPO, checked)

- This working tree is **`feat/ontology` at `e4fac2d`**, not `feat/automesher` (`git rev-parse`; the
  automesher branch is a separate worktree at `Iteration-CFD-mesh`). Solver-dc read `74ba832` in that
  worktree; workflow-goal read `e4fac2d` here. The DC files are byte-identical between them (every
  file:line agreed), so nothing in either sheet is invalidated — but **every unit below targets
  `feat/ontology`**.
- `feat/ontology` carries exactly two commits beyond `main`, both documents (`8d255b3`, `e4fac2d`).
  **None of docs/12's N0–N7, R1–R3, U1–U3 has landed**: no `gui/shared/src/ontology/`, no
  `gui/server/src/ontology/`, no sqlite dependency in any `gui/*/package.json`, and `run.json` still
  has 24 keys with no `gitSha`/`caseId` (`gui/runs/r_9/run.json`). Zero of the 221 runs is a DC run.
- Therefore the workflow-goal scope warning is confirmed: stage 1's traceability half needs N0 first.

## 1. The five sheets judged against each other

| # | Disagreement or unverified assertion | Ruling | Why |
|---|---|---|---|
| J1 | **How many ASHRAE classes.** standards: five (A1–A4 + H1, H1 recommended 18–22 °C). solver-dc, physics, workflow-goal: four, citing `dcmetrics.rs:74-91` as authority; workflow-goal §6.3 calls it "a pick-list of exactly four". | **standards wins.** The repo is incomplete, not authoritative. `AshraeClass` enumerates A1–A4 and `envelope()` hard-codes `(lo, 18.0, 27.0, hi)` (checked). H1 is a 2021 class with a different recommended band, so the hard-coded 18/27 is structurally wrong the day H1 is added. | standards' source is the ASHRAE Journal May 2022 column by TC 9.9 members, VERIFIED (numbers only); the other three explicitly marked their envelope claims PARTIAL and deferred to the repo. |
| J2 | **Which Wibron paper §55.8's gate needs.** solver-dc: *Energies* 12(8) 1473 (2019). workflow-goal E9/§8: *Energies* 2018, 11, 644, "the paper validate.rs names". | **solver-dc wins.** `validate.rs:14962` and `:15563` name **2019, 12(8) 1473** (checked). The 2018 paper is a turbulence-model comparison, useful later, not the ranking answer key. Both CC-BY, both unreachable (MDPI 403). | workflow-goal conflated two papers by the same authors. |
| J3 | **ISO/IEC 30134-2 edition.** physics: "unverifiable, did not retry". standards: ed. 2.0, 2026-01-16, IEC pub. 111538, VERIFIED (catalogue page). | **standards wins.** The wording at `dcmetrics.rs:24-26, :226-231` is stale; the refusal to compute PUE stands (text still paywalled). | Catalogue page is a primary source for edition/date. |
| J4 | **Rate-of-change limits** (20 °C/h, ≤5 °C/15 min, tape 5 °C/h). physics: PARTIAL from hvac.best. standards §1.4: VERIFIED — **for the 2011 whitepaper**, 5th-ed. status unknown. | Treat as **2011-edition values, 5th-ed. UNVERIFIED**. They enter the ontology only as a `StandardClause.publicValue` with `verification: PARTIAL`, never as a gate constant. | Both sheets agree once the edition is named. |
| J5 | **`-permissive` + `tiles[].baffle`** — solver-dc blocker (b): fallback is `PorousJump::Internal{faces: vec![]}`, "looks like no jump at all". | **Resolved, no unit needed.** `case_dc.rs:688-692` calls `refuse_baffle_insertion(..)?;` and **discards** the fallback; the tile then falls through to `t.lower()` and becomes an ordinary **boundary** jump on the same patch, scalars continuous. Behaviour is safe; the comment at `:689-691` ("substitutes the internal-face form") mis-names it. One-line comment fix, fold into D1. | Read the code. |
| J6 | **CAP-YPLUS exists?** physics blocker 4: not established. | **Exists in `lowmach.rs:2269-2354` (per-patch y+ min/mean/max, printed `:2500`), absent from `datacentre.rs`.** Unit D5 ports it; it must not re-derive. | `grep -rn yplus rust/src` (checked). |
| J7 | **SQLite twice.** graphrag §5.5 U1 "SQLite schema" vs docs/12 N2 "the mirror, SQLite". | **One file, one driver decision, owned by N2.** The corpus tables are a later migration (C1) into the same db. graphrag's licence blocker (sqlite-vec, better-sqlite3 — UNVERIFIED by them and by me) moves onto N2's brief; C1 ships without vectors so it is not blocked. | Two SQLite files in one server is the mush this document exists to prevent. |
| J8 | **"The typed ontology is the extractor's prompt, obtained for free"** (graphrag 0.5). | Right mechanism, wrong price: `ObjectTypeDef` does not exist yet (N1 unlanded). It is free *after* N1. | IN-REPO §0. |
| J9 | **RTI sign** (one search summary inverted it). | Closed: LBNL verbatim and the repo agree — <100 % bypass, >100 % recirculation. | standards M6b VERIFIED. |
| J10 | **Kuzay et al. 2022** is "the only reusable validation case with data" (standards). | Accepted, with its limit stated: 11 sensors on one rack's **rear door** at 2 kW — it validates a rack-exhaust profile, **not RCI**. Enters as a `Benchmark`, stage 3 (U-VAL). | Nobody else looked; the claim is VERIFIED and licensed CC BY 4.0. |
| J11 | **GraphRAG verdict rests on F7 (O-RAN, PARTIAL)** and on 2502.11371 numbers from a search summary. | Accept the verdict anyway. If F7 is wrong we have *under-built* (no community summaries); the reversible error. Never quote the F7 or 2502.11371 numbers in product text. | Asymmetric risk. |
| J12 | **Rack-schedule importer in stage 1** (workflow-goal §5). | **Moved to stage 3.** No rack schedule exists in-tree, the column layout is per-client, and `flow` is not in any schedule (their own §2). Stage 1's intake is the case file itself. | Nothing to test it against. |
| J13 | **"Free-cooling ceiling is the single most valuable number in a DC report"** (solver-dc, physics, both attributing S55.4). | Treat as **paraphrase**; neither sheet nor I quoted SPEC-LIT:9689-9712 verbatim. The design conclusion (D4) does not depend on the superlative. | Unverified wording. |

Nothing else conflicts. All five agree on the load-bearing facts: six-patch box, racks as heat zones,
no field output, no stop rule, PUE refused, sweep key parsed-and-unread (`case_dc.rs:351` is the
only reference in `rust/src` and `cases/` — checked), registry `writes.formats:['foam']` false.

## 2. The layering — three layers, one rule

The user asked for GraphRAG + Ontology + KAG. Here is what each name becomes, and what each may write.

| Layer | What it is here | Tables it may WRITE | Tables it may READ | LLM involved? |
|---|---|---|---|---|
| **L1 CORPUS** ("graph RAG the pattern", NOT the Microsoft pipeline) | Clause-aware chunks of documents, BM25 over them, and the **staging** of what an extractor found: `document`, `chunk`, `object_chunk`, `candidate_object`, `candidate_link`, `unmapped_span`. No community detection, no summaries, no Leiden. | **only** `candidate_*`, `unmapped_span`, `object_chunk`, `chunk`, `document` | everything | Yes — the extraction runner, prompted FROM the ObjectTypeDefs (C3). |
| **L2 ONTOLOGY** (docs/12 N1–N5, extended by O1–O4) | Typed objects, links, actions, edit log. The ten levels of §3. | `object_*`, `links`, `edit_log` — **only via `apply(proposal)`** (N4) | everything | Only through `ontology_act` (propose) → human approval → `ontology_apply`. |
| **L3 REASONING** (KAG reduced to its ablation-winning part) | Five operators `Retrieval / Sort / Math / Deduce / Output` planned by the LLM, executed by the store; `Math` includes the solver's own closed forms (`rci_hi`, `rti`, `shi_rhi`) exposed as functions. | **nothing** | L1 + L2 | Yes — emits a plan, never an answer; the store executes the plan. |

**The one rule.** *A row enters L2 only by an action a human approved, and every property in that
action's edit set names either the L1 chunk span it came from (`object_chunk`) or the human who typed
it.* Consequences, spelled out:

- **Automatic from an extracted graph:** candidate rows; `unmapped_span` residue; and an
  `object_chunk` row linking a chunk to an **already-existing** L2 object whose `primaryKey` matched
  **exactly** (M6 exact-resolve). That last one is a corpus-side index, stored in L1, not an L2 link.
- **Human-approved only:** creating any object; setting any property value; any L2 link; any
  `AcceptanceCriterion` threshold; any `AcceptanceVerdict`; anything on `Equation`
  (graphrag R4 — symbol tables are the highest-consequence error site). Batching is allowed: 200
  `StandardClause` stubs from one document are one proposal card.
- **Never, by anyone:** minting an `ObjectTypeDef` (types are code, N1; graphrag R7); writing paywalled
  text into any table (§5); an LLM summary as the source of a numeric requirement (graphrag R2).
- **Why this keeps them apart:** L1 is recall-biased and may be wrong; L2 is precision-biased and
  auditable; L3 has no storage. The mush happens when an extractor writes to `links` or a planner
  caches an answer as an object. Both are refused by name in code (C3 gate asserts zero rows written
  to `object_*`; C7's executor has no write handle).

## 3. The abstraction hierarchy — ten levels

Names in `code` are API names for N1's registry (PascalCase types, camelCase properties). "Identity"
is what makes two objects the same; every type also carries `sourcePath`/`importedAt` per docs/12 B7.
Existing DOMAIN-MODEL types are reused by name; new ones are marked ★.

| Level | IS | Identity (PK) | Required properties | Links leaving it |
|---|---|---|---|---|
| **1 `Concept`** ★ | A named physical idea with no numbers: *buoyancy, recirculation, bypass, stratification, containment, dehumidification, thermal mass*. | slug, e.g. `concept:recirculation` | `slug`, `title`, `oneLine` (≤ 160 chars, our words), `status ∈ {modelled, refused, absent}` | `expressedBy → Equation` (n:n); `refusedBy → Capability` when status=refused |
| **2 `Equation`** ★ | One closed mathematical statement with symbols, units, assumptions, validity range, citation. Never a scheme, mesh or tolerance. | the physics sheet's id, e.g. `EQ-B4`, `EQ-D-RTI` | `id`, `name`, `latex`, `symbols[] {symbol, unit, meaning}` (**human-checked**), `validity`, `specSection?` (`S55.2`), `citation → Document`, `verification` | `instantiatedBy → Model` (1:n); `defines → MetricDef` (1:0..1); `cites → Document \| StandardClause` |
| **3 `Model`** ★ | A choice among equations plus closures and constants that makes a system computable: `kEpsilon`, `standard` wall treatment, `quadratic` fan curve, `Thirds` sample set, `A1` envelope. Constants live HERE, never on a case. | registry name **exactly as the code spells it** (`rust/src/models/registry.rs`, `CurveKind`, `RciSamples`, `AshraeClass`) | `apiName`, `family`, `constants{}` (e.g. `C_mu: 0.09`), `equationIds[]`, `specSection` | `implementedBy → Capability` (n:1); `selectedBy → DcCase` (1:n, link prop: which key) |
| **4 `Capability`** ★ | What a **binary** can do, or **refuses by name**. `kind ∈ {provides, refuses, absent}`. A refusal is a first-class row whose `reason` is the load-bearing field (physics §9.2.6). | `CAP-*` tag for provides/absent; the `unsupported_note` setting string (e.g. `tiles/*/baffle`) or `refuse_*` fn name for refuses | `tag`, `kind`, `module` (`rust/src/fan.rs`), `anchor` (file:line, test-checked), `specSection`, `reason` (required when kind≠provides), `permissiveFallback?` | `offeredBy → Driver` (n:1); `computes → MetricDef` (1:n); `gatedBy → Gate` (n:n) |
| **5 `DcCase`** ★ (sibling of `Case`, `ChtCase`) | One concrete problem: the `.dc.jsonc` document. Children `DcFan`, `DcTile`, `DcRack` are their own types keyed `caseId + patch/name`, named **after the case-format keys**, not after plant they do not model. | `sha256(file bytes)`; title = `name`; `workspacePath` a property | `name`, `room{bounds,cells,boundaries}`, `air{}`, `metrics{ashraeClass,rciSamples,supplyPatch,returnPatch}`, `run{}`, `numerics{}`, `humidity?`, `schemaVersion` | `selects → Model` (n:n); `has → DcFan/DcTile/DcRack` (1:n); `supplies/returns → patch name` (1:1 each); `variantOf → DcCase` (n:0..1, link prop = edit set) |
| **6 `Run`** (existing) | One process. Unchanged from DOMAIN-MODEL; N0 adds `gitSha`, `gitDirty`, `caseId`, `meshId`, `machine`. | `r_<n>` | as today + N0's five | `runs → DcCase` (n:1, via `caseId`); `atCommit → Commit`; `executed → Driver`; `stoppedBy → ConvergenceCriterion` (after D3) |
| **7 `Result`** ★ family | What a run produced, with provenance: `DcMetricReport` (one per run), `FanOperatingPoint` (per fan), `PatchFlowBalance` (per run, **the convergence evidence**), `RackInletTemperature` (per rack), `ModelCaveat` (per caveat kind), `FieldStatistic` (existing; inert for DC until a field writer exists). | `runId + kind + key` | every number carries `runId`, `caseId`, `gitSha`, `nCells`, `iterations`, `continuityRatio`, `dtMeasured` (false today), `ashraeClass`, `rciSamples`, `nSamples` | `measuredIn → Run` (n:1); `caveatedBy → ModelCaveat`; `valueOf → MetricDef` (n:1); `assessedBy → AcceptanceVerdict` |
| **8 `MetricDef`** ★ | The **definition** of a metric, not a value: RCI_HI, RCI_LO, RTI, SHI, RHI, PUE, pPUE, CI, BR, RR, AE, ΔT, WUE… (standards M1–M20). `status ∈ {computed, refused, absent}` — this is how the chat answers "can you compute capture index". | metric API name, e.g. `RCI_HI` | `apiName`, `unit`, `range`, `ideal`, `status`, `reason` (when ≠ computed), `equationId`, `definedIn → Document\|StandardClause` | `computedBy → Capability` (n:0..1); `identityGatedBy → Gate` (55-A); `constrainedBy → AcceptanceCriterion` (1:n) |
| **9 `Standard` / `StandardClause`** ★ | An external document and a **locator** inside it. `text` is **not a property of the type** — it cannot be stored. What is stored: our one-line claim about the clause, and a `publicValue` only when a *public* source states it, with that source's id. | `Standard`: `issuer:number:edition` (e.g. `ASHRAE:TC9.9:5`). `StandardClause`: `standardId + locator` (`ASHRAE:TC9.9:5 / Table 3` or `JRC:CoC:2024 / 5.3.1`) | `Standard`: `issuer, number, edition, title, licence, isPaywalled, accessNote`. `StandardClause`: `locator, claim` (our words), `publicValue?`, `publicSource → Document`, `verification ∈ {VERIFIED,PARTIAL,UNVERIFIED}` | `supersedes → Standard`; `equivalentTo → Standard` (EN 50600-4-2 ↔ ISO 30134-2); `cited-by ← Equation\|MetricDef\|AcceptanceCriterion`; `restatedIn → chunk` (L1, only for public docs) |
| **10 Verdicts** — two types, never merged | **`GateVerdict`** (existing): solver verification against a published measurement, words `MISSES`/`OPEN`/pass, spelled once at `validate.rs:127-131`, test-protected. **`AcceptanceVerdict`** ★: design acceptance, words `MEETS/MARGINAL/FAILS/NOT-ASSESSED`, with `AcceptanceCriterion` ★ = `{metricApiName, operator, threshold, clauseId, setBy, setAt, rationale}`. | `GateVerdict`: `gate + runId`. `AcceptanceCriterion`: uuid. `AcceptanceVerdict`: `reportId + criterionId` | `AcceptanceVerdict`: `word, measured, threshold, margin, assertedBy` (**principal; agent may propose, only a human may apply `assertCompliance`**), `assertedAt`, `notAssessedReason?` | `assessedAgainst → AcceptanceCriterion` (n:1); `citesClause → StandardClause` (via criterion); `about → DcMetricReport` (n:1) |

Rules the coder implements as registry validation (fail at `vitest`, not at runtime):
1. A `Model` constant never appears on a `DcCase`; a `DcCase` value never appears on a `Model`.
2. A `Result` without `runId`, `caseId` and `gitSha` is refused by the importer naming the missing key.
3. `AcceptanceVerdict.word` and `GateVerdict.verdict` are disjoint enums; a test asserts the sets do
   not intersect (extends the existing one-word-per-arm test, `validate.rs` / docs/12:281).
4. `StandardClause` has no `text` property at all; a `PropertyDef` named `text` on it is a registry error.
5. `Capability.anchor` must resolve to an existing `file:line` (test greps the tree).
6. `MetricDef.status = computed` requires `computedBy` to resolve; `refused` requires `reason`.

## 4. What is NOT modelled, by name

| Not modelled | Reason |
|---|---|
| **PUE, pPUE, DCiE, MLC/ELC** as values | The solver refuses PUE by name (`dcmetrics.rs:226`); they are facility annual-energy ratios. `MetricDef` rows exist with `status: refused` and the reason; `PueInputs` are a `Result`. |
| **Community reports, Global Search, Leiden, DRIFT** | Cost ≈1.5× the corpus in generated tokens; answers "what are the themes", a question nobody here asks; on a standards corpus buys +0.04 (J11). |
| **Neo4j, KAG/OpenSPG runtime, leidenalg, python-igraph, any GPL component, a Python sidecar** | Prosperity 3.0.0 vs GPL is structurally incompatible (ANALYSIS, counsel before ship); four Docker services is not a desktop product. |
| **LLM-minted object types; LLM-populated `Equation.symbols`; LLM summaries as numeric sources** | graphrag R7, R4, R2. |
| **Paywalled text of any kind** | Never stored; §5 says how a clause is still referenced. The airatwork 2011-whitepaper mirror is not even fetched. |
| **Rack as solid, containment, leakage path, CRAH coil, chiller, economiser, transient ride-through, capture index, bypass/recirculation ratio, humidity envelope** | Absent from the solver (solver-dc §7). They appear as `Concept.status=absent` / `MetricDef.status=absent` / `Capability.kind=absent` rows with the first file that would change, so the chat can say what is missing and why — no object types promising them. |
| **`CoolingUnit`, `FloorPlan`, `LayerMap`, `RackSchedule`, `LoadScenario`, `DesignStudy`** (workflow-goal §6.3) | Stage 3 at the earliest; nothing in-tree backs them (J12). `DcFan`/`DcTile`/`DcRack` carry what the format carries. |
| **Radiation, multi-species tracers, other turbulence models for DC** | Implemented in the crate, unreachable from `DcCase` — `Capability.kind=absent` with the note "constructed only in tests/validate". |
| **`LogLine`, `Dataset` blobs, `SpecSection` parsing** | Already deferred by DOMAIN-MODEL §6; unchanged. |
| **A DC field writer** | Stage 2 in workflow-goal; not on this axis's first two stages. Until it exists `FieldStatistic` has no DC source and `registry.ts:365` is corrected to say so (D1). |

## 5. The corpus

**Tier A — ingest full text into L1 (public, licence permits):** `rust/SPEC-LIT.md` §52–§55 (ours,
lines 8281–9829, chunked by subsection id); `cases/coldAisle.dc.jsonc` (ours, chunked by top-level
key, its 40 lines of design prose are the best-annotated DC document we own); module headers of
`dcmetrics.rs`, `psychro.rs`, `fan.rs` (ours); `docs/12`, `DOMAIN-MODEL.md`; JRC EU Code of Conduct
2024 (CC BY 4.0, chunk by practice number `5.x.y`) and the 2025 edition once fetched; LBNL
Self-benchmarking Guide (US-Government sponsored, chunk by metric code `A1..B2`); Green Grid WP#35
(public; chunk by equation number); NIST CONTAM manual NISTIR 6921 (public domain); EnergyPlus
Engineering Reference "Coils" and "Chiller:Electric:EIR" pages (DOE documentation); Kuzay et al.
2022 (CC BY 4.0); arXiv 2404.16130 (CC BY 4.0); the five fact sheets themselves (ours — they are the
gold set, §6 C4). **Order of ingestion:** SPEC-LIT §55, coldAisle, JRC CoC, LBNL — the first four
cover every metric and every clause the shipped solver touches.

**Tier B — cite only; a `Standard`/`StandardClause` row, no chunk:** ASHRAE TC 9.9 5th ed. (2021);
ASHRAE 90.4-2022 and Addendum h (public PDF, explicit no-reproduction notice); ASHRAE Journal May
2022 column (numbers only — it is the `publicSource` for A1–A4, H1 envelopes); ISO/IEC 30134-2:2026;
FprEN 50600-4-2:2016 draft (CENELEC copyright; `publicValue` for eq. 1 only); EN 50600-2-3; Herrlin
2005, 2008; Sharma/Bash/Patel 2002; Shrivastava & VanGilder 2007; Tozer 2009; Idelchik; Morton–
Taylor–Turner 1956; Uptime Tier Standard; TIA-942-C; Wibron 2018/2019 (CC-BY but unreachable —
promote to Tier A when someone with access fetches them); EU 2024/1364 (public, EUR-Lex fetch
returned empty — retry; Tier A on success).

**How a clause we cannot reproduce is referenced.** One `StandardClause` row:
`{standardId: "ASHRAE:TC9.9:5", locator: "Table 3 / class A2", claim: "sets the A2 allowable
dry-bulb range at the ITE inlet", publicValue: "10–35 °C", publicSource: "doc:ashrae-journal-2022-05",
verification: "VERIFIED", isPaywalled: true}`. The value is traceable to the **public** column's
chunk via `object_chunk`; the paywalled book is named, located and never quoted. A clause with no
public statement has `publicValue: null` and `verification: UNVERIFIED` (e.g. 90.4 Table 6.5 MLC
maxima) — the chat may say the clause exists and what it governs, and must say the number is not
available. The `licence` and `accessNote` fields ("purchase, CHF 159, IEC pub. 111538") are how a user
learns what to buy.

## 6. The unit cut

Sizes: reading ≤ 6,000 lines, new code ≤ 1,200, one concept, one gate that names the cell it refuses,
coder never commits. **Existing tranche** = docs/12 units, unchanged and required. Prefixes: D = Rust
DC driver, O = ontology TypeScript, C = corpus TypeScript.

| Unit | Tranche | One concept | Depends on | Files touched | Reads (≈lines) / new | Gate that proves it |
|---|---|---|---|---|---|---|
| **N0** | existing | run record edges (`gitSha`,`gitDirty`,`caseId`,`meshId`,`machine`) | — | `gui/server/src/runs/manager.ts`, tests | 1,200 / 200 | as docs/12 |
| **N1** | existing | the definitions + registry validator; **add §3 rules 1,3,4,6 to its validator** | — | `gui/shared/src/ontology/*` | as docs/12 | registry validation test |
| **N2** | existing | the SQLite mirror; **settles the driver (node:sqlite vs better-sqlite3) and records both licences in the brief** | N1 | `gui/server/src/ontology/store.ts` | as docs/12 | as docs/12 |
| **N3** | existing | the importer (runs, sessions, cases, git) | N0, N2 | `gui/server/src/ontology/import.ts` | as docs/12 | rows per type, idempotent |
| **N4** | existing | the action engine, `startRun` first | N3 | `gui/server/src/ontology/actions.ts`, `loop.ts` | as docs/12 | edit log same transaction |
| **N5** | existing | three tools + REST | N4 | five places per docs/12 C | as docs/12 | as docs/12 |
| **R1** | existing | `ofgpu-validate -json` | — | `rust/src/bin/validate.rs` | as docs/12 | one document per run |
| **D1** | new | **`ofgpu-datacentre -json <path>` and `-schema`**: serialise `RoomSolution` + `MetricReport` + notes + the three caveats + `patch_flow` + fan list + `dt_measured`; emit `docs/schema/dc-1.json` via the `automesher.rs:175` pattern; fix `registry.ts:365` to `formats: []`, `json: true`; fix the `-csv` help text and the `:689-691` comment (J5) | — | `rust/src/bin/datacentre.rs`, `rust/src/io/case_dc.rs` (serde derives), `docs/schema/dc-1.json`, `gui/shared/src/registry.ts` + sync test | 810+450+120+60+40 ≈ 1,500 / 400 | JSON `rci_hi` equals the printed one bit-for-bit; schema validates `coldAisle.dc.jsonc` and refuses a case with `supplyTemperatureSweep` misspelt naming the key; registry sync test green |
| **D2** | new | **U-H1 + U-ISO + ΔT + AE**: `AshraeClass::H1` (5–25 allowable, 18–22 recommended), class-dependent recommended band, ISO 30134-2:2026 wording (refusal kept), `delta_t` and airflow efficiency W/cfm as report fields | — | `rust/src/dcmetrics.rs`, `dcmetrics/tests.rs`, `io/case_dc/tests.rs`, SPEC-LIT §55.1/§55.4 text | 525+450+260+80 ≈ 1,300 / 250 | pair test: H1 vs A1 differ in **both** RCI_HI and RCI_LO; `from_name` error names five; Gate 55-A still exact; `tests.rs:216` still finds "ISO/IEC 30134-2" |
| **O1** | new | **the DC types and links of §3** declared in N1's registry (Concept, Equation, Model, Capability, DcCase+DcFan/DcTile/DcRack, Result family, MetricDef, Standard, StandardClause, AcceptanceCriterion, AcceptanceVerdict, ConvergenceCriterion) | N1 | `gui/shared/src/ontology/objects.dc.ts`, `links.dc.ts`, tests | N1 files ≈ 1,000 + this doc / 600 | registry validates; §3 rules 3, 4, 6 each have a failing fixture that is refused naming the type and property |
| **O2** | new | **seed data as code**: every EQ-*, CAP-*, refusal (solver-dc §4.1, 11 rows), Model name, MetricDef M1–M20 with status, Standard S1–S26 with licence, Concept slugs — transcribed from the five sheets | O1 | `gui/shared/src/ontology/seed/dc.ts`, test | five sheets 2,700 + O1 / 900 (data) | every `Capability.anchor` resolves (test greps `rust/src`); every refusal name matches an `unsupported_note` setting or `refuse_*` fn; every `MetricDef.status=computed` resolves to a Capability; no `StandardClause.publicValue` without a `publicSource` |
| **O3** | new | **DC importer**: `*.dc.jsonc` → DcCase/DcFan/DcTile/DcRack (PK sha256); D1's JSON → DcMetricReport, FanOperatingPoint, PatchFlowBalance, RackInletTemperature, ModelCaveat, linked to Run by N0's `caseId` | N3, D1, O1 | `gui/server/src/ontology/import.dc.ts`, tests, one recorded D1 JSON fixture | N3 ≈ 800 + D1 output + O1 / 500 | fixture import yields the exact row counts; a JSON lacking `gitSha` is refused naming the key; re-import is idempotent |
| **O4** | new | **the DC actions**: `setAshraeClass`, `setRackAirflow(basis)`, `scaleFanCurve(from,to,reason)`, `scaleTileLoss`, `setRunBudget`, `acceptRun(continuityRatio, rationale)`, `assessAgainstStandard` (agent may propose), `assertCompliance` (**human principal only**) | N4, O1 | `gui/server/src/ontology/actions.dc.ts`, `loop.ts` approvalPreview, tests | N4 ≈ 900 + O1 / 700 | each propose returns an edit set the card can render; `assertCompliance` with principal `agent` refused by name; `scaleFanCurve` stores `fromCurve` so both appear on the report |
| **C1** | new | **corpus tables** as an N2 migration: `document, chunk, object_chunk, candidate_object, candidate_link, unmapped_span` + FTS5 on `chunk.text` (no vectors) | N2 | `gui/server/src/ontology/store.ts` (migration), `corpus/schema.ts`, tests | N2 ≈ 600 / 400 | migration applies on empty and on an N3-populated db; FTS5 returns a seeded chunk; `candidate_*` has no FK into `object_*` (asserted) |
| **C2** | new | **clause-aware chunker + Tier-A ingest** for SPEC-LIT §52–§55 (by `§n.m` heading), coldAisle (by key), JRC CoC (by `5.x.y`), LBNL (by metric code) | C1 | `gui/server/src/corpus/chunk.ts`, `ingest.ts`, tests, fixtures | 1,600 of SPEC-LIT + 3 docs ≈ 3,500 / 600 | every chunk carries a locator; `"55.4"` resolves to one chunk; concatenated chunks equal the source modulo whitespace (nothing lost); a document without a licence field is refused naming it |
| **C3** | new | **prompt generator + extraction runner**: ObjectTypeDef → prompt + zod validator; one type per pass over GLM-5.3-Flash via the existing provider; writes `candidate_*`, `object_chunk`, `unmapped_span` | O1, C2, N1 | `gui/server/src/corpus/extract.ts`, `prompt.ts`, tests with a recorded LLM transcript | O1 + C2 + provider ≈ 2,500 / 800 | on SPEC-LIT §55 chunks the recorded run proposes MetricDef candidates RCI_HI/RCI_LO/RTI/SHI/RHI each with a span; **zero rows written to `object_*`/`links`** (asserted); an empty-args reply is retried once then refused naming the type (memory: GLM oneOf empty `{}`) |
| **C4** | new | **M1..M7 gate + gold set**: O2's seed for §52–§55 is the gold set; per-property precision/recall | C3, O2 | `gui/server/src/corpus/gate.ts`, `gold/*.json`, tests | C3 + O2 / 500 | each M-check has a fixture refused naming the cell; a `metres` value with no unit in its span is refused quoting the span; the P/R table prints per property |
| **C5** | new | **`ProposeImport` action** + reviewer card: a batch of gated candidates → one proposal → approve → objects, links, edit log in one transaction | C4, N4, N5, U2 | `actions.dc.ts`, `iteration-gui` card, e2e test | 2,000 / 600 | e2e: JRC 5.3.1 chunk → candidate StandardClause → approve → row with `publicValue "10–35 °C"` and `object_chunk` provenance; deny leaves no row |
| **C6** | new | **retrieval**: BM25 ∪ typed lookup ∪ 1-hop → context with clause locators, as a `search` mode on `ontology_query` (no new tool — 79 KB budget) | C2, N5, O3 | `gui/server/src/tools/ontology.ts`, tests | 1,500 / 400 | "what does JRC 5.3.1 require" returns the chunk + locator; "which capability computes RTI" walks MetricDef→Capability to `dcmetrics.rs:183` |
| **D3** | new, **user decision** | **ConvergenceCriterion + divergence stop**: `run.stop{continuityRatioMax, operatingPointGapPctMax, consecutiveReports}`; loop exits on N consecutive passes; non-closure at budget end is a named error (pattern `925ac4a`); JSON records `stoppedBy` | D1 | `datacentre.rs:460-560,620-760`, `case_dc.rs:378-390`, `simple.rs:250-300` | 1,300 / 300 | coldAisle stops before its budget and reports the iteration; a case with contradictory fans stops with a named error, never "converged". **Control, not numerics** — but it changes when a run ends, so per the standing rule the user decides before it is cut |
| **D4** | new | **the sweep → `free_cooling_ceiling`**: implement `supplyTemperatureSweep`, N solves, highest `T_supply` with `RCI_HI = 100` | D1 | `datacentre.rs`, `case_dc.rs:336-352`, `dcmetrics.rs:204-236` | 1,400 / 250 | pair test: with/without sweep → different `PueInputs`; ceiling ≤ sweep hi; refused by name when the supply patch is a tile (no `supplyTemperature` to sweep) |
| **D5** | new | **CAP-YPLUS + CAP-GATE-WALLVALID for DC**: port `lowmach.rs:2269-2354` per-patch y+; add per-patch Gr/Re² (EQ-B4) as `ModelCaveat{kind: wallValidity}` | D1 | `datacentre.rs`, `lowmach.rs` (read only) | 1,200 / 300 | every wall patch of coldAisle reports y+ min/mean/max and a regime word; a synthetic 0.2 m/s case reports `natural` |
| **C7** | new, stage 3 | logical-form solver `Retrieval/Sort/Math/Deduce/Output` over L1+L2; `Math` binds `rci_hi/rti/shi_rhi` | C6, O3 | `gui/server/src/corpus/plan.ts` | — | "which racks in run r_N exceed the A2 recommended band" answered by `Deduce` over `RackInletTemperature` rows, plan printed |

**Parallelism.** Day 1: `N0 ‖ N1 ‖ D1 ‖ D2 ‖ R1` (five coders, disjoint files). Then
`N2 → N3 → N4 → N5` in series while `O1 → O2` and `C1 → C2` run beside it; `O3` waits on N3+D1+O1;
`O4` on N4+O1; `C3` on O1+C2; `C4` on C3+O2; `D3/D4/D5` on D1 only (Rust, parallel with everything
TypeScript). `C5` is the last stage-2 unit (needs U2's card). Stage 3 (not cut here): `DcCase.mesh`,
DXF intake, U-CI, U-BRRR, U-HUM, U-N1, U-VAL, C7.

**Stages.** Stage 1 = N0–N5, R1, D1, D2, O1–O4, C1, C2. Stage 2 = C3–C6, D3 (if approved), D4, D5.

## 7. The first useful thing

**The day D1 lands** (no ontology needed): `ofgpu-datacentre cases/coldAisle.dc.jsonc -json out.json`
writes a report in which every refusal, caveat, the continuity ratio and `dt_measured: false` are
fields, not prose; `-schema` lets `case_validate` check a DC case without a GPU, so the chat can tell a
user their `.dc.jsonc` is wrong before the queue. The registry stops lying about `foam`.

**The day O2 lands** (no solver needed): the chat answers "what does this solver refuse for a data
centre, and why" with the eleven named refusals and their `file:line`; "can you compute capture
index" with `absent — needs CAP-SPECIES wired into datacentre.rs`; "which standard sets the A2 band,
and may we quote it" with the `StandardClause`, its public value, its public source, and
"ASHRAE TC 9.9 5th ed. is paywalled: purchase".

**At the end of stage 1** a user drops a `.dc.jsonc` into the window; the assistant validates it
against `dc-1.json`, proposes `setAshraeClass`/`setRackAirflow` with the basis and rationale captured
as an edit set, the user approves, `startRun` launches with `gitSha` and `caseId` stamped, the
report is imported as a `DcMetricReport` linked Run → DcCase → Commit, and a human sets an
`AcceptanceCriterion` citing a `StandardClause` and gets an `AcceptanceVerdict` that names who
asserted it. Every number in the report names its run; every threshold names its clause; every clause
names its licence. Not yet: extraction from documents (stage 2), a stop rule (D3, user decision), a
picture of the cold aisle (field writer, later), anything the solver does not model (§4).
