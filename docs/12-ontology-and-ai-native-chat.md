# 12 — The ontology, and a chat that can actually do the work

**Status:** adopted 2026-09-15. Planned and reviewed by Opus 5; coded by GLM-5.3-Flash one unit at a time;
every unit verified and committed by an Opus 5 supervisor. Branch `feat/ontology`.

The goal in one sentence: **the product's nouns and verbs become one model, the chat window becomes the
main way to reach them, and a file or an image dropped into that window is an object in the same model.**
Palantir's Ontology is the reference throughout — object types, link types, and above all *actions* — and
the research behind that is in `AIP ontology/RESEARCH.md`, the contract we implement in
`AIP ontology/DOMAIN-MODEL.md` and the fact sheets beside these briefs.

## A. What the survey found, and what it forces

Four readers measured the tree before anything was planned. Six facts decide the shape of this work.

1. **The approval round trip already exists, end to end.** `PendingApproval`, `tool.approve`/`tool.deny`,
   `tool.approval_resolved`, the ten-minute TTL, the policy that refuses to auto-approve a mutating tool —
   all of it ships today. What is missing is only that the approval card's preview should become an
   action's **edit set**, and that the decision, the actor and the time are never persisted. So the
   kinetic layer is a re-packaging job on top of working machinery, not new machinery.
2. **`startRun` invents no rules.** All six of its submission checks already run in `runs/manager.ts` and
   `tools/run.ts`. The first action is a repackaging of validation that exists.
3. **Attachments are broken rather than absent.** `UserContext.attachments` carries workspace paths, and
   the server reads each one and does `buf.subarray(0, 16384).toString('utf8')` — so a dropped PNG reaches
   the model as mojibake, silently. The model→UI image path and the tool→model image path both already
   work. Only user→model is missing, and two type annotations are the whole type-level blocker.
4. **There is no REST entry to the assistant.** Verified live: `POST /api/chat` is 404. Every conversation
   runs over the WebSocket. "Drive it by API" therefore needs one new route, not a redesign.
5. **The tool budget is nearly spent.** Tool definitions are 79 KB per round, and `gui_control` alone is
   35 KB of it. New tools must be small, enum-shaped and few. The weak-model rails (`sanitizeSchema`,
   `envelopeUnion`, `forgive`) are load-bearing and are reused, not re-invented.
6. **The graph's most valuable edges do not exist yet and cannot be back-filled.** Not one of the 221 run
   records carries a commit, a mesh or a case id, and the timing analysis says they cannot be recovered
   after the fact (median 67 minutes to the previous commit; one commit would absorb 66 runs). Gate
   verdicts are printed to stdout and never written. **Mirroring a graph whose edges are absent is
   mirroring a list**, so the writers come before the mirror.

## B. The decisions

1. **Definitions are code, in one place.** `gui/shared/src/ontology/` holds the object types, link types
   and action types as TypeScript declarations with a runtime registry — the shape Palantir's
   ontology-as-code uses, so the same declarations port to a SuperRepo later. Every object type has an
   API name, a primary key, a title property and typed properties; a many-to-many link is backed by its
   own join.
2. **Every write is an action.** Nothing mutates the mirror except by applying an `ActionType`:
   parameters → validation rules → an **edit set** → approval → apply → an **edit log entry**. This is
   the one layer with no equivalent outside Foundry, so it is built deliberately and first among equals.
3. **The agent never writes without approval.** The model gets three tools and no others:
   `ontology_query` (read objects and walk links), `ontology_act` (propose an action; returns the edit set
   it *would* write), `ontology_apply` (apply an approved proposal). `ontology_act` is policy `ask`, so it
   lands in the approval card that already exists, showing the edit set.
4. **An attachment is an object.** Dropping a file mints an `Attachment` (id, name, mime, bytes, sha256,
   blob key), linked to the message and the session. Images go to the model as image blocks **once**, on
   the turn they arrive; afterwards the model refers to them by object id. Nothing base64 is replayed into
   the history every round, because sessions have no compaction and one file is already 1.2 MB.
5. **Vision is probed, not assumed.** z.ai's own documentation says `glm-5.3-flash` is a vision model but
   never says whether the Anthropic-compatible endpoint translates Anthropic image blocks. The attachment
   unit's first task is a probe that records the answer, and the fallback is stated before the run: if the
   endpoint refuses image blocks, images are described server-side and the description is what the model
   sees, with the refusal named in the UI.
6. **Writers before mirror.** The four run-record fields and the gate JSON writer land before the importer,
   so the first import has edges.
7. **The mirror is local and boring.** One SQLite file, one table per object type, one `links` table,
   every row carrying `source_path` and `imported_at` so nothing is invented. It is Stage 0 of
   `AIP ontology/PLAN.md` and it is what ports to a stack later.

## C. The units

Each is one GLM brief: reading list under 6,000 lines with ranges, new code under ~1,200 lines, one
concept, fixed verify commands, the coder never commits. A new tool touches **five** places
(`shared/tools.ts` names, meta and the `summarizeToolCall` case; `server/tools/index.ts` TOOLS; the tool
file; and `loop.ts`'s `approvalPreview`, whose `default` silently ships a blank card) — every tool brief
says so.

### Stream N — the ontology (TypeScript, `Iteration-CFD/gui`)

| unit | what it is | why it is this size |
|---|---|---|
| **N0** | **The run record grows its edges.** `gitSha`, `gitDirty`, `caseId`, `meshId`, `machine` written at launch in `runs/manager.ts`; existing records left alone and marked null by the reader. | ~200 lines, one file plus tests. Forward-only, because the survey proved the past cannot be recovered. |
| **N1** | **The definitions.** `gui/shared/src/ontology/{types.ts, registry.ts, objects.ts, links.ts, actions.ts}`: the `ObjectTypeDef` / `LinkTypeDef` / `ActionTypeDef` / `Proposal` / `EditSet` / `EditLogEntry` interfaces, the registry with its validation (unique API names, a primary key and title per type, link endpoints resolve, action parameters typed), and the first twelve object types and their links declared. No storage, no I/O. | Pure declarations and a validator; the whole unit is testable with `vitest` and nothing else. |
| **N2** | **The mirror.** `gui/server/src/ontology/store.ts`: schema generated from the registry, `put`/`get`/`list`/`query`/`links`, transactions, migration by version stamp. SQLite through Node 22's own driver if it is available, else `better-sqlite3` — the brief settles which after one probe. | One file plus its test; the query surface is deliberately narrow (type + equality filters + link traversal), no SQL from callers. |
| **N3** | **The importer.** Folds runs, sessions, cases, `git log`, the automesher and `step_mesh` summaries and `regions.json` into the mirror, idempotently, reporting rows per type and what it skipped and why. | The shapes are known exactly (the data fact sheet lists every key); the work is mapping, not discovery. |
| **N4** | **The action engine.** `propose(actionApiName, params) → Proposal{editSet, validation}`, `apply(proposalId, actor) → EditLogEntry`, with `startRun` as the first action wired to the existing run manager, and `attachFile` as the second. The approval card's preview becomes the edit set's summary. | The validation already exists in `manager.ts`/`run.ts` and is called, not rewritten. |
| **N5** | **The three tools and the REST surface.** `ontology_query`, `ontology_act`, `ontology_apply` (small enum-shaped schemas, registered in all five places), and `GET /api/ontology/types|objects|object/:id|links`, `POST /api/ontology/propose|apply`. | Tools plus routes plus their tests; the schemas are small by design because 63 % of the tool budget is already spent on two tools. |
| **N6** | **Attachments.** `POST /api/attachments` (multipart), blob storage reusing the dataset `BlobStore`, the `Attachment` object type and its link to message and session, the provider path (probe z.ai's Anthropic endpoint for image blocks first, record the answer, implement the stated fallback), and the fix for the 16 KB utf-8 truncation that currently mangles every binary. | The one unit with an external unknown, so it starts with the probe and carries its fallback in the brief. |
| **N7** | **The API entry to the chat.** `POST /api/chat` (start or continue a session; text plus attachment ids; returns the turn's messages), so a program — not only the browser — can drive the assistant. `ai-drive.ts` switches to it. | One route, one test, one client change. |

### Stream U — the window (`iteration-gui`)

| unit | what it is |
|---|---|
| **U1** | **The composer takes files.** A file picker, drag-and-drop and paste in `AssistantPanel`, uploading through N6 and showing chips; and the asymmetric `MessageView` fixed so a user message renders its image blocks instead of dropping them. |
| **U2** | **The action card.** A proposal renders as what it would change (the edit set, object by object), with approve and deny wired to the existing approval frames; an object returned by `ontology_query` renders as a card whose links are clickable and open the object. |
| **U3** | **The end-to-end test.** Playwright: drop an image and a case file, ask the assistant for something that needs both, approve the action it proposes, and see the object change in the mirror — plus the same drive through `POST /api/chat` with no browser. |

### Stream R — the writers the graph needs (Rust, `Iteration-CFD/rust`)

| unit | what it is |
|---|---|
| **R1** | **`ofgpu-validate -json`.** `serde` on `GateReport`, `Verdict`, `How`, `Uncertainty`; one document per run carrying every gate, the machine, the commit and the wall time. This is `docs/11` S2's other half and `AIP ontology/PLAN.md`'s O2, and the survey calls it the single most valuable unit in the plan. |
| **R2** | **The gate a check belongs to.** `Checks::check` records no gate parent even in memory, so "one row per check under its gate" needs that edge threaded through — a separate, larger unit that follows R1. |
| **R3** | **The mesh↔run edge.** `.meshSummary.json` is the only shape carrying both a run id and mesh numbers and has zero instances; the automesher and `step_mesh` summaries carry no run id. This unit makes both writers emit the identifiers, so the edge exists going forward. |

## D. Order, and what depends on what

```
N0 ─┬─ N1 ─ N2 ─ N3 ─┬─ N4 ─ N5 ─┬─ N6 ─ N7
R1 ─┘                │           └─ U1 ─ U2 ─ U3
R2, R3 ──────────────┘ (parallel, Rust)
```

N0 and R1 first, because the mirror is worth building only once the edges exist. U waits for N5 (the tools
and routes it draws). R2 and R3 run beside the TypeScript stream; they share no file with it.

## E. What this is not

Not a Foundry stack (see `AIP ontology/START-HERE.md`: there is nothing to download, the free Developer
Tier is the route). Not an ontology that spans anything beyond this product. Not a replacement for the
existing tools — the 35 tools stay, and the three ontology tools sit beside them. Not a rewrite of the
chat: the panel, the approval card and the session list all stay and grow.
