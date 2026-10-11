# Reaction mechanisms

The CHEMKIN reader (`rust/src/chem/mechanism.rs`, SPEC-LIT §129) reads a
mechanism file and, when the file has no THERMO block, a separate thermo file.
This folder is where mechanisms that may be shipped with meteor-cfd live.

## What is here, and what is not

**Nothing is vendored yet.**

* **WD1, JL4, BFER2.** These are the one-step Westbrook & Dryer (1981), the
  four-step Jones & Lindstedt (1988) and the two-step Franzelli et al. (2012)
  methane mechanisms (SPEC-LIT §129.6). Each would be written here by hand
  from a read copy of its paper, with the table and page cited. All three
  papers are closed and no copy has been read, so the files do not exist yet.
  No rate number is ever written from memory.
* **GRI-Mech 3.0 and DRM19.** Their files carry a disclaimer and no grant of
  permission to redistribute (SPEC-LIT §129.6 quotes it). They are therefore
  not in this repository. You download them yourself and name them by path.

## Using GRI-Mech 3.0 and DRM19

Download the files from `http://combustion.berkeley.edu/gri-mech/` and
`http://combustion.berkeley.edu/drm/`. Plain HTTP is needed, because the
host's TLS certificate does not cover that name. DRM19 needs GRI-Mech **1.2**'s
thermo file, not 3.0's. Then confirm that each file is the one counted here:

| file | sha256 | species / reactions |
|---|---|---|
| `grimech30.dat` | `a7726208619d27b50e43b9fb6b27dc0a737003c049cc79e0a849bc75301cd763` | 53 / 325 |
| `thermo30.dat` | `8d918b771aa965fe05e8c8f1f7e926a5326ebc90e7fc923ce129dc118b634dcc` | 53 records |
| `drm19.dat` | `1ed93a223186b890ce8b4ec0e3f3918edba56656c276b41fa6c093a05da621ed` | 21 / 84 |
| `thermo12.dat` (GRI 1.2) | `949cfbf6fa6d3c93dd3dee383095187f9aa9eb2e7f9c5f88aeefef37d7b68607` | 32 records |

To run the reader's tests on them, set these four variables:

```
OFGPU_GRI30_MECH=/path/to/grimech30.dat
OFGPU_GRI30_THERMO=/path/to/thermo30.dat
OFGPU_DRM19_MECH=/path/to/drm19.dat
OFGPU_DRM19_THERMO=/path/to/thermo12.dat
```

and run `cargo test --release --test chem_mechanism_files` in `rust/`.

If a pair is not set, `ck_gri30_counts` or `ck_drm19_counts` prints
`SKIPPED` with the reason and returns. That is a skip by name, not a pass.
Set `OFGPU_REQUIRE_MECH=1` to make a missing variable fail the test.

To cite GRI-Mech 3.0, the authors ask you to refer to its web page,
`http://combustion.berkeley.edu/gri-mech/version30/text30.html` (Smith,
Golden, Frenklach, Moriarty, Eiteneer, Goldenberg, Bowman, Hanson, Song,
Gardiner, Lissianski, Qin). For DRM19 they ask for: A. Kazakov and
M. Frenklach, `http://www.me.berkeley.edu/drm/`.
