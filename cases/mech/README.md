# Reaction mechanisms

The CHEMKIN reader (`rust/src/chem/mechanism.rs`, SPEC-LIT §129) reads a
mechanism file and, when the file has no THERMO block, a separate thermo file.
This folder is where mechanisms that may be shipped with meteor-cfd live.

## What is here, and what is not

**Only BFER2 at phi = 1 is here** (`bfer2_phi1.ck`, with its thermo
`tm4513_thermo.dat`, the ten TM-4513 Table II records of
`reference/mcbride-gordon-reno1993/table_II.csv` as a CHEMKIN THERMO file).

* **WD1, JL4, BFER2.** These are the one-step Westbrook & Dryer (1981), the
  four-step Jones & Lindstedt (1988) and the two-step Franzelli et al. (2012)
  methane mechanisms (SPEC-LIT §129.6). Each would be written here by hand
  from a read copy of its paper, with the table and page cited. All three
  papers are closed. No copy of WD1's or JL4's has been read, so their files
  do not exist yet.
  No rate number is ever written from memory. **BFER2 is now here**, as
  `bfer2_phi1.ck`, written from the read author manuscript on CERFACS's site
  (Franzelli, Riber, Gicquel & Poinsot, *Combust. Flame* 159 (2012) 621-637,
  DOI 10.1016/j.combustflame.2011.08.004,
  `https://www.cerfacs.fr/cantera/docs/mechanisms/methane-air/DOCS/BFER/CF_Preccinsta.pdf`,
  sha256 `42c230d28d2e25382928f43489497d8c5d727b746d86225fbb1bb780bd77f982`,
  read 2026-10-18; SPEC-LIT §133.12). Its Tables 3-4 and eqs. (6)-(7) are
  transcribed with the phi corrections f1 and f2 evaluated once at phi = 1
  and folded into the two A factors — exact only when every species has the
  same diffusivity, as the flame case that uses the file has (one Le for
  all, SPEC-LIT §133.11's Lewis route), so the local phi is 1 everywhere at
  all times. It is the phi = 1 file only: no general f(phi) mechanism is
  authored here. WD1 and JL4 stay as written above.
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

## Which case names these files

`cases/sandiaFlameD.jsonc` (Gate CHR-16, SPEC-LIT §134.8) names
`mech/drm19.dat` and `mech/thermo12.dat` — DRM19 with GRI-Mech 1.2's thermo,
the pair tabled above — as its `chemistry` block. The paths are relative to
the run copy, so the user places both files in that copy's `mech/` before
running; the repository itself carries neither, for the reason above.

## volatile1.ck (DST-15)

`volatile1.ck` is the one-step volatile mechanism of a Phase B dust cloud (SPEC-LIT 159.10). Its
species VOL is the corn-starch formula of SPEC-LIT 152.1 times 50 (C300 H394 O249), with the
starch's NASA-7 record times 50 (Huang, Bloching & Lipatnikov 2022, eqs. (1)-(2), Table 1). The
reaction `VOL + 274 O2 => 300 CO2 + 197 H2O` and its rate (A = 1.0E14 cm3/(mol s), E = 30 000
cal/mol, first order in VOL and O2) are **authored and uncalibrated**: no open source gives the
kinetics of corn-starch volatiles (SPEC-LIT 151.8). Use it with `tm4513_thermo.dat`, which holds
the other four species.
