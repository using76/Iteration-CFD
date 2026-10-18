#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
scaling.py - re-apply SPEC-LIT 140's scaling rule to `-bench` JSON.

Reads one or more `ofgpu-par -bench strong|weak -json FILE` documents and
prints, per file, a markdown table of the measured rows - P, cells,
cells/rank, the t_iter, L_ex and L_ag medians with their 95 % t intervals,
and the speedup / efficiency columns. The RULE is re-applied here, not
trusted from the JSON: a row whose np exceeds the file's distinct device
count prints "time-shared, not scaling" in its speedup and efficiency
columns even if the JSON carries numbers (the harness can only compute them
while P <= D; this script refuses to launder a time-shared row), P = 1
prints "reference", and the MS-MPI column is "NOT RUN" - the MPI transport
is PAR-06, deferred by the user.

Then the crossover of SPEC-LIT 73.6 with the measured latencies in place of
its free L - equation (140.3) of SPEC-LIT 140.4 - per P > 1 row and per
t_cell:

    n* = (L_ex + 2 L_ag) / t_cell       (PCG: one halo exchange + two
                                         reductions per iteration)
    n* = (2 L_ex + 4 L_ag) / t_cell     (PBiCGStab)

with t_cell from 73.6's table (defaults plume = 2.49e-9 s, gb_800000 =
0.84e-9 s), overridable and extendable with `--t-cell NAME=SECONDS`.

Times are printed in microseconds with 3 significant digits.

Usage:
    python tools/par/scaling.py FILE.json [FILE.json ...]
           [--t-cell NAME=SECONDS ...] [--selftest]

Provenance: ORIGINAL - the re-applied rule and the crossover arithmetic are
this project's own; the definitions they implement are cited in SPEC-LIT
140 (Amdahl 1967; Gustafson 1988; Hoefler & Belli 2015). No external source
code was consulted. No GPL-licensed source was consulted.
"""

import argparse
import io
import json
import math
import sys

TIME_SHARED = "time-shared, not scaling"
MPI_NOT_RUN = "NOT RUN"

DEFAULT_T_CELLS = {"plume": 2.49e-9, "gb_800000": 0.84e-9}


def sig3(seconds):
    """A time in SECONDS, printed in microseconds with 3 significant
    digits (fixed notation where the exponent allows, scientific
    otherwise)."""
    us = seconds * 1.0e6
    if us == 0:
        return "0.00"
    e = math.floor(math.log10(abs(us)))
    if e < -3 or e > 3:
        return f"{us:.2e}"
    return f"{us:.{max(0, 2 - e)}f}"


def interval(stat):
    """`median [95 % CI lo, hi]` for one stats object of the JSON."""
    lo, hi = stat["ci95"]
    return f"{sig3(stat['median'])} [{sig3(lo)}, {sig3(hi)}]"


def claim(row, devices):
    """The speedup / efficiency columns, the rule RE-APPLIED: whatever the
    JSON carries, a row whose np exceeds the distinct device count is
    time-shared and prints the refusal; P = 1 is the reference."""
    np_ = int(row["np"])
    if np_ == 1:
        return "reference", "reference"
    if np_ > devices:
        return TIME_SHARED, TIME_SHARED
    if row.get("speedup") is None or row.get("efficiency") is None:
        # 1 < P <= D, but the run had no P = 1 row to scale against.
        return "no reference", "no reference"
    return f"{float(row['speedup']):.2f}", f"{float(row['efficiency']):.2f}"


def table(doc, out):
    """The markdown table of one `-bench` JSON document."""
    devices = int(doc["devices"])
    rows = doc["rows"]
    print(f"## {doc['bench']} - {doc['device']}, {devices} of "
          f"{doc['device_count']} device(s), precision {doc['precision']}, "
          f"n = {doc['n']}, warm = {doc['warm']}, iters = {doc['iters']}, "
          f"repeats = {doc['repeats']}, {doc['seconds']:.1f} s", file=out)
    print(file=out)
    print("| P | cells | cells/rank | t_iter median [95 % CI] (us) | "
          "L_ex median [95 % CI] (us) | L_ag median [95 % CI] (us) | "
          "speedup | efficiency | MS-MPI |", file=out)
    print("|---|---|---|---|---|---|---|---|---|", file=out)
    for row in rows:
        speedup, efficiency = claim(row, devices)
        print(f"| {row['np']} | {row['cells']} | "
              f"{row['cells_per_rank']:g} | {interval(row['t_iter'])} | "
              f"{interval(row['l_ex'])} | {interval(row['l_ag'])} | "
              f"{speedup} | {efficiency} | {MPI_NOT_RUN} |", file=out)
    print(file=out)


def crossover_nstar(l_ex, l_ag, t_cell):
    """Equation (140.3): the cell count at which the exchanges' latency
    equals one iteration's compute, PCG's and PBiCGStab's exchanges per
    iteration in place of 73.6's free L."""
    pcg = (l_ex + 2.0 * l_ag) / t_cell
    pbicg = (2.0 * l_ex + 4.0 * l_ag) / t_cell
    return int(round(pcg)), int(round(pbicg))


def crossover_table(doc, t_cells, out):
    """Equation (140.3) per P > 1 row and per t_cell, with thousands
    commas."""
    rows = [r for r in doc["rows"] if int(r["np"]) > 1]
    if not rows:
        return
    print("### crossover (140.3): n* = (L_ex + 2 L_ag)/t_cell (PCG), "
          "(2 L_ex + 4 L_ag)/t_cell (PBiCGStab)", file=out)
    print(file=out)
    print("| P | t_cell | n* PCG | n* PBiCGStab |", file=out)
    print("|---|---|---|---|", file=out)
    for row in rows:
        l_ex = row["l_ex"]["median"]
        l_ag = row["l_ag"]["median"]
        for name, t_cell in t_cells.items():
            pcg, pbicg = crossover_nstar(l_ex, l_ag, t_cell)
            print(f"| {row['np']} | {name} = {t_cell:.2e} s | "
                  f"{pcg:,} | {pbicg:,} |", file=out)
    print(file=out)


def report(doc, t_cells, out):
    """One document's whole report: the measured table, then the
    crossover."""
    table(doc, out)
    crossover_table(doc, t_cells, out)


def selftest():
    """The rule, checked on a document that LIES: its P = 2 row carries
    efficiency 0.9 and scaling "measured" even though the file names one
    device - this script must refuse to print the number. Also checks the
    crossover's arithmetic against hand-computed n*. Prints
    "selftest: OK"."""
    doc = {
        "bench": "selftest",
        "device": "the one device",
        "devices": 1,
        "device_count": 1,
        "precision": "f64",
        "n": 8,
        "warm": 1,
        "iters": 2,
        "repeats": 5,
        "seconds": 1.0,
        "rows": [
            {
                "np": 1, "nx": 8, "ny": 8, "cells": 64, "cells_per_rank": 64.0,
                "t_iter": {"samples": [1] * 5, "median": 1.0e-3,
                           "mean": 1.0e-3, "ci95": [1.0e-3, 1.0e-3],
                           "range": [1.0e-3, 1.0e-3]},
                "l_ex": {"samples": [1] * 5, "median": 1.0e-6,
                         "mean": 1.0e-6, "ci95": [1.0e-6, 1.0e-6],
                         "range": [1.0e-6, 1.0e-6]},
                "l_ag": {"samples": [1] * 5, "median": 1.0e-6,
                         "mean": 1.0e-6, "ci95": [1.0e-6, 1.0e-6],
                         "range": [1.0e-6, 1.0e-6]},
                "scaling": "reference", "speedup": None, "efficiency": None,
            },
            {
                "np": 2, "nx": 16, "ny": 8, "cells": 128, "cells_per_rank": 64.0,
                "t_iter": {"samples": [1] * 5, "median": 2.0e-3,
                           "mean": 2.0e-3, "ci95": [2.0e-3, 2.0e-3],
                           "range": [2.0e-3, 2.0e-3]},
                "l_ex": {"samples": [1] * 5, "median": 20.0e-6,
                         "mean": 20.0e-6, "ci95": [20.0e-6, 20.0e-6],
                         "range": [20.0e-6, 20.0e-6]},
                "l_ag": {"samples": [1] * 5, "median": 5.0e-6,
                         "mean": 5.0e-6, "ci95": [5.0e-6, 5.0e-6],
                         "range": [5.0e-6, 5.0e-6]},
                # ON PURPOSE: a time-shared row carrying numbers, which the
                # re-applied rule must refuse to print.
                "scaling": "measured", "speedup": 1.8, "efficiency": 0.9,
            },
        ],
    }
    out = io.StringIO()
    report(doc, {"t": 2.5e-9}, out)
    text = out.getvalue()
    assert "time-shared, not scaling" in text, text
    assert "0.9" not in text, text
    assert "12,000" in text, text
    assert "24,000" in text, text
    pcg, pbicg = crossover_nstar(20.0e-6, 5.0e-6, 2.5e-9)
    assert pcg == 12000 and pbicg == 24000, (pcg, pbicg)
    print("selftest: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="re-apply SPEC-LIT 140's scaling rule to -bench JSON")
    ap.add_argument("files", nargs="*", help="-bench JSON documents")
    ap.add_argument("--t-cell", action="append", default=[],
                    metavar="NAME=SECONDS",
                    help="a per-cell iteration time for the crossover "
                         "(defaults plume=2.49e-9, gb_800000=0.84e-9)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the rule and the crossover, print "
                         "'selftest: OK', read no file")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.files:
        ap.error("at least one -bench JSON file, or --selftest")
    t_cells = dict(DEFAULT_T_CELLS)
    for spec in args.t_cell:
        name, _, value = spec.partition("=")
        t_cells[name.strip()] = float(value)
    for path in args.files:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        report(doc, t_cells, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
