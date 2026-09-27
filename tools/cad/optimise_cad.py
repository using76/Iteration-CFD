#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""optimise_cad.py - stage S11's numeric step (docs/16 §D S11, §I CAD-17): from a scrambled Sobol pool
over the template's active box, a CAD-only prefilter, then one Matern GP per hard mesh/cfd row and
constrained expected improvement on the exactly known objective; every proposal or refusal is a
cad-decision/1 row, and the optimiser abstains instead of guessing.

Constrained expected improvement follows the expected improvement of Jones, Schonlau & Welch 1998
("Efficient Global Optimization of Expensive Black-Box Functions", J. Global Optim. 13:455-492,
DOI 10.1023/A:1008306431147), REIMPLEMENTED from the method, no code consulted. The objective here
is known exactly from the CAD prefilter, so the improvement over the stable design is deterministic
and the acquisition is that improvement times the GPs' probability of feasibility.
The Sobol pool (scipy.stats.qmc, BSD-3-Clause), the CAD-only prefilter through the runner (this
repository's), the Gaussian process (scikit-learn GaussianProcessRegressor, BSD-3-Clause) and the
abstain ids CADOPT-NOFEAS / CADOPT-GPFIT / CADOPT-EXHAUSTED are this repository's. GP features are
the searched reals scaled to [0, 1] plus a one-hot law; ties go to the lower (law, Sobol) index;
candidates are deduplicated by their params sha. On an abstain the row carries requirements_lock,
base_stable_eval_key and the five requirement deltas of gate.deltas, which is what docs/16 §F item 5
hands the LLM: WHICH parameter or law to change and why, only through cad_propose_edit.

THE STAND-IN (docs/16 §I CAD-17's offline gate; docs/16 §H.4 G4). It is not a physics model. It
exists only to exercise the search on the real template box with a known answer, so the numeric
step is gated before any GPU minute is spent. standin_checks locks four rows: REQ-001 the true
normal minimum wall >= 3 mm (a brep row the CAD-only prefilter decides exactly), REQ-002 the a
priori acceleration guard of docs/16 §H.5 as K_max Re_De <= 6, REQ-003 an INVENTED exit
non-uniformity 0.02 (kre/6) exp(-(Lx_over_De - 0.25)/0.25) <= 0.01 that decays along the exit tube,
REQ-004 the objective, the body's axial extent L_over_Di D_i + Lx_over_De D_e, minimised. From the
G4 start poly7 at L_over_Di 0.5 (whose REQ-002/REQ-003 rows fail: the G4 precondition), the gate is
the pool optimum reached within 1 % in at most 18 evaluations for 5 of 5 seeds, with two fresh
processes byte-identical and every decision re-derived from the rows (replay).

Usage:
  python optimise_cad.py --selftest
  python optimise_cad.py --help
  python optimise_cad.py standin STUDY_ID OUT_DIR [MAX_EVALS]
  python optimise_cad.py replay OUT_DIR
  python optimise_cad.py prefilter PARAMS_JSON CHECKS_JSON WORK_DIR H_M OUT_JSON
"""

import copy
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import export  # noqa: E402
import gate  # noqa: E402
import mutate  # noqa: E402
import readiness  # noqa: E402
import reqs  # noqa: E402
import schema  # noqa: E402
import turb_integral  # noqa: E402
import verify  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm as _norm, qmc as _qmc  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern  # noqa: E402

ABSTAIN_IDS = ("CADOPT-NOFEAS", "CADOPT-GPFIT", "CADOPT-EXHAUSTED")
PROPOSE_IDS = ("CADOPT-START", "CADOPT-INIT", "CADOPT-POF", "CADOPT-EI")
PREFILTER_REPRS = ("brep", "stl")        # hard rows the CAD-only prefilter decides exactly
GP_REPRS = ("mesh", "cfd")               # hard rows the GPs model
POF_MIN = 1e-3
SIGMA_FLOOR = 1e-12
GP_MIN_POINTS = 3
LOO_Z_MAX = 10.0
GP_ALPHA = 1e-10
GP_RESTARTS = 2
GP_RANDOM_STATE = 0
LS_BOUNDS = (1e-2, 1e2)
C_BOUNDS = (1e-3, 1e3)
PREFILTER_KEYS = ("params_sha", "status", "rule_id", "objective", "rows", "reason")
CAND_KEYS = ("law", "law_index", "sobol_index", "params", "params_sha")
HISTORY_KEYS = ("params_sha", "params", "verdict")
PROPOSE_KEYS = ("study_id", "n", "decl", "template_sha", "gates", "gates_lock", "checks_doc", "objective",
                "pool", "prefilter", "history", "stable")
DELTA_KEYS = gate.DELTA_KEYS
STANDIN_VERSION = "1"
STANDIN_SEEDS = ("standin-1", "standin-2", "standin-3", "standin-4", "standin-5")
STANDIN_OBJECTIVE = {"quantity": "total_length", "sense": "min"}
STANDIN_START = {"D_i": 0.06, "CR": 9.0, "L_over_Di": 0.5, "law": "poly7", "x_m": None,
                 "Lx_over_De": 0.5, "Lu_over_Di": 0.5, "t_wall": 0.003}
STANDIN_CONTINUOUS_OPTIMUM_M = 0.05287678884639893
POOL_OPTIMA = {"standin-1": 0.05405279815196991, "standin-2": 0.05363740659318864,
               "standin-3": 0.05570419779513031, "standin-4": 0.05449305105023086,
               "standin-5": 0.05691312449518591}
POOL_NPASS = {"standin-1": 797, "standin-2": 796, "standin-3": 796, "standin-4": 796, "standin-5": 796}
POOL_NFEAS = {"standin-1": 346, "standin-2": 347, "standin-3": 340, "standin-4": 344, "standin-5": 347}
STANDIN_DETAIL = "CAD-17 analytic stand-in"
STANDIN_START_PROVENANCE = {"D_i": "user_text", "CR": "user_text", "L_over_Di": "llm_choice",
                            "law": "llm_choice", "Lx_over_De": "default", "Lu_over_Di": "default",
                            "t_wall": "user_text"}
_Z_TRACK = []                      # module-private: every LOO z RMS propose fitted (selftest T6c)
PREFILTER_JSON = "prefilter.jsonl"
DECISIONS_JSON = "decisions.jsonl"
EVALS_JSON = "evals.jsonl"
SUMMARY_JSON = "summary.json"
USAGE = ("usage: python optimise_cad.py --selftest" + chr(10)
         + "       python optimise_cad.py --help" + chr(10)
         + "       python optimise_cad.py standin STUDY_ID OUT_DIR [MAX_EVALS]" + chr(10)
         + "       python optimise_cad.py replay OUT_DIR" + chr(10)
         + "       python optimise_cad.py prefilter PARAMS_JSON CHECKS_JSON WORK_DIR H_M OUT_JSON")


def design_space(decl) -> dict:
    """The searched / fixed split of the declaration (docs/16 §H.1's box): laws in declaration order,
    the searched reals (role design, min < max) each with min, max, default_real and only_when, and
    every other real as fixed. A searched real is ACTIVE for a law when its only_when is null or
    equals law=<that law> (x_m is cubic only)."""
    laws = None
    searched, fixed = [], []
    for p in decl["params"]:
        if p["kind"] == "choice" and p["name"] == "law":
            laws = list(p["choices"])
    if not laws:
        raise ValueError("CADOPT-DOC: the declaration has no choice param named law")
    for p in decl["params"]:
        if p["kind"] != "real":
            continue
        if p["role"] == "design" and p["min"] is not None and p["max"] is not None and p["min"] < p["max"]:
            searched.append({"name": p["name"], "min": float(p["min"]), "max": float(p["max"]),
                             "default_real": float(p["default_real"]), "only_when": p["only_when"]})
        else:
            fixed.append(p["name"])
    return {"laws": laws, "searched": searched, "fixed": fixed}


def seed_of(study_id, law) -> int:
    """The scrambled Sobol seed of one law: crc32 of study_id + law (docs/16 §I CAD-17)."""
    return zlib.crc32((study_id + law).encode("utf-8"))


def params_sha(params) -> str:
    """sha256 of the canonical runner dict: every declared name present, reals as floats, law a
    string, an inactive x_m as None. Equals the params_sha export.py writes into probes.json."""
    return common.sha256_of(params)


def _active_names(ds, law) -> list:
    """The searched reals active for one law, in searched order."""
    tag = "law=" + law
    return [s for s in ds["searched"] if s["only_when"] is None or s["only_when"] == tag]


def pool(study_id, decl, fixed, n) -> list:
    """The scrambled Sobol pool: n per law, seeded seed_of(study_id, law), in the law's active box.

    Row i maps each active searched real (searched order) to lo + u (hi - lo); an inactive real is
    None; `fixed` supplies the fixed reals. Candidates come back in (law index, Sobol index) order,
    each with exactly CAND_KEYS."""
    ds = design_space(decl)
    if not isinstance(n, int) or isinstance(n, bool) or n < 2 or (n & (n - 1)) != 0:
        raise ValueError("CADOPT-DOC: pool size %r is not a power of two >= 2" % (n,))
    m = int(round(np.log2(n)))
    out = []
    for li, law in enumerate(ds["laws"]):
        active = _active_names(ds, law)
        u = _qmc.Sobol(d=len(active), scramble=True, seed=seed_of(study_id, law)).random_base2(m=m)
        for i in range(n):
            params = dict((s["name"], None) for s in ds["searched"])
            for k, s in enumerate(active):
                params[s["name"]] = s["min"] + float(u[i, k]) * (s["max"] - s["min"])
            for name in ds["fixed"]:
                params[name] = fixed[name]
            params["law"] = law
            out.append({"law": law, "law_index": li, "sobol_index": i,
                        "params": params, "params_sha": params_sha(params)})
    return out


def features(params, decl) -> list:
    """The GP features: each searched real scaled to [0, 1] (an inactive real takes default_real),
    then one 0/1 per law in law order; 8 for the nozzle."""
    ds = design_space(decl)
    law = params["law"]
    feats = []
    for s in ds["searched"]:
        active = s["only_when"] is None or s["only_when"] == "law=" + law
        v = float(params[s["name"]]) if active else s["default_real"]
        feats.append((v - s["min"]) / (s["max"] - s["min"]))
    feats.extend(1.0 if l == law else 0.0 for l in ds["laws"])
    return feats


def optimiser_provenance(decl) -> dict:
    """The provenance of a pool candidate: role intent -> user_text, a fixed design real
    (min == max) -> default, everything the pool chose (law and the active searched reals) ->
    optimiser."""
    prov = {}
    for p in decl["params"]:
        if p["kind"] == "choice":
            prov[p["name"]] = "optimiser"
        elif p["role"] == "intent":
            prov[p["name"]] = "user_text"
        elif p["min"] is not None and p["max"] is not None and p["min"] == p["max"]:
            prov[p["name"]] = "default"
        else:
            prov[p["name"]] = "optimiser"
    return prov


def params_doc(params, decl, template_sha, requirements_lock, base_stable_eval_key, provenance) -> dict:
    """A cad-params/1 doc: one value per declared param in declaration order, OMITTING an inactive
    real; validated, and refused as CADOPT-DOC on any schema error."""
    ds = design_space(decl)
    law = params["law"]
    values = []
    for p in decl["params"]:
        if p["kind"] == "choice":
            values.append({"name": p["name"], "real": None, "choice": params[p["name"]],
                           "provenance": provenance[p["name"]]})
            continue
        active = p["only_when"] is None or p["only_when"] == "law=" + law
        if not active:
            continue
        values.append({"name": p["name"], "real": float(params[p["name"]]), "choice": None,
                       "provenance": provenance[p["name"]]})
    doc = {"schema": "cad-params/1", "template_id": decl["template_id"], "template_sha": template_sha,
           "requirements_lock": requirements_lock, "base_stable_eval_key": base_stable_eval_key,
           "values": values}
    errs = schema.errors(doc, "cad-params/1")
    if errs:
        raise ValueError("CADOPT-DOC: %s" % (errs[0],))
    return doc


def gp_checks(checks_doc) -> list:
    """The checks the GPs model: hardness hard and repr mesh or cfd, in check order."""
    return [c for c in checks_doc["checks"] if c["hardness"] == "hard" and c["repr"] in GP_REPRS]


def _training(check, history, feats) -> tuple:
    """(X, y) for one constraint: every history entry whose verdict row for the check's req_id has a
    measured m; y is gate._margin (m - 0.5 for is_true), so positive means passing. `feats` is the
    feature rows of `history`, computed once by the caller."""
    X, y = [], []
    for h, f in zip(history, feats):
        row = None
        for v in h["verdict"]["verdicts"]:
            if v["req_id"] == check["req_id"]:
                row = v
                break
        if row is None or row["m"] is None:
            continue
        X.append(f)
        y.append(row["m"] - 0.5 if check["op"] == "is_true" else gate._margin(check, row["m"]))
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _gpr(kernel) -> GaussianProcessRegressor:
    """The one GPR constructor of this module: normalize_y, the fixed alpha, restarts and seed."""
    return GaussianProcessRegressor(kernel=kernel, alpha=GP_ALPHA, normalize_y=True,
                                    n_restarts_optimizer=GP_RESTARTS, random_state=GP_RANDOM_STATE)


def fit_constraint(X, y) -> tuple:
    """(model or None, loo) for one constraint's training data; loo = {"n", "z_rms", "ok", "reason"}.

    The Matern 2.5 kernel with the fixed bounds is fitted once; leave-one-out then holds the fitted
    kernel_ fixed (optimizer=None) and predicts each held-out point from the other n-1. Refused when
    fewer than GP_MIN_POINTS trainable points, the fit raises, any mu/sd/z is not finite, or the z
    RMS exceeds LOO_Z_MAX (CADOPT-GPFIT)."""
    n = len(y)
    loo = {"n": int(n), "z_rms": None, "ok": False, "reason": ""}
    if n < GP_MIN_POINTS:
        loo["reason"] = "only %d trainable points, at least %d are needed" % (n, GP_MIN_POINTS)
        return None, loo
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    kernel = ConstantKernel(1.0, constant_value_bounds=C_BOUNDS) * \
        Matern(length_scale=np.ones(d), length_scale_bounds=LS_BOUNDS, nu=2.5)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _gpr(kernel).fit(X, y)
            mu, sd = [], []
            for i in range(n):
                keep = np.arange(n) != i
                sub = GaussianProcessRegressor(kernel=model.kernel_, alpha=GP_ALPHA, normalize_y=True,
                                               optimizer=None).fit(X[keep], y[keep])
                p, s = sub.predict(X[i:i + 1], return_std=True)
                mu.append(float(p[0]))
                sd.append(float(s[0]))
    except Exception as e:                                  # noqa: BLE001 - any fit failure abstains
        loo["reason"] = "the Gaussian process fit raised %s: %s" % (type(e).__name__, e)
        return None, loo
    z = [(y[i] - mu[i]) / max(sd[i], SIGMA_FLOOR) for i in range(n)]
    if not (np.isfinite(mu).all() and np.isfinite(sd).all() and np.isfinite(z).all()):
        loo["reason"] = "the leave-one-out prediction is not finite"
        return None, loo
    z_rms = float(np.sqrt(np.mean(np.square(z))))
    loo["z_rms"] = z_rms
    if z_rms > LOO_Z_MAX:
        loo["reason"] = "leave-one-out z RMS %r exceeds %r" % (z_rms, LOO_Z_MAX)
        return model, loo
    loo["ok"] = True
    return model, loo


def _pof(models, checks, X) -> float:
    """The probability of feasibility: the product over the GP constraints (models in check order)
    of the predicted margin being positive, norm.cdf(mu / max(sd, SIGMA_FLOOR))."""
    p = 1.0
    for model in models:
        mu, sd = model.predict(np.asarray([X], dtype=float), return_std=True)
        p *= float(_norm.cdf(float(mu[0]) / max(float(sd[0]), SIGMA_FLOOR)))
    return p


def pick(scored) -> int:
    """The INDEX of the maximum acquisition; ties to the lower (law index, Sobol index)."""
    if not scored:
        raise ValueError("CADOPT-DOC: pick over an empty list")
    best = 0
    for i in range(1, len(scored)):
        if scored[i][0] > scored[best][0] \
                or (scored[i][0] == scored[best][0]
                    and (scored[i][1], scored[i][2]) < (scored[best][1], scored[best][2])):
            best = i
    return best


def _ctx_error(ctx) -> str:
    """The empty string when ctx is a valid propose context, else the CADOPT-DOC message."""
    if not isinstance(ctx, dict) or sorted(ctx.keys()) != sorted(PROPOSE_KEYS):
        return "ctx must have exactly %s" % (", ".join(PROPOSE_KEYS),)
    history = ctx["history"]
    if not isinstance(history, list) or not history:
        return "history must be a non-empty list (the start is evaluated first)"
    for h in history:
        if not isinstance(h, dict) or sorted(h.keys()) != sorted(HISTORY_KEYS):
            return "history entries must have exactly %s" % (", ".join(HISTORY_KEYS),)
    pool_rows, pf = ctx["pool"], ctx["prefilter"]
    if not isinstance(pool_rows, list) or not pool_rows:
        return "pool must be a non-empty list of CAND_KEYS candidates"
    for c in pool_rows:
        if not isinstance(c, dict) or sorted(c.keys()) != sorted(CAND_KEYS):
            return "pool candidates must have exactly %s" % (", ".join(CAND_KEYS),)
        if not isinstance(pf, dict) or c["params_sha"] not in pf:
            return "the prefilter map lacks the pool candidate %s" % (c["params_sha"],)
    for rec in pf.values():
        if not isinstance(rec, dict) or sorted(rec.keys()) != sorted(PREFILTER_KEYS):
            return "prefilter records must have exactly %s" % (", ".join(PREFILTER_KEYS),)
    obj = ctx["objective"]
    if obj is not None and (not isinstance(obj, dict) or sorted(obj.keys()) != ["quantity", "sense"]
                            or obj["sense"] not in ("min", "max")):
        return "objective must be None or {quantity, sense} with sense min or max"
    n_obj = sum(1 for c in ctx["checks_doc"]["checks"] if c["hardness"] == "objective")
    if (obj is not None) != (n_obj == 1):
        return "objective %r does not match the %d objective checks" % (obj, n_obj)
    if ctx["stable"] is not None:
        errs = schema.errors(ctx["stable"], "cad-verdict/1")
        if errs:
            return errs[0]
    return ""


def _row(ctx, decision, rule_id, reason, cand=None, ei=None) -> dict:
    """One validated cad-decision/1 row: an abstain carries gate.deltas of the last history verdict
    against the stable, a propose carries five empty arrays."""
    stable = ctx["stable"]
    base = None if stable is None else stable["eval_key"]
    row = {"schema": "cad-decision/1", "study_id": ctx["study_id"], "n": ctx["n"],
           "after_iteration": len(ctx["history"]) - 1, "decision": decision, "rule_id": rule_id,
           "stable_eval_key": base, "candidate_eval_key": None, "params_sha": None, "law": None,
           "sobol_index": None, "ei": ei, "reason": reason, "gates_lock": ctx["gates_lock"],
           "requirements_lock": ctx["checks_doc"]["requirements_lock"], "base_stable_eval_key": base}
    if decision == "abstain":
        row.update(gate.deltas(ctx["history"][-1]["verdict"], stable))
    else:
        for k in DELTA_KEYS:
            row[k] = []
    if cand is not None:
        row["params_sha"] = cand["params_sha"]
        row["law"] = cand["law"]
        row["sobol_index"] = cand["sobol_index"]
    errs = schema.errors(row, "cad-decision/1")
    if errs:
        raise ValueError("CADOPT-DOC: %s" % (errs[0],))
    return row


def _propose_doc(ctx, cand) -> dict:
    """The cad-params/1 doc of a proposed pool candidate, with the optimiser provenance."""
    return params_doc(cand["params"], ctx["decl"], ctx["template_sha"],
                      ctx["checks_doc"]["requirements_lock"],
                      None if ctx["stable"] is None else ctx["stable"]["eval_key"],
                      optimiser_provenance(ctx["decl"]))


def start(study_id, n, decl, template_sha, gates_lock, checks_doc, params, provenance) -> tuple:
    """The CADOPT-START decision: the study's given start vector, refused as CADOPT-DOC when its
    names are not exactly the declared ones, its law is not a choice, an active searched real is
    outside its box, or a fixed real misses its min == max value. Returns (params doc, row)."""
    names = sorted(p["name"] for p in decl["params"])
    if sorted(params.keys()) != names:
        raise ValueError("CADOPT-DOC: the start's parameters %s are not exactly the declared %s"
                         % (sorted(params.keys()), names))
    law = params["law"]
    law_p = [p for p in decl["params"] if p["kind"] == "choice" and p["name"] == "law"][0]
    if law not in law_p["choices"]:
        raise ValueError("CADOPT-DOC: the start's law %r is not one of %s" % (law, law_p["choices"]))
    ds = design_space(decl)
    for s in ds["searched"]:
        active = s["only_when"] is None or s["only_when"] == "law=" + law
        if active and not (s["min"] <= float(params[s["name"]]) <= s["max"]):
            raise ValueError("CADOPT-DOC: the start's %s value %r is outside its box [%r, %r]"
                             % (s["name"], params[s["name"]], s["min"], s["max"]))
    for p in decl["params"]:
        if p["kind"] == "real" and p["min"] is not None and p["min"] == p["max"] \
                and float(params[p["name"]]) != float(p["min"]):
            raise ValueError("CADOPT-DOC: the fixed %s must be %r, got %r"
                             % (p["name"], float(p["min"]), params[p["name"]]))
    doc = params_doc(params, decl, template_sha, checks_doc["requirements_lock"], None, provenance)
    sha = params_sha(params)
    row = {"schema": "cad-decision/1", "study_id": study_id, "n": n, "after_iteration": None,
           "decision": "propose", "rule_id": "CADOPT-START", "stable_eval_key": None,
           "candidate_eval_key": None, "params_sha": sha, "law": law, "sobol_index": None,
           "ei": None,
           "reason": "the study starts from the given vector: law %s, params_sha %s" % (law, sha),
           "gates_lock": gates_lock, "requirements_lock": checks_doc["requirements_lock"],
           "base_stable_eval_key": None}
    for k in DELTA_KEYS:
        row[k] = []
    errs = schema.errors(row, "cad-decision/1")
    if errs:
        raise ValueError("CADOPT-DOC: %s" % (errs[0],))
    return doc, row


def _init_pick(passed, remaining):
    """The first element of the init sequence (for j = 0, 1, ..: each law's j-th PASSED candidate in
    Sobol order) that is in remaining; None when the sequence is exhausted."""
    rem = set(c["params_sha"] for c in remaining)
    by_law = {}
    for c in passed:                          # pool order is (law index, Sobol index)
        by_law.setdefault(c["law_index"], []).append(c)
    j = 0
    while True:
        more = False
        for li in sorted(by_law):
            lst = by_law[li]
            if j < len(lst):
                more = True
                if lst[j]["params_sha"] in rem:
                    return lst[j]
        if not more:
            return None
        j += 1


def _objective_row(stable):
    """The stable's objective verdict row (hardness objective), or None."""
    for v in stable["verdicts"]:
        if v["hardness"] == "objective":
            return v
    return None


def propose(ctx) -> tuple:
    """One numeric step of S11 (docs/16 §D S11): prefilter pass set, dedup against history, the
    init / pof / ei phase, per-constraint GPs with leave-one-out, and the acquisition pick. Returns
    (params doc or None, cad-decision/1 row); an abstain carries the requirement deltas."""
    bad = _ctx_error(ctx)
    if bad:
        raise ValueError("CADOPT-DOC: %s" % (bad,))
    gates, pf, stable, obj = ctx["gates"], ctx["prefilter"], ctx["stable"], ctx["objective"]
    passed = [c for c in ctx["pool"] if pf[c["params_sha"]]["status"] == "pass"]
    if not passed:
        return None, _row(ctx, "abstain", "CADOPT-NOFEAS",
                          "the CAD prefilter passed none of the %d pool candidates" % (len(ctx["pool"]),))
    evaluated = set(h["params_sha"] for h in ctx["history"])
    remaining = [c for c in passed if c["params_sha"] not in evaluated]
    if len(ctx["history"]) < gates["init_design"]:
        phase = "init"
    elif obj is not None and stable is not None and stable["design_verdict"] == "feasible":
        phase = "ei"
    else:
        phase = "pof"
    f_best = None
    if phase == "ei":
        f_best = _objective_row(stable)["m"]
        sense = obj["sense"]

        def improves(c):
            f = pf[c["params_sha"]]["objective"]
            return f is not None and (f < f_best if sense == "min" else f > f_best)

        remaining = [c for c in remaining if improves(c)]
    if not remaining:
        if phase == "ei":
            why = ("no unevaluated passed candidate strictly improves on the stable objective %r: %d"
                   " passed against %d history entries" % (f_best, len(passed), len(ctx["history"])))
        else:
            why = ("every passed candidate has already been evaluated: %d passed against %d"
                   " history entries" % (len(passed), len(ctx["history"])))
        return None, _row(ctx, "abstain", "CADOPT-EXHAUSTED", why)
    if phase == "init":
        cand = _init_pick(passed, remaining)
        if cand is None:
            return None, _row(ctx, "abstain", "CADOPT-EXHAUSTED",
                              "the init sequence is exhausted with %d passed candidates" % (len(passed),))
        return _propose_doc(ctx, cand), _row(
            ctx, "propose", "CADOPT-INIT",
            "the initial design: candidate (%d, %d) of law %s, params_sha %s"
            % (cand["law_index"], cand["sobol_index"], cand["law"], cand["params_sha"]), cand)
    checks = gp_checks(ctx["checks_doc"])
    feats = [features(h["params"], ctx["decl"]) for h in ctx["history"]]
    models = []
    for c in checks:
        X, y = _training(c, ctx["history"], feats)
        model, loo = fit_constraint(X, y)
        if loo["z_rms"] is not None:
            _Z_TRACK.append(loo["z_rms"])
        if not loo["ok"]:
            return None, _row(ctx, "abstain", "CADOPT-GPFIT",
                              "constraint %s with n=%d: %s" % (c["req_id"], loo["n"], loo["reason"]))
        models.append(model)
    pofs = dict((c["params_sha"], _pof(models, checks, features(c["params"], ctx["decl"])))
                for c in remaining)
    if phase == "pof":
        scored = [(pofs[c["params_sha"]], c["law_index"], c["sobol_index"]) for c in remaining]
        best = max(s[0] for s in scored)
        if best < POF_MIN:
            return None, _row(ctx, "abstain", "CADOPT-NOFEAS",
                              "the best probability of feasibility %r is below %r" % (best, POF_MIN))
        cand = remaining[pick(scored)]
        return _propose_doc(ctx, cand), _row(
            ctx, "propose", "CADOPT-POF",
            "candidate (%d, %d) of law %s has probability of feasibility %r, the highest remaining"
            % (cand["law_index"], cand["sobol_index"], cand["law"], pofs[cand["params_sha"]]), cand)
    scored = [(abs(pf[c["params_sha"]]["objective"] - f_best) * pofs[c["params_sha"]],
               c["law_index"], c["sobol_index"]) for c in remaining]
    idx = pick(scored)
    cand = remaining[idx]
    acq = scored[idx][0]
    return _propose_doc(ctx, cand), _row(
        ctx, "propose", "CADOPT-EI",
        "candidate (%d, %d) of law %s has expected improvement %r: objective gap %r times"
        " probability of feasibility %r"
        % (cand["law_index"], cand["sobol_index"], cand["law"], acq,
           abs(pf[cand["params_sha"]]["objective"] - f_best), pofs[cand["params_sha"]]),
        cand, acq)


def _pf_record(params, status, rule_id, objective, rows, reason) -> dict:
    """One prefilter record: exactly PREFILTER_KEYS."""
    return {"params_sha": params_sha(params), "status": status, "rule_id": rule_id,
            "objective": objective, "rows": rows, "reason": reason}


def _probes_row(out_dir, check) -> dict:
    """The check's ONE probes.json row record: same primitive, same where list (mutate.measurements)."""
    rows = common.read_json(os.path.join(out_dir, "probes.json"))["rows"]
    hit = [r for r in rows if r["primitive"] == check["primitive"]
           and list(r["where"]) == list(check["args"]["where"])]
    if len(hit) != 1:
        raise RuntimeError("probes.json has %d rows for %s (%s), want exactly 1"
                           % (len(hit), check["req_id"], check["primitive"]))
    return hit[0]["record"]


def cad_prefilter(params, checks_doc, work_dir, h_m, template_path=export.TEMPLATE) -> dict:
    """The real CAD-only prefilter (docs/16 §D S3-S5): the runner build and export, then every hard
    brep/stl row plus the objective check judged on the probes.json records, then readiness. The
    first failing hard row refuses with its req id; exactly PREFILTER_KEYS; an exception becomes an
    error record ruled CADOPT-PREFILTER."""
    try:
        os.makedirs(work_dir, exist_ok=True)
        pipe = export.run_pipeline(template_path, params, work_dir)
        if pipe["status"] != "ok":
            status = "error" if pipe["status"] == "error" else "refused"
            return _pf_record(params, status, pipe["rule"], None, {},
                              "the build stage refused with %s" % (pipe["rule"],))
        rows, first_fail, obj_val = {}, None, None
        for c in checks_doc["checks"]:
            if not ((c["hardness"] == "hard" and c["repr"] in PREFILTER_REPRS)
                    or c["hardness"] == "objective"):
                continue
            record = _probes_row(work_dir, c)
            v = verify.judge(c, record)
            if c["hardness"] == "objective":
                raw = record["value"]
                obj_val = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) \
                    and math.isfinite(float(raw)) else None
            else:
                rows[c["req_id"]] = v
                if v["verdict"] != "pass" and first_fail is None:
                    first_fail = (c["req_id"], v["verdict"])
        if first_fail is not None:
            return _pf_record(params, "refused", first_fail[0], obj_val, rows,
                              "hard row %s is %s" % (first_fail[0], first_fail[1]))
        ready = readiness.check(work_dir, h_m)
        if ready["status"] != "ready":
            return _pf_record(params, "refused", ready["rule"], obj_val, rows,
                              "readiness refused with %s" % (ready["rule"],))
        return _pf_record(params, "pass", None, obj_val, rows,
                          "the build, %d hard brep/stl rows and readiness pass" % (len(rows),))
    except Exception as e:                                  # noqa: BLE001 - any failure is a record
        return _pf_record(params, "error", "CADOPT-PREFILTER", None, {},
                          "%s: %s" % (type(e).__name__, e))


_STANDIN_CHECKS = (
    {"req_id": "REQ-001", "primitive": "meridian_min_wall",
     "args": {"feature": None, "where": ["wetted", "outer"], "Re": None, "level": None},
     "op": ">=", "lo": 0.003, "hi": None, "tol": 1e-06, "u": 1e-08, "hardness": "hard", "repr": "brep"},
    {"req_id": "REQ-002", "primitive": "standin_kre",
     "args": {"feature": None, "where": ["wall_contraction"], "Re": None, "level": None},
     "op": "<=", "lo": None, "hi": 6.0, "tol": 0.0, "u": 1e-09, "hardness": "hard", "repr": "cfd"},
    {"req_id": "REQ-003", "primitive": "standin_nonuni",
     "args": {"feature": None, "where": ["exit_plane"], "Re": None, "level": None},
     "op": "<=", "lo": None, "hi": 0.01, "tol": 0.0, "u": 1e-09, "hardness": "hard", "repr": "cfd"},
    {"req_id": "REQ-004", "primitive": "extent_along_axis",
     "args": {"feature": None, "where": ["body"], "Re": None, "level": None},
     "op": "<=", "lo": None, "hi": None, "tol": 0.0, "u": 1e-09, "hardness": "objective", "repr": "brep"},
)


def standin_checks(study_id, hi_overrides=None) -> dict:
    """The stand-in cad-checks/1 doc (validated): four rows on the real template box, requirements
    lock hashed over the checks and the study id (there is no requirements.json behind it).
    `hi_overrides` maps req_id -> new hi, for fixtures only."""
    checks = copy.deepcopy(list(_STANDIN_CHECKS))
    if hi_overrides:
        by_id = dict((c["req_id"], c) for c in checks)
        for rid, hi in hi_overrides.items():
            by_id[rid]["hi"] = hi
    doc = {"schema": "cad-checks/1", "study_id": study_id,
           "requirements_lock": common.sha256_of({"standin_checks": checks, "study_id": study_id}),
           "declaration_sha": reqs.load_template(reqs.NOZZLE_DIR)[2], "checks": checks}
    errs = schema.errors(doc, "cad-checks/1")
    if errs:
        raise ValueError("CADOPT-DOC: %s" % (errs[0],))
    return doc


def standin_records(params, checks_doc) -> dict:
    """The four cad-measure/1 records the stand-in measures (docs/16 §H.5's K_max Re_De as kre, an
    invented exit non-uniformity, the body's axial extent): {req_id: record}, method standin/1."""
    di, cr = float(params["D_i"]), float(params["CR"])
    lodi, lxde = float(params["L_over_Di"]), float(params["Lx_over_De"])
    xm = None if params["x_m"] is None else float(params["x_m"])
    de = 2.0 * (di / 2.0) / math.sqrt(cr)
    kre = turb_integral.kre_max(params["law"], cr, lodi, xm)[0]
    vals = {"REQ-001": (float(params["t_wall"]), "m", 1e-08),
            "REQ-002": (kre, "1", None),
            "REQ-003": (0.02 * (kre / 6.0) * math.exp(-(lxde - 0.25) / 0.25), "1", None),
            "REQ-004": (lodi * di + lxde * de, "m", 1e-09)}
    out = {}
    for c in checks_doc["checks"]:
        v, unit, u = vals[c["req_id"]]
        out[c["req_id"]] = {"schema": "cad-measure/1", "primitive": c["primitive"], "feature": None,
                            "where": list(c["args"]["where"]), "value": v, "unit": unit,
                            "u_meas": u, "method": "standin/1", "status": "ok", "reason_id": None,
                            "detail": STANDIN_DETAIL}
    return out


def standin_prefilter(params, checks_doc) -> dict:
    """The stand-in's prefilter record, the same shape as cad_prefilter: REQ-001 is the only hard
    PREFILTER_REPRS row, the objective is REQ-004's value, readiness always passes."""
    records = standin_records(params, checks_doc)
    rows, first_fail, obj_val = {}, None, None
    for c in checks_doc["checks"]:
        v = verify.judge(c, records[c["req_id"]])
        if c["hardness"] == "objective":
            obj_val = float(records[c["req_id"]]["value"])
        else:
            rows[c["req_id"]] = v
            if c["repr"] in PREFILTER_REPRS and v["verdict"] != "pass" and first_fail is None:
                first_fail = c["req_id"]
    if first_fail is not None:
        return _pf_record(params, "refused", first_fail, obj_val, rows,
                          "hard row %s is %s" % (first_fail, rows[first_fail]["verdict"]))
    return _pf_record(params, "pass", None, obj_val, rows,
                      "the stand-in's %d hard brep/stl rows pass" % (len(rows),))


def standin_eval_key(params, checks_doc, template_sha, declaration_sha, gates_lock) -> str:
    """The stand-in's evaluation key (docs/16 §D): env {"standin": STANDIN_VERSION}, standin recipe
    and case-writer versions, no binary."""
    return reqs.eval_key({"template_sha": template_sha, "declaration_sha": declaration_sha,
                          "params": params, "requirements_lock": checks_doc["requirements_lock"],
                          "gates_lock": gates_lock, "env": {"standin": STANDIN_VERSION},
                          "mesh_recipe_version": "standin", "case_writer_version": "standin",
                          "bin_sha": None})


def standin_evaluate(params, checks_doc, eval_key) -> dict:
    """One stand-in evaluation: every check judged on its record, into a validated cad-verdict/1."""
    records = standin_records(params, checks_doc)
    rows = [verify.judge(c, records[c["req_id"]]) for c in checks_doc["checks"]]
    doc = {"schema": "cad-verdict/1", "study_id": checks_doc["study_id"],
           "requirements_lock": checks_doc["requirements_lock"],
           "checks_sha": common.sha256_of(checks_doc), "eval_key": eval_key,
           "design_verdict": verify.design_verdict(rows), "verdicts": rows}
    errs = schema.errors(doc, "cad-verdict/1")
    if errs:
        raise ValueError("CADOPT-DOC: %s" % (errs[0],))
    return doc


def _fixed_of(start_params, decl) -> dict:
    """The fixed reals of the design space, taken from the start vector."""
    return dict((name, start_params[name]) for name in design_space(decl)["fixed"])


def standin_pool_optimum(study_id) -> dict:
    """The supervisor's oracle: the stand-in evaluated on EVERY pool candidate. Returns the minimum
    objective over candidates that pass the stand-in prefilter AND are feasible, with the counts."""
    decl, template_sha, declaration_sha = reqs.load_template(reqs.NOZZLE_DIR)
    gates, gates_lock = gate.load_gates()
    checks_doc = standin_checks(study_id)
    best, n_pass, n_feas = None, 0, 0
    for c in pool(study_id, decl, _fixed_of(STANDIN_START, decl), gates["sobol_pool"]):
        if standin_prefilter(c["params"], checks_doc)["status"] != "pass":
            continue
        n_pass += 1
        v = standin_evaluate(c["params"], checks_doc,
                             standin_eval_key(c["params"], checks_doc, template_sha, declaration_sha,
                                              gates_lock))
        if v["design_verdict"] != "feasible":
            continue
        n_feas += 1
        m = _objective_row(v)["m"]
        if best is None or m < best:
            best = m
    return {"pool_optimum_m": best, "n_pass": n_pass, "n_feasible": n_feas}


def run_standin(study_id, out_dir, max_evals=18) -> dict:
    """One gated stand-in study into a fresh OUT_DIR: prefilter.jsonl, decisions.jsonl, evals.jsonl
    and summary.json; returns the summary. The gate is `pass`: a feasible evaluation within 1 % of
    the pool optimum in at most max_evals evaluations."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError("CADOPT-DOC: %s exists and is not empty" % (out_dir,))
    os.makedirs(out_dir, exist_ok=True)
    decl, template_sha, declaration_sha = reqs.load_template(reqs.NOZZLE_DIR)
    gates, gates_lock = gate.load_gates()
    checks_doc = standin_checks(study_id)
    cands = pool(study_id, decl, _fixed_of(STANDIN_START, decl), gates["sobol_pool"])
    pf_map = {}
    for c in cands:
        rec = standin_prefilter(c["params"], checks_doc)
        pf_map[c["params_sha"]] = rec
        common.jsonl_append(os.path.join(out_dir, PREFILTER_JSON),
                            dict([(k, c[k]) for k in CAND_KEYS] + [("prefilter", rec)]))
    opt = standin_pool_optimum(study_id)
    return _loop(study_id, checks_doc, decl, template_sha, declaration_sha, gates, gates_lock,
                 cands, pf_map, opt["pool_optimum_m"], out_dir, max_evals)


def _loop(study_id, checks_doc, decl, template_sha, declaration_sha, gates, gates_lock,
          cands, pf_map, pool_opt, out_dir, max_evals) -> dict:
    """The propose / evaluate / gate loop of run_standin, factored so a fixture can run it under an
    overridden checks doc (T5b). Writes decisions.jsonl and evals.jsonl; returns the summary."""
    p_dec = os.path.join(out_dir, DECISIONS_JSON)
    p_ev = os.path.join(out_dir, EVALS_JSON)
    state = {"n_evals": 0, "stable": None, "first_1pct": None}

    def evaluate(params, pdoc):
        ek = standin_eval_key(params, checks_doc, template_sha, declaration_sha, gates_lock)
        v = standin_evaluate(params, checks_doc, ek)
        g = gate.compare(v, state["stable"], STANDIN_OBJECTIVE, 0.0)
        if g["decision"] == "promote":
            state["stable"] = v
        i = state["n_evals"]
        if state["first_1pct"] is None and v["design_verdict"] == "feasible" \
                and pool_opt is not None and _objective_row(v)["m"] <= 1.01 * pool_opt:
            state["first_1pct"] = i + 1
        common.jsonl_append(p_ev, {"i": i, "params_sha": params_sha(params), "params": params,
                                   "params_doc": pdoc, "verdict": v,
                                   "gate": {"decision": g["decision"], "rule_id": g["rule_id"]},
                                   "stable_eval_key": None if state["stable"] is None
                                   else state["stable"]["eval_key"]})
        state["n_evals"] = i + 1
        return v

    doc0, row0 = start(study_id, 0, decl, template_sha, gates_lock, checks_doc, dict(STANDIN_START),
                       STANDIN_START_PROVENANCE)
    common.jsonl_append(p_dec, row0)
    n_dec = 1
    history = [{"params_sha": row0["params_sha"], "params": dict(STANDIN_START),
                "verdict": evaluate(dict(STANDIN_START), doc0)}]
    stop = None
    while state["n_evals"] < max_evals:
        ctx = {"study_id": study_id, "n": n_dec, "decl": decl, "template_sha": template_sha,
               "gates": gates, "gates_lock": gates_lock, "checks_doc": checks_doc,
               "objective": STANDIN_OBJECTIVE, "pool": cands, "prefilter": pf_map,
               "history": history, "stable": state["stable"]}
        pdoc, row = propose(ctx)
        common.jsonl_append(p_dec, row)
        n_dec += 1
        if row["decision"] == "abstain":
            stop = row["rule_id"]
            break
        cand = [c for c in cands if c["params_sha"] == row["params_sha"]][0]
        history.append({"params_sha": cand["params_sha"], "params": cand["params"],
                        "verdict": evaluate(cand["params"], pdoc)})
    stable = state["stable"]
    obj_m = None
    if stable is not None and stable["design_verdict"] == "feasible":
        obj_m = _objective_row(stable)["m"]
    summary = {"study_id": study_id, "n_evals": state["n_evals"], "n_decisions": n_dec,
               "stop": stop if stop is not None else "MAX-EVALS", "pool_optimum_m": pool_opt,
               "continuous_optimum_m": STANDIN_CONTINUOUS_OPTIMUM_M, "stable_objective_m": obj_m,
               "first_within_1pct": state["first_1pct"],
               "pass": state["first_1pct"] is not None and state["first_1pct"] <= max_evals}
    common.write_json(os.path.join(out_dir, SUMMARY_JSON), summary)
    return summary


def replay(out_dir) -> dict:
    """Re-derive every decision from the three jsonl files ONLY (docs/16 §I CAD-17): start() for row
    0, propose() on the recorded history for row k >= 1, the stable re-applied by gate.compare in
    order. Returns {"ok", "n_decisions", "first_mismatch"}; a mismatch names the file and row."""
    pre = common.read_jsonl(os.path.join(out_dir, PREFILTER_JSON))
    dec = common.read_jsonl(os.path.join(out_dir, DECISIONS_JSON))
    ev = common.read_jsonl(os.path.join(out_dir, EVALS_JSON))
    if not dec or not ev:
        return {"ok": False, "n_decisions": len(dec), "first_mismatch": "decisions.jsonl:0"}

    def mm(name, i):
        return {"ok": False, "n_decisions": len(dec), "first_mismatch": "%s:%d" % (name, i)}

    study_id = dec[0]["study_id"]
    checks_doc = standin_checks(study_id)
    decl, template_sha, declaration_sha = reqs.load_template(reqs.NOZZLE_DIR)
    gates, gates_lock = gate.load_gates()
    cands = [dict((k, r[k]) for k in CAND_KEYS) for r in pre]
    pf_map = dict((r["params_sha"], r["prefilter"]) for r in pre)
    prov = dict((v["name"], v["provenance"]) for v in ev[0]["params_doc"]["values"])
    doc0, row0 = start(study_id, 0, decl, template_sha, gates_lock, checks_doc,
                       ev[0]["params"], prov)
    if common.canonical_json(row0) != common.canonical_json(dec[0]):
        return mm(DECISIONS_JSON, 0)
    if common.canonical_json(doc0) != common.canonical_json(ev[0]["params_doc"]):
        return mm(EVALS_JSON, 0)
    state = {"stable": None}

    def apply_gate(j):
        e = ev[j]
        g = gate.compare(e["verdict"], state["stable"], STANDIN_OBJECTIVE, 0.0)
        if g["decision"] != e["gate"]["decision"] or g["rule_id"] != e["gate"]["rule_id"]:
            return mm(EVALS_JSON, j)
        if g["decision"] == "promote":
            state["stable"] = e["verdict"]
        return None

    bad = apply_gate(0)
    if bad:
        return bad
    for k in range(1, len(dec)):
        ctx = {"study_id": study_id, "n": k, "decl": decl, "template_sha": template_sha,
               "gates": gates, "gates_lock": gates_lock, "checks_doc": checks_doc,
               "objective": STANDIN_OBJECTIVE, "pool": cands, "prefilter": pf_map,
               "history": [dict((h, e[h]) for h in HISTORY_KEYS) for e in ev[:k]],
               "stable": state["stable"]}
        pdoc, row = propose(ctx)
        if common.canonical_json(row) != common.canonical_json(dec[k]):
            return mm(DECISIONS_JSON, k)
        if row["decision"] == "abstain":
            if k != len(dec) - 1:
                return mm(DECISIONS_JSON, k)
            break
        if k >= len(ev):
            return mm(EVALS_JSON, k)
        if common.canonical_json(pdoc) != common.canonical_json(ev[k]["params_doc"]):
            return mm(EVALS_JSON, k)
        bad = apply_gate(k)
        if bad:
            return bad
    return {"ok": True, "n_decisions": len(dec), "first_mismatch": None}


# ---- selftest T1..T12 (each prints exactly one "[ok]" line; any miss raises) ----

def _decl_gates():
    decl, template_sha, declaration_sha = reqs.load_template(reqs.NOZZLE_DIR)
    gates, gates_lock = gate.load_gates()
    return decl, template_sha, declaration_sha, gates, gates_lock


def _evaluate(params, checks_doc, template_sha, declaration_sha, gates_lock):
    return standin_evaluate(params, checks_doc,
                            standin_eval_key(params, checks_doc, template_sha, declaration_sha,
                                             gates_lock))


def _stable_of(verdicts):
    stable = None
    for v in verdicts:
        g = gate.compare(v, stable, STANDIN_OBJECTIVE, 0.0)
        if g["decision"] == "promote":
            stable = v
    return stable


def _hist(entries):
    return [dict((k, e[k]) for k in HISTORY_KEYS) for e in entries]


def _five_runs(root) -> dict:
    """The five stand-in sweeps of T9, run once per selftest process (T6c needs the largest LOO z
    RMS they produced); {study_id: {"summary", "z_max"}}."""
    if _FIVE_RUNS_CACHE.get("runs") is not None:
        return _FIVE_RUNS_CACHE["runs"]
    runs = {}
    for sid in STANDIN_SEEDS:
        mark = len(_Z_TRACK)
        summary = run_standin(sid, os.path.join(root, sid), 18)
        z = _Z_TRACK[mark:]
        runs[sid] = {"summary": summary, "z_max": max(z) if z else None,
                     "dir": os.path.join(root, sid)}
    _FIVE_RUNS_CACHE["runs"] = runs
    return runs


_FIVE_RUNS_CACHE = {}


def _t1() -> None:
    decl, _ts, _ds, _g, _gl = _decl_gates()
    fixed = {"D_i": 0.06, "CR": 9.0, "Lu_over_Di": 0.5}
    cands = pool("standin-1", decl, fixed, 256)
    assert len(cands) == 1024, len(cands)
    for li in range(4):
        assert sorted(c["sobol_index"] for c in cands[li * 256:(li + 1) * 256]) == list(range(256))
    assert seed_of("standin-1", "poly3") == 2587582285
    c0 = cands[0]
    want0 = {"L_over_Di": 1.2219172902405262, "x_m": None, "Lx_over_De": 0.37060008640401065,
             "t_wall": 0.008190682037733496, "D_i": 0.06, "CR": 9.0, "Lu_over_Di": 0.5,
             "law": "poly3"}
    assert c0["params"] == want0, c0["params"]
    assert c0["params_sha"] == "dc75a51dc965768a676be0351a8d0abf88e6d50257758b4a4ec3a515f1853261"
    c768 = cands[768]
    assert c768["params"]["L_over_Di"] == 1.356941606849432
    assert c768["params"]["x_m"] == 0.7960821827873588
    assert c768["params"]["Lx_over_De"] == 0.38975317450240254
    assert c768["params"]["t_wall"] == 0.005280397707596422
    ds = design_space(decl)
    for c in cands:
        for s in ds["searched"]:
            active = s["only_when"] is None or s["only_when"] == "law=" + c["law"]
            v = c["params"][s["name"]]
            if active:
                assert s["min"] <= v <= s["max"], (s["name"], v)
            else:
                assert v is None, (s["name"], c["law"], v)
    again = pool("standin-1", decl, fixed, 256)
    assert again == cands
    other = pool("standin-2", decl, fixed, 256)[0]
    assert other["params"]["L_over_Di"] != c0["params"]["L_over_Di"]
    try:
        pool("standin-1", decl, fixed, 100)
        raise AssertionError("pool(n=100) did not raise")
    except ValueError as e:
        assert "CADOPT-DOC" in str(e), e
    print("[ok] T1 pool: 1024 candidates, the expected seed 2587582285 and candidates 0/768 exact,"
          " boxes, determinism and the n=100 refusal")


def _t2() -> None:
    decl, template_sha, _ds2, _g, _gl = _decl_gates()
    fixed = {"D_i": 0.06, "CR": 9.0, "Lu_over_Di": 0.5}
    cands = pool("standin-1", decl, fixed, 256)
    prov = optimiser_provenance(decl)
    d0 = params_doc(cands[0]["params"], decl, template_sha, "b" * 64, None, prov)
    d768 = params_doc(cands[768]["params"], decl, template_sha, "b" * 64, "c" * 64, prov)
    assert schema.errors(d0, "cad-params/1") == []
    assert schema.errors(d768, "cad-params/1") == []
    assert len(d0["values"]) == 7, len(d0["values"])
    assert len(d768["values"]) == 8, len(d768["values"])
    by_name = dict((v["name"], v) for v in d0["values"])
    assert by_name["D_i"]["provenance"] == "user_text"
    assert by_name["CR"]["provenance"] == "user_text"
    assert by_name["Lu_over_Di"]["provenance"] == "default"
    for n in ("L_over_Di", "Lx_over_De", "t_wall", "law"):
        assert by_name[n]["provenance"] == "optimiser", n
    broken = copy.deepcopy(d0)
    del broken["requirements_lock"]
    errs = schema.errors(broken, "cad-params/1")
    assert errs and "requirements_lock" in errs[0], errs
    doc, _row = start("standin-1", 0, decl, template_sha, "d" * 64,
                      standin_checks("standin-1"), dict(STANDIN_START), dict(STANDIN_START_PROVENANCE))
    assert doc["base_stable_eval_key"] is None
    assert doc["requirements_lock"] == standin_checks("standin-1")["requirements_lock"]
    print("[ok] T2 params docs: candidate 0 has 7 values and 768 has 8, provenance by role, the"
          " schema names a missing requirements_lock, the start doc has a null base")


def _t3() -> None:
    decl, _ts, _ds2, _g, _gl = _decl_gates()
    fixed = {"D_i": 0.06, "CR": 9.0, "Lu_over_Di": 0.5}
    cands = pool("standin-1", decl, fixed, 256)
    f0 = features(cands[0]["params"], decl)
    f768 = features(cands[768]["params"], decl)
    assert len(f0) == 8 and all(isinstance(v, float) for v in f0)
    assert f0[1] == (0.5 - 0.2) / (0.8 - 0.2), f0        # the default scaled: 0.4999999999999999
    assert f0[4:] == [1.0, 0.0, 0.0, 0.0], f0[4:]
    assert f768[4:] == [0.0, 0.0, 0.0, 1.0], f768[4:]
    assert f768[1] == (0.7960821827873588 - 0.2) / (0.8 - 0.2), f768
    print("[ok] T3 features: 8 numbers, the inactive x_m slot at its scaled default %r, one-hot laws"
          % (f0[1],))


def _standin_world(sid):
    """The shared fixtures of the propose-level tests: decl/shas/gates, the checks doc, the pool and
    its stand-in prefilter map."""
    decl, template_sha, declaration_sha, gates, gates_lock = _decl_gates()
    checks_doc = standin_checks(sid)
    cands = pool(sid, decl, _fixed_of(STANDIN_START, decl), gates["sobol_pool"])
    pf_map = dict((c["params_sha"], standin_prefilter(c["params"], checks_doc)) for c in cands)
    return decl, template_sha, declaration_sha, gates, gates_lock, checks_doc, cands, pf_map


def _propose_ctx(sid, decl, template_sha, gates, gates_lock, checks_doc, cands, pf_map,
                 entries, stable):
    return {"study_id": sid, "n": len(entries), "decl": decl, "template_sha": template_sha,
            "gates": gates, "gates_lock": gates_lock, "checks_doc": checks_doc,
            "objective": STANDIN_OBJECTIVE, "pool": cands, "prefilter": pf_map,
            "history": _hist(entries), "stable": stable}


def _start_entry(sid, checks_doc, template_sha, declaration_sha, gates_lock):
    v = _evaluate(dict(STANDIN_START), checks_doc, template_sha, declaration_sha, gates_lock)
    return {"params_sha": params_sha(dict(STANDIN_START)), "params": dict(STANDIN_START), "verdict": v}


def _cand_entry(cand, checks_doc, template_sha, declaration_sha, gates_lock):
    v = _evaluate(cand["params"], checks_doc, template_sha, declaration_sha, gates_lock)
    return {"params_sha": cand["params_sha"], "params": cand["params"], "verdict": v}


def _t4() -> None:
    assert pick([(0.5, 1, 3), (0.5, 0, 7), (0.4, 0, 0), (0.5, 0, 9)]) == 1
    world = _standin_world("standin-1")
    decl, template_sha, dsha, gates, gl, checks_doc, cands, pf_map = world
    e_start = _start_entry("standin-1", checks_doc, template_sha, dsha, gl)
    stable = _stable_of([e["verdict"] for e in [e_start]])
    pdoc, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                     cands, pf_map, [e_start], stable))
    assert (row["law"], row["sobol_index"]) == ("poly3", 0), row
    e00 = _cand_entry(cands[0], checks_doc, template_sha, dsha, gl)
    entries = [e_start, e00]
    stable = _stable_of([e["verdict"] for e in entries])
    _p, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                   cands, pf_map, entries, stable))
    assert (row["law"], row["sobol_index"]) == ("poly5", 0), row
    e20 = _cand_entry(cands[512], checks_doc, template_sha, dsha, gl)
    entries = [e_start, e00, e20]
    stable = _stable_of([e["verdict"] for e in entries])
    _p, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                   cands, pf_map, entries, stable))
    assert (row["law"], row["sobol_index"]) == ("poly5", 0), row
    e10 = _cand_entry(cands[256], checks_doc, template_sha, dsha, gl)
    entries = [e_start, e00, e20, e10]
    stable = _stable_of([e["verdict"] for e in entries])
    _p, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                   cands, pf_map, entries, stable))
    assert (row["law"], row["sobol_index"]) == ("cubic_matched", 0), row
    print("[ok] T4 pick ties to the lower (law, Sobol) index and evaluated candidates are deduped:"
          " (0,0) (1,0) (1,0) (3,0)")


def _t5a():
    world = _standin_world("standin-1")
    decl, template_sha, dsha, gates, gl, checks_doc, cands, _pf = world
    pf_map = dict((c["params_sha"],
                   _pf_record(c["params"], "refused", "REQ-001", None, {},
                              "fixture: refused for every candidate")) for c in cands)
    e_start = _start_entry("standin-1", checks_doc, template_sha, dsha, gl)
    pdoc, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                     cands, pf_map, [e_start], None))
    assert pdoc is None and row["decision"] == "abstain" and row["rule_id"] == "CADOPT-NOFEAS", row
    assert "passed none of the 1024" in row["reason"], row["reason"]
    return (row, [e_start], None, checks_doc["requirements_lock"])


def _t5b(root):
    checks2 = standin_checks("standin-1", {"REQ-003": -1.0})
    decl, template_sha, dsha, gates, gl, _cd, cands, _pf = _standin_world("standin-1")
    pf_map = dict((c["params_sha"], standin_prefilter(c["params"], checks2)) for c in cands)
    out = os.path.join(root, "t5b")
    summary = _loop("standin-1", checks2, decl, template_sha, dsha, gates, gl, cands, pf_map,
                    None, out, 18)
    assert summary["stop"] == "CADOPT-NOFEAS" and summary["n_evals"] == 6, summary
    assert summary["n_decisions"] == 7, summary
    dec = common.read_jsonl(os.path.join(out, DECISIONS_JSON))
    ev = common.read_jsonl(os.path.join(out, EVALS_JSON))
    assert dec[-1]["decision"] == "abstain" and dec[-1]["rule_id"] == "CADOPT-NOFEAS"
    assert "probability of feasibility" in dec[-1]["reason"], dec[-1]["reason"]
    stable = _stable_of([e["verdict"] for e in ev])
    return (dec[-1], _hist(ev), stable, checks2["requirements_lock"])


def _t6a():
    world = _standin_world("standin-1")
    decl, template_sha, dsha, gates, gl, checks_doc, cands, pf_map = world
    nonuni_check = [c for c in checks_doc["checks"] if c["req_id"] == "REQ-003"][0]
    refused = {"schema": "cad-measure/1", "primitive": nonuni_check["primitive"], "feature": None,
               "where": list(nonuni_check["args"]["where"]), "value": None, "unit": "1",
               "u_meas": None, "method": "standin/1", "status": "refused",
               "reason_id": "POST-UNDEFINED", "detail": "T6a fixture"}
    entries = [_start_entry("standin-1", checks_doc, template_sha, dsha, gl)]
    for c in (cands[0], cands[256], cands[512], cands[768], cands[2]):
        entries.append(_cand_entry(c, checks_doc, template_sha, dsha, gl))
    for e in entries[:4]:                       # REQ-003 re-judged NE on 4 of the 6 evaluations
        rows = [verify.judge(nonuni_check, refused) if r["req_id"] == "REQ-003" else r
                for r in e["verdict"]["verdicts"]]
        e["verdict"] = dict(e["verdict"], verdicts=rows,
                            design_verdict=verify.design_verdict(rows))
    stable = _stable_of([e["verdict"] for e in entries])
    pdoc, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                     cands, pf_map, entries, stable))
    assert pdoc is None and row["rule_id"] == "CADOPT-GPFIT" and row["decision"] == "abstain", row
    assert "REQ-003" in row["reason"] and "n=2" in row["reason"], row["reason"]
    return (row, entries, stable, checks_doc["requirements_lock"])


def _t6b():
    decl = _decl_gates()[0]
    base = dict(STANDIN_START, law="poly3", x_m=None, t_wall=0.005)
    X, y = [], []
    for i in range(7):
        lod = 0.5 + 0.125 * i
        X.append(features(dict(base, L_over_Di=lod), decl))
        y.append(lod - 1.0)
    X.append(features(dict(base, L_over_Di=0.875 + 1e-5), decl))
    y.append(1.0)
    _model, loo = fit_constraint(np.array(X, dtype=float), np.array(y, dtype=float))
    assert loo["ok"] is False and loo["z_rms"] is not None and loo["z_rms"] > LOO_Z_MAX, loo
    return loo["z_rms"]


def _t6(runs) -> None:
    row, _entries, _stable, _lock = _t6a()
    jump = _t6b()
    zmax = max(r["z_max"] for r in runs.values())
    assert zmax <= LOO_Z_MAX, zmax
    print("[ok] T6 GPFIT: 4 of 6 NE rows leave n=2 and abstains naming REQ-003, the twin-jump"
          " fixture refuses at z_rms %r, the five T9 runs peaked at z_rms %r" % (jump, zmax))


def _t7():
    world = _standin_world("standin-1")
    decl, template_sha, dsha, gates, gl, checks_doc, cands, _pf = world
    refused = dict((c["params_sha"],
                    _pf_record(c["params"], "refused", "REQ-001", None, {},
                               "T7 fixture: refused"))
                   for c in cands)
    for c in (cands[0], cands[256]):
        refused[c["params_sha"]] = standin_prefilter(c["params"], checks_doc)
    entries = [_start_entry("standin-1", checks_doc, template_sha, dsha, gl),
               _cand_entry(cands[0], checks_doc, template_sha, dsha, gl),
               _cand_entry(cands[256], checks_doc, template_sha, dsha, gl)]
    stable = _stable_of([e["verdict"] for e in entries])
    pdoc, row = propose(_propose_ctx("standin-1", decl, template_sha, gates, gl, checks_doc,
                                     cands, refused, entries, stable))
    assert pdoc is None and row["decision"] == "abstain" and row["rule_id"] == "CADOPT-EXHAUSTED", row
    print("[ok] T7 EXHAUSTED: a pool whose only passed candidates (0,0) and (1,0) are already in"
          " the history abstains CADOPT-EXHAUSTED")
    return (row, entries, stable, checks_doc["requirements_lock"])


def _t8(abstains) -> None:
    for row, entries, stable, lock in abstains:
        assert schema.errors(row, "cad-decision/1") == [], row
        assert row["decision"] == "abstain"
        assert row["params_sha"] is None and row["law"] is None
        assert row["sobol_index"] is None and row["ei"] is None
        assert row["requirements_lock"] == lock, row
        assert row["base_stable_eval_key"] == (None if stable is None else stable["eval_key"])
        want = gate.deltas(entries[-1]["verdict"], stable)
        for k in DELTA_KEYS:
            assert row[k] == want[k], (k, row[k], want[k])
    print("[ok] T8 abstain rows: %d rows schema-valid with null pick fields, the checks' lock, the"
          " stable base and exactly gate.deltas" % (len(abstains),))


def _t9(runs) -> None:
    for sid in STANDIN_SEEDS:
        s = runs[sid]["summary"]
        assert s["pass"] is True, s
        opt = standin_pool_optimum(sid)
        assert opt["pool_optimum_m"] == POOL_OPTIMA[sid], (sid, opt)
        assert opt["n_pass"] == POOL_NPASS[sid] and opt["n_feasible"] == POOL_NFEAS[sid], (sid, opt)
        print("  %s: first_within_1pct=%s pool_optimum_m=%r stable_objective_m=%r"
              % (sid, s["first_within_1pct"], s["pool_optimum_m"], s["stable_objective_m"]))
    n = sum(1 for sid in STANDIN_SEEDS if runs[sid]["summary"]["first_within_1pct"] is not None
            and runs[sid]["summary"]["first_within_1pct"] <= 18)
    assert n == 5, n
    print("[ok] T9 the gate: the five pool optima match the oracle table and the stable design is"
          " within 1 %% of the pool optimum in at most 18 evaluations for %d of 5 seeds" % (n,))


def _t5b_lock():
    return standin_checks("standin-1", {"REQ-003": -1.0})["requirements_lock"]


def _t10(root) -> None:
    script = os.path.join(HERE, "optimise_cad.py")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    dirs = []
    for tag in ("t10a", "t10b"):
        d = os.path.join(root, tag)
        p = subprocess.run([sys.executable, script, "standin", "standin-1", d, "18"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=root, env=env, timeout=600)
        assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
        dirs.append(d)
    names = (PREFILTER_JSON, DECISIONS_JSON, EVALS_JSON)
    shas = []
    for d in dirs:
        shas.append(tuple(common.sha256_file(os.path.join(d, n)) for n in names))
    assert shas[0] == shas[1], shas
    r = replay(dirs[0])
    n_rows = len(common.read_jsonl(os.path.join(dirs[0], DECISIONS_JSON)))
    assert r["ok"] is True and r["n_decisions"] == n_rows, (r, n_rows)
    tam = os.path.join(root, "t10c")
    shutil.copytree(dirs[0], tam)
    p_ev = os.path.join(tam, EVALS_JSON)
    rows = common.read_jsonl(p_ev)
    for v in rows[2]["verdict"]["verdicts"]:
        if v["req_id"] == "REQ-002":
            v["m"] = v["m"] + 1e-3
    common.atomic_write(p_ev,
                        "".join(common.canonical_json(r) + "\n" for r in rows))
    r = replay(tam)
    assert r["ok"] is False and r["first_mismatch"] is not None \
        and "decisions.jsonl" in r["first_mismatch"], r
    print("[ok] T10 replay: two fresh processes byte-identical across the three files, replay"
          " re-derives %d decisions, and a tampered evals row fails at %s"
          % (n_rows, r["first_mismatch"]))


def _t11(root) -> None:
    _doc, checks_doc, _decl = mutate.load_checks()
    h_m = readiness.H_SELFTEST_M
    probes = dict(export.NOMINAL)
    rec = cad_prefilter(probes, checks_doc, os.path.join(root, "t11a"), h_m)
    assert rec["status"] == "pass" and rec["rule_id"] is None, rec
    rows = common.read_json(os.path.join(root, "t11a", "probes.json"))["rows"]
    ext = [r for r in rows if r["primitive"] == "extent_along_axis"
           and list(r["where"]) == ["body"]][0]["record"]["value"]
    assert rec["objective"] == 0.07000000000000006 == ext, (rec["objective"], ext)
    assert rec["params_sha"] == common.read_json(os.path.join(root, "t11a", "probes.json"))["params_sha"]
    assert common.sha256_of(dict(export.NOMINAL)) == rec["params_sha"]
    rec = cad_prefilter(dict(export.NOMINAL, L_over_Di=1.5), checks_doc,
                        os.path.join(root, "t11b"), h_m)
    assert rec["status"] == "refused" and rec["rule_id"] == "REQ-004", rec
    rec = cad_prefilter(dict(export.NOMINAL, t_wall=0.001), checks_doc,
                        os.path.join(root, "t11c"), h_m)
    assert rec["status"] == "refused" and rec["rule_id"] == "REQ-006", rec
    print("[ok] T11 the real prefilter: the nominal passes with the exact objective and params sha,"
          " L_over_Di 1.5 is refused by REQ-004 and t_wall 0.001 by REQ-006")


def _t12(runs) -> None:
    sid = "standin-1"
    lock = standin_checks(sid)["requirements_lock"]
    d = runs[sid]["dir"]
    ev = common.read_jsonl(os.path.join(d, EVALS_JSON))
    dec = common.read_jsonl(os.path.join(d, DECISIONS_JSON))
    for e in ev:
        assert e["params_doc"]["requirements_lock"] == lock, e["i"]
    for r in dec:
        assert r["requirements_lock"] == lock, r["n"]
    prev = None
    for k, e in enumerate(ev):
        assert e["params_doc"]["base_stable_eval_key"] == dec[k]["base_stable_eval_key"] == prev, k
        prev = e["stable_eval_key"]
    by = dict((r["req_id"], r["verdict"]) for r in ev[0]["verdict"]["verdicts"])
    assert by["REQ-002"] == "fail" and by["REQ-003"] == "fail", by
    print("[ok] T12 lineage: every params doc and decision row carries the checks' lock, every"
          " base_stable_eval_key chains through the stable, and the start fails REQ-002 and REQ-003")


def _t5(root) -> list:
    a = _t5a()
    b = _t5b(root)
    assert b[0]["rule_id"] == "CADOPT-NOFEAS" and b[0]["decision"] == "abstain"
    print("[ok] T5 NOFEAS: an all-refused pool abstains on the first propose and the REQ-003 hi -1"
          " study stops at 6 evaluations with the best probability of feasibility below 1e-3")
    return [a, b]


def selftest() -> None:
    """The CAD-17 gate: T1-T12, one [ok] line each, SELFTEST PASS at the end."""
    with tempfile.TemporaryDirectory() as root:
        runs = _five_runs(root)
        _t1()
        _t2()
        _t3()
        _t4()
        abstains = _t5(root)
        t6a = _t6a()
        _t6(runs)
        abstains += [t6a, _t7()]
        _t8(abstains)
        _t9(runs)
        _t10(root)
        _t11(root)
        _t12(runs)
    print("SELFTEST PASS")


def main(argv) -> int:
    """The CLI: --selftest, --help, the standin and replay verbs, and the real prefilter verb."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if argv in (["--help"], ["-h"]):
        print(USAGE)
        return 0
    if argv and argv[0] == "standin" and len(argv) in (3, 4):
        try:
            summary = run_standin(argv[1], argv[2], int(argv[3]) if len(argv) == 4 else 18)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(common.canonical_json(summary))
        return 0 if summary["pass"] else 1
    if argv and argv[0] == "replay" and len(argv) == 2:
        r = replay(argv[1])
        if r["ok"]:
            print(common.canonical_json(r))
            return 0
        print("replay refused: %s" % (r["first_mismatch"],))
        return 1
    if argv and argv[0] == "prefilter" and len(argv) == 6:
        record = cad_prefilter(common.read_json(argv[1]), common.read_json(argv[2]), argv[3],
                               float(argv[4]))
        common.write_json(argv[5], record)
        return 0 if record["status"] == "pass" else 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
