#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Template fixture: runs the nozzle template's profile rules on an injected profile - a cone law (PRF-DERIV) or a fluid wire without its axis edge (PRF-FACE2D)."""
import importlib.util


def build(params, out_dir):
    spec = importlib.util.spec_from_file_location("nozzle_template_under_test", params["template"])
    tpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpl)
    p = tpl.resolved(params["params"])
    d = tpl.derived(p)
    if params["case"] == "deriv":
        prof, ref = tpl.profile_stage(p, d, [tpl.bezier([(0.0, d["R_i"]), (d["L"], d["R_e"])])])
    elif params["case"] == "face2d":
        prof, ref = tpl.profile_stage(p, d, tpl.law_curves(p, d))
        if ref is not None:
            raise RuntimeError("the nominal profile was refused: %r" % (ref,))
        faces, ref = tpl.edge_list_rules([prof["fluid_edges"][:-1], prof["body_edges"]])
    else:
        raise ValueError("unknown case %r" % (params["case"],))
    return {"rule": ref[0] if ref else None, "detail": ref[1] if ref else ""}
