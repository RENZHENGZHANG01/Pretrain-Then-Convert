"""FD-DINOv2 with each model's own guidance selection (eval/<run>/fdsel.json, scripts/eval_fdsel.py of the code) for
the runs in analysis/fd_dinov2_runs.txt: per route and budget the best FD-DINOv2 over the evaluated candidates (the
FID-frontier and FD-frontier cells of analysis/stage_seeds_fd.py), the one-step tax and native/convert ratio under that
metric, and the 3e20 XL f-curve. Compared with FD-DINOv2 at the FID-selected guidance.

  python analysis/fd_dinov2.py  ->  analysis/out/fd_dinov2.json (+ table on stdout)
"""
import json
import os
import re

from common import ROOT
from common import EVAL
BUD = ("1e19", "3e19", "1e20", "3e20")


def parse(rid):
    m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s0$", rid)
    if not m:
        return None
    route = {"r1": "R1", "r2": "NAT", "r3": "CONV"}[m.group(1)]
    return route, m.group(2), m.group(3), (float(m.group(4)[1:]) if route == "CONV" else (0.0 if route == "NAT" else None))


def main():
    rows = []
    for rid in open(f"{ROOT}/analysis/fd_dinov2_runs.txt").read().split():
        p = f"{EVAL}/{rid}/fdsel.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if parse(rid) is None and not rid.startswith("v"):
            continue
        # FID-selected values from the main evaluation (summary.json / extra.json, after the extended guidance search);
        # the FD-selected values come from the FD-selected evaluation itself
        summ = json.load(open(f"{EVAL}/{rid}/summary.json"))
        ex = json.load(open(f"{EVAL}/{rid}/extra.json")) if os.path.exists(f"{EVAL}/{rid}/extra.json") else {}
        rows.append(dict(run_id=rid, key=parse(rid), grid_version=d.get("grid_version", 1),
                         fd_fdsel=d["fd_dinov2_50k_fdsel"], fd_fidsel=ex.get("fd_dinov2", d.get("fd_dinov2_50k_fidsel")),
                         fid_fidsel=summ["fid50k"], fid_at_fdsel=d["fid50k_at_fdsel"],
                         fid_point=summ["point"], fd_point=d["fd_selected_point"]))
    # FM results selected on the first FM guidance grid (omega <= 4) are superseded by those on the wider grid
    final = [r for r in rows if not (r["key"] and r["key"][0] == "R1" and r["grid_version"] < 2)]
    best = {}
    for r in final:
        if r["key"] is None:
            continue
        route, size, b, f = r["key"]
        k = (route, b)
        if k not in best or r["fd_fdsel"] < best[k]["fd_fdsel"]:
            best[k] = r
    out = dict(rows=rows, best={f"{k[0]}_{k[1]}": v["run_id"] for k, v in best.items()})
    tax, natc = {}, {}
    for b in BUD:
        if ("R1", b) in best and ("CONV", b) in best:
            tax[b] = best[("CONV", b)]["fd_fdsel"] / best[("R1", b)]["fd_fdsel"]
        if ("NAT", b) in best and ("CONV", b) in best:
            natc[b] = best[("NAT", b)]["fd_fdsel"] / best[("CONV", b)]["fd_fdsel"]
    out["tax_fd_fdsel"], out["nat_over_conv_fd_fdsel"] = tax, natc
    curve = sorted((r["key"][3], r["fd_fdsel"], r["fd_fidsel"], r["fid_fidsel"]) for r in final
                   if r["key"] and r["key"][1] == "XL" and r["key"][2] == "3e20" and r["key"][0] in ("CONV", "NAT"))
    out["xl3e20_fcurve"] = [dict(f=c[0], fd_fdsel=c[1], fd_fidsel=c[2], fid=c[3]) for c in curve]
    # 1e21 (pre-registered runs): best one-step = convert XL (V2) or H (V2b), multi-step reference = H
    v = {r["run_id"]: r for r in rows if r["run_id"].startswith("v")}
    if "v2_H_1e21_cd_s0" in v:
        one = [v[k] for k in ("v2_XL_1e21_f0.60_s0", "v2_H_1e21_f0.60_s0") if k in v]
        if one:
            b1 = min(one, key=lambda r: r["fd_fdsel"])
            out["fd_1e21"] = dict(multistep_H=v["v2_H_1e21_cd_s0"]["fd_fdsel"], best_onestep=b1["run_id"], best_onestep_fd=b1["fd_fdsel"],
                                  tax_fd_fdsel=b1["fd_fdsel"] / v["v2_H_1e21_cd_s0"]["fd_fdsel"],
                                  tax_fd_fidsel=b1["fd_fidsel"] / v["v2_H_1e21_cd_s0"]["fd_fidsel"])
            if "v1_XL_1e21_cd_s0" in v and "v2_XL_1e21_f0.60_s0" in v:  # native over convert at equal size (XL)
                out["fd_1e21"]["nat_over_conv_XL_fd_fdsel"] = v["v1_XL_1e21_cd_s0"]["fd_fdsel"] / v["v2_XL_1e21_f0.60_s0"]["fd_fdsel"]
                out["fd_1e21"]["nat_over_conv_XL_fd_fidsel"] = v["v1_XL_1e21_cd_s0"]["fd_fidsel"] / v["v2_XL_1e21_f0.60_s0"]["fd_fidsel"]
    json.dump(out, open(f"{ROOT}/analysis/out/fd_dinov2.json", "w"), indent=1)
    print("1e21:", out.get("fd_1e21"))
    print(f"{len(rows)} fdsel results ({len(final)} final)")
    for r in rows:
        print(f"  {r['run_id']:24s} v{r['grid_version']} FD {r['fd_fidsel'] or float('nan'):6.1f} -> {r['fd_fdsel']:6.1f} | "
              f"FID {r['fid_fidsel']:5.2f} -> {r['fid_at_fdsel']:5.2f}")
    print("tax (FD, own guidance):", {b: round(v, 3) for b, v in tax.items()})
    print("native/convert (FD, own guidance):", {b: round(v, 3) for b, v in natc.items()})
    print("XL 3e20 f-curve (f, FD own, FD at FID guidance, FID):", [(c["f"], round(c["fd_fdsel"], 1), round(c["fd_fidsel"] or 0, 1), round(c["fid"], 2)) for c in out["xl3e20_fcurve"]])


if __name__ == "__main__":
    main()
