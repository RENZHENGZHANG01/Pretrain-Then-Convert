"""Collect the evaluation summaries (data/eval/<run>/summary.json) and the run registry (data/runs.csv) into one table.

  python analysis/collect.py  ->  analysis/out/suite_metrics.csv
"""
import csv
import glob
import json

from common import DATA, EVAL, OUT


def main():
    reg = {r["run_id"]: r for r in csv.DictReader(open(f"{DATA}/runs.csv"))}
    rows = []
    for f in sorted(glob.glob(f"{EVAL}/*/summary.json")):
        s = json.load(open(f))
        r = reg.get(s["run_id"], {})
        rows.append(dict(run_id=s["run_id"], route=r.get("route", s["route"]), kind=s["kind"], size=s["size"], seed=r.get("seed", ""),
                         budget=r.get("budget", ""), f=r.get("f", "") or (0.0 if s["route"] == "imf" and s["kind"] == "cooldown" else ""),
                         flops=s["flops"], weights=s["weights"], fid50k=s["fid50k"], fid10k_sel=s["fid10k_sel"],
                         fid50k_grid1=(s.get("grid1") or s)["fid50k"],  # before the extended guidance search
                         fid_2nfe_10k=s.get("fid_2nfe_10k", ""), fid_noguid_10k=s.get("fid_noguid_10k", ""),
                         point=json.dumps(s["point"])))
    with open(f"{OUT}/suite_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "evaluated runs")


if __name__ == "__main__":
    main()
