"""One compute law for native and converted one-step models.

The separately fitted floored laws (Table 2: three parameters on four frontier points per route) do not identify the
floors, and their point estimates cross near 3e21. Here both one-step frontiers are fitted with one law: the converted
frontier is the native law at m*C, where m is either
  (A) the sample multiplier of the best conversion at each budget, m = 1 + f(kappa - 1) (no free parameter), or
  (B) a single free compute multiplier m_hat, estimated from the frontiers.
Fitted on the four grid budgets (1e19-3e20); the 1e21 runs are predictions. Intervals: the parametric bootstrap of
analysis/noise.py (cells refitted per replicate). Also: the native/convert advantage at equal "efficient" FLOPs, i.e.
with an iMF implementation that computes the JVP only for r != t samples and the interval-adjusted guidance pass only
where t lies outside the guidance interval (kappa 10-12% lower).

  python analysis/joint_law.py  ->  analysis/out/joint_law.json
"""
import json
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as fdist

import noise as nz

from common import EVAL as EV, FT, ROOT
from frontier import fit_floor, frontier

BUD = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}
RUN21 = dict(nat=json.load(open(f"{EV}/v1_XL_1e21_cd_s0/summary.json"))["fid50k"],
             conv=json.load(open(f"{EV}/v2_XL_1e21_f0.60_s0/summary.json"))["fid50k"],
             ms=json.load(open(f"{EV}/v2_H_1e21_cd_s0/summary.json"))["fid50k"], size="XL", f=0.6)
N_BOOT = 2000
CGRID = np.logspace(19, 22, 31)


def law(p, c):
    return p[0] + p[1] * np.asarray(c) ** (-p[2])


def kappa(size):
    return FT[size]["imf"] / FT[size]["fm"]


def frontier_m(cells):
    """Best conversion per budget -> (log FID, sample multiplier m of that cell)."""
    out = {}
    for b in BUD:
        q, k = min((v, kk) for kk, v in cells.items() if kk[0] == "CONV" and kk[2] == b)
        out[b] = (q, 1 + k[3] * (kappa(k[1]) - 1))
    return out


def fit_joint(cells, free_m, budgets=tuple(BUD)):
    fn = frontier(cells, "NAT")
    fc = frontier_m(cells)
    x = np.array([BUD[b] for b in budgets]) / 1e20
    yn = np.array([fn[b] for b in budgets]); yc = np.array([fc[b][0] for b in budgets]); m = np.array([fc[b][1] for b in budgets])
    ymin = min(np.exp(yn).min(), np.exp(yc).min())

    def resid(p):
        mm = np.full(len(budgets), p[3]) if free_m else m
        return np.concatenate([np.log(law(p[:3], x)) - yn, np.log(law(p[:3], mm * x)) - yc])
    best = None
    for E0 in (0.5, 1.0, 1.4):
        for g0 in (0.4, 0.7, 1.0):
            p0 = [min(E0, 0.95 * ymin), 2.0, g0] + ([1.8] if free_m else [])
            lo = [0, 1e-6, 0.01] + ([1.0] if free_m else [])
            hi = [ymin, 1e3, 3.0] + ([20.0] if free_m else [])
            r = least_squares(resid, p0, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
    return best.x, float(np.sum(best.fun ** 2)), m


def summary(cells):
    pa, rss_a, m = fit_joint(cells, False)
    pb, rss_b, _ = fit_joint(cells, True)
    pn, pc = fit_floor(frontier(cells, "NAT")), fit_floor({b: v[0] for b, v in frontier_m(cells).items()})
    x = np.array([BUD[b] for b in BUD]) / 1e20
    rss_s = float(np.sum((np.log(law(pn, x)) - np.array([frontier(cells, "NAT")[b] for b in BUD])) ** 2)
                  + np.sum((np.log(law(pc, x)) - np.array([frontier_m(cells)[b][0] for b in BUD])) ** 2))
    m21 = 1 + RUN21["f"] * (kappa(RUN21["size"]) - 1)
    out = dict(A=dict(E=pa[0], K=pa[1], gamma=pa[2], rss=rss_a, m=m.tolist(), m21=m21,
                      nat21=float(law(pa, 10.0)), conv21=float(law(pa, m21 * 10.0)),
                      ratio={f"{c:.1e}": float(law(pa, c / 1e20) / law(pa, m21 * c / 1e20)) for c in (1e21, 2.5e21, 1e22, 1e23)}),
               B=dict(E=pb[0], K=pb[1], gamma=pb[2], m_hat=pb[3], rss=rss_b,
                      curve=[float(law(pb[:3], c / 1e20) / law(pb[:3], pb[3] * c / 1e20)) for c in CGRID],
                      nat21=float(law(pb[:3], 10.0)), conv21=float(law(pb[:3], pb[3] * 10.0)),
                      ratio={f"{c:.1e}": float(law(pb[:3], c / 1e20) / law(pb[:3], pb[3] * c / 1e20)) for c in (1e21, 2.5e21, 1e22, 1e23)}),
               separate=dict(nat=pn.tolist(), conv=pc.tolist(), rss=rss_s, nat21=float(law(pn, 10.0)), conv21=float(law(pc, 10.0))))
    return out


def held_out(cells):
    """Out-of-sample checks: both one-law versions fitted on 1e19-1e20 predict 3e20, and every budget
    is predicted from a fit on the other three. Errors are predicted/measured - 1."""
    fn, fc = frontier(cells, "NAT"), frontier_m(cells)
    out = {}
    for mdl, free in (("A", False), ("B", True)):
        def err(train, b):
            p = fit_joint(cells, free, train)[0]
            mm = p[3] if free else fc[b][1]
            x = BUD[b] / 1e20
            return dict(native=float(law(p[:3], x) / math.exp(fn[b]) - 1), convert=float(law(p[:3], mm * x) / math.exp(fc[b][0]) - 1))
        out[mdl] = dict(backtest_3e20=err(("1e19", "3e19", "1e20"), "3e20"),
                        lobo={b: err(tuple(bb for bb in BUD if bb != b), b) for b in BUD})
    # separate floored laws (three parameters per route, so three points determine each)
    fcq = {b: v[0] for b, v in fc.items()}
    sep = {}
    for b in BUD:
        pn = fit_floor({bb: fn[bb] for bb in BUD if bb != b}); pc = fit_floor({bb: fcq[bb] for bb in BUD if bb != b})
        sep[b] = dict(native=float(law(pn, BUD[b] / 1e20) / math.exp(fn[b]) - 1), convert=float(law(pc, BUD[b] / 1e20) / math.exp(fcq[b]) - 1))
    out["separate"] = dict(backtest_3e20=sep["3e20"], lobo=sep)
    return out


def efficient(cells):
    """Frontier advantage at equal 'efficient' FLOPs (kappa of an iMF implementation without the r=t JVP and with the
    interval-adjusted guidance pass only where needed): native at nominal C*k/k' and convert at C/(f + (1-f)k'/k),
    read off the separate laws of each route."""
    rng = np.random.default_rng(0)
    n = 2_000_000
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    t = np.maximum(sig(rng.normal(-0.4, 1, n)), sig(rng.normal(-0.4, 1, n)))
    p_out = float(np.mean((t < rng.uniform(0, 0.5, n)) | (t > rng.uniform(0.5, 1, n))))
    save = (0.5 + 0.5 * (1 - p_out)) / 3  # in FM steps: JVP tangent on the r=t half; extra pass inside the interval
    pn, pc = fit_floor(frontier(cells, "NAT")), fit_floor({b: v[0] for b, v in frontier_m(cells).items()})
    out = dict(p_out=p_out, kappa_saving_fm_steps=save, kappa={s: dict(measured=kappa(s), efficient=kappa(s) - save) for s in FT},
               ratio={})
    for b, (q, _) in frontier_m(cells).items():
        k = min((v, kk) for kk, v in cells.items() if kk[0] == "CONV" and kk[2] == b)[1]
        size, f = k[1], k[3]
        kk, kp = kappa(size), kappa(size) - save
        C = BUD[b] / 1e20
        meas = math.exp(frontier(cells, "NAT")[b] - q)
        adj = (law(pn, C) / law(pn, C * kk / kp)) / (law(pc, C) / law(pc, C / (f + (1 - f) * kp / kk)))
        out["ratio"][b] = dict(measured=meas, efficient=float(meas / adj))
    kk, kp = kappa("XL"), kappa("XL") - save
    adj = (law(pn, 10.0) / law(pn, 10.0 * kk / kp)) / (law(pc, 10.0) / law(pc, 10.0 / (0.6 + 0.4 * kp / kk)))
    meas = RUN21["nat"] / RUN21["conv"]
    out["ratio"]["1e21"] = dict(measured=meas, efficient=float(meas / adj))
    return out


def main():
    raw = nz.load_runs()
    base = nz.base_of(raw)
    noise = nz.estimate(raw)
    est = summary(base)
    rng = np.random.default_rng(20261025)
    boots = [summary(nz.perturb(raw, base, noise, rng)) for _ in range(N_BOOT)]

    def ci(get):
        v = np.array([get(b) for b in boots])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for mdl in ("A", "B"):
        for k in ("E", "K", "gamma", "nat21", "conv21"):
            est[mdl][k + "_ci95"] = ci(lambda b, mdl=mdl, k=k: b[mdl][k])
        for c in est[mdl]["ratio"]:
            est[mdl]["ratio_ci95_" + c] = ci(lambda b, mdl=mdl, c=c: b[mdl]["ratio"][c])
    est["B"]["m_hat_ci95"] = ci(lambda b: b["B"]["m_hat"])
    cv = np.array([b["B"]["curve"] for b in boots])
    est["B"]["curve_C"] = CGRID.tolist()
    est["B"]["curve_ci95"] = [np.percentile(cv, 2.5, axis=0).tolist(), np.percentile(cv, 97.5, axis=0).tolist()]
    # measured frontier ratios with intervals (grid budgets; the 1e21 runs get their routes' seed noise)
    fr = {}
    for b in BUD:
        q = frontier_m(base)[b][0]
        fr[b] = math.exp(frontier(base, "NAT")[b] - q)
    est["measured_ratio"] = {b: dict(point=v) for b, v in fr.items()}
    rng2 = np.random.default_rng(7)
    mr = []
    for _ in range(N_BOOT):
        c = nz.perturb(raw, base, noise, rng2)
        mr.append({b: math.exp(frontier(c, "NAT")[b] - frontier_m(c)[b][0]) for b in BUD})
    for b in BUD:
        v = [x[b] for x in mr]
        est["measured_ratio"][b]["ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    d21 = rng2.normal(0, noise["sd"]["NAT"], N_BOOT) - rng2.normal(0, noise["sd"]["CONV"], N_BOOT)
    r21 = RUN21["nat"] / RUN21["conv"]
    est["measured_ratio"]["1e21"] = dict(point=r21, ci95=[float(r21 * math.exp(np.percentile(d21, 2.5))),
                                                         float(r21 * math.exp(np.percentile(d21, 97.5)))])
    n = 8
    for mdl, k in (("A", 3), ("B", 4)):
        F = ((est[mdl]["rss"] - est["separate"]["rss"]) / (6 - k)) / (est["separate"]["rss"] / (n - 6))
        est[mdl].update(F=F, p=float(1 - fdist.cdf(F, 6 - k, n - 6)),
                        aic=n * math.log(est[mdl]["rss"] / n) + 2 * k, bic=n * math.log(est[mdl]["rss"] / n) + k * math.log(n),
                        rms=math.sqrt(est[mdl]["rss"] / n))
    est["separate"].update(aic=n * math.log(est["separate"]["rss"] / n) + 12, bic=n * math.log(est["separate"]["rss"] / n) + 6 * math.log(n),
                           rms=math.sqrt(est["separate"]["rss"] / n))
    # small-sample AIC (8 points); the separate laws (k = 6) leave one residual degree of freedom
    for mdl, k in (("A", 3), ("B", 4), ("separate", 6)):
        est[mdl]["aicc"] = est[mdl]["aic"] + 2 * k * (k + 1) / (n - k - 1)
    # predictive intervals for single runs at 1e21: law uncertainty plus the route's run-to-run noise
    for mdl in ("A", "B"):
        for k, route in (("nat21", "NAT"), ("conv21", "CONV")):
            v = np.array([b[mdl][k] for b in boots]) * np.exp(rng2.normal(0, noise["sd"][route], len(boots)))
            est[mdl][k + "_pred95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    est["held_out"] = held_out(base)
    est["runs_1e21"] = RUN21
    est["efficient"] = efficient(base)
    json.dump(est, open(f"{ROOT}/analysis/out/joint_law.json", "w"), indent=1, default=float)
    for mdl in ("A", "B"):
        e = est[mdl]
        extra = f" m_hat={e['m_hat']:.2f} [{e['m_hat_ci95'][0]:.2f}, {e['m_hat_ci95'][1]:.2f}]" if mdl == "B" else ""
        print(f"{mdl}: E={e['E']:.3f} {np.round(e['E_ci95'],2)} K={e['K']:.3f} g={e['gamma']:.3f}{extra} rms={e['rms']:.4f} "
              f"F={e['F']:.2f} p={e['p']:.3f} AIC={e['aic']:.1f} BIC={e['bic']:.1f}")
        print(f"   1e21: native {e['nat21']:.3f} {np.round(e['nat21_ci95'],3)} (run {RUN21['nat']:.3f}); convert {e['conv21']:.3f} "
              f"{np.round(e['conv21_ci95'],3)} (run {RUN21['conv']:.3f}); ratios " +
              ", ".join(f"{c}: {v:.3f} {np.round(est[mdl]['ratio_ci95_'+c],3)}" for c, v in e["ratio"].items()))
    s = est["separate"]
    print(f"separate: rms={s['rms']:.4f} AIC={s['aic']:.1f} BIC={s['bic']:.1f} nat21={s['nat21']:.3f} conv21={s['conv21']:.3f}")
    print("efficient:", {b: round(v["efficient"], 3) for b, v in est["efficient"]["ratio"].items()})
    for mdl in ("A", "B", "separate"):
        print(f"{mdl}: AICc={est[mdl]['aicc']:.1f}", *(f"{k}_pred95={np.round(est[mdl][k + '_pred95'], 2)}" for k in ("nat21", "conv21")
                                                       if k + "_pred95" in est[mdl]))
    for mdl, h in est["held_out"].items():
        print(f"{mdl} backtest 3e20:", {k: f"{100 * v:+.1f}%" for k, v in h["backtest_3e20"].items()},
              "| LOBO:", {b: (f"{100 * e['native']:+.0f}%", f"{100 * e['convert']:+.0f}%") for b, e in h["lobo"].items()})


if __name__ == "__main__":
    main()
