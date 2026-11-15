"""Two-component account of the conversion f-curves.

A native one-step model of size N trained on S samples has log FID g_N(S) (its measured curve, log-log interpolated
as in analysis/matched_steps.py). Split it into a velocity term and a one-step deficit,

    g_N(S) = V_N(S) + D_N(S),    D_N(S) = log(1 + (tau_N / S)^beta),    tau_N = tau * (N / 10^8)^-delta.

A conversion at split f has seen S_FM = f C / c_FM flow-matching samples and S_1 = (1 - f) C / c_iMF one-step samples.
Both train the velocity field; only the one-step samples train the one-step map:

    log FID_conv = V_N(lam * S_FM + S_1) + D_N(S_1),

with lam the value of an FM sample in the velocity term relative to a one-step sample. f = 0 gives back the native
curve. Fitted by least squares on log FID to every conversion whose equal-samples count lies inside its size's native
curve (the cells of the exchange analysis); validated by leaving out one budget or one size. Conversions at f < 0.2
are held out of the fit and predicted (they lie below every split of the main grid). Intervals: parametric bootstrap
with the seed-noise model of analysis/noise.py, refitting each replicate.

  python analysis/two_component.py  ->  analysis/out/two_component.json, paper/tables/ta_twocomp.tex
"""
import json
import math
import warnings

import numpy as np
from scipy.optimize import least_squares

import matched_steps as ms
import noise as nz

ROOT = ms.ROOT
FT, BUD = ms.FT, ms.BUD
F_LOW = 0.2  # conversions below this split are held out and predicted
FORMS = {  # name: (initial values, lower bounds, upper bounds) of (log lam, log tau, beta[, delta])
    "c3": ([0.5, 15.0, 0.5], [-3, 5, 0.05], [3, 25, 5]),
    "c4": ([0.5, 15.0, 0.5, 0.5], [-3, 5, 0.05, -3], [3, 25, 5, 5]),
}
FGRID = np.linspace(0.02, 0.995, 391)
BUD_TEX = {"1e19": "$10^{19}$", "3e19": "$3{\\times}10^{19}$", "1e20": "$10^{20}$", "3e20": "$3{\\times}10^{20}$"}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def samples(size, b, f):
    C = BUD[b]
    return f * C / FT[size]["fm"], (1 - f) * C / FT[size]["imf"]


def deficit(S, tau, beta):
    return math.log1p((tau / S) ** beta)


def predict(p, curves, size, b, f):
    lam, tau, beta = math.exp(p[0]), math.exp(p[1]), p[2]
    tau *= (ms.FT[size]["N_inference"] / 1e8) ** (-(p[3] if len(p) > 3 else 0.0))
    sf, s1 = samples(size, b, f)
    sv = lam * sf + s1
    return ms.native_at(curves[size], math.log(sv))[0] - deficit(sv, tau, beta) + deficit(s1, tau, beta)


def inside(curves, size, b, f):
    sf, s1 = samples(size, b, f)
    ls = curves[size][0]
    return ls[0] <= math.log(sf + s1) <= ls[-1]


def split_cells(cells, curves):
    use = [(k, v) for k, v in cells.items() if k[0] == "CONV" and k[1] in curves and inside(curves, *k[1:])]
    return [c for c in use if c[0][3] >= F_LOW], [c for c in use if c[0][3] < F_LOW]


def fit(form, data, curves, p0=None):
    init, lb, ub = FORMS[form]
    y = np.array([v for _, v in data])

    def res(p):
        return np.array([predict(p, curves, *k[1:]) for k, _ in data]) - y
    starts = [p0] if p0 is not None else [init] + [np.clip(np.array(init) + np.random.default_rng(j).normal(0, 0.7, len(init)),
                                                           np.array(lb) + 1e-6, np.array(ub) - 1e-6) for j in range(1, 8)]
    best = None
    for q in starts:
        r = least_squares(res, q, bounds=(lb, ub))
        if best is None or r.cost < best.cost:
            best = r
    return best.x, best.fun


def rms(x):
    return float(math.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def by_split(data, r):
    return rms([e for (k, _), e in zip(data, r) if k[3] < 0.95]), rms([e for (k, _), e in zip(data, r) if k[3] >= 0.95])


def held_out(form, data, curves):
    out = {}
    for key, idx in (("budget", 2), ("size", 1)):
        errs = []
        for g in sorted({k[idx] for k, _ in data}):
            tr = [c for c in data if c[0][idx] != g]
            te = [c for c in data if c[0][idx] == g]
            p, _ = fit(form, tr, curves)
            errs += [(c, predict(p, curves, *c[0][1:]) - c[1]) for c in te]
        out[key] = by_split([c for c, _ in errs], [e for _, e in errs])
    return out


def fstar(p, curves, data):
    out = {}
    for size, b in sorted({(k[1], k[2]) for k, _ in data}):
        cc = [(k, v) for k, v in data if k[1] == size and k[2] == b]
        if len(cc) < 3:
            continue
        q = [predict(p, curves, size, b, f) for f in FGRID]
        i = int(np.argmin(q))
        best = min(cc, key=lambda c: c[1])
        out[f"{size}@{b}"] = dict(f_model=float(FGRID[i]), fid_model=math.exp(q[i]), f_best=best[0][3], fid_best=math.exp(best[1]))
    return out


def analyse(cells, nat21, form, p0=None):
    curves = ms.native_curves(cells, nat21)
    data, low = split_cells(cells, curves)
    p, r = fit(form, data, curves, p0)
    return dict(p=p, data=data, low=low, curves=curves, res=r, fstar=fstar(p, curves, data),
                low_pred=[(k, predict(p, curves, *k[1:]), v) for k, v in low])


def params(p):
    d = dict(lam=math.exp(p[0]), tau_samples=math.exp(p[1]), beta=float(p[2]))
    if len(p) > 3:
        d["delta"] = float(p[3])
    return d


def main(n_boot=1000):
    raw = nz.load_runs()
    base = nz.base_of(raw)
    noise = nz.estimate(raw)
    sd = noise["sd"]
    floor = (math.hypot(sd["CONV"], sd["NAT"]), math.hypot(sd["SHORT"], sd["NAT"]))
    out = dict(noise_floor=dict(mid=floor[0], late=floor[1]), forms={})
    for form in FORMS:
        a = analyse(base, ms.NAT21_XL, form)
        n, k = len(a["data"]), len(a["p"])
        rss = float(np.sum(a["res"] ** 2))
        fitq = by_split(a["data"], a["res"])
        ho = held_out(form, a["data"], a["curves"])
        rng = np.random.default_rng(20261026)
        boots = []
        for _ in range(n_boot):
            cells = nz.perturb(raw, base, noise, rng)
            b = analyse(cells, ms.NAT21_XL + rng.normal(0, sd["NAT"]), form, p0=a["p"])
            boots.append(b)

        def ci(vals):
            return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        pb = [params(b["p"]) for b in boots]
        e = dict(params=params(a["p"]), params_ci95={kk: ci([x[kk] for x in pb]) for kk in params(a["p"])},
                 n_cells=n, n_mid=sum(1 for c in a["data"] if c[0][3] < 0.95), n_params=k,
                 rms_mid=fitq[0], rms_late=fitq[1], aic=n * math.log(rss / n) + 2 * k, bic=n * math.log(rss / n) + k * math.log(n),
                 held_out_budget=dict(mid=ho["budget"][0], late=ho["budget"][1]),
                 held_out_size=dict(mid=ho["size"][0], late=ho["size"][1]),
                 fstar={}, low=[])
        for cell, v in a["fstar"].items():
            fb = [b["fstar"][cell]["f_model"] for b in boots if cell in b["fstar"]]
            e["fstar"][cell] = dict(v, f_model_ci95=ci(fb))
        for i, (kk, q, v) in enumerate(a["low_pred"]):
            qb = [b["low_pred"][i][1] for b in boots if len(b["low_pred"]) > i]
            e["low"].append(dict(cell=f"{kk[1]}@{kk[2]}", f=kk[3], predicted=math.exp(q), measured=math.exp(v), log_err=v - q,
                                 predicted_ci95=[math.exp(x) for x in ci(qb)]))
        # model curves for the figure: equal-samples count and predicted FID along f, every fitted cell
        e["curves"] = {}
        for size, b in sorted({(c[0][1], c[0][2]) for c in a["data"]}):
            xs = [sum(samples(size, b, f)) for f in FGRID]
            e["curves"][f"{size}@{b}"] = dict(f=FGRID.tolist(), samples=xs,
                                              fid=[math.exp(predict(a["p"], a["curves"], size, b, f)) for f in FGRID])
        out["forms"][form] = e
        fs = [v["f_model"] for v in e["fstar"].values()]
        print(f"{form}: {json.dumps({kk: round(v, 3) for kk, v in e['params'].items()})} CI {json.dumps({kk: [round(x, 3) for x in v] for kk, v in e['params_ci95'].items()})}")
        print(f"   n={n} rms mid {fitq[0]:.3f} late {fitq[1]:.3f} (noise floor {floor[0]:.3f} / {floor[1]:.3f}); AIC {e['aic']:.1f} BIC {e['bic']:.1f}")
        print(f"   held out: budget {ho['budget'][0]:.3f}/{ho['budget'][1]:.3f}, size {ho['size'][0]:.3f}/{ho['size'][1]:.3f}; f* {min(fs):.2f}-{max(fs):.2f}")
        for x in e["low"]:
            print(f"   {x['cell']} f={x['f']}: predicted {x['predicted']:.2f} [{x['predicted_ci95'][0]:.2f}, {x['predicted_ci95'][1]:.2f}], measured {x['measured']:.2f}")
    json.dump(out, open(f"{ROOT}/analysis/out/two_component.json", "w"), indent=1)
    table(out)


def table(out):
    c3, c4 = out["forms"]["c3"], out["forms"]["c4"]

    def pci(e, key, fmt):
        if key not in e["params"]:
            return "---"
        lo, hi = e["params_ci95"][key]
        return f"{fmt(e['params'][key])} [{fmt(lo)}, {fmt(hi)}]"
    mil = lambda x: f"{x / 1e6:.1f}M"  # noqa: E731
    two = lambda x: f"{x:.2f}"  # noqa: E731
    fl = out["noise_floor"]
    L = ["\\begin{tabular}{lcc}", "\\toprule", "& $\\tau$ shared by sizes & $\\tau\\propto N^{-\\delta}$ \\\\", "\\midrule",
         f"$\\lambda$ (value of an FM sample in the velocity term) & {pci(c3, 'lam', two)} & {pci(c4, 'lam', two)} \\\\",
         f"$\\tau$ (one-step samples; $N=10^8$) & {pci(c3, 'tau_samples', mil)} & {pci(c4, 'tau_samples', mil)} \\\\",
         f"$\\beta$ & {pci(c3, 'beta', two)} & {pci(c4, 'beta', two)} \\\\",
         f"$\\delta$ & --- & {pci(c4, 'delta', two)} \\\\", "\\midrule",
         f"rms log FID, $f<0.95$ / $f\\geq0.95$ (noise {fl['mid']:.3f} / {fl['late']:.3f}) & "
         f"{c3['rms_mid']:.3f} / {c3['rms_late']:.3f} & {c4['rms_mid']:.3f} / {c4['rms_late']:.3f} \\\\",
         f"held-out budget & {c3['held_out_budget']['mid']:.3f} / {c3['held_out_budget']['late']:.3f} & "
         f"{c4['held_out_budget']['mid']:.3f} / {c4['held_out_budget']['late']:.3f} \\\\",
         f"held-out size & {c3['held_out_size']['mid']:.3f} / {c3['held_out_size']['late']:.3f} & "
         f"{c4['held_out_size']['mid']:.3f} / {c4['held_out_size']['late']:.3f} \\\\",
         f"AIC / BIC ({c4['n_cells']} conversions) & {c3['aic']:.0f} / {c3['bic']:.0f} & {c4['aic']:.0f} / {c4['bic']:.0f} \\\\"]
    for x3, x4 in zip(c3["low"], c4["low"]):
        size, b = x3["cell"].split("@")
        L.append(f"{size} at {BUD_TEX[b]}, $f={x3['f']:g}$ (not fitted; measured {x3['measured']:.2f}) & "
                 f"{x3['predicted']:.2f} & {x4['predicted']:.2f} \\\\")
    fs3 = [v["f_model"] for v in c3["fstar"].values()]
    fs4 = [v["f_model"] for v in c4["fstar"].values()]
    L += [f"implied $f^*$ over cells & {min(fs3):.2f}--{max(fs3):.2f} & {min(fs4):.2f}--{max(fs4):.2f} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    open(f"{ROOT}/paper/tables/ta_twocomp.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
