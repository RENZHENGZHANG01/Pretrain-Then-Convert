"""Comparison with published class-conditional ImageNet-256 results (FID-50k). Published rows are read from
data/lit/verified_results*.json, where every number carries a verbatim quote from the paper's arXiv LaTeX source and
was checked against that source when the record was made; with LIT_SOURCES set to a directory of the downloaded sources
(record field "source_checked" is relative to it), main() re-checks every quote before writing. Our rows come from the
evaluation summaries; epochs of our runs are FLOPs / (FLOPs per sample * 1,281,167 images).

  python analysis/lit_table.py  ->  paper/tables/t8_compare.tex, t3_compare_short.tex, ta_lit_sources.tex
"""
import glob
import json
import os
import re

from common import DATA, EVAL, ROOT

LIT = f"{DATA}/lit/verified_results.json"
LIT_EXTRA = f"{DATA}/lit/verified_results_extra.json"
SOURCES = os.environ.get("LIT_SOURCES")  # optional: downloaded arXiv sources, to re-check the stored quotes
N_IMG = 1281167
FT = json.load(open(f"{ROOT}/data/flops_table.json"))  # measured training FLOPs per sample

G1, G2, G3 = ("One-step, one training run from scratch", "One-step, starting from a pretrained flow model or teacher",
              "Multi-step (NFE = steps $\\times$ 2 passes for guidance)")
# group, variant (prefix match in the records), NFE filter, name, cite key, NFE shown, labels, epochs shown, note,
# optional (variant, label) of a second record quoted in the note
PUB = [
    (G1, "MeanFlow-XL/2", "1", "MeanFlow-XL/2", "meanflow", "1", "--", "240", "", None),
    (G1, "Shortcut (DiT-XL", "1", "Shortcut-XL", "shortcut", "1", "--", "160$^a$", "", None),
    (G1, "IMM-XL/2 (w=1.25", "1 step", "IMM-XL/2", "imm", "1$\\times$2", "--", "$\\approx$3800", "guidance at sampling", None),
    (G1, "TiM-XL", "1", "TiM-XL", "tim", "1", "--", "300", "", None),
    (G1, "SoFlow-XL/2", "1", "SoFlow-XL/2", "soflow", "1", "--", "240", "", None),
    (G1, "alpha-Flow-XL/2+ (class-balanced", "1", "$\\alpha$-Flow-XL/2+", "alphaflow", "1", "bal.", "240+60", "random labels: {alt}",
     ("alpha-Flow-XL/2+", "1")),
    (G1, "AFM-XL/2", "1", "AFM-XL/2", "afm", "1", "bal.", "125$^b$", "adversarial; classifier-guidance loss", None),
    (G1, "pMF-H/16", "1", "pMF-H/16", "pmf", "1", "--", "360", "pixel space; perceptual losses", None),
    (G1, "iMF-L/2", "1", "iMF-L/2", "imf", "1", "bal.", "640", "{flops_L}$^c$", None),
    (G1, "iMF-XL/2", "1", "iMF-XL/2", "imf", "1", "bal.", "800", "{flops_XL}$^c$", None),
    (G1, "Drifting L/2", "1", "Drifting-L/2", "drifting", "1", "--", "1280$^d$", "pretrained feature encoder in the loss", None),
    (G1, "W-Flow XL/2", "1", "W-Flow-XL/2", "wflow", "1", "--", "1280$^d$", "pretrained feature encoder in the loss", None),
    (G1, "UOT-GF XL/2 (200k)", "1", "UOT-GF-XL/2", "uotgf", "1", "--", "200k it", "pretrained feature encoder in the loss", None),
    (G2, "DMF-XL/2 (240", "1", "Decoupled MF-XL/2", "decoupledmeanflow", "1", "--", "160+80", "own FM pretraining", None),
    (G2, "DMF-XL/2+", "1", "Decoupled MF-XL/2+", "decoupledmeanflow", "1", "--", "800+80", "from REPA SiT-XL/2", None),
    (G2, "pi-Flow GM-REPA", "1", "$\\pi$-Flow (GM-REPA)", "piflow", "1", "--", "800+$\\approx$450", "distilled from REPA SiT-XL/2", None),
    (G2, "MF-RAE", "1", "MeanFlow-RAE", "mfrae", "1", "--", "800+36", "RAE latent; distilled from RAE", None),
    (G2, "FACM (LightningDiT-XL)", "1", "FACM", "facm", "1", "--", "n.s.$^e$", "VA-VAE latent; from LightningDiT", None),
    (G2, "PVD (Ours)", "1", "PVD", "pvd", "1", "--", "n.s.", "distilled, FACM teacher setup", None),
    (G2, "Priming-L (160 ep.)", "1", "Priming-L", "priming", "1", "bal.", "160+25k it", "pixel; own denoising pretraining; pretrained encoders$^g$", None),
    (G3, "DiT-XL/2-G", "250", "DiT-XL/2", "dit", "250$\\times$2", "rand.", "1400", "", None),
    (G3, "SiT-XL (cfg=1.5", "250", "SiT-XL", "sit", "250$\\times$2", "rand.", "1400", "balanced labels: {alt}$^f$",
     ("SiT-XL/2, w/ guidance, class-balanced", "as SiT")),
    (G3, "SiT-XL/2 + REPA", "250", "REPA (SiT-XL/2)", "repa", "250$\\times$2", "rand.", "800",
     "DINOv2 alignment; balanced labels: {alt}$^f$", ("SiT-XL/2+REPA, w/ guidance, class-balanced", "as REPA")),
    (G3, "LightningDiT-XL", "250", "LightningDiT-XL", "lightningdit", "250$\\times$2", "--", "800", "VA-VAE latent (DINOv2-aligned)", None),
    (G3, "JiT-H/16", "50", "JiT-H/16", "jit", "100$\\times$2", "bal.", "600", "pixel space; 50 Heun steps", None),
    (G3, "DiT^DH-XL (DINOv2-B RAE), AutoGuidance", "50", "RAE (DiT$^\\text{DH}$-XL)", "rae", "50$\\times$2", "bal.", "800",
     "RAE latent; AutoGuidance", None),
]

# the short comparison of the main text: (group, variant, NFE filter, name, cite key, NFE shown, text after the citation)
SHORT = [
    ("One-step, trained from scratch", "MeanFlow-XL/2", "1", "MeanFlow-XL/2", "meanflow", "1", ""),
    ("One-step, trained from scratch", "SoFlow-XL/2", "1", "SoFlow-XL/2", "soflow", "1", ""),
    ("One-step, trained from scratch", "alpha-Flow-XL/2+ (class-balanced", "1", "$\\alpha$-Flow-XL/2+", "alphaflow", "1", ""),
    ("One-step, trained from scratch", "iMF-L/2", "1", "iMF-L/2", "imf", "1", ", {flops_L}"),
    ("One-step, trained from scratch", "iMF-XL/2", "1", "iMF-XL/2", "imf", "1", ", {flops_XL}"),
    ("Multi-step references", "DiT-XL/2-G", "250", "DiT-XL/2", "dit", "250$\\times$2", ""),
    ("Multi-step references", "SiT-XL (cfg=1.5", "250", "SiT-XL", "sit", "250$\\times$2", ""),
]


def pick(recs, variant, nfe):
    m = [r for r in recs if str(r.get("variant", "")).startswith(variant) and str(r.get("nfe", "")).startswith(nfe)]
    if not variant.startswith(("SiT-XL/2, w/", "SiT-XL/2+REPA, w/")):
        m = [r for r in m if "re-evaluated" not in r["method"] and "reproduced" not in r["method"]]
    if len(m) > 1:  # e.g. "alpha-Flow-XL/2+" vs its balanced-label record, FACM v2 vs v1: take the exact name
        m = [r for r in m if r["variant"] == variant]
    assert len(m) == 1, (variant, nfe, [r["variant"] for r in m])
    return m[0]


def quote_in_source(r):
    """The stored quote (whitespace-normalized) must occur in one of the paper's downloaded .tex files (only checked
    with LIT_SOURCES; otherwise the check made when the record was built stands)."""
    if not SOURCES:
        return r.get("quote_verified_in_source")
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    q = norm(r["quote"])
    for p in glob.glob(f"{SOURCES}/{r['source_checked']}/**/*.tex", recursive=True):
        if q in norm(open(p, errors="ignore").read()):
            return p
    return None


def ep(c, size, obj):
    return c / FT[size][obj] / N_IMG


def sci(x):
    e = int(f"{x:e}".split("e")[1])
    return f"$\\approx{x / 10 ** e:.1f}{{\\times}}10^{{{e}}}$ FLOPs"


def ours():
    s = lambda rid: json.load(open(f"{EVAL}/{rid}/summary.json"))["fid50k"]
    off = [json.loads(l) for l in open(f"{EVAL}/official_imf/results.jsonl")]
    off_xl = min(o["fid"] for o in off if o["size"] == "XL" and o["n"] == 50000)
    return [
        ("iMF-XL/2, official weights (our evaluation)", "610M", "1", off_xl, "800", ""),
        ("Native one-step XL (ours)", "604M", "1", s("v1_XL_1e21_cd_s0"), f"{ep(1e21, 'XL', 'imf'):.0f}", "$10^{21}$ FLOPs"),
        ("Pretrain$\\to$convert XL, $f{=}0.60$ (ours)", "604M", "1", s("v2_XL_1e21_f0.60_s0"),
         f"{ep(6e20, 'XL', 'fm'):.0f}+{ep(4e20, 'XL', 'imf'):.0f}", "$10^{21}$ FLOPs"),
        ("Multi-step FM, H (ours)", "1.36B", "70", s("v2_H_1e21_cd_s0"), f"{ep(1e21, 'H', 'fm'):.0f}",
         "$10^{21}$ FLOPs; guidance interval"),
    ]


def short_table(recs, fl):
    s = lambda rid: json.load(open(f"{EVAL}/{rid}/summary.json"))["fid50k"]  # noqa: E731
    rows = {g: [] for g, *_ in SHORT}
    for g, var, nfe, name, key, nfe_s, extra in SHORT:
        r = pick(recs, var, nfe)
        rows[g].append(f"{name} \\citep{{{key}}}{extra.format(**fl)} & {r['params'].split('+')[0].split(' ')[0]} & {nfe_s} & "
                       f"{float(r['fid50k']):.2f} \\\\")
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Method & Params & NFE & FID \\\\", "\\midrule",
         "\\multicolumn{4}{l}{\\emph{One-step, trained from scratch}} \\\\"] + rows["One-step, trained from scratch"]
    L += ["\\midrule", "\\multicolumn{4}{l}{\\emph{Ours, $10^{21}$ FLOPs}} \\\\",
          f"Native one-step XL & 604M & 1 & {s('v1_XL_1e21_cd_s0'):.2f} \\\\",
          f"Pretrain$\\to$convert XL ($f{{=}}0.6$) & 604M & 1 & \\textbf{{{s('v2_XL_1e21_f0.60_s0'):.2f}}} \\\\",
          "\\midrule", "\\multicolumn{4}{l}{\\emph{Multi-step references}} \\\\"] + rows["Multi-step references"]
    L += [f"Ours: multi-step H, $10^{{21}}$ FLOPs & 1.36B & 70 & {s('v2_H_1e21_cd_s0'):.2f} \\\\", "\\bottomrule", "\\end{tabular}"]
    open(f"{ROOT}/paper/tables/t3_compare_short.tex", "w").write("\n".join(L) + "\n")


def main():
    recs = json.load(open(LIT)) + json.load(open(LIT_EXTRA))
    fl = dict(flops_L=sci(640 * N_IMG * FT["L"]["imf"]), flops_XL=sci(800 * N_IMG * FT["XL"]["imf"]))
    short_table(recs, {k: v.replace("\\approx", "\\approx\\!") for k, v in fl.items()})
    L = ["\\begin{tabular}{lcccccl}", "\\toprule",
         "Method & Params & NFE & FID-50k & Labels & Epochs & Notes \\\\"]
    src, group, bad = [], None, []
    for g, var, nfe, name, key, nfe_s, lab, epo, note, alt in PUB:
        r = pick(recs, var, nfe)
        if not (r.get("quote_verified_in_source") and quote_in_source(r)):
            bad.append(name)
        a = ""
        if alt:
            ra = pick(recs, *alt)
            a = f"{float(ra['fid50k']):.2f}"
            if not quote_in_source(ra):
                bad.append(name + " (alt)")
            src.append((name + " (" + ("random labels" if "random" in note else "balanced labels") + ")", ra))
        if g != group:
            L += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{g}}}}} \\\\"]
            group = g
        params = r["params"].split("+")[0].split(" ")[0]
        nt = note.format(alt=a, **fl)
        L.append(f"{name} \\citep{{{key}}} & {params} & {nfe_s} & {float(r['fid50k']):.2f} & {lab} & {epo} & {nt} \\\\")
        src.append((name, r))
        print(f"{name:26s} {params:6s} NFE {nfe_s:14s} FID {float(r['fid50k']):5.2f}  {lab:5s} ep {epo:18s} {nt}")
    L += ["\\midrule", "\\multicolumn{7}{l}{\\emph{This work (SD-VAE latent, imfDiT, batch 256)}} \\\\"]
    for name, p, n, f, epo, note in ours():
        L.append(f"{name} & {p} & {n} & {f:.2f} & bal. & {epo} & {note} \\\\")
        print(f"{name:26s} {p:6s} NFE {n:14s} FID {f:5.2f}  bal.  ep {epo:18s} {note}")
    L += ["\\bottomrule", "\\end{tabular}"]
    assert not bad, f"quote not found in source: {bad}"
    open(f"{ROOT}/paper/tables/t8_compare.tex", "w").write("\n".join(L) + "\n")
    # sources: arXiv id, version and location of every published number
    S = ["\\begin{tabular}{lll}", "\\toprule", "Row & arXiv (version) & Location in the source \\\\", "\\midrule"]
    for name, r in src:
        ver = str(r["arxiv_version"]).split(" ")[0].lstrip("v")
        loc = re.sub(r"\s*\((?:tab|tbl|table):[^)]*\)|\s*\(maintable\)", "", r["location"])  # LaTeX labels of the source
        loc = re.sub(r"(?:main table|Table)\s*\(tables/[^)]*\)", "main results table", loc)  # file names of the source
        loc = loc.replace("_", "\\_").replace("&", "\\&").replace("#", "\\#").replace("^", "\\^{}")
        loc = re.sub(r'"([^"]*)"', r"``\1''", loc)
        S.append(f"{name} & {r['arxiv_id']} (v{ver}) & {loc[:70]} \\\\")
    S += ["\\bottomrule", "\\end{tabular}"]
    open(f"{ROOT}/paper/tables/ta_lit_sources.tex", "w").write("\n".join(S) + "\n")
    print("all", len(src), "quoted numbers " + ("found verbatim in the downloaded sources" if SOURCES else "verified when the records were made"))


if __name__ == "__main__":
    main()
