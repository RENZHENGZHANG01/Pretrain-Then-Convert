"""Overleaf package: main.tex, refs.bib, the style files, and only the tables and figures that main.tex uses.

  python tools/make_overleaf.py [zip path]   (default: pretrain_then_convert_overleaf.zip)
"""
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")


def used_files():
    tex = open(f"{P}/main.tex").read()
    tex = "\n".join(line.split("%")[0] if not line.lstrip().startswith("\\%") else line for line in tex.splitlines())
    tables = sorted(set(re.findall(r"\\input\{(tables/[^}]+)\}", tex)))
    sources = [tex] + [open(f"{P}/{t}").read() for t in tables]
    figs = sorted({f"figures/{g}" for s in sources for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)})
    style = sorted(f for f in os.listdir(P) if f.endswith((".sty", ".bst")))
    files = ["main.tex", "refs.bib"] + style + tables + figs
    missing = [f for f in files if not os.path.exists(f"{P}/{f}")]
    assert not missing, f"missing (run analysis/run_all.sh first): {missing}"
    return files


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "pretrain_then_convert_overleaf.zip")
    files = used_files()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f"{P}/{f}", f)
    print(f"{out}: {len(files)} files, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
