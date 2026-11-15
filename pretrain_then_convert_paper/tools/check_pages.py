"""Check the ICML 8-page limit on a build: the main text must end on page 8, in the right column and near the bottom,
and every float of the main text must sit on pages 1-8. Reads the build's main.aux (\\markendmain in main.tex writes
the page and position of the end of the Conclusion).

  python tools/check_pages.py [build dir]   (default: build/)
"""
import os
import re
import sys

B = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
aux = open(f"{B}/main.aux").read()
tex = open(f"{B}/main.tex").read()
main_part = tex.split("\\appendix")[0]
labels = re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", main_part)
pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux)}
m = re.search(r"\\gdef\s*\\endmainpos\s*\{(\d+),(\d+),(\d+)\}", aux)
if not m:
    sys.exit("no \\endmainpos in main.aux (is \\markendmain at the end of the Conclusion?)")
page, x, y = int(m.group(1)), int(m.group(2)) / 65536, int(m.group(3)) / 65536
late = {k: pages.get(k) for k in labels if pages.get(k, 0) > 8}
col = "right" if x > 307 else "left"
print(f"end of main text: page {page}, {col} column, {y:.0f}pt above the bottom edge (text area ends at ~72pt)")
print("main-text floats:", ", ".join(f"{k}:p{pages.get(k)}" for k in labels))
ok = page == 8 and col == "right" and not late
full = y < 100
print("PASS" if ok and full else ("PASS (page 8 not full: %.0fpt of white space)" % (y - 72) if ok else "FAIL"),
      "" if not late else f"floats after page 8: {late}")
