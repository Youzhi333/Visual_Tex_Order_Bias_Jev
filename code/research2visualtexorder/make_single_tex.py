"""Build paper/main_single.tex from the modular sources.

    python research2visualtexorder/make_single_tex.py

Expands every \\input of a preamble, section or table and inlines the compiled
bibliography (paper/main.bbl) so that main_single.tex is a single source file.
It still reads cvpr.sty and the PDFs in paper/figures/.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
INPUT = re.compile(r"\\input\{([^}]+)\}")


def expand(path: Path, seen: set[Path]) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        def inline(m: re.Match[str]) -> str:
            child = PAPER / m.group(1)
            if child.suffix != ".tex":
                child = child.with_suffix(".tex")
            child = child.resolve()
            if child in seen:
                raise SystemExit(f"recursive input: {child}")
            seen.add(child)
            body = "\n".join(expand(child, seen))
            seen.discard(child)
            tag = f"% ===== inlined from {child.relative_to(PAPER)} ====="
            return f"\n{tag}\n{body}\n"

        out.extend(INPUT.sub(inline, line).splitlines())
    return out


def main() -> int:
    lines = expand(PAPER / "main.tex", set())
    text = "\n".join(lines)
    bbl = (PAPER / "main.bbl").read_text(encoding="utf-8").strip()
    marker = "\\bibliographystyle{ieeenat_fullname}"
    start = text.index(marker)
    end = text.index("\\bibliography{main}", start) + len("\\bibliography{main}")
    head = ("% Single-file build: every section, table and the bibliography are\n"
            "% inlined. Regenerate with research2visualtexorder/make_single_tex.py.\n"
            "% The build still reads cvpr.sty and the PDFs in figures/.\n")
    text = head + text[:start] + bbl + text[end:]
    (PAPER / "main_single.tex").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {PAPER / 'main_single.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
