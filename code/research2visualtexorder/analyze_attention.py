"""Summarize every-layer, per-head image-attention probes.

    python research2visualtexorder/analyze_attention.py \
        --results research2visualtexorder/results/2026-11-12

Reads the ``attention_*_all.json`` files written by ``attention_probe.py`` and
reports, per model and dataset, the layer-0 and final-layer prefix-to-suffix
change, the layer with the largest change, the number of heads above 50% image
attention in each placement, and the single largest head. Writes a JSON summary
and a LaTeX table. Every number is derived from the named run files; no value is
typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = {
    "qwen3-vl-2b": "Qwen3-VL-2B",
    "qwen3-vl-4b": "Qwen3-VL-4B",
    "qwen3-vl-8b": "Qwen3-VL-8B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
    "idefics3-8b": "Idefics3-8B",
    "llava-onevision": "LLaVA-OneVision-7B",
    "llava-v1.6": "LLaVA-NeXT-7B",
}
DATASETS = {"aokvqa": "A-OKVQA", "eurosat": "EuroSAT"}


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = data["positions"]
    if not all(p in positions for p in ("prefix", "suffix")):
        return {}
    prefix = positions["prefix"]["layers"]
    suffix = positions["suffix"]["layers"]
    layers = sorted(set(prefix) & set(suffix), key=int)
    if not layers:
        return {}

    def heads(layer: str, place: str) -> list[float]:
        return [float(x) for x in positions[place]["layers"][layer]["head_means"]]

    gaps = []
    for layer in layers:
        p = float(prefix[layer]["mean"])
        s = float(suffix[layer]["mean"])
        gaps.append({"layer": int(layer), "prefix": p, "suffix": s, "gap": s - p})

    cells = {"prefix": [], "suffix": []}
    for layer in layers:
        for place in cells:
            for head, value in enumerate(heads(layer, place)):
                cells[place].append({"layer": int(layer), "head": head, "value": value})

    def top(place: str) -> dict:
        best = max(cells[place], key=lambda x: x["value"])
        return best

    above = {place: sum(1 for c in cells[place] if c["value"] > 0.5) for place in cells}
    mean_head = {place: sum(c["value"] for c in cells[place]) / len(cells[place])
                 for place in cells}

    def count_ci(place: str, n_boot: int = 1000):
        mats = [np.asarray(positions[place]["layers"][layer].get("head_values", []),
                           dtype=float) for layer in layers]
        if any(m.size == 0 for m in mats):
            return None
        n = min(m.shape[0] for m in mats)
        rng = np.random.default_rng(0)
        counts = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            counts.append(sum(int((m[idx].mean(axis=0) > 0.5).sum()) for m in mats))
        return [int(x) for x in np.percentile(counts, [2.5, 97.5])]
    max_gap = max(gaps, key=lambda x: x["gap"])
    first, last = gaps[0], gaps[-1]
    share = {}
    option = {}
    for place in ("prefix", "suffix"):
        rows = positions[place]["layers"]
        share[place] = float(np.mean([rows[l].get("share_mean", 0.0) for l in layers]))
        option[place] = float(np.mean([rows[l].get("option_mean", 0.0) for l in layers]))
    return {
        "source": path.name,
        "model": data["model"],
        "dataset": data["dataset"],
        "n": data["n"],
        "n_layers": len(layers),
        "n_heads": len(cells["prefix"]) // len(layers),
        "first_layer": first,
        "last_layer": last,
        "max_gap_layer": max_gap,
        "mean_head_mass": mean_head,
        "heads_above_50pct": above,
        "heads_above_50pct_ci": {place: count_ci(place) for place in cells},
        "top_head": top("suffix"),
        "top_head_prefix": top("prefix"),
        "image_share": share,
        "option_mass": option,
        "per_layer": gaps,
        "per_layer_heads_above_50pct": {
            "prefix": {layer: sum(1 for h in heads(layer, "prefix") if h > 0.5)
                       for layer in layers},
            "suffix": {layer: sum(1 for h in heads(layer, "suffix") if h > 0.5)
                       for layer in layers},
        },
    }


def pretty_model(model: str) -> str:
    name = model.rstrip("/").split("/")[-1].lower()
    for key, label in MODELS.items():
        if key in name:
            return label
    return model.rstrip("/").split("/")[-1]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def latex_row(s: dict) -> str:
    def arrow(pair: dict) -> str:
        return f"{pct(pair['prefix'])}\\,$\\to$\\,{pct(pair['suffix'])}"

    top = s["top_head"]
    max_gap = s["max_gap_layer"]
    above = s["heads_above_50pct"]
    return (
        f"{pretty_model(s['model'])} & {DATASETS.get(s['dataset'], s['dataset'])} & "
        f"{s['n']} & {arrow(s['first_layer'])} & {arrow(s['last_layer'])} & "
        f"L{max_gap['layer']} ({arrow(max_gap)}) & "
        f"{above['prefix']}/{above['suffix']} & "
        f"{pct(top['value'])} (L{top['layer']},H{top['head']}) \\\\"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path(__file__).parent / "results/2026-11-12"))
    parser.add_argument("--pattern", default="attention_*_all.json")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--tex-out", default=None)
    args = parser.parse_args()

    results = Path(args.results)
    files = sorted(results.glob(args.pattern))
    summaries = [s for s in (summarize(p) for p in files) if s]
    order = list(MODELS.values())
    summaries.sort(key=lambda s: (order.index(pretty_model(s["model"]))
                                  if pretty_model(s["model"]) in order else 99,
                                  s["dataset"]))
    payload = {"results_dir": str(results), "files": [s["source"] for s in summaries],
               "summaries": summaries}
    json_out = Path(args.json_out) if args.json_out else results / "analysis_attention_heads.json"
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "% generated by analyze_attention.py from attention_*_all.json",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Model & Dataset & $n$ & Layer 0 & Last layer & Largest-gap layer & "
        "Heads $>$50\\% (pre/suf) & Max head \\\\",
        "\\midrule",
    ]
    lines += [latex_row(s) for s in summaries]
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_out = (Path(args.tex_out) if args.tex_out
               else Path(__file__).parent / "paper/tables/attention_heads.tex")
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    share_lines = [
        "% generated by analyze_attention.py; image share excludes option mass",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Dataset & image share pre & image share suf & option mass pre & "
        "option mass suf \\\\",
        "\\midrule",
    ]
    for s in summaries:
        share_lines.append(
            f"{pretty_model(s['model'])} & {DATASETS.get(s['dataset'], s['dataset'])} & "
            f"{pct(s['image_share']['prefix'])} & {pct(s['image_share']['suffix'])} & "
            f"{pct(s['option_mass']['prefix'])} & {pct(s['option_mass']['suffix'])} \\\\")
    share_lines += ["\\bottomrule", "\\end{tabular}"]
    share_out = tex_out.parent / "attention_share.tex"
    share_out.write_text("\n".join(share_lines) + "\n", encoding="utf-8")

    count_lines = [
        "% generated by analyze_attention.py; bootstrap over 100 items",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Model & Dataset & prefix count [95\\% CI] & suffix count [95\\% CI] \\\\",
        "\\midrule",
    ]
    for s in summaries:
        ci = s.get("heads_above_50pct_ci") or {}
        def fmt(place):
            value = s["heads_above_50pct"][place]
            bounds = ci.get(place)
            return f"{value}" if not bounds else f"{value} [{bounds[0]},{bounds[1]}]"
        count_lines.append(
            f"{pretty_model(s['model'])} & {DATASETS.get(s['dataset'], s['dataset'])} & "
            f"{fmt('prefix')} & {fmt('suffix')} \\\\")
    count_lines += ["\\bottomrule", "\\end{tabular}"]
    (tex_out.parent / "attention_counts.tex").write_text(
        "\n".join(count_lines) + "\n", encoding="utf-8")

    print(f"{'model':28s} {'data':8s} {'L0 pre':7s} {'L0 suf':7s} "
          f"{'last pre':8s} {'last suf':8s} {'gap layer':9s} {'>50 p/s':8s} {'max head'}")
    for s in summaries:
        print(f"{pretty_model(s['model']):28s} {s['dataset']:8s} "
              f"{pct(s['first_layer']['prefix']):7s} {pct(s['first_layer']['suffix']):7s} "
              f"{pct(s['last_layer']['prefix']):8s} {pct(s['last_layer']['suffix']):8s} "
              f"L{s['max_gap_layer']['layer']:<8d} "
              f"{s['heads_above_50pct']['prefix']}/{s['heads_above_50pct']['suffix']:<6d} "
              f"{pct(s['top_head']['value'])} (L{s['top_head']['layer']},H{s['top_head']['head']})")
    print(f"wrote {json_out}\nwrote {tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
