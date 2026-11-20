"""Dataset adapters for the Stage 2 visual decision grid.

Every adapter returns a list of dicts:

    id, question, options, answer, images (tuple of local paths), state, meta

`options` are unique; `answer` is one of the option texts. AnyJev's letter
readout supports at most 26 options, so the 37-class Pets split is reduced to
the 26 largest test classes.
"""
from __future__ import annotations

import ast
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MAX_OPTIONS = 26


def dedupe_options(options: Sequence[Any]) -> List[str]:
    return list(dict.fromkeys(str(o) for o in options))


def coerce_options(options: Any) -> List[str]:
    """MMMU stores its option list as a string such as "['$6', '$7']"."""
    if isinstance(options, str):
        try:
            parsed = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            parsed = [options]
        options = parsed if isinstance(parsed, (list, tuple)) else [options]
    return list(options or [])


def canonical_options(options: Sequence[Any]) -> List[str]:
    """Sort the option set by text. Input-order invariant; no labels used."""
    return sorted(str(o) for o in options)


def answer_text(options: Sequence[str], answer: Any) -> Optional[str]:
    """Resolve an integer index, a letter (B) or a letter in parentheses ((C))."""
    if isinstance(answer, int):
        return options[answer] if 0 <= answer < len(options) else None
    s = str(answer).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        i = ord(s.upper()) - 65
        return options[i] if 0 <= i < len(options) else None
    return s


def expand_options(options: Sequence[str], answer: str, pool: Sequence[str],
                   k: int, rng: random.Random) -> Optional[List[str]]:
    """Append unique distractors from `pool` to reach k options, then shuffle."""
    opts = dedupe_options(options)
    if answer not in opts or k < len(opts) or k > MAX_OPTIONS:
        return None
    forbidden = set(opts)
    candidates = [x for x in dedupe_options(pool) if x not in forbidden]
    rng.shuffle(candidates)
    need = k - len(opts)
    if len(candidates) < need:
        return None
    opts += candidates[:need]
    if len(opts) != k or len(set(opts)) != k:
        return None
    rng.shuffle(opts)
    return opts


def _save(image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return str(path)


def _select(n: int, limit: int, seed: int) -> List[int]:
    idx = list(range(n))
    if limit and limit > 0:
        random.Random(seed).shuffle(idx)
        idx = idx[:limit]
    return idx


def _aokvqa(limit: int, seed: int, image_dir: Path) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    items = []
    for i in _select(len(ds), limit, seed):
        row = ds[i]
        original = [str(x) for x in row["choices"]]
        answer = original[int(row["correct_choice_idx"])]
        options = dedupe_options(original)
        if answer not in options:
            continue
        qid = str(row["question_id"])
        path = image_dir / f"{qid}.png"
        if not path.exists():
            _save(row["image"], path)
        items.append({"id": qid, "question": row["question"], "options": options,
                      "answer": answer, "images": (str(path),), "state": "",
                      "meta": {"source": "A-OKVQA"}})
    return items


def _scienceqa(limit: int, seed: int, image_dir: Path) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("derek-thomas/ScienceQA", split="test")
    rows = [i for i in range(len(ds)) if ds[i]["image"] is not None]
    items = []
    for k in _select(len(rows), limit, seed):
        i = rows[k]
        row = ds[i]
        original = [str(x) for x in row["choices"]]
        answer = answer_text(original, int(row["answer"]))
        options = dedupe_options(original)
        if answer is None or answer not in options:
            continue
        path = image_dir / f"scienceqa_{i}.png"
        if not path.exists():
            _save(row["image"], path)
        hint = row.get("hint") or ""
        items.append({"id": f"scienceqa_{i}", "question": row["question"], "options": options,
                      "answer": answer, "images": (str(path),),
                      "state": f"Hint: {hint}" if hint else "",
                      "meta": {"subject": row.get("subject"), "topic": row.get("topic")}})
    return items


def _mmmu(limit: int, seed: int, image_dir: Path) -> List[Dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names("MMMU/MMMU")
    items = []
    skipped_large = 0
    for cfg in configs:
        ds = load_dataset("MMMU/MMMU", cfg, split="validation")
        for i in range(len(ds)):
            if limit and limit > 0 and len(items) >= limit:
                break
            row = ds[i]
            if row.get("question_type") != "multiple-choice":
                continue
            options = dedupe_options(coerce_options(row.get("options")))
            if len(options) > MAX_OPTIONS:
                skipped_large += 1
                continue
            answer = answer_text(options, row.get("answer", ""))
            if answer is None or answer not in options:
                continue
            images = []
            for j in range(1, 8):
                image = row.get(f"image_{j}")
                if image is not None:
                    path = image_dir / f"mmmu_{row['id']}_{j}.png"
                    if not path.exists():
                        _save(image, path)
                    images.append(str(path))
            if not images:
                continue
            items.append({"id": str(row["id"]), "question": row["question"], "options": options,
                          "answer": answer, "images": tuple(images), "state": "",
                          "meta": {"subject": cfg}})
        if limit and limit > 0 and len(items) >= limit:
            break
    print(f"mmmu: skipped {skipped_large} multiple-choice items with more than {MAX_OPTIONS} options",
          flush=True)
    return items


def _cvbench(limit: int, seed: int, image_dir: Path) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    items = []
    for i in _select(len(ds), limit, seed):
        row = ds[i]
        options = dedupe_options(row["choices"])
        answer = answer_text(options, row["answer"])
        if answer is None or answer not in options:
            continue
        path = image_dir / f"cvbench_{i}.png"
        if not path.exists():
            _save(row["image"], path)
        items.append({"id": f"cvbench_{i}", "question": row["question"], "options": options,
                      "answer": answer, "images": (str(path),), "state": "",
                      "meta": {"type": row.get("type"), "task": row.get("task")}})
    return items


def _eurosat(limit: int, seed: int, image_dir: Path) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("tanganke/eurosat", split="test")
    names = [str(x) for x in ds.features["label"].names]
    items = []
    for i in _select(len(ds), limit, seed):
        row = ds[i]
        answer = names[int(row["label"])]
        path = image_dir / f"eurosat_{i}.png"
        if not path.exists():
            _save(row["image"], path)
        items.append({"id": f"eurosat_{i}", "question": "What land-use or land-cover class is shown?",
                      "options": names, "answer": answer, "images": (str(path),), "state": "",
                      "meta": {"label": answer}})
    return items


def _pets(limit: int, seed: int, image_dir: Path, data_root: Path) -> List[Dict[str, Any]]:
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root=str(data_root), split="test", download=True)
    classes = list(ds.classes)
    counts = Counter(ds._labels)
    keep = sorted(sorted(counts, key=lambda i: (-counts[i], classes[i]))[:MAX_OPTIONS])
    keep_set = set(keep)
    options = [classes[i] for i in keep]
    rows = [i for i in range(len(ds)) if ds._labels[i] in keep_set]
    items = []
    for k in _select(len(rows), limit, seed):
        i = rows[k]
        image, target = ds[i]
        answer = classes[target]
        path = image_dir / f"pets_{i}.jpg"
        if not path.exists():
            _save(image, path)
        items.append({"id": f"pets_{i}", "question": "What breed of pet is shown?",
                      "options": options, "answer": answer, "images": (str(path),), "state": "",
                      "meta": {"breed": answer}})
    return items


def _aokvqa_expanded(k: int, limit: int, seed: int, image_dir: Path) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    pool = sorted({str(x) for row in ds for x in row["choices"]})
    items = []
    for i in _select(len(ds), limit, seed):
        row = ds[i]
        original = [str(x) for x in row["choices"]]
        answer = original[int(row["correct_choice_idx"])]
        options = expand_options(original, answer, pool, k, random.Random(seed + i))
        if options is None:
            continue
        qid = str(row["question_id"])
        path = image_dir / f"{qid}.png"
        if not path.exists():
            _save(row["image"], path)
        items.append({"id": f"aokvqa_k{k}_{qid}", "question": row["question"], "options": options,
                      "answer": answer, "images": (str(path),), "state": "",
                      "meta": {"source": "A-OKVQA", "synthetic_k": k}})
    return items


def load_items(name: str, *, limit: int = 0, seed: int = 0,
               image_dir: str = "research/visual_bias/stage2_images",
               data_root: str = "/tmp/anyjev_data") -> List[Dict[str, Any]]:
    """Load one dataset. `limit=0` means the whole evaluation set."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    key = name.lower().replace("_", "-")
    if key == "aokvqa":
        return _aokvqa(limit, seed, image_dir)
    if key == "scienceqa":
        return _scienceqa(limit, seed, image_dir)
    if key == "mmmu":
        return _mmmu(limit, seed, image_dir)
    if key == "cvbench":
        return _cvbench(limit, seed, image_dir)
    if key == "eurosat":
        return _eurosat(limit, seed, image_dir)
    if key == "pets":
        return _pets(limit, seed, image_dir, Path(data_root))
    if key in ("aokvqa-k8", "aokvqa-k16"):
        return _aokvqa_expanded(int(key.rsplit("k", 1)[1]), limit, seed, image_dir)
    raise ValueError(f"unknown dataset {name!r}")
