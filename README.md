# Where the Image Sits: Vision–Text Order Bias in Jev

Code and result files for the paper **“Where the Image Sits: Vision–Text Order
Bias in Jev.”**

The project studies what happens to a **system-1 decision readout** when the
image is placed at different positions in the prompt. The readout is one
prefill, a restricted softmax over the option-label tokens, and an argmax — no
generation, no parsing, and no sampled tokens.

The four image placements are:

```
prefix        [image] instruction / question / options / cue
pre-question  instruction [image] question / options / cue
pre-options   instruction / question [image] options / cue
suffix        instruction / question / options / cue [image]
```

The option listing is fixed; this repository does not study option rotation.

## Results at a glance

Across 42 model–dataset cells (7 checkpoints × 6 datasets, 300 items each):

| measurement | result |
|---|---|
| suffix vs prefix decision flip | **41.0%** |
| suffix accuracy | **0.708 → 0.522** (−18.6 points) |
| placement-invariant reference | **35.9%** suffix flip |
| Qwen family | 27.6% flip, −9.7 points |
| non-Qwen family | 58.9% flip, −30.5 points |
| full A-OKVQA/EuroSAT/Pets sets | 0.693 → 0.450, 47.4% flip |
| P+Q readout | 0.711 accuracy at **2.3%** flip |
| P+O readout | 0.708 at 8.2% (Qwen 0.753) |
| dual-order fusion β=0.75 | 0.706 at 5.8% |
| prefix readout | 0.708 at 0% |

The robust claim is **decision instability**, not a uniform accuracy loss. The
accuracy sign is dataset-dependent. Prefix-first is the cheapest strong default;
P+Q is the most stable two-forward readout; P+O is best on the Qwen rows but does
not replicate; dual-order fusion is a diagnostic and conditional stabilizer.

## Models

- Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct
- Qwen2.5-VL-7B-Instruct
- Idefics3-8B-Llama3
- LLaVA-OneVision-7B
- LLaVA-NeXT-7B

## Datasets

A-OKVQA, ScienceQA, CV-Bench, EuroSAT, Oxford-IIIT Pets, MMMU.

Dataset images and question files are **not redistributed**. See
`THIRD_PARTY.md` for the license of each dataset and model. The loaders take a
local image directory or data root; the paper reports the evaluation sizes.

## Repository layout

```
code/
  anyjev/                         readout package (Decider, Question, VisualState)
  scripts/                        dataset loader and pointwise scoring
  research2visualtexorder/
    run_order.py                  main run: placements, grey, pointwise
    attention_probe.py            every-layer, per-head attention probe
    label_permutation.py          label-permutation control
    fixed_distance.py             fixed-token-distance control
    downstream_utility.py         confidence-gated action proxy
    make_*_table.py               table generators
    paper_stats.py                paired tests and baselines
    paper/tables/                 generated LaTeX tables
data/
  results_2026-11-12.zip          result JSONs (no images, 41 MB compressed)
  README.md                       archive contents
requirements.txt
THIRD_PARTY.md
LICENSE
```

The result archive contains the `_4pos`, `_token` / 4B `_token6`, `_full`,
`_pointwise`, `_grey`, `_listing` and `analysis_*` files. Raw attention maps
(381 MB) are omitted; `analysis_attention_heads.json` is included.

## Setup

Python 3.10+ is required; the reported environment is Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate

# install the PyTorch build for your CUDA version first
pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

The source tree is not installed as a package. From the repository root, add
the code directories to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/code:$PWD/code/scripts"
```


`requirements.txt` intentionally leaves torch to this step. See
`anyjev/backends/hf_vlm.py` for the model wrapper.

## Reproduce one cell

Run from the repository root with the `PYTHONPATH` above:

```bash
python code/research2visualtexorder/run_order.py \
  --dataset aokvqa \
  --models Qwen/Qwen3-VL-4B-Instruct \
  --limit 300 \
  --image-dir /path/to/images \
  --positions prefix,middle,question,suffix \
  --pointwise \
  --out /tmp/result.json
```

For LLaVA checkpoints the chat template forces the image first, so use the
`input_ids` splice positions:

```bash
python code/research2visualtexorder/run_order.py \
  --dataset aokvqa \
  --models llava-hf/llava-v1.6-mistral-7b-hf \
  --limit 300 \
  --image-dir /path/to/images \
  --positions prefix,splice_middle,splice_question,splice_suffix \
  --out /tmp/result.json
```

`run_order.py` writes per-item distributions, predictions and metrics. The
readout is deterministic; nothing is sampled.

## Reproduce the tables

Unzip the committed results first:

```bash
cd code/research2visualtexorder
mkdir -p results/2026-11-12
unzip ../../data/results_2026-11-12.zip -d results/2026-11-12
```

Run the table scripts from `code/research2visualtexorder/` with the repository
`code` and `code/scripts` on `PYTHONPATH`:

```bash
python make_positions_table.py
python make_fuse_table.py
python make_fullcell_tables.py
python make_avg_tables.py
python make_aggregate_tables.py
python make_token_table.py
python make_grey_table.py
python paper_stats.py
python position_invariant.py
python position_invariant_loo.py
python token_invariant.py
python invariant_robustness.py
python label_interaction_all.py
```

The scripts write `.tex` files into `paper/tables/`, which is included for
reference. Figure PDFs are not required to regenerate the tables.

## Table-to-file map

| paper element | source |
|---|---|
| dataset properties | `code/scripts/visual_datasets.py` |
| per-cell position table | `*_4pos.json` |
| per-cell readout table | `*_4pos.json`, `*_pointwise.json` |
| averaging ablations | `*_4pos.json` |
| model/dataset aggregates | `*_4pos.json` |
| token-level sweep | `*_token.json`, 4B `*_token6.json` |
| paired checks / invariant reference | `analysis_paper_stats.json`, `analysis_position_invariant.json` |
| downstream proxy | `analysis_downstream*.json` |
| token distance | `analysis_distance*.json`, `analysis_fixed_distance*.json` |
| full sets | `*_full.json`, `analysis_full_*.json` |
| label permutation | `label_perm_*.json`, `analysis_label_*.json` |
| grey control | `*_grey.json` and 4B grey sections |
| attention | `analysis_attention_heads.json` |

## Hardware and software

The reported runs use one NVIDIA RTX 4090 D (24 GB), Python 3.12.4, PyTorch
2.14.1+cu126, transformers 5.20.2, CUDA 12.6 and bfloat16. Batch size is 4 for
the 2B model and 1 for the others; the image cap is 448×448 (200,704 pixels).
 
## Notes and limitations

- Pointwise covers A-OKVQA and EuroSAT only; it costs \(K\) forwards per item.
- Grey control covers the 4B pilot and the three non-Qwen checkpoints on those
  two datasets.
- Full-set confirmation covers A-OKVQA, EuroSAT and Pets.
- LLaVA templates force the image first; the splice is a structural instrument,
  not a native image-after-text template.
- Attention probes are descriptive, 100 items per dataset, not a causal circuit
  claim.
- Raw attention maps and dataset images are not included.

## License

The code is released under the repository’s Apache-2.0 license. Dataset and
model terms are listed in `THIRD_PARTY.md`.

## Citation

Citation information will be added when the paper is available.

```bibtex
@misc{visualjev2026,
  title  = {Where the Image Sits: Vision--Text Order Bias in Jev},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgements

The implementation builds on the AnyJev typed-decision interface and the public
Qwen, LLaVA, and Idefics model families. Dataset and model credits are listed in
`THIRD_PARTY.md`.
