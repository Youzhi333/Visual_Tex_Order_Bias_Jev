# Results archive

`results_2026-11-12.zip` contains the JSON files behind the paper's tables and
prose. Unzip it into `code/research2visualtexorder/results/2026-11-12/`.

File groups:

- `*_4pos.json` — four placement distributions and metrics.
- `*_token.json`, `*_token6.json` — token-level sweeps (4B uses token6).
- `*_full.json` — full A-OKVQA/EuroSAT/Pets runs.
- `*_pointwise.json` — pointwise control on A-OKVQA/EuroSAT.
- `*_grey.json` and the 4B grey sections — grey-image control.
- `*_listing_*.json` — alternate fixed listings.
- `analysis_*.json` — derived summaries (statistics, distance, invariant,
  downstream, head ablation, attention heads, label interaction).

Raw full per-layer attention maps are not included (381 MB uncompressed);
their summary is in `analysis_attention_heads.json`. Dataset images and
questions are not redistributed.
