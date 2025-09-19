# Phase 1.1 — Prices + LLM Dual-Stream News

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/run_pipeline.py --config configs/btc4h.yaml
```

Artifacts:
- `data/btc_4h.csv`
- `data/news_micro.json` (≤10)
- `data/news_macro.json` (≤10)
