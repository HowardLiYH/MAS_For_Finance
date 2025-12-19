# Configuration Files

This directory contains YAML configuration files for the multi-agent trading system.

## Directory Layout

```
configs/
├── multi_asset.yaml      # 5-coin trading with cross-asset features (RECOMMENDED)
├── default.yaml          # Default single-asset configuration
├── single/               # Individual coin configs for backtesting
│   ├── btc.yaml
│   ├── eth.yaml
│   ├── sol.yaml
│   ├── doge.yaml
│   └── xrp.yaml
└── news_urls.txt         # Candidate URLs for news search
```

## Usage

### Multi-Asset Mode (Recommended)
Trade all 5 coins with cross-asset market intelligence:
```bash
trading-agents multi --config configs/multi_asset.yaml
```

### Single-Asset Mode
Trade a single coin:
```bash
trading-agents run --config configs/single/btc.yaml
trading-agents run --config configs/single/eth.yaml
```

## Configuration Sections

### `data:`
```yaml
data:
  multi_asset: true/false    # Enable multi-asset mode
  symbols: [BTC, ETH, ...]   # Coins to trade (multi-asset only)
  bybit_csv_dir: "data/bybit" # Path to Bybit CSV files
  add_cross_features: true   # Generate cross-asset signals
```

### `news:`
```yaml
news:
  use_llm_news: true         # Use LLM for news search
  news_query: "crypto..."    # Search query
  llm_model: "gpt-4o-mini"   # LLM model
```

### `learning:`
```yaml
learning:
  knowledge_transfer_frequency: 10  # Transfer every K rounds
  pruning_frequency: 50             # Prune every M rounds
```

### `agents:`
Configure inventory methods for each agent type:
- `analyst.features`: Feature extraction (talib_stack, stl)
- `analyst.trends`: Trend detection (gaussian_hmm, kalman_filter)
- `researcher.forecasting`: Price forecasting (arima_x, tft)
- `researcher.uncertainty`: Uncertainty estimation
- `trader.styles`: Execution styles (aggressive_market, passive_laddered_limit)
- `risk.checks`: Risk validation rules

## Coins Supported

| Symbol | Name | Config |
|--------|------|--------|
| BTC | Bitcoin | `single/btc.yaml` |
| ETH | Ethereum | `single/eth.yaml` |
| SOL | Solana | `single/sol.yaml` |
| DOGE | Dogecoin | `single/doge.yaml` |
| XRP | Ripple | `single/xrp.yaml` |
| MULTI | All 5 coins | `multi_asset.yaml` |

