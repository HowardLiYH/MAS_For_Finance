# Multi-Agent LLM Financial Trading Model

## 🚀 Multi-Asset Crypto Trading with Cross-Market Intelligence

A modular, LLM-powered trading system that trades **5 cryptocurrencies** (BTC, ETH, SOL, DOGE, XRP) with cross-asset market context features.

| Coin | Symbol | Description |
|------|--------|-------------|
| Bitcoin | BTC | Primary market benchmark |
| Ethereum | ETH | Smart contract platform |
| Solana | SOL | High-performance L1 |
| Dogecoin | DOGE | Meme coin / retail sentiment |
| Ripple | XRP | Payment-focused crypto |

---

## ⚙️ Quick Start

### Multi-Asset Mode (Recommended)
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Copy Bybit data to data/bybit/
cp /path/to/Bybit_CSV_Data/*.csv data/bybit/

# Run multi-asset trading
python -m trading_agents.cli multi --config configs/multi_asset.yaml
```

### Single-Asset Mode
```bash
# Trade single coin
python -m trading_agents.cli run --config configs/single/btc.yaml
python -m trading_agents.cli run --config configs/single/eth.yaml
```

### Paper Trading (Bybit Testnet)
```bash
# Install paper trading dependencies
pip install -e ".[paper-trading]"

# Set Bybit Testnet API credentials
export BYBIT_TESTNET_KEY="your-api-key"
export BYBIT_TESTNET_SECRET="your-api-secret"

# Run paper trading
python -m trading_agents.cli paper --symbols BTC ETH SOL
```

### Admin Reports
```bash
# Generate performance report
python -m trading_agents.cli report --days 30

# Check admin status
python -m trading_agents.cli status
```

---

## 📊 Cross-Asset Market Context

When running in multi-asset mode, the system generates 8 cross-asset signals:

| Feature | Description | Trading Signal |
|---------|-------------|----------------|
| `btc_dominance` | BTC market cap proxy | High = risk-off |
| `altcoin_momentum` | Altcoin returns | Positive = risk-on |
| `eth_btc_ratio` | ETH/BTC strength | Rising = ETH outperforming |
| `cross_oi_delta` | Total OI change | Rising = conviction |
| `aggregate_funding` | Weighted funding | High = crowded long |
| `risk_on_off` | Altcoin beta | High = risk-on |
| `market_volatility` | Annualized vol | High = uncertainty |
| `cross_correlation` | Pairwise correlation | High = macro-driven |

---

## 📁 Project Structure

```
MAS_Final_With_Agents/
├── trading_agents/              # Core multi-agent trading system
│   ├── agents/                  # Agent implementations
│   │   ├── analyst.py           # Feature & trend extraction
│   │   ├── researcher.py        # Forecasting & uncertainty
│   │   ├── trader.py            # LLM-powered order generation
│   │   ├── risk.py              # Risk validation
│   │   └── evaluator.py         # Performance scoring
│   ├── inventory/               # Pluggable strategy methods
│   │   ├── analyst/             # TALib, STL, HMM, Kalman
│   │   ├── researcher/          # ARIMAX, TFT, Bootstrap
│   │   ├── trader/              # Market, Limit execution
│   │   └── risk/                # VaR, Leverage, Margin checks
│   ├── config/                  # Configuration management
│   ├── optimization/            # Continual learning
│   ├── services/                # Services layer
│   │   ├── llm.py               # LLM proposal generation
│   │   ├── metrics.py           # Performance tracking
│   │   ├── events.py            # Event bus system
│   │   ├── alerts.py            # Alert rules engine
│   │   ├── notifications.py     # Slack/console/file notifications
│   │   ├── reports.py           # Report generation
│   │   ├── bybit_client.py      # Bybit Testnet API client
│   │   ├── order_manager.py     # Order lifecycle management
│   │   └── positions.py         # Position tracking
│   ├── workflow.py              # WorkflowEngine
│   └── cli.py                   # Command-line interface
│
├── data_pipeline/               # Data fetching & processing
│   └── pipeline/
│       ├── multi_asset.py       # 5-coin Bybit loader
│       ├── cross_features.py    # Cross-asset signals
│       └── data_pipeline.py     # Unified entry point
│
├── configs/                     # YAML configurations
│   ├── multi_asset.yaml         # 5-coin trading (RECOMMENDED)
│   ├── default.yaml             # Single-asset default
│   └── single/                  # Per-coin configs
│       ├── btc.yaml
│       ├── eth.yaml
│       ├── sol.yaml
│       ├── doge.yaml
│       └── xrp.yaml
│
└── data/                        # Market data
    ├── bybit/                   # Bybit CSV source files
    ├── multi_asset/             # Multi-asset outputs
    └── single/                  # Single-asset outputs
```

---

## 🤖 Agent Descriptions

### Analyst Agent
Processes time-series price data to extract:
- **Features**: TALib technical indicators, STL decomposition
- **Trends**: Gaussian HMM regime detection, Kalman filter

### Researcher Agent
Generates trading signals with uncertainty:
- **Forecasting**: ARIMAX, Temporal Fusion Transformer
- **Uncertainty**: Bootstrap ensemble, Quantile regression
- **Calibration**: Temperature scaling, Conformal prediction

### Trader Agent
LLM-powered order generation:
- Interprets research signals + news narratives
- Selects execution style (aggressive market / passive limit)
- Outputs: position size, leverage, TP/SL, liquidation price

### Risk Manager Agent
Validates orders with three verdicts:
- **pass**: Order within all limits
- **soft_fail**: Minor violation, can adjust
- **hard_fail**: Critical violation, abort

### Evaluator Agent
Tracks performance metrics:
- Sharpe ratio, PnL, Hit rate
- Max drawdown, Calibration ECE

---

## 🔄 Workflow

```
┌─────────────┐
│ Price Data  │ ← 5 coins from Bybit
│ (BTC,ETH,..)│
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│   Analyst   │────▶│  Researcher  │
│ (per coin)  │     │ (per coin)   │
└─────────────┘     └──────┬───────┘
                           │
       ┌─────────────┐     │
       │Market Context│◀───┘
       │(cross-asset) │
       └──────┬───────┘
              │
              ▼
       ┌─────────────┐
       │   Trader    │◀──── News Data
       │ (per coin)  │
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │ Risk Manager│
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │  Evaluator  │
       └─────────────┘
```

---

## 📔 Change History

* (2025.07.03) First Meeting
* (2025.08.28) Project Proposal and Workflow First Draft
* (2025.09.18) Completed Micro & Macro News and Price Data Fetch
* (2025.10.17) Created config-driven, raw multi-agent pipeline
* (2025.12.19) **Major Architecture Refactoring v0.2.0**
   * Structural reorganization (inventory/, config/, optimization/)
   * Plugin-based inventory system with `@register` decorator
   * Complete Risk Manager with hard_fail / soft_fail / pass
   * Performance tracking with Sharpe, PnL, HitRate, MaxDD, CalibECE
   * Knowledge transfer and inventory pruning
* (2025.12.19) **Multi-Asset Data Pipeline v0.3.0**
   * Added support for 5 coins: BTC, ETH, SOL, DOGE, XRP
   * Bybit perpetual futures data with derivative features
   * Cross-asset market context (8 signals)
   * Per-coin and multi-asset configuration files
   * Updated project structure for multi-coin trading
* (2025.12.19) **Admin Agent & Paper Trading v0.4.0**
   * Admin Agent with automated reporting and alerting
   * Event bus system for system-wide communication
   * Alert rules: max drawdown, daily loss, risk breaches, Sharpe warnings
   * Scheduled reports: daily summary, weekly summary, performance reports
   * Bybit Testnet integration for paper trading validation
   * Order manager with position tracking
   * Slack/console/file notification channels

---

## 🎯 Next (NeurIPS 2026 Target)

### Experimental Validation
- Run backtesting on 2-year data (4h intervals) for all 5 coins
- Validate: Aug 2024, Test: Sep-Dec 2024
- Ablation studies: with/without cross-asset features, with/without risk manager

### Benchmark Comparisons
- Compare against [Alpha Arena](https://alpha-arena.com) baselines
- Evaluate LLMs: GPT-4, DeepSeek, Claude on decision quality
- Measure cross-asset vs single-asset performance

### Paper Contributions
- Multi-agent orchestration for algorithmic trading
- Cross-asset market context features
- Continual learning and inventory pruning
- Risk-aware execution with LLM reasoning

### Technical Improvements
- ~~Add Admin Agent for automated reporting~~ ✅ Done
- ~~Real-time paper trading validation~~ ✅ Done
- ~~Extend to more assets (AVAX, LINK, etc.)~~ ✅ Done (5 coins)
- WebSocket real-time feeds for live trading
- Email notifications for critical alerts
- Backtesting engine improvements

---

## ✨ Related Repositories

* TradingAgents Enhanced Chinese Edition: [TradingAgents-CN](https://github.com/your-repo/TradingAgents-CN)
* TradingAgents Original by Tauric Research: [TradingAgents](https://github.com/tauric-research/TradingAgents)

---

## Configuration

### Multi-Asset (configs/multi_asset.yaml)
```yaml
data:
  multi_asset: true
  symbols: [BTC, ETH, SOL, DOGE, XRP]
  bybit_csv_dir: "data/bybit"
  add_cross_features: true
```

### Single-Asset (configs/single/btc.yaml)
```yaml
data:
  multi_asset: false
  offline_prices_csv: "data/bybit/Bybit_BTC.csv"
```

---

## Data Setup

### Option 1: Copy Bybit CSVs
```bash
cp /path/to/Bybit_CSV_Data/*.csv data/bybit/
```

### Option 2: Symlink
```bash
ln -s /path/to/Bybit_CSV_Data data/bybit
```

### Expected Files
```
data/bybit/
├── Bybit_BTC.csv
├── Bybit_ETH.csv
├── Bybit_SOL.csv
├── Bybit_DOGE.csv
└── Bybit_XRP.csv
```

---

## License & Attribution

This implementation borrows design patterns from **TradingAgents** and **TradingAgents-CN** (Apache-2.0). See their repositories for details.
