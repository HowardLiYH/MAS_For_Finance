# Multi-Agent LLM Financial Trading Model

## 🚀 Multi-Asset Crypto Trading with Cross-Market Intelligence

A modular, LLM-powered trading system that trades **5 cryptocurrencies** (BTC, ETH, SOL, DOGE, XRP) with cross-asset market context features and **population-based continual learning**.

| Coin | Symbol | Description |
|------|--------|-------------|
| Bitcoin | BTC | Primary market benchmark |
| Ethereum | ETH | Smart contract platform |
| Solana | SOL | High-performance L1 |
| Dogecoin | DOGE | Meme coin / retail sentiment |
| Ripple | XRP | Payment-focused crypto |

---

## 🧬 Key Innovation: Population-Based Agent Learning

Unlike traditional multi-agent systems with fixed architectures, our system maintains **populations of diverse agents** for each role that evolve through continual learning:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    POPULATION-BASED LEARNING                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Analyst Population    Researcher Population   Trader Population       │
│  ┌───┐ ┌───┐ ┌───┐    ┌───┐ ┌───┐ ┌───┐      ┌───┐ ┌───┐ ┌───┐       │
│  │A-1│ │A-2│ │A-3│    │R-1│ │R-2│ │R-3│      │T-1│ │T-2│ │T-3│       │
│  │ ★ │ │   │ │   │    │   │ │ ★ │ │   │      │   │ │   │ │ ★ │       │
│  └───┘ └───┘ └───┘    └───┘ └───┘ └───┘      └───┘ └───┘ └───┘       │
│    │                      │                      │                     │
│    ▼                      ▼                      ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │                     EVALUATOR                                │      │
│  │  Score all agents → Identify best (★) → Transfer knowledge  │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Variants (5 per role)

| Role | Variants | Description |
|------|----------|-------------|
| **Analyst** | Technical, Statistical, Momentum, Volatility, Hybrid | Feature extraction strategies |
| **Researcher** | Statistical, Ensemble, Bayesian, Quantile, Adaptive | Forecasting approaches |
| **Trader** | Aggressive, Conservative, Momentum, Contrarian, Adaptive | Execution styles |
| **Risk** | Strict, Moderate, Dynamic, VaR-based, Drawdown | Risk tolerance levels |

### Learning Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Soft Update** | Gradually blend parameters toward best performer |
| **Distillation** | Train agents to match best agent's outputs |
| **Selective Transfer** | Only transfer high-importance parameters |
| **Diversity Preservation** | Mutation to prevent population collapse |

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

### Population-Based Learning Mode
```bash
# Run with population-based learning
python -m trading_agents.cli population --config configs/multi_asset.yaml

# With custom population size
python -m trading_agents.cli population --pop-size 5 --transfer-freq 10
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
├── trading_agents/                 # Core multi-agent trading system
│   ├── agents/                     # Agent implementations
│   │   ├── analyst.py              # Feature & trend extraction
│   │   ├── researcher.py           # Forecasting & uncertainty
│   │   ├── trader.py               # LLM-powered order generation
│   │   ├── risk.py                 # Risk validation
│   │   ├── evaluator.py            # Performance scoring
│   │   └── admin.py                # Monitoring & reporting
│   │
│   ├── population/                 # 🆕 Population-based learning
│   │   ├── base.py                 # AgentPopulation class
│   │   ├── variants.py             # 5 variants per agent role
│   │   ├── transfer.py             # Knowledge transfer strategies
│   │   ├── diversity.py            # Diversity preservation
│   │   ├── scoring.py              # Shapley-based credit assignment
│   │   └── workflow.py             # PopulationWorkflow engine
│   │
│   ├── inventory/                  # Pluggable strategy methods
│   │   ├── analyst/                # TALib, STL, HMM, Kalman
│   │   ├── researcher/             # ARIMAX, TFT, Bootstrap
│   │   ├── trader/                 # Market, Limit execution
│   │   └── risk/                   # VaR, Leverage, Margin checks
│   │
│   ├── services/                   # Services layer
│   │   ├── llm.py                  # LLM proposal generation
│   │   ├── metrics.py              # Performance tracking
│   │   ├── events.py               # Event bus system
│   │   ├── alerts.py               # Alert rules engine
│   │   ├── notifications.py        # Slack/console notifications
│   │   ├── bybit_client.py         # Bybit Testnet API
│   │   └── order_manager.py        # Order lifecycle
│   │
│   ├── config/                     # Configuration management
│   ├── optimization/               # Continual learning
│   ├── workflow.py                 # WorkflowEngine
│   └── cli.py                      # Command-line interface
│
├── data_pipeline/                  # Data fetching & processing
│   ├── news/                       # News intelligence
│   │   ├── providers/              # Bocha, SerpAPI
│   │   ├── enrichment.py           # LLM news enrichment
│   │   ├── aggregation.py          # News clustering
│   │   └── sources.py              # Source credibility
│   └── pipeline/
│       ├── multi_asset.py          # 5-coin Bybit loader
│       └── cross_features.py       # Cross-asset signals
│
├── configs/                        # YAML configurations
│   ├── multi_asset.yaml            # 5-coin trading
│   └── single/                     # Per-coin configs
│
├── data/                           # Market data
│   └── bybit/                      # Bybit CSV files
│
└── docs/
    └── ARCHITECTURE.md             # Detailed architecture diagrams
```

---

## 🤖 Agent Descriptions

### Analyst Agent (5 Variants)
| Variant | Focus | Key Parameters |
|---------|-------|----------------|
| Technical | TALib indicators | RSI, MACD, BB, ADX |
| Statistical | Autocorrelation, volatility | Lookback 20-120 |
| Momentum | Rate of change | Short lookbacks 5-20 |
| Volatility | ATR, range, BB width | Regime detection |
| Hybrid | Adaptive mix | Dynamic weights |

### Researcher Agent (5 Variants)
| Variant | Method | Uncertainty |
|---------|--------|-------------|
| Statistical | ARIMA-based | Bootstrap CI |
| Ensemble | Multiple models | Ensemble std |
| Bayesian | Prior-based | Posterior |
| Quantile | Quantile regression | Full distribution |
| Adaptive | Online learning | Adaptive window |

### Trader Agent (5 Variants)
| Variant | Style | Risk Profile |
|---------|-------|--------------|
| Aggressive | High leverage, large size | 3% risk/trade |
| Conservative | Low leverage, small size | 1% risk/trade |
| Momentum | Trend following | 2% risk/trade |
| Contrarian | Fade moves | 1.5% risk/trade |
| Adaptive | Context-dependent | Dynamic |

### Risk Manager (5 Variants)
| Variant | Max Leverage | Max Drawdown |
|---------|--------------|--------------|
| Strict | 3x | 5% |
| Moderate | 5x | 10% |
| Dynamic | 6x | 12% |
| VaR-based | 5x | 8% |
| Drawdown | 4x | 6% |

---

## 🔄 Population Learning Workflow

```
Iteration N:
│
├── 1. Sample Pipeline Combinations
│   └── Up to 25 (analyst, researcher, trader, risk) tuples
│
├── 2. Evaluate Each Pipeline
│   └── Run full trading simulation → PnL result
│
├── 3. Score Agents
│   ├── Individual performance (Sharpe, PnL, hit rate)
│   ├── Pipeline contribution (Shapley values)
│   └── Diversity bonus
│
├── 4. Knowledge Transfer (every N iterations)
│   └── Best agent → Other agents (soft update τ=0.1)
│
├── 5. Diversity Preservation
│   └── If diversity < threshold → Mutate non-elite agents
│
└── 6. Record Results
    └── Update population scores, history
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
* (2025.12.19) **Admin Agent & Paper Trading v0.4.0**
   * Admin Agent with automated reporting and alerting
   * Event bus system for system-wide communication
   * Bybit Testnet integration for paper trading validation
* (2025.12.19) **Bocha Search Integration v0.4.1**
   * Replaced SerpAPI with Bocha Search API
   * LLM-based news enrichment and aggregation
* (2025.12.19) **Population-Based Learning v0.5.0** 🆕
   * 5 agent variants per role (Analyst, Researcher, Trader, Risk)
   * Knowledge transfer strategies (Soft Update, Distillation, Selective)
   * Diversity preservation with mutation
   * Shapley-based credit assignment for fair scoring
   * PopulationWorkflow engine for evolutionary learning

---

## 🎯 Next (NeurIPS 2026 Target)

### Research Contribution
- **Novel Framework**: Population-based continual learning for multi-agent LLM trading
- **Key Innovation**: Heterogeneous agent populations that co-evolve
- **Technical Depth**: Shapley values for credit assignment, conformal calibration

### Experimental Validation
- Run backtesting on 2-year data (4h intervals) for all 5 coins
- Compare: Single-agent vs Population-based (5 variants)
- Ablation: With/without knowledge transfer, with/without diversity preservation

### Benchmark Comparisons
- GPT-4 vs DeepSeek vs Claude on decision quality
- Population learning vs static best-agent

### Paper Structure
1. Introduction: Problem of brittle LLM agent architectures
2. Method: Population-based continual learning framework
3. Experiments: Crypto trading on 5 assets
4. Analysis: What knowledge transfers? Emergent specialization?
5. Conclusion: Evolving agent populations outperform fixed architectures

---

## Configuration

### Multi-Asset (configs/multi_asset.yaml)
```yaml
data:
  multi_asset: true
  symbols: [BTC, ETH, SOL, DOGE, XRP]
  bybit_csv_dir: "data/bybit"
  add_cross_features: true

population:
  enabled: true
  size: 5
  transfer_frequency: 10
  transfer_tau: 0.1
  diversity_weight: 0.1
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

## ✨ Related Work

* TradingAgents by Tauric Research: [TradingAgents](https://github.com/tauric-research/TradingAgents)
* Population-Based Training: [PBT Paper](https://arxiv.org/abs/1711.09846)

---

## License & Attribution

This implementation borrows design patterns from **TradingAgents** (Apache-2.0). See their repository for details.
