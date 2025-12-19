# PopAgent: Multi-Agent LLM Trading with Adaptive Method Selection

## 🧬 Core Innovation: Agents Learn to SELECT Methods

Unlike fixed-strategy trading systems, **PopAgent** maintains populations of agents that **learn to SELECT** which methods to use from a shared inventory. This creates a meta-learning system where agents discover optimal method combinations through continual learning.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    POPAGENT: METHOD SELECTION LEARNING                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INVENTORY (15 methods)              AGENT POPULATION (5 agents)       │
│  ┌─────────────────────┐             ┌───────────────────────────┐     │
│  │ ☐ RSI               │             │ Agent 1                   │     │
│  │ ☐ MACD              │◄── selects ─│ Preferences: RSI↑ HMM↑    │     │
│  │ ☐ BollingerBands    │             │ Picks: [RSI, HMM, Kalman] │     │
│  │ ☐ HMM_Regime        │             └───────────────────────────┘     │
│  │ ☐ KalmanFilter      │             ┌───────────────────────────┐     │
│  │ ☐ WaveletTransform  │◄── selects ─│ Agent 2                   │     │
│  │ ☐ STL_Decomposition │             │ Preferences: MACD↑ STL↑   │     │
│  │ ☐ VolatilityClustering           │ Picks: [MACD, STL, Wavelet│     │
│  │ ☐ ... (more)        │             └───────────────────────────┘     │
│  └─────────────────────┘                        ...                     │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CONTINUAL LEARNING                            │   │
│  │                                                                  │   │
│  │  1. Agents select methods → Execute pipeline → Get reward       │   │
│  │  2. Update preferences: pref[method] += α × (reward - baseline) │   │
│  │  3. Transfer: Best agent's preferences → Other agents           │   │
│  │  4. Diversity: Ensure agents don't all select same methods      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Why This Is Novel

| Traditional Approach | PopAgent Approach |
|---------------------|-------------------|
| Fixed agent strategies | Agents SELECT methods dynamically |
| Learn parameters | Learn WHICH methods to use |
| Single best agent | Population discovers combinations |
| Static configurations | Adapts to market conditions |

### Research Contribution
- **Meta-Learning for Trading**: Agents learn to select strategies, not just tune parameters
- **Selection Pressure**: Inventory (15) > Selection (3) creates meaningful choices
- **Preference Transfer**: Knowledge sharing is about WHAT to select
- **Context-Aware Selection**: Different methods for different market regimes

---

## 📊 Method Inventories

Each role has **10-15 methods** available, but agents only select **3** at a time:

### Analyst (15 methods)
| Category | Methods |
|----------|---------|
| Technical | RSI, MACD, BollingerBands, ADX, Stochastic |
| Statistical | Autocorrelation, VolatilityClustering, MeanReversion, Cointegration |
| Decomposition | STL, WaveletTransform, FourierAnalysis |
| ML | HMM_Regime, KalmanFilter, IsolationForest |

### Researcher (12 methods)
| Category | Methods |
|----------|---------|
| Statistical | ARIMA, ExponentialSmoothing, VectorAutoregression, GARCH |
| ML | RandomForest, GradientBoosting, LSTM, TemporalFusion |
| Uncertainty | BootstrapEnsemble, QuantileRegression, BayesianInference, ConformalPrediction |

### Trader (10 methods)
| Category | Methods |
|----------|---------|
| Execution | AggressiveMarket, PassiveLimit, TWAP, VWAP |
| Sizing | KellyCriterion, FixedFractional, VolatilityScaled |
| Entry | MomentumEntry, ContrarianEntry, BreakoutEntry |

### Risk (10 methods)
| Category | Methods |
|----------|---------|
| Position | MaxLeverage, MaxPositionSize, ConcentrationLimit |
| Loss | MaxDrawdown, DailyStopLoss, TrailingStop |
| Metrics | VaRLimit, ExpectedShortfall |
| Dynamic | VolatilityAdjusted, RegimeAware |

---

## ⚙️ Quick Start

### Method Selection Mode (Recommended)
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run with method selection learning
python -m trading_agents.cli selector --config configs/multi_asset.yaml
```

### Configuration
```yaml
# configs/multi_asset.yaml
population:
  mode: "selector"  # Use method selection (vs "fixed" for legacy)
  size: 5           # 5 agents per role
  max_methods: 3    # Each agent picks 3 methods
  transfer_frequency: 10
  learning_rate: 0.1
  exploration_rate: 0.15
```

---

## 🔄 Learning Workflow

```
Iteration N:
│
├── 1. METHOD SELECTION
│   └── Each agent selects 3 methods from inventory (UCB + preferences)
│       Agent 1: [RSI, HMM_Regime, KalmanFilter]
│       Agent 2: [MACD, STL_Decomposition, WaveletTransform]
│       ...
│
├── 2. PIPELINE SAMPLING
│   └── Sample 25 combinations of (analyst, researcher, trader, risk)
│
├── 3. EVALUATION
│   └── Run each pipeline → measure PnL
│
├── 4. PREFERENCE UPDATE (Reinforcement Learning)
│   └── For each method used:
│       preference[method] += learning_rate × (reward - baseline)
│
├── 5. KNOWLEDGE TRANSFER (every 10 iterations)
│   └── Best agent's preferences → Other agents (soft update τ=0.1)
│
├── 6. DIVERSITY CHECK
│   └── If selection diversity < threshold → increase exploration
│
└── 7. Next Iteration
```

---

## 📁 Project Structure

```
trading_agents/
├── population/                    # 🆕 Population-based method selection
│   ├── selector.py                # MethodSelector class (core innovation)
│   ├── inventories.py             # 15 methods per role
│   ├── selector_workflow.py       # Selection-based workflow
│   ├── base.py                    # Base population classes
│   ├── transfer.py                # Knowledge transfer strategies
│   ├── diversity.py               # Diversity preservation
│   └── scoring.py                 # Shapley-based credit assignment
│
├── agents/                        # Agent implementations
├── inventory/                     # Method implementations
├── services/                      # LLM, events, notifications
└── config/                        # Configuration management
```

---

## 📔 Change History

* (2025.07.03) First Meeting
* (2025.08.28) Project Proposal and Workflow First Draft
* (2025.09.18) Completed Micro & Macro News and Price Data Fetch
* (2025.10.17) Created config-driven, raw multi-agent pipeline
* (2025.12.19) **Major Architecture Refactoring v0.2.0**
* (2025.12.19) **Multi-Asset Data Pipeline v0.3.0** (5 coins)
* (2025.12.19) **Admin Agent & Paper Trading v0.4.0**
* (2025.12.19) **Bocha Search Integration v0.4.1**
* (2025.12.19) **PopAgent v0.5.0: Population-Based Learning**
* (2025.12.19) **PopAgent v0.6.0: Adaptive Method Selection** 🆕
   * Agents now SELECT methods from inventory (not fixed strategies)
   * Extended inventories: 15/12/10/10 methods per role
   * Selection learning via UCB + reinforcement learning
   * Preference-based knowledge transfer
   * Context-aware method selection

---

## 🎯 NeurIPS 2026 Target

### Paper Title
*"PopAgent: Adaptive Method Selection in Multi-Agent LLM Trading via Continual Learning"*

### Core Contributions
1. **Method Selection as Meta-Learning** - Agents learn WHAT to use, not just HOW
2. **Inventory > Agents** - Selection pressure creates meaningful learning
3. **Preference Transfer** - Novel knowledge sharing mechanism
4. **Context-Aware Selection** - Adapt to market regimes

### Experiments
- 5 crypto assets (BTC, ETH, SOL, DOGE, XRP)
- 2 years of 4h data
- Compare: Fixed strategies vs Method Selection
- Ablations: Transfer frequency, inventory size, exploration rate

---

## 🚀 Multi-Asset Trading

Trades **5 cryptocurrencies** with cross-asset market context:

| Coin | Symbol | Description |
|------|--------|-------------|
| Bitcoin | BTC | Primary market benchmark |
| Ethereum | ETH | Smart contract platform |
| Solana | SOL | High-performance L1 |
| Dogecoin | DOGE | Meme coin / retail sentiment |
| Ripple | XRP | Payment-focused crypto |

### Cross-Asset Features (8 signals)
- BTC dominance, altcoin momentum, ETH/BTC ratio
- Cross OI delta, aggregate funding, risk-on/off
- Market volatility, cross-correlation

---

## Configuration

### Multi-Asset with Method Selection
```yaml
data:
  multi_asset: true
  symbols: [BTC, ETH, SOL, DOGE, XRP]
  bybit_csv_dir: "data/bybit"

population:
  mode: "selector"
  size: 5
  max_methods: 3
  transfer_frequency: 10
  learning_rate: 0.1
```

---

## License & Attribution

This implementation builds on **TradingAgents** (Apache-2.0) and **Population-Based Training** research.
