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
| Train-then-deploy | **Online learning** (models update every bar) |

### Research Contribution
- **Meta-Learning for Trading**: Agents learn to select strategies, not just tune parameters
- **Selection Pressure**: Inventory (15) > Selection (3) creates meaningful choices
- **Preference Transfer**: Knowledge sharing is about WHAT to select
- **Context-Aware Selection**: Different methods for different market regimes
- **Online Learning**: Models update after EVERY observation (like real hedge funds)

---

## 🧠 Feature-Aligned Learning (v0.9.8) - The Right Way

**Key insight: Update frequency should match FEATURE TIMESCALE, not model complexity!**

```
┌─────────────────────────────────────────────────────────────────┐
│           FEATURE-ALIGNED LEARNING ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FAST FEATURES (momentum, vol) ─────────► Update: EVERY BAR    │
│  Model: Any (even XGBoost!)     Why: These change every 4h     │
│                                                                 │
│  MEDIUM FEATURES (trend, daily) ────────► Update: EVERY 6 BARS │
│  Model: Any                     Why: Trend changes daily       │
│                                                                 │
│  SLOW FEATURES (regime, corr) ──────────► Update: EVERY 42 BARS│
│  Model: Any (even simple!)      Why: Regime changes weekly     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Wrong Approach (Model-Based) | Right Approach (Feature-Based) |
|------------------------------|-------------------------------|
| Simple model → fast update | Fast-changing feature → fast update |
| Complex model → slow update | Slow-changing feature → slow update |
| Computational constraint drives design | Data dynamics drive design |

### How It Works:

```
Bar 1:   Observe → Predict → Trade → See outcome → UPDATE WEIGHTS
Bar 2:   Observe → Predict (better) → Trade → See outcome → UPDATE WEIGHTS
Bar 3:   Observe → Predict (even better) → Trade → See outcome → UPDATE WEIGHTS
...
Bar 8700: Model has been learning for 4 years
```

### Feature Groups and Models:

| Feature Group | Features | Update Freq | Models Used |
|---------------|----------|-------------|-------------|
| **Fast** | ret_1bar, ret_5bar, vol_intrabar, momentum | Every bar | OnlineLinear + OnlineRidge |
| **Medium** | trend_strength, daily_vol, sma_ratio | Every 6 bars | Ridge with batch refit |
| **Slow** | regime, cross_correlation | Every 42 bars | RandomForest + regime means |

### Online Models (used in Fast features):

| Model | Algorithm | What It Learns |
|-------|-----------|----------------|
| `OnlineLinearRegression` | SGD | Return prediction |
| `OnlineRidge` | Recursive Least Squares | Trend prediction |
| `OnlineVolatility` | EWMA | Volatility estimation |
| `OnlineRegimeDetector` | Bayesian HMM | Market regime (Bull/Bear/Neutral) |

### Code Example:

```python
# Online models update after EVERY bar:
for bar in price_data:
    features = extract_features(bar)

    # Predict BEFORE seeing outcome
    prediction = model.predict(features)

    # Execute trade
    execute_trade(prediction)

    # Next bar: see actual outcome
    actual_return = next_bar.close / bar.close - 1

    # UPDATE model weights with observation
    model.update(features, actual_return)  # ← This is online learning!
```

---

## 🎰 RL Enhancements (v0.7.0)

Three lightweight, theoretically-grounded RL improvements for robust learning:

### 1. Thompson Sampling (Bayesian Exploration)

Instead of deterministic UCB, agents sample from Beta distributions to naturally balance exploration and exploitation:

```
For each method m:
  sample ~ Beta(α_m, β_m)
  # High uncertainty → high variance → more exploration
  # High success rate → high mean → more exploitation
```

| Scenario | Alpha | Beta | Behavior |
|----------|-------|------|----------|
| New method | 1 | 1 | Uniform sampling (explore) |
| 10 wins, 2 losses | 11 | 3 | High mean, exploit |
| 3 wins, 10 losses | 4 | 11 | Low mean, avoid |

### 2. Contextual Baselines (Regime-Aware Learning)

Per-regime baselines for proper credit assignment:

```
Bull market: +2% is average (baseline = 2.5%)  → advantage ≈ 0
Bear market: +2% is exceptional (baseline = -0.5%) → advantage ≈ +2.5%
```

Agents learn **context-specific** method preferences, not global averages.

### 3. Multi-Step Returns (Temporal Credit Assignment)

Discounted future rewards for methods that sacrifice short-term for long-term:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...

Method A: Immediate +1%, then +0.5%, +0.5%  →  G = 1.86%
Method B: Immediate -0.5%, then +3%, +2%   →  G = 4.32% ✓
```

Multi-step returns properly credit methods that set up future gains.

### Configuration

```yaml
population:
  use_thompson_sampling: true
  gamma: 0.9        # Discount factor
  n_step: 3         # Steps for multi-step returns
```

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

### Step 1: Create Conda Environment (Recommended)
```bash
# Create and activate conda environment
conda create -n mas python=3.11 -y
conda activate mas

# Install core packages
conda install pandas numpy matplotlib requests pyyaml -y
conda install -c conda-forge openai -y

# Install project
cd /path/to/MAS_Final_With_Agents
pip install -e .
```

### Step 2: Run Population Backtest
```bash
# Single asset backtest
python -m trading_agents.cli backtest --symbol BTC

# Multi-asset backtest
python -m trading_agents.cli backtest --symbols BTC,ETH,SOL,DOGE,XRP

# With options
python -m trading_agents.cli backtest --symbol BTC \
    --population-size 5 \
    --capital 10000 \
    --start 2024-01-01 \
    --end 2024-06-01
```

### Step 3: Visualization Dashboard (Optional)

**Terminal 1 - Start API server:**
```bash
conda activate mas
python -m trading_agents.cli api --port 8000
```

**Terminal 2 - Start React dashboard:**
```bash
cd dashboard
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

### Step 4: Run Ablation Study (LLM vs News Effects)
```bash
# Run all 4 conditions: A=Baseline, B=LLM, C=News, D=Full
python -m trading_agents.cli ablation --condition all \
    --symbols BTC,ETH,SOL,XRP,DOGE \
    --start 2022-01-01 \
    --end 2024-12-01

# Run single condition (e.g., baseline only)
python -m trading_agents.cli ablation --condition A
```

| Condition | LLM | News | Description |
|-----------|-----|------|-------------|
| A (Baseline) | No | No | Pure Thompson Sampling |
| B (LLM Only) | Yes | No | LLM reasoning, no news |
| C (News Only) | No | Yes | News as features |
| D (Full) | Yes | Yes | Complete system |

### Step 5: Real-Time Learning Mode (Live Trading)
```bash
# Run real-time learning with 4-hour iterations
python -m trading_agents.cli live --symbols BTC,ETH,SOL,XRP,DOGE

# With options
python -m trading_agents.cli live \
    --symbols BTC,ETH,SOL \
    --interval 4.0 \
    --use-llm \
    --use-news \
    --testnet  # Execute on Bybit testnet

# Test single iteration (no waiting)
python -m trading_agents.cli live --test-once
```

**Key difference from backtesting:**
- **Backtest**: Simulates historical data rapidly (1000+ iterations in minutes)
- **Live Mode**: Waits actual 4 hours between iterations, fetches live data

| Mode | Data Source | Wait Time | Use Case |
|------|-------------|-----------|----------|
| Backtest | Historical CSV | None | Research, hyperparameter tuning |
| Live | Real-time API | 4 hours | Continuous learning, paper trading |

### Step 6: Export for NeurIPS Paper
```bash
python -m trading_agents.cli export --experiment-id <exp_id> --output-dir exports/neurips
```

### Alternative: Method Selection Learning Mode
```bash
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
├── population/                    # Population-based method selection
│   ├── selector.py                # MethodSelector class (core innovation)
│   ├── inventories.py             # 15 methods per role
│   ├── selector_workflow.py       # Selection-based workflow
│   └── ...                        # Transfer, diversity, scoring
├── agents/                        # Agent implementations
├── inventory/                     # Method implementations
│   ├── online_models.py           # Online learning (SGD, RLS, HMM)
│   ├── feature_aligned_learner.py # Feature-timescale-aligned learning (v0.9.8)
│   └── ...                        # Analyst, Researcher, Trader, Risk methods
├── backtesting/                   # Backtesting engine
│   ├── engine.py                  # BacktestEngine with population support
│   └── executor.py                # Order execution simulation
├── services/                      # LLM, events, notifications
│   ├── experiment_logger.py       # Structured logging (JSONL)
│   ├── scheduler.py               # 4-hour paper trading scheduler
│   └── neurips_export.py          # Publication-ready exports
├── api/                           # Dashboard API
│   └── server.py                  # FastAPI + WebSocket server
└── config/                        # Configuration management

dashboard/                         # React Visualization Dashboard
├── src/components/                # AgentPopulation, MethodInventory, etc.
└── ...                            # Next.js app

tests/                             # Test suite
├── conftest.py                    # Pytest fixtures
└── test_*.py                      # Mock and integration tests
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
* (2025.12.19) **PopAgent v0.6.0: Adaptive Method Selection**
   * Agents now SELECT methods from inventory (not fixed strategies)
   * Extended inventories: 15/12/10/10 methods per role
   * Selection learning via UCB + reinforcement learning
   * Preference-based knowledge transfer
   * Context-aware method selection
* (2025.12.19) **PopAgent v0.7.0: RL Enhancements**
   * Thompson Sampling for Bayesian exploration
   * Contextual baselines for regime-aware learning
   * Multi-step returns for temporal credit assignment
* (2025.12.20) **PopAgent v0.8.0: Testing & Visualization**
   * Complete test suite with mock data fixtures
   * Population-based backtesting (`run_population_backtest`)
   * React dashboard for visualization (Next.js + Tailwind)
   * FastAPI backend with WebSocket live updates
   * 4-hour paper trading scheduler
   * NeurIPS export utilities (figures, tables, traces)
* (2025.12.20) **PopAgent v0.9.0: Online Learning (Hedge Fund Style)**
   * **TRUE Online Learning**: Models update weights after EVERY observation
   * `OnlineLinearRegression`: SGD-based return predictor
   * `OnlineRidge`: Recursive Least Squares with forgetting factor
   * `OnlineVolatility`: EWMA variance estimation
   * `OnlineRegimeDetector`: Bayesian HMM with incremental updates
   * Persistent model state across sessions
   * Real-time learning mode (`python -m trading_agents.cli live`)
* (2025.12.20) **PopAgent v0.9.1-v0.9.6: Incremental Improvements**
   * v0.9.1: Stay-flat metrics tracking (avoid trading in uncertainty)
   * v0.9.4: Simplified trading logic (online model decides trade/no-trade)
   * v0.9.5: Fixed momentum as PRIMARY driver (not overridden by untrained models)
   * v0.9.6: Real pipeline execution, regime detector responsiveness fixes
* (2025.12.21) **PopAgent v0.9.8: Feature-Aligned Learning** 🆕
   * **KEY INSIGHT**: Update frequency should match FEATURE TIMESCALE, not model complexity!
   * Deprecated hybrid learning (model complexity → frequency approach was flawed)
   * New `FeatureAlignedLearner` with 3 feature groups:
     * **Fast features** (momentum, vol spikes): Update EVERY bar
     * **Medium features** (trend, daily vol): Update every 6 bars (~daily)
     * **Slow features** (regime, correlations): Update every 42 bars (~weekly)
   * Each group can use ANY model complexity - complexity ≠ update frequency
   * Adaptive blending weights based on market conditions
   * `feature_aligned_learner.py`: 500+ lines of principled learning architecture

---

## 🧪 Ablation Experiments

### Planned Experiments for NeurIPS Paper

#### 1. Learning Approach Comparison
| Experiment | Description | Hypothesis |
|------------|-------------|------------|
| **A: Online-Only** | Pure SGD, update every bar | Fast adaptation, poor pattern capture |
| **B: Batch-Only** | Refit RF/XGB weekly | Good patterns, slow adaptation |
| **C: Hybrid (Model-Based)** | Simple→fast, Complex→slow | Suboptimal: wrong dimension |
| **D: Feature-Aligned** | Fast features→fast, Slow→slow | ✓ Best: matches data dynamics |

```bash
python -m trading_agents.cli ablation --experiment learning_approach \
    --conditions online,batch,hybrid,feature_aligned
```

#### 2. Feature Timescale Sensitivity
| Config | Fast Freq | Medium Freq | Slow Freq |
|--------|-----------|-------------|-----------|
| Aggressive | 1 bar | 3 bars | 21 bars |
| **Default** | 1 bar | 6 bars | 42 bars |
| Conservative | 1 bar | 12 bars | 84 bars |

#### 3. Method Selection vs Fixed Strategies
| Condition | Strategy | Expected Outcome |
|-----------|----------|-----------------|
| Fixed-Best | Always use top-3 methods | Good baseline, no adaptation |
| Fixed-Random | Random method selection | Poor performance |
| **PopAgent** | Learned selection | Adapts to regime changes |

#### 4. Population Size Effect
| Pop Size | Diversity | Convergence Speed | Final Performance |
|----------|-----------|-------------------|-------------------|
| 3 | Low | Fast | Risk of local optima |
| **5** | Medium | Balanced | Default setting |
| 10 | High | Slow | Better exploration |

#### 5. Knowledge Transfer Frequency
| Transfer Every | Effect |
|---------------|--------|
| 5 iterations | Rapid homogenization |
| **10 iterations** | Balanced (default) |
| 20 iterations | More diversity, slower learning |

#### 6. Cross-Asset Learning
| Condition | Description |
|-----------|-------------|
| Independent | Each asset learns separately |
| **Shared Population** | Single population, cross-asset features |
| Transfer Across Assets | BTC insights → altcoins |

### Metrics to Report
- Sharpe Ratio (primary)
- Total Return %
- Maximum Drawdown
- Win Rate
- Stay-Flat Rate (% of iterations with no trade)
- Learning Improvement (avg PnL last 10% vs first 10%)
- Selection Diversity (entropy of method usage)

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
