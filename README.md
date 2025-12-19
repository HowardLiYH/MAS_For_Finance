# Multi-agent LLM Financial Trading Model on BTC Perpetual

## ⚙️ Basic Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m trading_agents.cli --symbol BTCUSD.PERP --interval 4h
```

## ✨ Related Repositories

* TradingAgents Enhanced Chinese Edition: [TradingAgents-CN](https://github.com/your-repo/TradingAgents-CN)
* TradingAgents Original by Tauric Research: [TradingAgents](https://github.com/tauric-research/TradingAgents)

## 📔 Change History

* (2025.07.03) First Meeting
* (2025.08.28) Project Proposal and Workflow First Draft
* (2025.09.18) Completed Micro & Macro News and Price Data Fetch
* (2025.10.17) Created config-driven, raw multi-agent pipeline, and raw inventory instantiation
* (2025.12.19) **Major Architecture Refactoring v0.2.0**
   * **Structural Reorganization:**
      * `strategies/` → `inventory/` with agent-specific subdirectories (analyst/, researcher/, trader/, risk/)
      * `models/types.py` → `models.py` (flattened single-file module)
      * `orchestrator/graph.py` → `workflow.py` with `WorkflowEngine` class (removed global state)
      * `learning/` → `optimization/` with `KnowledgeTransfer` and `InventoryPruner`
      * `tools/` → `utils/` (standard Python convention)
      * `config.py` + `compose.py` → `config/` module with `schemas.py` and `loader.py`
   * **New Features Implemented:**
      * Plugin-based inventory system with `@register` decorator
      * Lazy-loading package imports to improve startup time
      * Complete Risk Manager with `hard_fail` / `soft_fail` / `pass` verdicts
      * Liquidation price calculation in Trader Agent
      * Performance tracking with Sharpe, PnL, HitRate, MaxDD, CalibECE metrics
      * Knowledge transfer mechanism (top agents teach bottom performers every K rounds)
      * Inventory pruning (remove rarely-used methods every M rounds)
   * **Agent Implementations Completed:**
      * Analyst: TALibStack, STLDecomposition, GaussianHMM, KalmanFilter
      * Researcher: ARIMAX, TFT, BootstrapEnsemble, QuantileRegression, TemperatureScaling, ConformalICP
      * Trader: AggressiveMarket, PassiveLadderedLimit with LLM integration
      * Risk: VaRSafeBand, LeveragePositionLimits, LiquidationSafety, MarginCallRisk, GlobalVaRBreach
      * Evaluator: PerformanceTracker with comprehensive metrics

* **Next (NeurIPS 2026 Submission Target):**
   * **Experimental Validation:**
      * Run backtesting on 2-year BTC perpetual data (4h intervals)
      * Validate on Aug 2024, Test on Sep-Dec 2024
      * Generate ablation studies: with/without continual learning, with/without risk manager
   * **Benchmark Comparisons:**
      * Compare against [Alpha Arena](https://alpha-arena.com) baselines
      * Evaluate different LLMs (GPT-4, DeepSeek, Claude) on Trader decision quality
      * Measure market interpretability and reasoning quality
   * **Paper Writing:**
      * Problem formulation: Multi-agent orchestration for algorithmic trading
      * Contributions: Continual learning, inventory pruning, risk-aware execution
      * Experiments: Sharpe ratio, max drawdown, hit rate, calibration ECE
      * Analysis: Ablations, LLM comparison, knowledge transfer effectiveness
   * **Technical Improvements:**
      * Add Admin Agent for automated reporting and monitoring
      * Integrate real-time paper trading for live validation
      * Implement cross-asset generalization (ETH, SOL)

---

# 🤖 Summary of the Multi-agent LLM Financial Trading Model on BTC Perpetual with SJTU

## Key Concepts

* Multi-agent Systems
* Large Language Models
* Continual Learning
* Uncertainty Quantification
* Risk-aware Decision Making
* Explainable AI
* Cross-modal Data Fusion
* Agent-based Orchestration
* Feedback-driven Optimization
* Structured Knowledge Representation

---

## High-level Summary

Our model incorporates five types of agents to replicate the workflow pipeline of a hedge fund:
**Analysts, Researchers, Traders, Risk Managers, and Evaluators.**

* The system ingests two distinct streams of information:
   1. **Text-based news data** (collected via LLM prompting with strict date control to prevent leakage)
   2. **Time-series BTC price data** (4-hour intervals)
* Agents perform trading activities and continually improve through feedback.
* Each agent type has specialized inventories (methods/prompts) to complete tasks.
* After each trading round (when an order is closed), the **Evaluator Agent** measures performance against fixed metrics.
* After every K iterations, top performers in each category transfer knowledge to peers.
* Price data and news data are deliberately treated separately:
   * **Analyst & Researcher Agents** → preprocess & analyze time-series inputs.
   * **Trader Agents** → interpret news narratives + market micro-structure.

---

## Overall Workflow

```
┌─────────────┐
│ Price Data  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│   Analyst   │────▶│  Researcher  │
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Trader    │◀────┐
                    └──────┬──────┘     │
                           │            │
                           ▼            │
                    ┌─────────────┐     │
                    │ Risk Manager│     │
                    └──────┬──────┘     │
                           │            │
                           ▼            │
                    ┌─────────────┐     │
                    │  Evaluator  │─────┘
                    └─────────────┘
```

Part (1): **Agents Simulating a Hedge Fund Pipeline**

* Each step draws methods from the inventory.

---

## Inventory Reference

* **(F):** Agents built in first stage development
* **(S):** Agents built in second stage development
* All agents call **one shared LLM**

---

# Agent Descriptions

## Analyst Agent (F)

**Description:**
Processes time-series price data to output:

* Constructed Features (DataFrame)
* Constructed Trend Information (DataFrame)

**Processing Steps:**

* A-A Data Alignment
* A-B Feature Construction (ᴹ)
* A-C Trend Detection (ᴹ)

---

## Researcher Agent (F)

**Description:**
Consumes Analyst outputs (features + trends) and produces:

* JSON research summary with trading recommendations

**Processing Steps:**

* R-A Forecasting (ᴹ)
* R-B Uncertainty & Risk Quantification (ᴹ)
* R-C Probability Calibration (ᴹ)
* R-D Signal Packaging

**Output JSON Keys:**
`Meta, Market_State, Forecast, Signals, Risk, Recommendation, Scenarios, Explainability, Constraints, Confidence, Post_trade_evaluation_keys`

### 📖 Example JSON Structures

**Research Summary Example:**

```json
{
  "Meta": "...",
  "Market_State": "...",
  "Forecast": "...",
  "Signals": "...",
  "Risk": "...",
  "Recommendation": "...",
  "Scenarios": "...",
  "Explainability": "...",
  "Constraints": "...",
  "Confidence": "...",
  "Post_trade_evaluation_keys": "..."
}
```

---

## Trader Agent (F)

**Description:**

* Consumes Researcher outputs + fresh News Data
* Selects trading style from inventory (optimized over time)
* Executes orders considering both market conditions and LLM signals

**Notes:**

* Focused on a single instrument/market/product (BTC perpetuals)
* Style convergence expected but must avoid bias from extreme conditions
* Traders scored differently (longer iteration cycles recommended)

**Processing Steps:**

* T-A Obtain Execution Style (ᴹ)
* T-B Execute Order

**Output (JSON):**

* Order ID
* Current Price
* Limit/Market Order
* Position Size
* Direction (Long/Short)
* Take Profit / Stop Loss Prices
* Closed Price (N/A if open)
* Leverage Size
* Liquidation Price
* Execution Expired Time (EET)

**Order State Machine:**

```
Open → Filled → Closed (via TP/SL/EET)
```

---

## Risk Manager Agent (S)

**Description:**
Ensures Trader execution is safe:

* **hard_fail** → abort order
* **soft_fail** → regenerate order (back to T-B or T-A)
* **pass** → order executed & logged

**Processing Steps:**

* M-A Risk Analysis (ᴹ)
* M-B Output Log

**Outputs:**

* JSON risk analysis logs
* Pass / Soft Fail / Hard Fail decisions

**Decision Logic:**

* **Pass:** Order within VaR / size / margin limits
* **Soft Fail:** Violates minor rule (e.g., too high leverage) but can be adjusted
* **Hard Fail:** Breaches critical rule (e.g., margin call imminent) → discarded

---

## Continual Learning & Optimization

### Part (2) Continual Learning from the Best

* After K rounds, top agents in each category **teach peers**.
* Trader Agents may follow a distinct iteration cycle.

### Part (3) Inventory Pruning

* Rank methods by frequency of use.
* Remove less-used methods over time (careful to preserve scenario-specific methods).

### Part (n) Future Extensions

* Add Admin agent to generate reports, monitor performance, and deliver evaluations.

---

## Project Structure

```
MAS_Final_With_Agents/
├── trading_agents/           # Main multi-agent trading system (v0.2.0)
│   ├── agents/               # Agent implementations
│   │   ├── analyst.py        # Feature construction & trend detection
│   │   ├── researcher.py     # Forecasting, uncertainty, calibration
│   │   ├── trader.py         # LLM-powered order generation
│   │   ├── risk.py           # Risk validation (hard_fail/soft_fail/pass)
│   │   └── evaluator.py      # Performance scoring
│   ├── inventory/            # Pluggable methods (registry pattern)
│   │   ├── analyst/          # TALibStack, STL, GaussianHMM, KalmanFilter
│   │   ├── researcher/       # ARIMAX, TFT, Bootstrap, QuantileReg, Calibration
│   │   ├── trader/           # AggressiveMarket, PassiveLadderedLimit
│   │   └── risk/             # VaR, Leverage, Liquidation, Margin checks
│   ├── config/               # Configuration management
│   │   ├── schemas.py        # AppConfig, DataConfig, NewsConfig, LearningConfig
│   │   └── loader.py         # YAML loading, agent building
│   ├── optimization/         # Continual learning
│   │   ├── knowledge_transfer.py  # Top-to-bottom knowledge sharing
│   │   └── inventory_pruning.py   # Remove rarely-used methods
│   ├── services/             # External services
│   │   ├── llm.py            # LLM integration for Trader
│   │   └── metrics.py        # PerformanceTracker (Sharpe, PnL, etc.)
│   ├── utils/                # Utilities
│   │   ├── news_filter.py    # 3-stage news filtering
│   │   └── thought_logger.py # LLM thought process logging
│   ├── backtesting/          # Backtesting framework
│   ├── workflow.py           # WorkflowEngine (main orchestration)
│   ├── models.py             # Data models (ResearchSummary, ExecutionSummary, etc.)
│   └── cli.py                # Command-line interface
├── data_pipeline/            # Market data fetching and processing
│   ├── pipeline/             # Core data pipeline
│   └── news/                 # News fetching (SerpAPI, LLM prompts)
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Full config (730 days)
│   └── btc4h.yaml            # Short config (30 days, offline data)
├── data/                     # Data storage (prices, news)
└── README.md                 # This file
```

---

## Configuration

Configuration is managed via YAML files in the `configs/` directory:

* `configs/default.yaml` — Full configuration (730 days of data, all inventory methods)
* `configs/btc4h.yaml` — Short-term config (30 days, offline data for fast iteration)

Example usage:
```bash
# Run with default config
python -m trading_agents.cli run --symbol BTCUSD.PERP --interval 4h

# Run with custom config
python -m trading_agents.cli run --symbol BTCUSD.PERP --interval 4h --config configs/btc4h.yaml
```

---

## License & Attribution

This implementation borrows **design patterns** (orchestrator signature, multi-LLM adapter, news filter idea) from the **TradingAgents** and **TradingAgents-CN** projects (Apache-2.0). See their repositories for details; attribution retained in code comments and this README.
