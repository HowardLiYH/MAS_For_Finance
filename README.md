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
│   │   ├── evaluator.py         # Performance scoring
│   │   └── admin.py             # Monitoring & reporting
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
│   ├── news/                    # News intelligence
│   │   ├── providers/           # Search providers
│   │   │   ├── search_bocha.py  # Bocha AI search
│   │   │   └── search_serpapi.py# SerpAPI (legacy)
│   │   ├── llm_prompt_search.py # LLM-planned queries
│   │   ├── multi_asset_queries.py # Asset-specific queries
│   │   ├── sources.py           # Source credibility
│   │   ├── enrichment.py        # LLM news enrichment
│   │   └── aggregation.py       # News clustering
│   └── pipeline/
│       ├── multi_asset.py       # 5-coin Bybit loader
│       ├── cross_features.py    # Cross-asset signals
│       └── data_pipeline.py     # Unified entry point
│
├── configs/                     # YAML configurations
│   ├── multi_asset.yaml         # 5-coin trading (RECOMMENDED)
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

### Admin Agent
System monitoring and reporting:
- Event-driven alert system
- Scheduled performance reports
- Multi-channel notifications (Slack, Console, File)

---

## 🔄 System Workflow

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MULTI-AGENT TRADING SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │ DATA PIPELINE│────▶│TRADING AGENTS│────▶│  EXECUTION   │            │
│  │              │     │              │     │              │            │
│  │ • Price Data │     │ • Analyst    │     │ • Risk Check │            │
│  │ • News Data  │     │ • Researcher │     │ • Order Exec │            │
│  │ • Cross-Asset│     │ • Trader     │     │ • Eval Score │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│         │                    │                    │                     │
│         ▼                    ▼                    ▼                     │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │                      EVENT BUS                          │           │
│  │  Publishes: trade_signal, order_executed, pnl_update   │           │
│  └─────────────────────────────────────────────────────────┘           │
│         │                    │                    │                     │
│         ▼                    ▼                    ▼                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │ ADMIN AGENT  │     │   ALERTS     │     │   REPORTS    │            │
│  │              │     │              │     │              │            │
│  │ • Monitoring │     │ • Drawdown   │     │ • Daily      │            │
│  │ • Scheduling │     │ • Risk Breach│     │ • Weekly     │            │
│  │ • Notify     │     │ • Low Sharpe │     │ • Performance│            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Data Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRICE DATA FLOW                                                        │
│  ══════════════                                                         │
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │  Bybit CSVs  │────▶│ Load & Parse │────▶│  Per-Asset   │            │
│  │              │     │              │     │  DataFrames  │            │
│  │ • BTC.csv    │     │ • Timestamp  │     │              │            │
│  │ • ETH.csv    │     │ • OHLCV      │     │ • close      │            │
│  │ • SOL.csv    │     │ • OI, Fund   │     │ • volume     │            │
│  │ • DOGE.csv   │     │ • LS Ratio   │     │ • oi         │            │
│  │ • XRP.csv    │     │              │     │ • funding    │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                   │                     │
│                                                   ▼                     │
│                              ┌──────────────────────────────┐          │
│                              │    CROSS-ASSET FEATURES      │          │
│                              │                              │          │
│                              │  btc_dominance    = BTC/Total│          │
│                              │  altcoin_momentum = ALT rets │          │
│                              │  eth_btc_ratio    = ETH/BTC  │          │
│                              │  cross_oi_delta   = ΔOI sum  │          │
│                              │  aggregate_funding= wgt fund │          │
│                              │  risk_on_off      = ALT beta │          │
│                              │  market_volatility= avg vol  │          │
│                              │  cross_correlation= pairwise │          │
│                              └──────────────────────────────┘          │
│                                                                         │
│  NEWS DATA FLOW                                                         │
│  ══════════════                                                         │
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │ LLM Query    │────▶│ Bocha Search │────▶│ Raw Articles │            │
│  │ Generation   │     │ API          │     │              │            │
│  │              │     │              │     │ • title      │            │
│  │ "Generate 5  │     │ • freshness  │     │ • summary    │            │
│  │  queries for │     │ • count: 20  │     │ • url        │            │
│  │  BTC news"   │     │              │     │ • date       │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                   │                     │
│                                                   ▼                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │ Source       │────▶│ LLM Enrich   │────▶│ Aggregate    │            │
│  │ Credibility  │     │              │     │              │            │
│  │              │     │ • sentiment  │     │ • cluster    │            │
│  │ Tier 1: 1.0  │     │ • event_type │     │ • dominant   │            │
│  │ Tier 2: 0.7  │     │ • entities   │     │   narratives │            │
│  │ Tier 3: 0.4  │     │ • impact     │     │ • digest     │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                   │                     │
│                                                   ▼                     │
│                              ┌──────────────────────────────┐          │
│                              │        NEWS DIGEST           │          │
│                              │                              │          │
│                              │  sentiment_score: +0.35      │          │
│                              │  overall_sentiment: bullish  │          │
│                              │  dominant_narratives: [...]  │          │
│                              │  key_events: [...]           │          │
│                              │  asset_sentiment: {BTC: ...} │          │
│                              └──────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Trading Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRADING AGENT WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOR EACH ASSET (BTC, ETH, SOL, DOGE, XRP):                            │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │ Price Data   │─────────────────────────────┐                         │
│  │ + Cross-Asset│                             │                         │
│  │   Features   │                             ▼                         │
│  └──────────────┘                    ┌─────────────────┐               │
│                                      │  ANALYST AGENT  │               │
│                                      │                 │               │
│                                      │ ┌─────────────┐ │               │
│                                      │ │ TALib_Basic │ │               │
│                                      │ │ STL_Decomp  │ │               │
│                                      │ │ Gaussian_HMM│ │               │
│                                      │ │ Kalman_Filt │ │               │
│                                      │ └─────────────┘ │               │
│                                      │                 │               │
│                                      │ Output:         │               │
│                                      │ • features_df   │               │
│                                      │ • trend_dict    │               │
│                                      └────────┬────────┘               │
│                                               │                         │
│                                               ▼                         │
│                                      ┌─────────────────┐               │
│                                      │RESEARCHER AGENT │               │
│                                      │                 │               │
│                                      │ ┌─────────────┐ │               │
│                                      │ │ ARIMAX_Fcst │ │               │
│                                      │ │ TFT_Forecast│ │               │
│                                      │ │ Bootstrap_UQ│ │               │
│                                      │ │ Quantile_UQ │ │               │
│                                      │ │ Temp_Calib  │ │               │
│                                      │ └─────────────┘ │               │
│                                      │                 │               │
│                                      │ Output:         │               │
│                                      │ • ResearchSum   │               │
│                                      │   - forecast    │               │
│                                      │   - confidence  │               │
│                                      │   - risk        │               │
│                                      └────────┬────────┘               │
│                                               │                         │
│  ┌──────────────┐                             │                         │
│  │ News Digest  │─────────────────────────────┤                         │
│  │              │                             │                         │
│  │ • sentiment  │                             ▼                         │
│  │ • narratives │                    ┌─────────────────┐               │
│  │ • key events │                    │  TRADER AGENT   │               │
│  └──────────────┘                    │                 │               │
│                                      │ ┌─────────────┐ │               │
│                                      │ │  LLM Call   │ │               │
│                                      │ │  (GPT-4o)   │ │               │
│                                      │ └─────────────┘ │               │
│                                      │                 │               │
│                                      │ Input Prompt:   │               │
│                                      │ • Price summary │               │
│                                      │ • Research data │               │
│                                      │ • News digest   │               │
│                                      │ • Exec style    │               │
│                                      │                 │               │
│                                      │ Output:         │               │
│                                      │ • direction     │               │
│                                      │ • position_size │               │
│                                      │ • leverage      │               │
│                                      │ • entry/TP/SL   │               │
│                                      └────────┬────────┘               │
│                                               │                         │
│                                               ▼                         │
│                                      ┌─────────────────┐               │
│                                      │  RISK MANAGER   │               │
│                                      │                 │               │
│                                      │ Checks:         │               │
│                                      │ • Max leverage  │               │
│                                      │ • Position size │               │
│                                      │ • Margin safety │               │
│                                      │ • VaR limits    │               │
│                                      │                 │               │
│                                      │ Verdict:        │               │
│                                      │ ✅ pass         │               │
│                                      │ ⚠️ soft_fail    │               │
│                                      │ ❌ hard_fail    │               │
│                                      └────────┬────────┘               │
│                                               │                         │
│                          ┌────────────────────┴───────────────┐        │
│                          │                                    │        │
│                     [pass/soft_fail]                    [hard_fail]    │
│                          │                                    │        │
│                          ▼                                    ▼        │
│                 ┌─────────────────┐                  ┌──────────────┐  │
│                 │ EXECUTE ORDER   │                  │ ABORT ORDER  │  │
│                 │                 │                  │              │  │
│                 │ • Paper/Live    │                  │ Log reason   │  │
│                 │ • Bybit API     │                  │ No execution │  │
│                 └────────┬────────┘                  └──────────────┘  │
│                          │                                             │
│                          ▼                                             │
│                 ┌─────────────────┐                                    │
│                 │ EVALUATOR AGENT │                                    │
│                 │                 │                                    │
│                 │ Metrics:        │                                    │
│                 │ • Sharpe ratio  │                                    │
│                 │ • PnL           │                                    │
│                 │ • Hit rate      │                                    │
│                 │ • Max drawdown  │                                    │
│                 │ • Calibration   │                                    │
│                 └─────────────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### News Processing Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       NEWS PROCESSING WORKFLOW                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: QUERY GENERATION                                              │
│  ════════════════════════                                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Asset-Specific Query Templates (multi_asset_queries.py)  │          │
│  │                                                          │          │
│  │  BTC Micro Queries:                                      │          │
│  │  ├── "Bitcoin spot ETF inflows outflows today"          │          │
│  │  ├── "Bitcoin whale wallet movements"                   │          │
│  │  ├── "Bitcoin mining hash rate difficulty"              │          │
│  │  └── "BTC price technical analysis support resistance"  │          │
│  │                                                          │          │
│  │  BTC Macro Queries:                                      │          │
│  │  ├── "Federal Reserve interest rate decision"           │          │
│  │  ├── "US inflation CPI data release"                    │          │
│  │  └── "Cryptocurrency regulation news"                   │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  STEP 2: SEARCH EXECUTION                                              │
│  ════════════════════════                                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Bocha Search API (search_bocha.py)                       │          │
│  │                                                          │          │
│  │  Request:                                                │          │
│  │  ├── query: "Bitcoin spot ETF..."                       │          │
│  │  ├── freshness: "oneWeek"                               │          │
│  │  ├── count: 20                                          │          │
│  │  └── summary: true                                      │          │
│  │                                                          │          │
│  │  Response:                                               │          │
│  │  ├── title: "BlackRock ETF sees $500M inflow"           │          │
│  │  ├── summary: "Institutional demand..."                 │          │
│  │  ├── url: "https://..."                                 │          │
│  │  ├── siteName: "Bloomberg"                              │          │
│  │  └── datePublished: "2025-12-18T10:30:00Z"              │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  STEP 3: SOURCE CREDIBILITY                                            │
│  ══════════════════════════                                            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Source Tiers (sources.py)                                │          │
│  │                                                          │          │
│  │  Tier 1 (weight=1.0): High credibility                  │          │
│  │  ├── bloomberg.com, reuters.com, wsj.com                │          │
│  │  ├── coindesk.com, theblock.co, cointelegraph.com       │          │
│  │  └── sec.gov, federalreserve.gov                        │          │
│  │                                                          │          │
│  │  Tier 2 (weight=0.7): Medium credibility                │          │
│  │  ├── decrypt.co, bitcoinmagazine.com                    │          │
│  │  └── cryptoslate.com, newsbtc.com                       │          │
│  │                                                          │          │
│  │  Tier 3 (weight=0.4): Lower credibility                 │          │
│  │  └── Unknown/unranked sources                           │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  STEP 4: LLM ENRICHMENT                                                │
│  ══════════════════════                                                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ News Enrichment (enrichment.py)                          │          │
│  │                                                          │          │
│  │  Input: Raw article                                      │          │
│  │  ├── title: "BlackRock ETF sees $500M inflow"           │          │
│  │  └── summary: "Institutional demand continues..."       │          │
│  │                                                          │          │
│  │  LLM Extraction (GPT-4o-mini):                          │          │
│  │  ├── sentiment: "bullish" (0.7)                         │          │
│  │  ├── event_type: "etf_flow"                             │          │
│  │  ├── entities: ["BlackRock", "BTC"]                     │          │
│  │  ├── impact_timeframe: "short"                          │          │
│  │  ├── confidence: 0.85                                   │          │
│  │  └── key_facts: ["$500M inflow", "institutional"]       │          │
│  │                                                          │          │
│  │  Output: EnrichedNewsItem                               │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  STEP 5: AGGREGATION                                                   │
│  ═══════════════════                                                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ News Aggregation (aggregation.py)                        │          │
│  │                                                          │          │
│  │  Clustering (by event_type):                            │          │
│  │  ├── Cluster 1: ETF flows (5 articles)                  │          │
│  │  │   └── Narrative: "Strong institutional inflows"      │          │
│  │  ├── Cluster 2: Regulation (3 articles)                 │          │
│  │  │   └── Narrative: "SEC review ongoing"                │          │
│  │  └── Cluster 3: Price analysis (7 articles)             │          │
│  │      └── Narrative: "Technical breakout expected"       │          │
│  │                                                          │          │
│  │  Final Digest:                                           │          │
│  │  ├── sentiment_score: +0.35                             │          │
│  │  ├── overall_sentiment: "bullish"                       │          │
│  │  ├── dominant_narratives: [...]                         │          │
│  │  ├── key_events: [...]                                  │          │
│  │  └── asset_sentiment: {BTC: +0.4, ETH: +0.2, ...}       │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Admin & Monitoring Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADMIN & MONITORING WORKFLOW                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  EVENT BUS (Central Communication)                                      │
│  ═════════════════════════════════                                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                       EVENT BUS                          │          │
│  │                                                          │          │
│  │  Publishers:                    Subscribers:             │          │
│  │  ├── WorkflowEngine ──────────▶ AdminAgent              │          │
│  │  ├── OrderManager ────────────▶ AlertsEngine            │          │
│  │  ├── PositionTracker ─────────▶ ReportGenerator         │          │
│  │  └── RiskManager ─────────────▶ NotificationService     │          │
│  │                                                          │          │
│  │  Event Types:                                            │          │
│  │  ├── trade_signal    │ order_executed  │ pnl_update     │          │
│  │  ├── risk_breach     │ drawdown_alert  │ system_health  │          │
│  │  └── iteration_complete │ error │ warning               │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│              ┌───────────────┼───────────────┐                         │
│              │               │               │                         │
│              ▼               ▼               ▼                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │ ALERT RULES  │   │   REPORTS    │   │NOTIFICATIONS │               │
│  │              │   │              │   │              │               │
│  │ Conditions:  │   │ Scheduled:   │   │ Channels:    │               │
│  │ • MaxDD >10% │   │ • Daily 8AM  │   │ • Console    │               │
│  │ • DailyLoss  │   │ • Weekly Mon │   │ • Slack      │               │
│  │ • RiskBreach │   │              │   │ • File log   │               │
│  │ • LowSharpe  │   │ On-demand:   │   │ • Email      │               │
│  │ • Position   │   │ • Performance│   │              │               │
│  │   Concentr.  │   │ • Custom     │   │              │               │
│  └──────┬───────┘   └──────┬───────┘   └──────────────┘               │
│         │                  │                    ▲                      │
│         │                  │                    │                      │
│         ▼                  ▼                    │                      │
│  ┌─────────────────────────────────────────────┴──┐                   │
│  │                  ADMIN AGENT                    │                   │
│  │                                                 │                   │
│  │  Responsibilities:                              │                   │
│  │  ├── Monitor all events from EventBus          │                   │
│  │  ├── Evaluate alert conditions                 │                   │
│  │  ├── Trigger notifications on breaches         │                   │
│  │  ├── Generate scheduled reports                │                   │
│  │  └── Track system health metrics               │                   │
│  │                                                 │                   │
│  │  Alert Flow:                                    │                   │
│  │  Event ──▶ Check Rules ──▶ If triggered ──▶ Notify                 │
│  │                                                 │                   │
│  │  Report Flow:                                   │                   │
│  │  Schedule ──▶ Collect Metrics ──▶ Generate ──▶ Send               │
│  └─────────────────────────────────────────────────┘                   │
│                                                                         │
│  PAPER TRADING FLOW                                                    │
│  ══════════════════                                                    │
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│  │ Trade Signal │────▶│OrderManager  │────▶│ Bybit API    │           │
│  │              │     │              │     │ (Testnet)    │           │
│  │ direction    │     │ • validate   │     │              │           │
│  │ size         │     │ • submit     │     │ • place_order│           │
│  │ leverage     │     │ • track      │     │ • get_pos    │           │
│  │ TP/SL        │     │ • confirm    │     │ • get_bal    │           │
│  └──────────────┘     └──────┬───────┘     └──────────────┘           │
│                              │                                          │
│                              ▼                                          │
│                     ┌──────────────┐                                   │
│                     │PositionTrack │                                   │
│                     │              │                                   │
│                     │ • Open pos   │                                   │
│                     │ • Unrealized │                                   │
│                     │   PnL        │                                   │
│                     │ • Emit events│                                   │
│                     └──────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Optimization & Learning Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION & LEARNING WORKFLOW                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PERFORMANCE TRACKING                                                   │
│  ════════════════════                                                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ After each iteration:                                    │          │
│  │                                                          │          │
│  │  Evaluator Agent ──▶ PerformanceTracker                 │          │
│  │                                                          │          │
│  │  Metrics Collected:                                      │          │
│  │  ├── Trade results (win/loss, PnL)                      │          │
│  │  ├── Per-method performance                             │          │
│  │  ├── Agent-level scores                                 │          │
│  │  └── System-wide metrics                                │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  KNOWLEDGE TRANSFER (Every N iterations)                               │
│  ═══════════════════════════════════════                               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ KnowledgeTransfer Module                                 │          │
│  │                                                          │          │
│  │  Step 1: Collect agent experiences                      │          │
│  │  ├── Analyst: Which features predicted well?            │          │
│  │  ├── Researcher: Which forecasts were accurate?         │          │
│  │  ├── Trader: Which styles worked in what conditions?    │          │
│  │  └── Risk: Which checks prevented bad trades?           │          │
│  │                                                          │          │
│  │  Step 2: Cross-agent insights                           │          │
│  │  ├── Analyst features → Researcher calibration          │          │
│  │  ├── Risk patterns → Trader position sizing             │          │
│  │  └── Evaluator feedback → All agents                    │          │
│  │                                                          │          │
│  │  Step 3: Update agent parameters                        │          │
│  │  └── Store in shared knowledge base                     │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  INVENTORY PRUNING (Every M iterations)                                │
│  ══════════════════════════════════════                                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ InventoryPruner Module                                   │          │
│  │                                                          │          │
│  │  For each agent's inventory methods:                    │          │
│  │                                                          │          │
│  │  ┌─────────────────────────────────────────────┐        │          │
│  │  │ Method: ARIMAX_Forecast                     │        │          │
│  │  │ Usage count: 150                            │        │          │
│  │  │ Success rate: 0.62                          │        │          │
│  │  │ Avg return: +0.8%                           │        │          │
│  │  │ Status: ✅ KEEP                             │        │          │
│  │  └─────────────────────────────────────────────┘        │          │
│  │                                                          │          │
│  │  ┌─────────────────────────────────────────────┐        │          │
│  │  │ Method: Experimental_Strategy_X             │        │          │
│  │  │ Usage count: 10                             │        │          │
│  │  │ Success rate: 0.35                          │        │          │
│  │  │ Avg return: -1.2%                           │        │          │
│  │  │ Status: ❌ PRUNE (low usage + poor perf)    │        │          │
│  │  └─────────────────────────────────────────────┘        │          │
│  │                                                          │          │
│  │  Pruning criteria:                                       │          │
│  │  ├── Usage count < threshold AND                        │          │
│  │  └── Performance < min_score                            │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
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
* (2025.12.19) **Bocha Search Integration v0.4.1**
   * Replaced expensive SerpAPI with Bocha Search API
   * Bocha provides better Chinese/global web search at lower cost
   * Supports time-based freshness filtering (oneDay, oneWeek, oneMonth)
   * ISO date parsing for reliable date filtering
* (2025.12.19) **Enhanced News Intelligence v0.4.2**
   * Multi-asset query templates for asset-specific news
   * Source credibility scoring (tier-1, tier-2, tier-3)
   * LLM-based news enrichment (sentiment, events, entities, impact)
   * News clustering and aggregation for market digest
   * Enhanced trader prompt with structured news integration

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
- ~~Enhanced news intelligence pipeline~~ ✅ Done
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
