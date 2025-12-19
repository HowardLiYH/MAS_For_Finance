# Multi-Agent Trading System - Complete Architecture

## 🎯 System Overview

```mermaid
flowchart TB
    subgraph INPUTS["📥 INPUT LAYER"]
        BYBIT[("🔷 Bybit CSVs<br/>BTC, ETH, SOL<br/>DOGE, XRP")]
        BOCHA["🔍 Bocha Search API"]
        CONFIG[("⚙️ YAML Configs<br/>multi_asset.yaml<br/>single/*.yaml")]
    end

    subgraph DATA_PIPELINE["📊 DATA PIPELINE"]
        subgraph PRICE_PROC["Price Processing"]
            LOADER["MultiAssetLoader<br/>load_bybit_csv()"]
            ALIGN["align_timestamps()"]
            CROSS["CrossAssetFeatures<br/>8 signals"]
        end

        subgraph NEWS_PROC["News Processing"]
            QUERY["QueryGenerator<br/>multi_asset_queries.py"]
            SEARCH["BochaSearchProvider<br/>search_bocha.py"]
            CRED["SourceCredibility<br/>sources.py"]
            ENRICH["NewsEnrichment<br/>enrichment.py"]
            AGG["NewsAggregation<br/>aggregation.py"]
        end
    end

    subgraph TRADING_AGENTS["🤖 TRADING AGENTS"]
        ANALYST["📈 Analyst Agent"]
        RESEARCHER["🔬 Researcher Agent"]
        TRADER["💹 Trader Agent"]
        RISK["🛡️ Risk Manager"]
        EVAL["📊 Evaluator Agent"]
        ADMIN["👨‍💼 Admin Agent"]
    end

    subgraph SERVICES["⚡ SERVICES LAYER"]
        LLM["🧠 LLM Service<br/>GPT-4o"]
        EVENTS["📡 Event Bus"]
        METRICS["📈 Metrics Tracker"]
        ALERTS["🚨 Alert Rules"]
        NOTIFY["📢 Notifications"]
        REPORTS["📄 Reports"]
    end

    subgraph EXECUTION["🎯 EXECUTION"]
        ORDER["OrderManager"]
        BYBIT_API["Bybit Testnet API"]
        POS["PositionTracker"]
    end

    subgraph OPTIMIZATION["🔄 OPTIMIZATION"]
        KNOWLEDGE["KnowledgeTransfer"]
        PRUNER["InventoryPruner"]
    end

    %% Connections
    BYBIT --> LOADER
    CONFIG --> LOADER
    LOADER --> ALIGN --> CROSS

    BOCHA --> SEARCH
    QUERY --> SEARCH
    SEARCH --> CRED --> ENRICH --> AGG

    CROSS --> ANALYST
    AGG --> TRADER

    ANALYST --> RESEARCHER --> TRADER --> RISK --> EVAL

    TRADER --> LLM
    RISK --> ORDER --> BYBIT_API --> POS

    EVAL --> METRICS --> EVENTS
    EVENTS --> ADMIN
    ADMIN --> ALERTS --> NOTIFY
    ADMIN --> REPORTS

    METRICS --> KNOWLEDGE --> PRUNER
```

---

## 📊 Data Pipeline - Complete Flow

```mermaid
flowchart TB
    subgraph PRICE_INPUT["📈 PRICE DATA INPUT"]
        BTC_CSV["Bybit_BTC.csv"]
        ETH_CSV["Bybit_ETH.csv"]
        SOL_CSV["Bybit_SOL.csv"]
        DOGE_CSV["Bybit_DOGE.csv"]
        XRP_CSV["Bybit_XRP.csv"]
    end

    subgraph LOAD["📂 LOAD & PARSE"]
        LOADER2["load_bybit_csv()<br/>├── timestamp parsing<br/>├── column rename<br/>├── sort by time<br/>└── dropna"]
    end

    subgraph ALIGN2["⏰ TIMESTAMP ALIGNMENT"]
        ALIGN_FN["align_timestamps()<br/>├── find common index<br/>├── reindex all assets<br/>└── forward fill gaps"]
    end

    subgraph PER_ASSET["📊 PER-ASSET FEATURES"]
        BTC_DF["BTC DataFrame<br/>close, volume, oi<br/>funding, ls_ratio"]
        ETH_DF["ETH DataFrame"]
        SOL_DF["SOL DataFrame"]
        DOGE_DF["DOGE DataFrame"]
        XRP_DF["XRP DataFrame"]
    end

    subgraph CROSS_FEAT["🌐 CROSS-ASSET FEATURES"]
        DOM["btc_dominance<br/>= BTC_close / Σ(all_close)"]
        ALT["altcoin_momentum<br/>= mean(ALT returns)"]
        ETH_BTC["eth_btc_ratio<br/>= ETH_close / BTC_close"]
        OI_DELTA["cross_oi_delta<br/>= Σ(pct_change(OI))"]
        FUND["aggregate_funding<br/>= weighted avg funding"]
        RISK_ON["risk_on_off<br/>= ALT_ret / BTC_ret"]
        VOL["market_volatility<br/>= mean(annualized vol)"]
        CORR["cross_correlation<br/>= mean(pairwise corr)"]
    end

    subgraph OUTPUT_PRICE["📤 OUTPUT"]
        MARKET_CTX["MarketContext<br/>dataclass"]
        ASSET_DFS["Dict[symbol, DataFrame]"]
    end

    %% Connections
    BTC_CSV & ETH_CSV & SOL_CSV & DOGE_CSV & XRP_CSV --> LOADER2
    LOADER2 --> ALIGN_FN
    ALIGN_FN --> BTC_DF & ETH_DF & SOL_DF & DOGE_DF & XRP_DF

    BTC_DF & ETH_DF & SOL_DF & DOGE_DF & XRP_DF --> DOM & ALT & ETH_BTC & OI_DELTA & FUND & RISK_ON & VOL & CORR

    DOM & ALT & ETH_BTC & OI_DELTA & FUND & RISK_ON & VOL & CORR --> MARKET_CTX
    BTC_DF & ETH_DF & SOL_DF & DOGE_DF & XRP_DF --> ASSET_DFS
```

---

## 📰 News Pipeline - Complete Flow

```mermaid
flowchart TB
    subgraph QUERY_GEN["🔍 QUERY GENERATION"]
        ASSET_Q["ASSET_QUERIES dict<br/>├── BTC: 8 micro + 5 macro<br/>├── ETH: 6 micro + 5 macro<br/>├── SOL: 5 micro + 4 macro<br/>├── DOGE: 4 micro + 3 macro<br/>└── XRP: 5 micro + 4 macro"]

        MICRO["Micro Queries<br/>├── ETF flows<br/>├── Whale movements<br/>├── Mining/staking<br/>├── Technical analysis<br/>└── Exchange reserves"]

        MACRO["Macro Queries<br/>├── Fed interest rates<br/>├── CPI/inflation<br/>├── Regulation<br/>├── Geopolitical<br/>└── Market sentiment"]
    end

    subgraph SEARCH_EXEC["🔎 SEARCH EXECUTION"]
        BOCHA_API["Bocha Web Search API<br/>POST /v1/web-search<br/>├── query: string<br/>├── freshness: oneWeek<br/>├── count: 20<br/>└── summary: true"]

        RAW_RESULTS["Raw Results<br/>├── title<br/>├── snippet/summary<br/>├── url<br/>├── siteName<br/>└── datePublished"]
    end

    subgraph CREDIBILITY["⭐ SOURCE CREDIBILITY"]
        TIER1["Tier 1 (weight=1.0)<br/>├── bloomberg.com<br/>├── reuters.com<br/>├── wsj.com<br/>├── coindesk.com<br/>├── theblock.co<br/>└── sec.gov"]

        TIER2["Tier 2 (weight=0.7)<br/>├── decrypt.co<br/>├── bitcoinmagazine.com<br/>├── cryptoslate.com<br/>└── newsbtc.com"]

        TIER3["Tier 3 (weight=0.4)<br/>└── Unknown sources"]

        FILTER["filter_by_credibility()<br/>sort_by_credibility()"]
    end

    subgraph ENRICHMENT["🧠 LLM ENRICHMENT"]
        LLM_CALL["GPT-4o-mini<br/>enrich_with_llm()"]

        EXTRACT["Extract:<br/>├── sentiment: bullish/bearish/neutral<br/>├── sentiment_score: -1.0 to 1.0<br/>├── event_type: etf_flow/regulation/whale...<br/>├── entities: [BTC, BlackRock, SEC...]<br/>├── impact_timeframe: immediate/short/medium/long<br/>├── confidence: 0.0 to 1.0<br/>└── key_facts: [string list]"]

        ENRICHED["EnrichedNewsItem<br/>dataclass"]
    end

    subgraph AGGREGATION["📊 AGGREGATION"]
        DEDUP["deduplicate_by_content()<br/>TF-IDF + cosine similarity"]

        CLUSTER["Cluster by event_type<br/>├── etf_flow cluster<br/>├── regulation cluster<br/>├── whale_movement cluster<br/>└── technical cluster"]

        NARRATIVE["Generate narratives<br/>per cluster"]

        DIGEST["NewsDigest<br/>├── sentiment_score: weighted avg<br/>├── overall_sentiment: bull/bear/neutral<br/>├── sentiment_trend: improving/stable/declining<br/>├── dominant_narratives: [top 3]<br/>├── key_events: [sorted by impact]<br/>├── asset_sentiment: {BTC: 0.3, ETH: 0.1...}<br/>└── tier1_percentage: 0.65"]
    end

    subgraph OUTPUT_NEWS["📤 OUTPUT"]
        TO_TRADER["→ Trader Agent<br/>format_news_digest()"]
    end

    %% Connections
    ASSET_Q --> MICRO & MACRO
    MICRO & MACRO --> BOCHA_API
    BOCHA_API --> RAW_RESULTS

    RAW_RESULTS --> TIER1 & TIER2 & TIER3
    TIER1 & TIER2 & TIER3 --> FILTER

    FILTER --> LLM_CALL
    LLM_CALL --> EXTRACT --> ENRICHED

    ENRICHED --> DEDUP --> CLUSTER --> NARRATIVE --> DIGEST

    DIGEST --> TO_TRADER
```

---

## 🤖 Agent Workflow - Complete Flow

```mermaid
flowchart TB
    subgraph ANALYST_AGENT["📈 ANALYST AGENT"]
        subgraph ANALYST_INV["Inventory Methods"]
            TALIB["TALib_Basic<br/>├── RSI, MACD, BB<br/>├── ADX, ATR<br/>└── SMA, EMA"]
            STL["STL_Decompose<br/>├── trend<br/>├── seasonal<br/>└── residual"]
            HMM["Gaussian_HMM<br/>├── regime detection<br/>├── bull/bear/neutral<br/>└── transition probs"]
            KALMAN["Kalman_Filter<br/>├── trend extraction<br/>└── noise reduction"]
        end

        ANALYST_OUT["Output:<br/>├── features_df: DataFrame<br/>└── trend_dict: Dict"]
    end

    subgraph RESEARCHER_AGENT["🔬 RESEARCHER AGENT"]
        subgraph RESEARCHER_INV["Inventory Methods"]
            ARIMAX["ARIMAX_Forecast<br/>├── 8h forecast<br/>├── 24h forecast<br/>└── confidence"]
            TFT["TFT_Forecast<br/>├── transformer-based<br/>└── multi-horizon"]
            BOOT["Bootstrap_UQ<br/>├── ensemble sampling<br/>└── confidence intervals"]
            QUANT["Quantile_Regression<br/>├── q05, q25, q50, q75, q95<br/>└── distribution"]
            CALIB["Temperature_Scaling<br/>└── calibration ECE"]
        end

        RESEARCHER_OUT["Output: ResearchSummary<br/>├── market_state<br/>├── recommendation: BUY/SELL/HOLD<br/>├── confidence: 0.0-1.0<br/>├── forecast: {8h, 24h}<br/>└── risk: {q05, q95, var}"]
    end

    subgraph TRADER_AGENT["💹 TRADER AGENT"]
        STYLE_SELECT["Style Selection<br/>├── Aggressive_Market<br/>├── Conservative_Limit<br/>└── Neutral_Scaled"]

        LLM_GEN["LLM Generation<br/>GPT-4o / GPT-4o-mini"]

        PROMPT["Prompt Includes:<br/>├── Execution style<br/>├── Price summary<br/>├── Research summary<br/>├── News digest<br/>└── Trading rules"]

        TRADER_OUT["Output: ExecutionSummary<br/>├── direction: LONG/SHORT<br/>├── position_size: 0.0-1.0<br/>├── leverage: 1-10x<br/>├── order_type: MARKET/LIMIT<br/>├── entry_price<br/>├── take_profit<br/>├── stop_loss<br/>└── liquidation_price"]
    end

    subgraph RISK_AGENT["🛡️ RISK MANAGER"]
        subgraph RISK_CHECKS["Risk Checks"]
            LEV["Leverage_Limit<br/>max: 10x"]
            SIZE["Position_Size<br/>max: 50%"]
            MARGIN["Margin_Safety<br/>min: 20%"]
            VAR["VaR_Limit<br/>max: 5%"]
            LIQ["Liquidation_Safety<br/>buffer: 10%"]
        end

        VERDICTS["Verdicts:<br/>├── ✅ pass<br/>├── ⚠️ soft_fail (adjust)<br/>└── ❌ hard_fail (abort)"]

        RISK_OUT["Output: RiskReview<br/>├── verdict<br/>├── adjustments<br/>└── violations"]
    end

    subgraph EVAL_AGENT["📊 EVALUATOR AGENT"]
        TRACK["PerformanceTracker"]

        METRICS2["Metrics:<br/>├── Sharpe ratio<br/>├── PnL<br/>├── Hit rate<br/>├── Max drawdown<br/>└── Calibration ECE"]

        EVAL_OUT["Output: AgentScores<br/>├── per_agent_scores<br/>├── per_method_scores<br/>└── system_metrics"]
    end

    subgraph ADMIN_AGENT2["👨‍💼 ADMIN AGENT"]
        MONITOR["Monitor Events"]

        ALERT_CHECK["Check Alerts<br/>├── MaxDrawdown > 10%<br/>├── DailyLoss > 5%<br/>├── RiskBreach count<br/>└── Sharpe < 0.5"]

        REPORT_GEN["Generate Reports<br/>├── Daily summary<br/>├── Weekly summary<br/>└── Performance report"]

        SEND_NOTIFY["Send Notifications<br/>├── Console<br/>├── Slack<br/>└── File"]
    end

    %% Flow
    TALIB & STL & HMM & KALMAN --> ANALYST_OUT
    ANALYST_OUT --> ARIMAX & TFT & BOOT & QUANT & CALIB
    ARIMAX & TFT & BOOT & QUANT & CALIB --> RESEARCHER_OUT

    RESEARCHER_OUT --> STYLE_SELECT
    STYLE_SELECT --> LLM_GEN
    PROMPT --> LLM_GEN
    LLM_GEN --> TRADER_OUT

    TRADER_OUT --> LEV & SIZE & MARGIN & VAR & LIQ
    LEV & SIZE & MARGIN & VAR & LIQ --> VERDICTS
    VERDICTS --> RISK_OUT

    RISK_OUT --> TRACK
    TRACK --> METRICS2
    METRICS2 --> EVAL_OUT

    EVAL_OUT --> MONITOR
    MONITOR --> ALERT_CHECK & REPORT_GEN
    ALERT_CHECK --> SEND_NOTIFY
    REPORT_GEN --> SEND_NOTIFY
```

---

## ⚡ Services Layer - Complete Architecture

```mermaid
flowchart TB
    subgraph EVENT_BUS["📡 EVENT BUS"]
        PUBLISH["publish(event_type, data)"]
        SUBSCRIBE["subscribe(event_type, callback)"]

        EVENT_TYPES["Event Types:<br/>├── trade_signal<br/>├── order_submitted<br/>├── order_filled<br/>├── order_rejected<br/>├── position_update<br/>├── pnl_update<br/>├── risk_breach<br/>├── drawdown_alert<br/>├── iteration_complete<br/>├── system_health<br/>├── error<br/>└── warning"]
    end

    subgraph LLM_SERVICE["🧠 LLM SERVICE"]
        CREATE_CLIENT["_create_openai_client()<br/>├── API key from env<br/>└── custom base_url support"]

        GEN_PROPOSAL["generate_trading_proposal()<br/>├── execution_style<br/>├── research_summary<br/>├── news_digest<br/>├── price_data<br/>└── model selection"]

        FORMAT_NEWS["format_news_digest()<br/>├── sentiment score<br/>├── narratives<br/>├── key events<br/>└── asset sentiment"]

        FALLBACK["_fallback_proposal()<br/>Rule-based backup"]
    end

    subgraph METRICS_SERVICE["📈 METRICS SERVICE"]
        PERF_TRACK["PerformanceTracker<br/>├── record_trade()<br/>├── get_sharpe()<br/>├── get_pnl()<br/>├── get_hit_rate()<br/>├── get_max_dd()<br/>└── get_calib_ece()"]

        METHOD_TRACK["Method Tracking<br/>├── method_id<br/>├── usage_count<br/>├── success_rate<br/>└── avg_return"]
    end

    subgraph ALERT_SERVICE["🚨 ALERT SERVICE"]
        RULES["Alert Rules:<br/>├── MaxDrawdownRule(threshold=0.1)<br/>├── DailyLossRule(threshold=0.05)<br/>├── RiskBreachRule(max_breaches=3)<br/>├── LowSharpeRule(min_sharpe=0.5)<br/>└── PositionConcentration(max=0.5)"]

        CHECK["check(metrics) → bool"]
        TRIGGER["trigger() → Notification"]
    end

    subgraph NOTIFY_SERVICE["📢 NOTIFICATION SERVICE"]
        CHANNELS["Channels:<br/>├── ConsoleChannel<br/>├── SlackChannel<br/>├── FileChannel<br/>└── EmailChannel (TODO)"]

        SEND["send(channel, message, level)"]

        LEVELS["Levels:<br/>├── INFO<br/>├── WARNING<br/>├── ERROR<br/>└── CRITICAL"]
    end

    subgraph REPORT_SERVICE["📄 REPORT SERVICE"]
        REPORT_TYPES["Report Types:<br/>├── PerformanceReport<br/>├── DailySummary<br/>└── WeeklySummary"]

        GENERATE["generate(metrics, trades) → str"]
        SCHEDULE["Scheduled:<br/>├── Daily @ 8:00 AM<br/>└── Weekly @ Monday"]
    end

    subgraph BYBIT_SERVICE["🔷 BYBIT SERVICE"]
        CLIENT["BybitTestnetClient<br/>├── api_key, api_secret<br/>└── base_url (testnet)"]

        METHODS["Methods:<br/>├── place_order(symbol, side, qty, price)<br/>├── cancel_order(order_id)<br/>├── get_positions()<br/>├── get_wallet_balance()<br/>└── get_order_status(order_id)"]

        SIGN["_sign_request()<br/>HMAC-SHA256"]
    end

    subgraph ORDER_SERVICE["📋 ORDER SERVICE"]
        ORDER_MGR["OrderManager<br/>├── submit_order(proposal)<br/>├── monitor_fills()<br/>├── cancel_order(id)<br/>└── get_open_orders()"]

        ORDER_STATES["Order States:<br/>├── pending<br/>├── submitted<br/>├── partial<br/>├── filled<br/>├── cancelled<br/>└── rejected"]
    end

    subgraph POSITION_SERVICE["📊 POSITION SERVICE"]
        POS_TRACK["PositionTracker<br/>├── update_positions()<br/>├── get_position(symbol)<br/>├── calculate_unrealized_pnl()<br/>└── emit_pnl_events()"]

        POS_DATA["Position Data:<br/>├── symbol<br/>├── side<br/>├── size<br/>├── entry_price<br/>├── mark_price<br/>├── unrealized_pnl<br/>└── leverage"]
    end

    %% Connections
    PUBLISH --> EVENT_TYPES
    EVENT_TYPES --> SUBSCRIBE

    CREATE_CLIENT --> GEN_PROPOSAL
    FORMAT_NEWS --> GEN_PROPOSAL
    GEN_PROPOSAL --> FALLBACK

    PERF_TRACK --> METHOD_TRACK

    RULES --> CHECK --> TRIGGER

    CHANNELS --> SEND
    LEVELS --> SEND

    REPORT_TYPES --> GENERATE
    SCHEDULE --> GENERATE

    CLIENT --> METHODS
    SIGN --> METHODS

    ORDER_MGR --> ORDER_STATES

    POS_TRACK --> POS_DATA
```

---

## 🔄 Optimization Loop - Complete Flow

```mermaid
flowchart TB
    subgraph ITERATION["🔁 TRADING ITERATION"]
        ITER_START["Start Iteration N"]
        RUN_AGENTS["Run All Agents<br/>Analyst → Researcher → Trader → Risk → Eval"]
        RECORD["Record Results<br/>├── trade outcome<br/>├── method performance<br/>└── agent scores"]
        ITER_END["End Iteration N"]
    end

    subgraph KNOWLEDGE["📚 KNOWLEDGE TRANSFER"]
        CHECK_K["Check: N % K == 0?"]

        COLLECT["Collect Agent Experiences<br/>├── Analyst: feature importance<br/>├── Researcher: forecast accuracy<br/>├── Trader: style performance<br/>└── Risk: breach patterns"]

        CROSS_LEARN["Cross-Agent Learning<br/>├── Analyst → Researcher calibration<br/>├── Risk → Trader sizing<br/>├── Evaluator → All agents<br/>└── Market context → Trading style"]

        UPDATE_K["Update Agent Parameters<br/>Store in knowledge base"]
    end

    subgraph PRUNING["✂️ INVENTORY PRUNING"]
        CHECK_P["Check: N % M == 0?"]

        ANALYZE["Analyze Method Performance<br/>For each method:<br/>├── usage_count<br/>├── success_rate<br/>└── avg_return"]

        CRITERIA["Pruning Criteria<br/>├── usage < min_threshold AND<br/>└── performance < min_score"]

        PRUNE["Remove Underperformers<br/>├── Disable method<br/>├── Log removal<br/>└── Update registry"]

        KEEP["Keep Performers<br/>├── High usage + good perf<br/>├── Low usage + excellent perf<br/>└── Recently added (grace period)"]
    end

    subgraph REGISTRY["📦 INVENTORY REGISTRY"]
        ANALYST_REG["Analyst Registry<br/>├── TALib_Basic ✅<br/>├── STL_Decompose ✅<br/>├── Gaussian_HMM ✅<br/>├── Kalman_Filter ✅<br/>└── Experimental_X ❌"]

        RESEARCHER_REG["Researcher Registry<br/>├── ARIMAX_Forecast ✅<br/>├── TFT_Forecast ✅<br/>├── Bootstrap_UQ ✅<br/>├── Quantile_UQ ✅<br/>└── Conformal_Calib ✅"]

        TRADER_REG["Trader Registry<br/>├── Aggressive_Market ✅<br/>├── Conservative_Limit ✅<br/>└── Neutral_Scaled ✅"]

        RISK_REG["Risk Registry<br/>├── Leverage_Limit ✅<br/>├── Position_Size ✅<br/>├── Margin_Safety ✅<br/>├── VaR_Limit ✅<br/>└── Liquidation_Safety ✅"]
    end

    %% Flow
    ITER_START --> RUN_AGENTS --> RECORD --> ITER_END

    ITER_END --> CHECK_K
    CHECK_K -->|Yes| COLLECT
    CHECK_K -->|No| CHECK_P
    COLLECT --> CROSS_LEARN --> UPDATE_K --> CHECK_P

    CHECK_P -->|Yes| ANALYZE
    CHECK_P -->|No| ITER_START
    ANALYZE --> CRITERIA
    CRITERIA --> PRUNE & KEEP
    PRUNE --> ANALYST_REG & RESEARCHER_REG & TRADER_REG & RISK_REG
    KEEP --> ANALYST_REG & RESEARCHER_REG & TRADER_REG & RISK_REG

    ANALYST_REG & RESEARCHER_REG & TRADER_REG & RISK_REG --> ITER_START
```

---

## 🎛️ Configuration Structure

```mermaid
flowchart TB
    subgraph CONFIGS["⚙️ CONFIGURATION FILES"]
        subgraph MULTI["configs/multi_asset.yaml"]
            MULTI_DATA["data:<br/>  multi_asset: true<br/>  symbols: [BTC,ETH,SOL,DOGE,XRP]<br/>  bybit_csv_dir: data/bybit<br/>  add_cross_features: true"]

            MULTI_NEWS["news:<br/>  enabled: true<br/>  search_provider: bocha<br/>  max_articles: 50"]

            MULTI_AGENTS["agents:<br/>  analyst: [TALib, STL, HMM]<br/>  researcher: [ARIMAX, Bootstrap]<br/>  trader: [Aggressive, Conservative]<br/>  risk: [Leverage, Position, Margin]"]

            MULTI_ADMIN["admin:<br/>  enabled: true<br/>  alerts: [MaxDD, DailyLoss, Sharpe]<br/>  reports: [daily, weekly]<br/>  notify: [console, slack]"]

            MULTI_PAPER["paper_trading:<br/>  enabled: true<br/>  testnet: true<br/>  initial_balance: 10000"]
        end

        subgraph SINGLE["configs/single/btc.yaml"]
            SINGLE_DATA["data:<br/>  multi_asset: false<br/>  offline_prices_csv: data/bybit/Bybit_BTC.csv<br/>  symbol: BTCUSDT<br/>  interval: 4h"]

            SINGLE_NEWS["news:<br/>  enabled: true<br/>  search_provider: bocha<br/>  micro_queries: 5<br/>  macro_queries: 3"]
        end
    end

    subgraph SCHEMAS["📋 CONFIG SCHEMAS"]
        DATA_SCHEMA["DataConfig<br/>├── multi_asset: bool<br/>├── symbols: List[str]<br/>├── bybit_csv_dir: str<br/>├── offline_prices_csv: str<br/>├── add_cross_features: bool<br/>└── interval: str"]

        NEWS_SCHEMA["NewsConfig<br/>├── enabled: bool<br/>├── search_provider: str<br/>├── max_articles: int<br/>├── freshness: str<br/>└── include_macro: bool"]

        AGENT_SCHEMA["AgentConfig<br/>├── analyst: List[str]<br/>├── researcher: List[str]<br/>├── trader: List[str]<br/>└── risk: List[str]"]

        ADMIN_SCHEMA["AdminConfig<br/>├── enabled: bool<br/>├── alerts: List[str]<br/>├── reports: List[str]<br/>└── notify_channels: List[str]"]

        PAPER_SCHEMA["PaperTradingConfig<br/>├── enabled: bool<br/>├── testnet: bool<br/>├── api_key_env: str<br/>├── api_secret_env: str<br/>└── initial_balance: float"]

        APP_SCHEMA["AppConfig<br/>├── data: DataConfig<br/>├── news: NewsConfig<br/>├── agents: AgentConfig<br/>├── admin: AdminConfig<br/>└── paper_trading: PaperTradingConfig"]
    end

    %% Connections
    MULTI_DATA & MULTI_NEWS & MULTI_AGENTS & MULTI_ADMIN & MULTI_PAPER --> APP_SCHEMA
    SINGLE_DATA & SINGLE_NEWS --> APP_SCHEMA

    APP_SCHEMA --> DATA_SCHEMA & NEWS_SCHEMA & AGENT_SCHEMA & ADMIN_SCHEMA & PAPER_SCHEMA
```

---

## 📁 Complete Directory Structure

```
MAS_Final_With_Agents/
│
├── 📁 trading_agents/                    # Core trading system
│   │
│   ├── 📁 agents/                        # Agent implementations
│   │   ├── __init__.py                   # Agent exports
│   │   ├── base.py                       # BaseAgent class
│   │   ├── analyst.py                    # AnalystAgent
│   │   ├── researcher.py                 # ResearcherAgent
│   │   ├── trader.py                     # TraderAgent
│   │   ├── risk.py                       # RiskManagerAgent
│   │   ├── evaluator.py                  # EvaluatorAgent
│   │   ├── admin.py                      # AdminAgent
│   │   └── compose.py                    # Agent factory
│   │
│   ├── 📁 inventory/                     # Pluggable strategy methods
│   │   ├── __init__.py
│   │   ├── registry.py                   # @register decorator
│   │   │
│   │   ├── 📁 analyst/                   # Analyst methods
│   │   │   ├── __init__.py
│   │   │   ├── talib_basic.py            # TALib_Basic
│   │   │   ├── stl_decompose.py          # STL_Decompose
│   │   │   ├── hmm_regime.py             # Gaussian_HMM
│   │   │   └── kalman_filter.py          # Kalman_Filter
│   │   │
│   │   ├── 📁 researcher/                # Researcher methods
│   │   │   ├── __init__.py
│   │   │   ├── arimax_forecast.py        # ARIMAX_Forecast
│   │   │   ├── tft_forecast.py           # TFT_Forecast
│   │   │   ├── bootstrap_uq.py           # Bootstrap_UQ
│   │   │   ├── quantile_uq.py            # Quantile_UQ
│   │   │   └── temp_calib.py             # Temperature_Scaling
│   │   │
│   │   ├── 📁 trader/                    # Trader methods
│   │   │   ├── __init__.py
│   │   │   ├── aggressive_market.py      # Aggressive_Market
│   │   │   ├── conservative_limit.py     # Conservative_Limit
│   │   │   └── neutral_scaled.py         # Neutral_Scaled
│   │   │
│   │   └── 📁 risk/                      # Risk methods
│   │       ├── __init__.py
│   │       ├── leverage_limit.py         # Leverage_Limit
│   │       ├── position_size.py          # Position_Size
│   │       ├── margin_safety.py          # Margin_Safety
│   │       ├── var_limit.py              # VaR_Limit
│   │       └── liquidation_safety.py     # Liquidation_Safety
│   │
│   ├── 📁 config/                        # Configuration management
│   │   ├── __init__.py
│   │   ├── loader.py                     # load_config()
│   │   └── schemas.py                    # Dataclass schemas
│   │
│   ├── 📁 optimization/                  # Continual learning
│   │   ├── __init__.py
│   │   ├── knowledge_transfer.py         # KnowledgeTransfer
│   │   └── inventory_pruner.py           # InventoryPruner
│   │
│   ├── 📁 services/                      # Services layer
│   │   ├── __init__.py
│   │   ├── llm.py                        # LLM service
│   │   ├── metrics.py                    # PerformanceTracker
│   │   ├── events.py                     # EventBus
│   │   ├── alerts.py                     # Alert rules
│   │   ├── notifications.py              # Notification channels
│   │   ├── reports.py                    # Report generation
│   │   ├── bybit_client.py               # Bybit API client
│   │   ├── order_manager.py              # Order management
│   │   └── positions.py                  # Position tracking
│   │
│   ├── 📁 models/                        # Data models
│   │   ├── __init__.py
│   │   └── types.py                      # Dataclasses
│   │
│   ├── workflow.py                       # WorkflowEngine
│   └── cli.py                            # CLI interface
│
├── 📁 data_pipeline/                     # Data fetching
│   │
│   ├── 📁 news/                          # News intelligence
│   │   ├── __init__.py                   # Module exports
│   │   ├── llm_prompt_search.py          # LLM query planning
│   │   ├── multi_asset_queries.py        # Asset-specific queries
│   │   ├── sources.py                    # Source credibility
│   │   ├── enrichment.py                 # LLM enrichment
│   │   ├── aggregation.py                # News clustering
│   │   │
│   │   └── 📁 providers/                 # Search providers
│   │       ├── __init__.py
│   │       ├── search_bocha.py           # Bocha API
│   │       └── search_serpapi.py         # SerpAPI (legacy)
│   │
│   └── 📁 pipeline/                      # Price data
│       ├── __init__.py
│       ├── data_pipeline.py              # Unified entry point
│       ├── multi_asset.py                # Multi-asset loader
│       ├── cross_features.py             # Cross-asset signals
│       └── schemas.py                    # Data schemas
│
├── 📁 configs/                           # YAML configurations
│   ├── multi_asset.yaml                  # 5-coin config
│   ├── README.md                         # Config docs
│   │
│   └── 📁 single/                        # Per-coin configs
│       ├── btc.yaml
│       ├── eth.yaml
│       ├── sol.yaml
│       ├── doge.yaml
│       └── xrp.yaml
│
├── 📁 data/                              # Market data
│   ├── 📁 bybit/                         # Bybit CSV files
│   │   ├── Bybit_BTC.csv
│   │   ├── Bybit_ETH.csv
│   │   ├── Bybit_SOL.csv
│   │   ├── Bybit_DOGE.csv
│   │   └── Bybit_XRP.csv
│   │
│   ├── 📁 multi_asset/                   # Multi-asset outputs
│   └── 📁 single/                        # Single-asset outputs
│
├── 📁 docs/                              # Documentation
│   └── ARCHITECTURE.md                   # This file
│
├── .env                                  # Environment variables
├── pyproject.toml                        # Dependencies
├── README.md                             # Project README
└── RUN_DEMO.sh                           # Demo script
```

---

## 🔐 Environment Variables

```bash
# .env file

# OpenAI API
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1  # Optional

# Bocha Search API
BOCHA_API_KEY=sk-...

# Bybit Testnet (Paper Trading)
BYBIT_TESTNET_KEY=...
BYBIT_TESTNET_SECRET=...

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # Optional
```

---

## 🚀 Execution Flow Summary

```
1. Load Config (multi_asset.yaml)
       ↓
2. Initialize Services (EventBus, Metrics, LLM)
       ↓
3. Load Data
   ├── Price: Bybit CSVs → align → cross-features
   └── News: Bocha → credibility → enrich → aggregate
       ↓
4. For each asset (BTC, ETH, SOL, DOGE, XRP):
   │
   ├── 4a. Analyst Agent
   │   └── TALib, STL, HMM, Kalman → features + trends
   │
   ├── 4b. Researcher Agent
   │   └── ARIMAX, TFT, Bootstrap, Quantile → ResearchSummary
   │
   ├── 4c. Trader Agent
   │   └── LLM(research + news + market_context) → ExecutionSummary
   │
   ├── 4d. Risk Manager
   │   └── Check limits → pass / soft_fail / hard_fail
   │
   └── 4e. Execute (if pass/soft_fail)
       └── OrderManager → Bybit API → PositionTracker
       ↓
5. Evaluator Agent
   └── Calculate Sharpe, PnL, HitRate, MaxDD, ECE
       ↓
6. Admin Agent
   ├── Check alert rules
   ├── Generate scheduled reports
   └── Send notifications
       ↓
7. Optimization (every N iterations)
   ├── Knowledge transfer between agents
   └── Prune underperforming methods
       ↓
8. Next iteration → Step 3
```
