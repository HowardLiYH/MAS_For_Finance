# Multi-agent LLM Financial Trading Model on BTC Perpetual

## ⚙️ Basic Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m mas_finance.cli --symbol BTCUSD.PERP --interval 4h
```


## 📅 Meeting Notes
- Sep 18：
   -  Draw a UML diagram first
      - Tools
            - Is there a Base class
            - What are the additional features for each?
      - Pools
            - How is it related to each agent
            - What kind of tools does it include
      - Agents
            - Core Properties
   -  Create a raw functionable complete workflow
       - Full run of five agents
       - Related Repositories
            - [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN)
           - [TradingAgents](https://github.com/TauricResearch/TradingAgents)



## 📔 Development Log for Phase 1
- Done ✅：
  - Get Price 📈 and News 📰
      - Retrieve BTC Price Data from the crypto exchange of choice on 4h intervals for 120 days (Parameters are subjectable to changes)
        - `data/btc_4h.csv`
      - Retrieve both micro and macro news data by combining prompt engineering with OpenAI’s LLM and the SerpAPI search engine.
        - `data/news_micro.json` (Size: 10)
        - `data/news_macro.json` (Size: 10)

NOTE: The parameters for getting Price data and prompts for News data are all at a very basic functional level and need to be finetuned/optimized later.


- To Do ❕
  - Analyst Agent
      - Evaluate on Price Data
  - Researcher Agent
      - Evaluate on Price Data
  - Trader
      - Evaluate on both the Price Data AND News Data
  - Risk Manager
      - Evaluate on Trader's performance
  - Evaluator
      - TBD Maybe push to Phase 2


----

# 🤖 Summary of the Multi-agent LLM Financial Trading Model on BTC Perpetual with SJTU

## Key Concepts
- Multi-agent Systems
- Large Language Models
- Continual Learning
- Uncertainty Quantification
- Risk-aware Decision Making
- Explainable AI
- Cross-modal Data Fusion
- Agent-based Orchestration
- Feedback-driven Optimization
- Structured Knowledge Representation

---

## High-level Summary
Our model incorporates five types of agents to replicate the workflow pipeline of a hedge fund:
**Analysts, Researchers, Traders, Risk Managers, and Evaluators.**

- The system ingests two distinct streams of information:
  1. **Text-based news data** (collected via LLM prompting with strict date control to prevent leakage)
  2. **Time-series BTC price data** (4-hour intervals)

- Agents perform trading activities and continually improve through feedback.
- Each agent type has specialized inventories (methods/prompts) to complete tasks.
- After each trading round (when an order is closed), the **Evaluator Agent** measures performance against fixed metrics.
- After every K iterations, top performers in each category transfer knowledge to peers.
- Price data and news data are deliberately treated separately:
  - **Analyst & Researcher Agents** → preprocess & analyze time-series inputs.
  - **Trader Agents** → interpret news narratives + market micro-structure.

---

## Overall Workflow
<img width="1369" height="1132" alt="image" src="https://github.com/user-attachments/assets/de2d3d97-c266-4992-9a80-182946cb3611" />
<img width="3840" height="2545" alt="1 3 Single‑iteration sequence (with pass _ Mermaid Chart-2025-10-03-053857" src="https://github.com/user-attachments/assets/4bf2edcc-78cf-406f-9563-306a42766e91" />

Part (1): **Agents Simulating a Hedge Fund Pipeline**
- Each step draws methods from the inventory.

---

## Inventory Reference
<img width="867" height="675" alt="image" src="https://github.com/user-attachments/assets/6bd4a047-889c-4e69-8077-13ef4c70321f" />


- **(F):** Agents built in first stage development
- **(S):** Agents built in second stage development
- All agents call **one shared LLM**

---

# Agent Descriptions

## Analyst Agent (F)

**Description:**
Processes time-series price data to output:
- Constructed Features (DataFrame)
- Constructed Trend Information (DataFrame)

**Processing Steps:**
- A-A Data Alignment
- A-B Feature Construction (ᴹ)
- A-C Trend Detection (ᴹ)
<img width="1325" height="532" alt="image" src="https://github.com/user-attachments/assets/97fee38f-4728-4bcb-846e-3cbb30c97722" />

---

## Researcher Agent (F)
**Description:**
Consumes Analyst outputs (features + trends) and produces:
- JSON research summary with trading recommendations

**Processing Steps:**
- R-A Forecasting (ᴹ)
- R-B Uncertainty & Risk Quantification (ᴹ)
- R-C Probability Calibration (ᴹ)
- R-D Signal Packaging

<img width="1308" height="568" alt="image" src="https://github.com/user-attachments/assets/39cf353e-e7f6-4db4-b215-61900b6c3ef3" />

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
- Consumes Researcher outputs + fresh News Data
- Selects trading style from inventory (optimized over time)
- Executes orders considering both market conditions and LLM signals

**Notes:**
- Focused on a single instrument/market/product (BTC perpetuals)
- Style convergence expected but must avoid bias from extreme conditions
- Traders scored differently (longer iteration cycles recommended)

**Processing Steps:**
- T-A Obtain Execution Style (ᴹ)
- T-B Execute Order

**Output (JSON):**
- Order ID
- Current Price
- Limit/Market Order
- Position Size
- Direction (Long/Short)
- Take Profit / Stop Loss Prices
- Closed Price (N/A if open)
- Leverage Size
- Liquidation Price
<img width="1117" height="356" alt="image" src="https://github.com/user-attachments/assets/da748f5e-b978-455a-b594-ff9c029b1033" />
<img width="500" height="800" alt="1 5 Trader order state machine _ Mermaid Chart-2025-10-03-054208" src="https://github.com/user-attachments/assets/7cc08d4c-ad3c-4372-ae61-67e5ae458669" />

---

## Risk Manager Agent (S)

**Description:**
Ensures Trader execution is safe:
- **hard_fail** → abort order
- **soft_fail** → regenerate order (back to T-B or T-A)
- **pass** → order executed & logged

**Processing Steps:**
- M-A Risk Analysis (ᴹ)
- M-B Output Log

**Outputs:**
- JSON risk analysis logs
- Pass / Soft Fail / Hard Fail decisions
<img width="1162" height="598" alt="image" src="https://github.com/user-attachments/assets/8d72726f-64c2-46d3-8cdf-1215c80bb256" />

**Decision Logic:**
- **Pass:** Order within VaR / size / margin limits
- **Soft Fail:** Violates minor rule (e.g., too high leverage) but can be adjusted
- **Hard Fail:** Breaches critical rule (e.g., margin call imminent) → discarded
<img width="1420" height="651" alt="image" src="https://github.com/user-attachments/assets/4ab44c8b-060d-439c-9c96-e6093f1862d4" />
<img width="700" height="800" alt="1 4 Risk analysis activity _ Mermaid Chart-2025-10-03-054441" src="https://github.com/user-attachments/assets/6c74d7d2-800d-400c-a863-114b7c459f31" />

---

## Continual Learning & Optimization

### Part (2) Continual Learning from the Best
- After K rounds, top agents in each category **teach peers**.
- Trader Agents may follow a distinct iteration cycle.

### Part (3) Inventory Pruning
- Rank methods by frequency of use.
- Remove less-used methods over time (careful to preserve scenario-specific methods).
<img width="700" height="800" alt="1 6 Knowledge transfer   pruning _ Mermaid Chart-2025-10-03-054658" src="https://github.com/user-attachments/assets/b860b4f3-4cb3-4792-b8e4-e961d59abf81" />

### Part (n) Future Extensions
- Add Admin agent to generate reports, monitor performance, and deliver evaluations.

---
