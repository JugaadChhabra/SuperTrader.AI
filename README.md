Awesome—here’s a super-focused, 5-day MVP roadmap in **README tick-box format**, tailored to your strengths and the goal of being able to **start (paper) trading by Day 5**. Each day lists concrete deliverables, tests, and “Definition of Done” so nobody’s blocked.

---

# 🚀 SuperTrader.AI — 5-Day MVP Roadmap

> Goal by **Day 5 (EOD)**: runnable system with **event-driven backtest + live paper-trading** loop:
> Data Agent → Feature/State → RL Strategy Agent → Execution Agent → Risk → Logs/Metrics.

---

## Day 1 — Bootstrap, Contracts, and Wiring

**Muaaz (AI/Agentic, learning finance)**

* [ ] Create **LangGraph** workflow: `graph/workflow.py` (nodes: DataAgent → NewsAgent → RLAgent → ExecutionAgent → RiskGuard).
* [ ] Implement `agents/rl_strategy_agent.py` stubs: `init_rl_agent`, `build_state_representation`, `sample_action`, `set_risk_params`.
* [ ] Define RL config: `configs/rl.yaml` (algo=DQN baseline, γ, lr, buffer size, ε-schedule, target update).
* [ ] Draft `agents/news_agent.py` interface: `init_news_agent`, `fetch_news_blobs`, `score_sentiment`, `build_sentiment_features` (return zeros as placeholder).
* [ ] Finance learning: outline risk metrics to show in UI (exposure %, PnL, drawdown, turnover, Kelly cap).

**Jugaad (Finance lead, learning AI)**

* [ ] Curate **NSE/BSE universe** (top liquid F\&O 50): `configs/market.yaml` (symbols, lot size, tick size, session hours, fees/bp).
* [ ] Implement `data/loaders.py` minimal OHLCV loader (CSV or free API placeholder) with schema: `timestamp, open, high, low, close, volume`.
* [ ] Implement `agents/data_agent.py` stubs: `fetch_ohlcv`, `validate_data` (schema, missing bars), `compute_indicators` (SMA9/21, RSI14/30, MACD), `build_feature_frame`.
* [ ] Finance mapping doc: tick value ↔ bp costs ↔ notional for each symbol.

**Manikya (DSA/algorithms, learning AI & finance)**

* [ ] Implement `utils/metrics.py`: rolling volatility (EWMA), Sharpe, Sortino, MDD, Calmar, turnover.
* [ ] Implement **replay buffer** (ring buffer) for RL with O(1) push/sample, and unit tests.
* [ ] Implement `utils/logging.py` structured logs; add `Makefile` tasks: `make setup | make backtest | make train | make run`.
* [ ] Create `scripts/setup.py` to generate sample configs & sample CSVs.

**Definition of Done (Day 1)**

* [ ] `make setup` installs env, runs unit tests (buffers + metrics).
* [ ] One symbol loads from CSV, indicators compute, and state vector builds without errors.
* [ ] LangGraph graph builds (dry run) with mock outputs.

---

## Day 2 — Features, Sentiment, and RL Baseline

**Muaaz**

* [ ] Finalize **state representation**: 60-step window of {norm close, returns(1/5/15), vol(EWMA60), MACD(S,L), RSI30, volume surge}, + placeholder sentiment features.
* [ ] Implement `rl_strategy_agent.py` DQN network (2-layer LSTM 64/32 + LeakyReLU head → actions {-1,0,1}); target net + Double/Dueling.
* [ ] Implement `train_step(batch)` + epsilon-greedy `sample_action`.
* [ ] Draft `scripts/train.py` (offline on historical OHLCV; save checkpoints).
* [ ] Finance learning: document **volatility targeting** (σ\_target normalization) to be applied to rewards later.

**Jugaad**

* [ ] Implement `validate_data` with: missing bar repair, off-session filter, spike z-scores, forward-fill bounds.
* [ ] Implement `compute_indicators`: MACD (multi-tf), RSI30, ATR14, volume surge z-score; add `build_feature_frame` combining price + indicators.
* [ ] Populate `configs/market.yaml` fees, slippage (bp), trading hours (pre/post filters).
* [ ] Add `fetch_corporate_actions` stub (return empty today) & note splits/div for future.

**Manikya**

* [ ] Implement `scripts/backtest.py` (event-driven): iterate bars → build state → RL action → position update → apply costs → record PnL.
* [ ] Implement **transaction cost model** (bp × notional) and **position transition** cost doubling on sign flips.
* [ ] Unit tests: PnL accounting (additive), turnover, no-cost same-action, doubled cost on flips; edge cases.

**Definition of Done (Day 2)**

* [ ] `make train` runs on 1–2 symbols for 6–12 months data and produces a checkpoint.
* [ ] `make backtest` computes metrics and writes `artifacts/run.json` + `plots/`.
* [ ] Mocks in NewsAgent return deterministic sentiment vector plugged into state.

---

## Day 3 — Execution, Risk, and News Wiring

**Muaaz**

* [ ] Implement `news_agent.py` minimal pipeline: basic web fetch (or local text files) → clean → keyword → simple polarity (rule or tiny model) → ticker map (regex) → `build_sentiment_features` (rolling mean & momentum; temporal decay).
* [ ] Integrate sentiment features into state and ablation flag `use_sentiment`.
* [ ] Add `evaluate_policy(env, episodes)` to report Sharpe/Sortino/MDD on validation period; checkpoint best.

**Jugaad**

* [ ] Implement `agents/execution_agent.py` stubs tied to finance: `pre_trade_checks` (max exposure, per-symbol limit, correlation cap), `build_order(signal, price, size, tif)`, `apply_stops_and_targets`.
* [ ] Sizing: **volatility-scaled** target notional per position (σ\_target / σ\_est) with cap by VaR/lot sizes.
* [ ] Write **Risk Config** `configs/risk.yaml`: max gross/net exposure %, per-name cap, max leverage, Kelly clip (e.g., <= 0.5 Kelly).

**Manikya**

* [ ] Implement **RiskGuard** utility called each step: exposure checks, drawdown halt, circuit break on abnormal vol/spread.
* [ ] Implement **portfolio accounting** (cash, positions, realized/unrealized PnL) and day roll logic.
* [ ] Add `scripts/run.py` for **paper mode**: reads live-like feed (simulated stream) and routes through Workflow.

**Definition of Done (Day 3)**

* [ ] Backtest with **risk on** runs from CLI; orders sized with vol targeting; stops/targets apply.
* [ ] Sentiment features present in state (feature length & masks validated).
* [ ] RiskGuard can halt trading when MDD > threshold.

---

## Day 4 — End-to-End Backtest & Paper Engine

**Muaaz**

* [ ] Hyperparam pass (batch, γ, lr, ε-schedule, target τ) and **early stopping** by validation Sharpe.
* [ ] Add `save_checkpoint(path, step, metrics)` + model versioning to `/artifacts/`.
* [ ] Add **PPO or A2C** skeleton switch in `rl_strategy_agent.py` (keep DQN as default for MVP).

**Jugaad**

* [ ] Build **universe ranking**: `rank_universe(features, rules)` (liquidity, spread proxy, vol regime) + `select_top_k(k=10)`.
* [ ] Finalize **fees/slippage table** by symbol; sanity-check notional sizes vs. lot size & tick value.
* [ ] Create **trade ledger & attribution** in `post_trade_attribution(trades, benchmarks)`.

**Manikya**

* [ ] Optimize backtester (vectorized reward & cost application; pre-allocate arrays).
* [ ] Add **unit tests**: state shapes, NaN guards, universe selection deterministic given seed.
* [ ] Build **CLI UX**:

  * [ ] `make backtest SYMBOLS="RELIANCE,TCS,INFY" START=2024-01-01 END=2025-06-30`
  * [ ] `make paper START=2025-06-01`

**Definition of Done (Day 4)**

* [ ] Full backtest over top-10 names completes with metrics + plots (equity curve, drawdown, turnover).
* [ ] Validation **Sharpe ≥ 0.5** on daily bars (or your internal target) OR clear diagnostics logged.
* [ ] Paper engine streams decisions deterministically from historical feed.

---

## Day 5 — MVP Readiness & Live Paper Trading

**Muaaz**

* [ ] Wire **LangGraph**: data → features → sentiment → RL → risk → execution; add retries/timeouts.
* [ ] Add **model selection**: load best checkpoint; log config hash + git commit.
* [ ] Draft **Playbook.md**: start/stop, risk knobs, what to watch (PnL, DD, exposure, log levels).

**Jugaad**

* [ ] Final **risk sweep**: exposure caps, Kelly clip, per-name/sector caps, **kill switch** command in `ExecutionAgent`.
* [ ] Sanity trial with **σ\_target** variations (e.g., 10%, 15%) and bp sensitivities; pick conservative defaults.
* [ ] Prepare **go-live checklist** (data latency, clocks, session times, symbol holidays).

**Manikya**

* [ ] Build **health monitor**: heartbeat for each agent; on failure → safe flat; alert log.
* [ ] Add **order simulator**: partial fills, slippage on high turnover, queue priority toy model.
* [ ] Create `tests/` smoke suite and `make smoke` that runs in <1 min over 2 symbols for 1 month.

**Definition of Done (Day 5)**

* [ ] One-click run: `make paper` starts pipeline and prints orders + PnL in real-time (simulated).
* [ ] Backtest & paper logs include: Sharpe, Sortino, MDD, turnover, hit-rate, avg P/L, exposure, max position.
* [ ] **Risk kills** verified (trigger MDD breach → system flattens and halts).

---

## Cross-Day Stretch (if time permits)

* [ ] Add **volatility-scaled reward** in training loop (σ\_target normalization).
* [ ] Simple **sentiment momentum** (EMA, half-life decay) and ablation switch.
* [ ] **Correlation-aware** position limits (block diagonal cap by sector/β).
* [ ] Basic **dashboard** (Streamlit) to view Equity Curve, DD, Exposure, Open Positions.

---

## Success Criteria (MVP)

* [ ] Event-driven backtest completes on **Top-10** names with **Sharpe ≥ 0.5**, **MDD ≤ 15%**, **turnover ≤ 3×/month/name** (tunable).
* [ ] Paper engine runs for **≥ 2 trading sessions** without crash; risk caps respected; logs clean.
* [ ] Reproducible via `make` targets; configs fully drive behavior; checkpoints saved & loaded.

---

## Quick File Checklist (must exist by Day 3)

* [ ] `agents/data_agent.py`, `agents/news_agent.py`, `agents/rl_strategy_agent.py`, `agents/execution_agent.py`
* [ ] `configs/market.yaml`, `configs/rl.yaml`, `configs/risk.yaml`, `configs/agents.yaml`
* [ ] `data/features.py`, `data/loaders.py`
* [ ] `graph/state.py`, `graph/workflow.py`
* [ ] `models/environment.py`, `models/lstm_networks.py`, `models/rl_agents.py` (can be merged for MVP)
* [ ] `scripts/backtest.py`, `scripts/train.py`, `scripts/run.py`, `scripts/setup.py`
* [ ] `utils/config.py`, `utils/logging.py`, `utils/metrics.py`

---

### Notes on Division of Learning vs. Doing

* **Muaaz** drives **AI/Agentic** decisions (DQN baseline, wiring, sentiment) while noting finance constraints.
* **Jugaad** leads **Finance/Data/Risk** (universe, fees, exposure, Kelly caps) while implementing AI-facing feature engineering.
* **Manikya** owns **algorithms, performance, and correctness** (buffers, cost model, risk guard, tests, CLI).

Got it. Below is a **high-precision, no-code “construction spec”** for what each function should *do*, what to log, inputs/outputs, edge cases, and acceptance criteria—split by **person** and **day** so you can drop this straight into your README.

---

## Muaaz

### Day 1 — RL + Agentic Wiring (skeletons, contracts, logging)

**File:** `agents/rl_strategy_agent.py`

* **`init_rl_agent(config, action_space, obs_space)`**

  * **Purpose:** Instantiate DQN (default) with 2-layer LSTM(64,32) + LeakyReLU head; register target network; set ε-greedy scheduler; set optimizer & LR from config.
  * **Inputs:** `config` (dict-like), `action_space` (`{-1,0,1}`), `obs_space` (state vector length).
  * **Outputs:** agent handle with attributes (`net`, `target_net`, `replay`, `epsilon`, `step`, `risk_params`).
  * **Logging:** model hash from hyperparams; action space; obs shape; seed.
  * **Edge cases:** mismatched obs dims → raise ConfigError; missing keys → warn + defaults.
  * **Done when:** `agent.summary()` returns layer shapes; replay initializes.

* **`build_state_representation(price_feats, tech_feats, senti_feats)`**

  * **Purpose:** Concatenate normalized close, returns (1/5/15), EWMA vol(60), MACD(S/L), RSI(30), volume-surge z, and sentiment level + momentum into a fixed 60-step tensor.
  * **Inputs:** dicts of aligned frames; flags for `use_sentiment`.
  * **Outputs:** 2D array `[window_len, feature_dim]`, mask for NaNs.
  * **Logging:** feature names, counts, any clipped/ffilled values.
  * **Edge cases:** partial windows <60 → left-pad with zeros + mask.

**File:** `graph/workflow.py`

* Define nodes: `DataAgent → NewsAgent → RLAgent → RiskGuard → ExecutionAgent`.
* **Contracts between nodes** (keys & dtypes) and **retry/timeouts** (e.g., 200ms retry x3).
* **Done when:** dry-run passes with stubs, prints DAG order.

**File:** `configs/rl.yaml`

* DQN defaults: `gamma=0.3`, `lr=1e-4`, `batch=64`, `target_update=1000`, `replay_size=50k`, `epsilon: start=1.0,end=0.05,decay_steps=50k`.

---

### Day 2 — DQN training loop + rewards + evaluation

**File:** `agents/rl_strategy_agent.py`

* **`sample_action(state, mode)`**

  * **Purpose:** ε-greedy sampling; in `eval` use argmax Q.
  * **Inputs:** `state` tensor; `mode in {'train','eval'}`.
  * **Outputs:** action in `{-1,0,1}`.
  * **Logging:** ε value every 1k steps; action hist.

* **`add_experience(s, a, r, s_next, done)`**

  * **Purpose:** Push to replay with ring-buffer; avoid storing NaN states.
  * **Done when:** buffer length increments; eviction logged.

* **`train_step(batch)`**

  * **Purpose:** Compute TD target with Double+Dueling; Huber/MSE loss; backprop; update target net at τ.
  * **Inputs:** sampled batch; config.
  * **Outputs:** dict of losses (q\_loss, grad\_norm, mean\_q).
  * **Edge:** empty buffer → skip; exploding grads → clip and log.

* **`evaluate_policy(env, episodes)`**

  * **Purpose:** Run greedy policy on validation; return Sharpe/Sortino/MDD/turnover.
  * **Done when:** JSON metrics dumped under `artifacts/`.

**Reward function (to use in backtester)**

* Volatility-scaled additive PnL; apply bp costs; double cost on sign-flip; σ\_target normalization flag.
* **Document precisely** in README (var names, units).

---

### Day 3 — News/Sentiment v0 and feature fusion

**File:** `agents/news_agent.py`

* **`init_news_agent(config)`**

  * **Purpose:** Load lightweight pipeline config; set decay half-life; set ticker map.
  * **Done when:** returns handle with callable scorers.

* **`fetch_news_blobs(sources, start, end)`**

  * **Purpose:** Pull recent headlines/text (for MVP: local JSON/CSV or simple RSS fetcher already prepared by you later).
  * **Outputs:** list of `{ts, source, text}`.

* **`preprocess_text(blobs)`**

  * **Purpose:** lowercase, de-dup, strip tickers/URLs, keep numbers/units.
  * **Outputs:** clean texts + metadata.

* **`run_ner(texts)`**

  * **Purpose:** Rule-based ticker mapping (regex on company/ISIN/short names), output spans `{ticker, ts}`.
  * **Done when:** ≥90% mapping for known test strings.

* **`score_sentiment(texts)`**

  * **Purpose:** Return float in \[-1,1] per text (lexicon or mini-model); confidence score.
  * **Edge:** non-finance text → neutral 0 with low conf.

* **`aggregate_entity_sentiment(spans)`**

  * **Purpose:** Time-bucket per ticker (e.g., 15m), mean score weighted by recency; output momentum & level.
  * **Output contract:** `{ticker: {level, momentum}}`.

* **`build_sentiment_features(entities, ts)`**

  * **Purpose:** Align to price timestamps; forward-fill up to N bars; missing → zeros with mask.

**Fusion:** ensure `build_state_representation` accepts sentiment vector and mask.

---

### Day 4 — Risk hooks and model selection

**File:** `agents/rl_strategy_agent.py`

* **`set_risk_params(params)`**

  * **Purpose:** Store exposure caps, Kelly clip, max position size factor.

* **`save_checkpoint(path, step, metrics)`**

  * **Purpose:** Serialize weights + config hash + metrics; write atomic (temp → move).
  * **Edge:** disk errors → retry.

**File:** `graph/workflow.py`

* Add **guards**: if RiskGuard flags breach → force action `0` (flat) for K bars; log reason.

**Model selection doc:** define single best checkpoint rule (Sharpe primary, MDD secondary).

---

### Day 5 — Paper-trading runbook

**File:** `scripts/run.py` (you’re defining behavior, not code)

* **Modes:** `--paper` streams historical bars as “now”; fixed latency; prints orders + PnL.
* **Inputs:** configs; start/end; symbols.
* **Outputs:** rolling metrics to console + JSON log lines.

**Playbook.md**

* Start/stop steps, risk switches, how to roll back to last known good model, what to monitor each hour.

---

## Jugaad

### Day 1 — Market config + Data Agent skeletons

**File:** `configs/market.yaml`

* **Contents:** NSE F\&O top-liquid universe (symbol, lot size, tick size), trading hours (pre-open, regular), holidays ref, fee/slippage (bp), spread proxy rules.

**File:** `agents/data_agent.py`

* **`init_data_agent(config)`**

  * **Purpose:** Set data roots/cache; validate session times; warm up symbol metadata.
  * **Output:** object with live/historical modes.

* **`fetch_ohlcv(symbols, interval, start, end)`**

  * **Purpose:** Use **your ICICI fetch** (already implemented) under the hood; for MVP accept CSV fallback.
  * **Edge:** throttle; missing bars; partial sessions.

* **`validate_data(df)`**

  * **Purpose:** schema check; timestamp monotonicity; session filter; spike z-scores; duplicate removal.
  * **Outputs:** cleaned df + report counters.

* **Done when:** single symbol loads and validates from your source for a test day.

---

### Day 2 — Indicators & Feature Frame

**File:** `data/features.py`

* **`compute_indicators(df)`**

  * **Purpose:** SMA(9,21), RSI(14,30), MACD(S,L) with standard params; ATR14; volume-surge z.
  * **Outputs:** df with aligned columns; no look-ahead.

* **`build_feature_frame(df_prices, df_indicators)`**

  * **Purpose:** align, drop NaNs at head, normalize (returns, vol); add `session_mask`.
  * **Outputs:** frame ready for `build_state_representation`.

**File:** `agents/data_agent.py`

* **`build_feature_frame(...)`** delegate to `data/features.py`; keep API stable for RL agent.

**Docs:** mapping **bp costs** to notional for each symbol; show example notional calc; confirm fee floor.

---

### Day 3 — Risk & Sizing in Execution

**File:** `agents/execution_agent.py`

* **`init_execution_agent(config)`**

  * **Purpose:** Load fee/slip table, limits, broker route configs (placeholders).
* **`pre_trade_checks(order, portfolio, limits)`**

  * **Purpose:** guard rails: max gross/net exposure %, per-name cap, sector cap, leverage, MDD halt.
  * **Outputs:** pass/fail + reasons.
* **`compute_position_size(state, raw_action, risk_params)`** (if called from RL agent or here)

  * **Purpose:** **volatility-scaled** target size: scale by `σ_target/σ_est`; apply Kelly clip; min lot rounding.
* **`build_order(signal, price, size, tif)`**

  * **Purpose:** Convert target position to buy/sell quantity; include tif, intent, reason codes.
* **`apply_stops_and_targets(position, rules)`**

  * **Purpose:** ATR-based SL/TP; trail option; write stop distance in ticks.
* **Done when:** backtest uses these to size & block orders correctly.

---

### Day 4 — Universe & Costs

**File:** `agents/data_agent.py`

* **`rank_universe(features, rules)`**

  * **Purpose:** rank by liquidity (volume proxy), spread proxy, stable data availability, vol regime; tie-break with turnover penalty.
* **`select_top_k(ranks, k, constraints)`**

  * **Purpose:** choose top-k with sector/β constraints; return symbol list + reasons.

**Costs Table:** finalize per-symbol bp & expected spread; doc any conservative cushions.
**Attribution:** define fields for trade ledger (entry/exit ts, fees, realized/unrealized PnL, slippage, reason codes).

---

### Day 5 — Go-live guardrails

**Checklist:**

* Time sync; session windows; holiday exclusions; sanity pre-open flatness; data latency threshold.
* Kill-switch procedure (config flag + manual command).
* **Parameter freeze** (yaml hashes) for reproducibility.
* Paper-trade dry run across open/close transitions; confirm no orders outside session.

---

## Manikya

### Day 1 — Metrics, Buffers, Make targets

**File:** `utils/metrics.py`

* **`rolling_vol_ewma(ret, span)`**, **`sharpe(ann_factor)`**, **`sortino(ann_factor)`**, **`max_drawdown()`**, **`calmar()`**, **`turnover(positions)`**

  * **Purpose:** return scalars; handle NaNs; documented units.
  * **Done when:** unit tests pass on toy arrays.

**File:** `utils/replay_buffer.py`

* **`Replay(capacity, seed)`** with O(1) push/sample; returns batches with aligned dtypes; supports `n_step` later.
* **Edge:** under-filled buffer → sample with replacement and warn.

**Makefile**

* `make setup`, `make train`, `make backtest`, `make paper`, `make smoke`.
* Smoke runs 1 month, 2 symbols, prints metrics.

---

### Day 2 — Event-driven Backtester

**File:** `scripts/backtest.py`

* **Loop contract:** for each bar → state → action → position transition → apply costs → update PnL → risk guard → log.
* **Costs:** bp × notional; sign flip doubles cost; slippage hook.
* **Outputs:** equity curve, drawdown curve, per-trade ledger JSON + CSV.
* **Acceptance:** exact PnL match for known toy case; turnover math verified.

---

### Day 3 — RiskGuard & Portfolio Accounting

**File:** `utils/risk.py`

* **`risk_guard(step_snapshot, limits)`**

  * **Purpose:** evaluate MDD, max exposure, per-name cap, volatility spike halt; return `OK/VIOLATION` + reason.
* **`portfolio_accounting`**

  * **Purpose:** track cash, positions, realized/unrealized PnL; corporate-action-safe (placeholder).
* **Edge:** NaN price → hold position, mark as stale, skip PnL.

**File:** `scripts/run.py`

* **Paper mode engine:** stream bars from historical source with wall-clock pacing; on error → safe-flat and continue.

---

### Day 4 — Performance & CLI UX

* **Vectorization:** pre-allocate arrays for PnL, positions, costs; avoid Python loops where not needed.
* **CLI flags:** `--symbols`, `--start`, `--end`, `--use_sentiment`, `--sigma_target`.
* **Tests:** deterministic universe selection with seed; NaN guard tests; risk halt tests.

---

### Day 5 — Health, Fills, and Smoke Suite

**File:** `utils/health.py`

* **Heartbeat:** each agent must ping; missed K beats → system flat + alert line.
  **File:** `utils/fills.py`
* **Order simulator:** partial fills; slip function of turnover percentile; queue depth toy model.
  **Tests:** `make smoke` runs <1 min; verifies: non-decreasing equity timestamps, no orders outside session, halt on MDD breach.

---

### Notes / Shared Assumptions

* **Data source:** you already have ICICI OHLCV; use *that* for `fetch_ohlcv`; CSV fallback only for local tests.
* **Actions:** start with discrete `{-1,0,1}`; map to target positions; cost model doubles on sign flips.
* **Vol targeting:** use config flag to enable reward normalization and position scaling.
* **Reproducibility:** every run logs `git_commit`, `config_hash`, `seed`, and model checkpoint path.

If you want, I can convert this into a **README.md** sectioned by person/day so the team can check boxes directly.
