# System Architecture Diagram
## Market Regime Interpreter - Dual System Interaction

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          MARKET DATA ENVIRONMENT                                     │
│                 (S&P 500, Sector ETFs, Volatility Indices)                          │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ Raw Market Data
                               ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                   INFORMATION GENERATION SYSTEM (IGS)                                 ║
║                        [Unsupervised Learning Pipeline]                              ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║  ┌────────────────────────────────────────────────────────────────────────────┐     ║
║  │                        MODULE 1: Feature Engineering                        │     ║
║  │                     [Time Series Analysis & Indicators]                     │     ║
║  ├────────────────────────────────────────────────────────────────────────────┤     ║
║  │                                                                             │     ║
║  │  Input:  OHLCV Data (S&P 500 + 5 Sector ETFs)                             │     ║
║  │                                                                             │     ║
║  │  Processing:                                                                │     ║
║  │  • Calculate Log Returns                                                    │     ║
║  │  • Rolling 21-Day Volatility (σ)                                           │     ║
║  │  • Technical Indicators (RSI, Bollinger Bands, MACD)                       │     ║
║  │  • Cross-Asset Correlation Matrix (5×5 sectors)                            │     ║
║  │  • Systemic Health Score (Sₜ) = Avg. correlation across sectors           │     ║
║  │                                                                             │     ║
║  │  Output:  features_complete_2012_2018.csv                                  │     ║
║  │           [12 engineered features × ~1,500 trading days]                   │     ║
║  └─────────────────────────────┬───────────────────────────────────────────────┘     ║
║                                │ Feature Matrix                                      ║
║                                ▼                                                     ║
║  ┌────────────────────────────────────────────────────────────────────────────┐     ║
║  │                    MODULE 2: Regime Detection & Classification              │     ║
║  │                        [Hidden Markov Model Training]                       │     ║
║  ├────────────────────────────────────────────────────────────────────────────┤     ║
║  │                                                                             │     ║
║  │  Input:  Standardized Feature Matrix (from Module 1)                       │     ║
║  │                                                                             │     ║
║  │  Training Phase (2012-2017):                                                │     ║
║  │  ┌──────────────────────────────────────────────────────────────┐          │     ║
║  │  │  Hidden Markov Model (3 States)                              │          │     ║
║  │  │  • State Transition Matrix (A): 3×3                          │          │     ║
║  │  │  • Emission Distributions: Multivariate Gaussian (μ, Σ)     │          │     ║
║  │  │  • Initial State Distribution (π)                            │          │     ║
║  │  │                                                               │          │     ║
║  │  │  Training Algorithm:                                          │          │     ║
║  │  │  1. K-Means Initialization                                   │          │     ║
║  │  │  2. Baum-Welch EM (100 iterations)                           │          │     ║
║  │  │  3. Convergence Check (Δ Log-Likelihood < 0.01)             │          │     ║
║  │  └──────────────────────────────────────────────────────────────┘          │     ║
║  │                                                                             │     ║
║  │  Inference Phase (Viterbi Algorithm):                                       │     ║
║  │  • Decodes most probable state sequence                                    │     ║
║  │  • Assigns regime label Rₜ ∈ {0, 1, 2} for each day                       │     ║
║  │                                                                             │     ║
║  │  Automatic Regime Labeling Logic:                                          │     ║
║  │  ┌──────────────────────────────────────────────────────────────┐          │     ║
║  │  │  State → Regime Mapping (Unsupervised → Interpretable)       │          │     ║
║  │  │  • Analyze mean features per state:                          │          │     ║
║  │  │    - High Return, Low Vol, High Health → "Growth"            │          │     ║
║  │  │    - Moderate Return, Med Vol → "Correction"                 │          │     ║
║  │  │    - Negative Return, High Vol, Low Health → "Crisis"        │          │     ║
║  │  └──────────────────────────────────────────────────────────────┘          │     ║
║  │                                                                             │     ║
║  │  Outputs:                                                                   │     ║
║  │  • regimes_train_2012_2017.csv  (Training set with Rₜ labels)             │     ║
║  │  • regimes_test_2018.csv        (Test set with Rₜ labels)                 │     ║
║  │  • hmm_model.pkl                (Trained HMM parameters)                    │     ║
║  │  • scaler.pkl                   (Feature standardization)                  │     ║
║  │  • regime_labels.pkl            (State → Regime mapping)                   │     ║
║  └─────────────────────────────┬───────────────────────────────────────────────┘     ║
║                                │                                                     ║
╚════════════════════════════════│═════════════════════════════════════════════════════╝
                                 │
                                 │ Daily Regime Classifications
                                 │ Rₜ ∈ {Growth, Correction, Crisis}
                                 │ + Systemic Health Score (Sₜ)
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                     ACTION EXECUTION SYSTEM (AES)                                     ║
║                     [Model-Based Reflex Agent]                                        ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║  ┌────────────────────────────────────────────────────────────────────────────┐     ║
║  │                   MODULE 3: Trading Strategy Execution                      │     ║
║  │                      [Rule-Based Decision Framework]                        │     ║
║  ├────────────────────────────────────────────────────────────────────────────┤     ║
║  │                                                                             │     ║
║  │  Input:  Daily Regime Labels (Rₜ) + Market Prices                          │     ║
║  │                                                                             │     ║
║  │  ┌────────────────────────────────────────────────────────────┐            │     ║
║  │  │         REGIME-TO-ACTION MAPPING (If-Then Rules)            │            │     ║
║  │  ├────────────────────────────────────────────────────────────┤            │     ║
║  │  │                                                             │            │     ║
║  │  │  IF Rₜ = "Growth" (Bullish):                               │            │     ║
║  │  │    THEN Action = BUY                                        │            │     ║
║  │  │         Rationale: Low volatility, positive momentum        │            │     ║
║  │  │                    → Accumulate equity exposure             │            │     ║
║  │  │                                                             │            │     ║
║  │  │  IF Rₜ = "Correction" (Transitional):                      │            │     ║
║  │  │    THEN Action = HOLD                                       │            │     ║
║  │  │         Rationale: Mixed signals, regime uncertainty        │            │     ║
║  │  │                    → Maintain current positions             │            │     ║
║  │  │                                                             │            │     ║
║  │  │  IF Rₜ = "Crisis" (Bearish):                               │            │     ║
║  │  │    THEN Action = SELL                                       │            │     ║
║  │  │         Rationale: High volatility, negative returns        │            │     ║
║  │  │                    → Reduce risk exposure                   │            │     ║
║  │  │                                                             │            │     ║
║  │  └────────────────────────────────────────────────────────────┘            │     ║
║  │                                                                             │     ║
║  │  Execution Logic:                                                           │     ║
║  │  • Iterate through each trading day in test period (2018)                  │     ║
║  │  • Check current regime Rₜ                                                 │     ║
║  │  • Execute corresponding action based on rulebook                          │     ║
║  │  • Update portfolio state (cash, shares, value)                            │     ║
║  │  • Track performance metrics                                               │     ║
║  │                                                                             │     ║
║  │  ┌────────────────────────────────────────────────────────────┐            │     ║
║  │  │              PERFORMANCE TRACKING                           │            │     ║
║  │  ├────────────────────────────────────────────────────────────┤            │     ║
║  │  │  Metrics Computed:                                          │            │     ║
║  │  │  • Total Return (%)                                         │            │     ║
║  │  │  • Number of Trades                                         │            │     ║
║  │  │  • Final Portfolio Value                                    │            │     ║
║  │  │  • Profit/Loss ($)                                          │            │     ║
║  │  │                                                             │            │     ║
║  │  │  Benchmark Comparison:                                      │            │     ║
║  │  │  • Buy-and-Hold Strategy (passive baseline)                │            │     ║
║  │  │  • Outperformance = Regime Strategy - Buy-and-Hold         │            │     ║
║  │  └────────────────────────────────────────────────────────────┘            │     ║
║  │                                                                             │     ║
║  │  Outputs:                                                                   │     ║
║  │  • Backtest Results (printed to console)                                   │     ║
║  │  • Trade Log (dates, actions, prices)                                      │     ║
║  │  • Performance Metrics vs Benchmark                                        │     ║
║  └─────────────────────────────┬───────────────────────────────────────────────┘     ║
║                                │                                                     ║
╚════════════════════════════════│═════════════════════════════════════════════════════╝
                                 │
                                 ▼
                    ┌────────────────────────────────┐
                    │    PORTFOLIO PERFORMANCE       │
                    │   (Returns, Risk, Trades)      │
                    └────────────────────────────────┘
                                 │
                                 │ Feedback Loop (Optional)
                                 ▼
                    ┌────────────────────────────────┐
                    │   CONTINUOUS LEARNING          │
                    │  (Quarterly Retraining)        │
                    │                                │
                    │  IF Performance Degrades:      │
                    │  → Retrain HMM on new data     │
                    │  → Update regime definitions   │
                    │  → Recalibrate action rules    │
                    └────────────────────────────────┘
```

---

## System Interaction Flow

### 1. **Data Flow Pipeline**
```
Market Data → Feature Engineering → HMM Training → Regime Labels → Trading Actions → Performance
```

### 2. **Key Information Transfers**

| From System | To System | Information Type | Format |
|-------------|-----------|------------------|--------|
| Market Environment | IGS Module 1 | Raw price/volume data | CSV from Yahoo Finance |
| IGS Module 1 | IGS Module 2 | Engineered features (12 columns) | `features_complete_2012_2018.csv` |
| IGS Module 2 | AES Module 3 | Regime labels + health score | `regimes_test_2018.csv` |
| IGS Module 2 | AES Module 3 | Trained model artifacts | `.pkl` files (HMM, scaler, labels) |
| AES Module 3 | User/Analyst | Performance metrics | Console output |

### 3. **Temporal Separation (Walk-Forward Validation)**

```
Timeline:
├─────────────────────────────────────────┬──────────────────┐
│          TRAINING PERIOD                │   TEST PERIOD    │
│         (2012-01 to 2017-12)            │  (2018-01 to     │
│                                         │   2018-12)       │
│  IGS learns regime patterns             │  AES executes    │
│  (Unsupervised HMM training)            │  trading rules   │
│  NO future data leakage                 │  Out-of-sample   │
└─────────────────────────────────────────┴──────────────────┘
```

### 4. **Decision-Making Architecture**

```
┌──────────────────────┐
│   Perception Layer   │  ← IGS generates "world model" (regimes)
│    (IGS Output)      │
└──────────┬───────────┘
           │ Regime State
           ▼
┌──────────────────────┐
│   Decision Layer     │  ← AES maps states to actions
│   (Rulebook Logic)   │
└──────────┬───────────┘
           │ Action
           ▼
┌──────────────────────┐
│  Execution Layer     │  ← Portfolio updates
│  (Trade Management)  │
└──────────────────────┘
```

---

## System Properties

### IGS (Information Generation System)
- **Type:** Unsupervised Learning (HMM)
- **Input:** Multivariate time series (12 features)
- **Output:** Discrete state labels (3 regimes)
- **Training:** Offline, on historical data
- **Adaptability:** Can retrain quarterly on new data

### AES (Action Execution System)
- **Type:** Model-Based Reflex Agent
- **Input:** Regime labels from IGS
- **Output:** Trading decisions (BUY/SELL/HOLD)
- **Execution:** Online, day-by-day
- **Logic:** Deterministic rule-based mapping

### Interaction Pattern
- **One-Way Data Flow:** IGS → AES (no feedback loop in basic version)
- **Asynchronous Updates:** IGS retrains periodically, AES operates daily
- **Modular Design:** Each system can be improved independently
- **Clean Separation:** Perception (IGS) vs Action (AES)

---

**Author:** Nyan Linn Htun (Nathan)  
**Date:** December 2025  
**Project:** AI-Driven Market Regime Trading System
