# Market Regime Interpreter

> **AI-driven trading system that detects market regimes and executes adaptive strategies**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Hidden%20Markov%20Models-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

**Author:** Nyan Linn Htun (Nathan) | December 2025

---

## 💡 Overview

Financial markets shift between distinct behavioral regimes—growth periods, corrections, and crises. This project uses **Hidden Markov Models** to automatically detect these regimes from S&P 500 data and execute regime-adaptive trading strategies:

- 🟢 **Growth** → Buy (Low volatility, positive returns)
- 🟡 **Correction** → Hold (Moderate volatility, negative returns)
- 🔴 **Crisis** → Sell (High volatility, high sector correlation)

**Key Results:** Validated on out-of-sample 2018 data with walk-forward testing, demonstrating risk-adaptive portfolio management.

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib

# Run the three-module pipeline (~2 minutes)
python information_generation_system_module_1.py  # Feature engineering
python information_generation_system_module_2.py  # Regime detection
python action_execution_system.py                 # Backtesting
```

**Output:** Regime classifications, trading performance metrics, and visualization. See [QUICK_START.md](QUICK_START.md) for details.

---

## 🎯 Approach

**The Challenge:** Traditional buy-and-hold strategies fail during market volatility. Manual timing is unreliable.

**The Solution:** Unsupervised learning to discover hidden market regimes:
1. **Feature Engineering** - Extract returns, volatility, and sector correlation
2. **Regime Detection** - HMM discovers 3 behavioral states
3. **Adaptive Trading** - Execute regime-specific strategies

**Validation:**
- Walk-forward testing (train: 2012-2017, test: 2018)
- Reproducible with saved models (no look-ahead bias)
- Risk-managed portfolio performance

---

## � How It Works

### The Three-Module Pipeline

```
📊 MARKET DATA                 →  🤖 AI DETECTION       →  💰 TRADING STRATEGY
───────────────────────────       ────────────────────    ──────────────────
S&P 500 + 5 Sector ETFs          Hidden Markov Model      IF Growth  → BUY
(XLK, XLF, XLV, XLE, XLI)        discovers 3 regimes      IF Crisis  → SELL
                                 from features            IF Correction → HOLD
```

**Module 1: Feature Engineering**  
Transform raw prices into 3 key features:
- **Index Returns**: Daily log returns of S&P 500 (directional signal)
- **Index Volatility**: 21-day rolling standard deviation (risk measure)  
- **Systemic Health Score**: 60-day sector correlation (coupling/fragility indicator)

**Module 2: Regime Detection**  
Gaussian HMM discovers 3 hidden states, automatically labeled:
```python
Low volatility + Positive returns + Low correlation  → Growth
Negative returns                                     → Correction  
High volatility + High correlation                   → Crisis
```

**Module 3: Trading Execution**  
Simple regime-based rules with percentage allocations.

📖 **Technical details:** [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)

---

## 📊 Key Features

| Feature | Implementation |
|---------|----------------|
| **Validation** | Walk-forward testing (2012-2017 → 2018) |
| **Learning** | Unsupervised (no manual labeling) |
| **Features** | Domain-driven (volatility + sector coupling) |
| **Interpretability** | Explainable regime rules |
| **Reproducibility** | Saved models (.pkl files) |

**Innovation:** Systemic Health Score—sector correlation as a market stress indicator.

---

## 🎓 Key Learnings

1. **Walk-forward validation is critical** - K-fold creates look-ahead bias in time series
2. **Feature engineering > model complexity** - Smart features outperform complex models on raw data
3. **Domain knowledge enables unsupervised learning** - HMM finds patterns; humans make them actionable
4. **Simplicity scales** - 3 regimes, 3 features, 3 rules

---

## 🚀 Future Enhancements

**Immediate:**
- Add transaction costs and slippage
- Risk-adjusted metrics (Sharpe, max drawdown)
- Regime confidence thresholds

**Advanced:**
- Online learning (periodic retraining)
- Multi-asset portfolio regimes
- Real-time deployment pipeline

---

## 📁 Project Files

```
information_generation_system_module_1.py    # Feature engineering (S&P 500 + sectors)
information_generation_system_module_2.py    # HMM training & regime detection
action_execution_system.py                   # Trading strategy & backtesting

features_complete_2012_2018.csv             # 3 engineered features (2012-2018)
regimes_train_2012_2017.csv                 # Training regimes with labels
regimes_test_2018.csv                       # Out-of-sample predictions (2018)

hmm_model.pkl                               # Trained Gaussian HMM (3 states)
scaler.pkl                                  # StandardScaler for features
regime_labels.pkl                           # State ID → Regime name mapping
regime_visualization_train.png              # Cumulative returns by regime
```

---

## 🛠️ Tech Stack

**Core:**
- `yfinance` - Market data
- `hmmlearn` - Hidden Markov Models  
- `scikit-learn` - Preprocessing
- `pandas` / `numpy` - Data manipulation
- `matplotlib` - Visualization

**Why HMM?**
- Markets have hidden states (regimes)
- State persistence (regimes last weeks, not minutes)
- Probabilistic transitions
- No labeled data required

---

## 🎯 Technical Skills

**Machine Learning:** Unsupervised learning · Time series · Feature engineering · Walk-forward validation  
**Quantitative Finance:** Regime detection · Backtesting · Risk management · Sector analysis  
**Software Engineering:** Modular design · Model persistence · Pipeline architecture

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICK_START.md](QUICK_START.md)** | Setup and execution guide |
| **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)** | HMM configuration and algorithms |

---

## 📊 Disclaimer

This is a research/educational project, not investment advice. Results are historical backtests and do not guarantee future performance.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Nyan Linn Htun (Nathan)** | AI & Machine Learning Final Project | December 2025
