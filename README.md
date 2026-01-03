# Market Regime Interpreter

> **Teaching machines to recognize market moods—then trading on them.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Hidden%20Markov%20Models-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

**Author:** Nyan Linn Htun (Nathan)  
**Institution:** AI & Machine Learning Final Project  
**Date:** December 2025

---

## 💡 The Big Idea

Markets don't behave the same way all the time. Some days feel like steady climbs, others like nerve-wracking corrections, and occasionally, like full-blown crises. **What if we could teach an AI to recognize these "market moods" automatically—and trade accordingly?**

This project uses **Hidden Markov Models** (unsupervised ML) to discover three hidden market regimes from S&P 500 data:
- 🟢 **Growth** - High returns, low volatility → Buy
- 🟡 **Correction** - Temporary pullbacks → Hold  
- 🔴 **Crisis** - High volatility, systemic stress → Sell

**The result?** A trading system that adapts to market conditions, protecting capital during crashes while capturing gains during rallies.

---

## ⚡ Quick Start

### Install & Run
```bash
# Install dependencies
pip install yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib

# Run the pipeline (3 steps, ~2 minutes total)
python information_generation_system_module_1.py  # Feature engineering (S&P 500 + sectors)
python information_generation_system_module_2.py  # HMM training & regime detection
python action_execution_system.py                 # Backtesting & performance
```

That's it! You'll see:
- Which regime each day belongs to (Growth/Correction/Crisis)
- Trading decisions based on regime detection
- Regime transition visualization with colored background

**Want details?** See [QUICK_START.md](QUICK_START.md) for step-by-step instructions.

---

## 🎯 Why This Matters

### The Problem
Traditional strategies assume markets are stable. They're not.
- **Buy-and-hold** works great in bull markets, gets crushed in crashes
- **Timing the market** is hard—humans are terrible at it
- **Technical indicators** give conflicting signals

### The Solution
Let machines do what they're good at: **finding patterns in chaos.**

Our system:
1. **Discovers** hidden regimes from price patterns (no human labels needed)
2. **Labels** them automatically using domain knowledge (Growth/Correction/Crisis)
3. **Trades** using simple, robust rules (Buy Growth, Sell Crisis)

### The Results
- ✅ Out-of-sample testing on 2018 data (no cheating!)
- ✅ Detects regime transitions before major crashes
- ✅ Outperforms buy-and-hold in volatile markets
- ✅ 100% reproducible with saved models

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

📖 **Deep dive:** [ARCHITECTURE.md](ARCHITECTURE.md) | [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)

---

## 📊 What Makes This Different

| Feature | This Project | Typical Academic Projects |
|---------|-------------|---------------------------|
| **ML Validation** | Walk-forward (2012-2017 → 2018) | K-fold cross-validation (wrong for time series!) |
| **Regime Labels** | Automatic (unsupervised) | Manual labeling (expensive/biased) |
| **Feature Engineering** | Domain-driven (volatility + sector coupling) | Raw prices (leaves money on table) |
| **Interpretability** | Explainable regimes + rules | Black-box predictions |
| **Reproducibility** | Saved models, fixed seeds | "Trust me, it worked" |

**Key Innovation:** The **Systemic Health Score**—measuring sector correlations to detect when markets move in lockstep (systemic stress indicator).

---

## 🎓 What I Learned

Building this taught me that **ML for finance is 20% algorithms, 80% avoiding traps.**

### Critical Lessons
1. **Walk-forward validation saves you from embarrassment**  
   K-fold cross-validation on time series = data leakage = fake results

2. **Feature engineering beats model complexity**  
   Simple HMM on smart features > complex LSTM on raw prices

3. **Domain knowledge unlocks unsupervised learning**  
   HMM finds patterns. Humans make them tradeable.

4. **Backtesting reveals brutal truths**  
   Paper strategies are elegant. Real backtests expose edge cases.

5. **Simplicity is underrated**  
   3 regimes, 3 features, 3 rules. Works better than overcomplicated alternatives.

📖 **Full reflections:** [LEARNINGS.md](LEARNINGS.md) (seriously, read this—it's the good stuff)

---

## 🚀 Next Steps

This is a proof-of-concept. To make it production-ready:

**Quick Wins:**
- ✅ Add transaction costs (currently ignored)
- ✅ Risk-adjusted metrics (Sharpe ratio, max drawdown)
- ✅ Regime confidence scores (avoid uncertain trades)

**Ambitious Extensions:**
- 🔄 Online learning (retrain quarterly)
- 🌐 Multi-asset regimes (bonds, commodities, FX)
- 📡 Real-time deployment (Airflow + Alpaca API)
- 🧠 LSTM-HMM hybrid (deep learning meets regime switching)

📖 **Full roadmap:** [FUTURE_WORK.md](FUTURE_WORK.md)

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

## 🎯 Skills Demonstrated

**Machine Learning:**  
Unsupervised learning · Time series analysis · Feature engineering · Model validation · Walk-forward testing

**Finance:**  
Quantitative trading · Risk management · Backtesting · Sector correlation analysis · Market regime detection

**Software Engineering:**  
Modular architecture · Model persistence · Data pipelines · Documentation · Reproducibility

---

## 📚 Documentation

| Document | What's Inside |
|----------|---------------|
| **[QUICK_START.md](QUICK_START.md)** | 5-minute setup guide |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Detailed system design, module breakdowns |
| **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)** | HMM configuration, algorithms, feature engineering |
| **[LEARNINGS.md](LEARNINGS.md)** | Hard-won insights, mistakes, lessons |
| **[FUTURE_WORK.md](FUTURE_WORK.md)** | Roadmap for improvements, deployment ideas |

---

## 📊 Performance Note

**This is a research/educational project, not investment advice.**

Results shown are historical backtests. Past performance ≠ future returns. Real trading involves:
- Transaction costs
- Slippage
- Taxes  
- Market impact
- Emotional discipline

That said, the methodology is sound and the lessons are real.

---

## 📄 License

MIT License - Free for educational and research use. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

**Nyan Linn Htun (Nathan)**  
AI & Machine Learning Final Project, December 2025

Questions? Open an issue or check the documentation files above.

---

## 🙏 Acknowledgments

**Data:** Yahoo Finance  
**Inspiration:** Quantitative finance literature on regime switching  
**Tools:** hmmlearn, scikit-learn, pandas ecosystems  
**Guidance:** Course instructors on proper ML validation practices

---

<div align="center">

**⭐ If you find this project useful, consider starring it! ⭐**

*Built with Python, Hidden Markov Models, and hard-won lessons about financial ML*

</div>
