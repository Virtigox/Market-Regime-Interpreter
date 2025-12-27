# Market Regime Interpreter

> **An AI-driven quantitative trading system that identifies hidden market regimes using Hidden Markov Models and executes data-driven trading strategies**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Hidden%20Markov%20Models-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

**Author:** Nyan Linn Htun (Nathan)  
**Institution:** AI & Machine Learning Final Project  
**Date:** December 2025

---

## 🎯 Project Overview

This project implements a **dual-system AI trading architecture** that combines unsupervised machine learning with rule-based decision making to detect and exploit market regime transitions in the S&P 500 index.

### Problem Statement

Traditional buy-and-hold strategies fail to account for **structural changes in market behavior**—periods of growth, correction, and crisis operate under different dynamics. This system addresses the challenge of:
- Identifying latent market states from observable price and volatility patterns
- Adapting trading decisions to current market regimes
- Maintaining predictive performance on unseen data without overfitting

### Solution Architecture

**Two-System Design:**

1. **Information Generation System (IGS)** - Unsupervised Learning Pipeline
   - **Module 1:** Feature Engineering & Time Series Analysis
   - **Module 2:** Hidden Markov Model Training & Regime Classification

2. **Action Execution System (AES)** - Rule-Based Trading Agent
   - **Module 3:** Model-Based Reflex Agent with Regime-Dependent Strategies

### Key Technical Features

✅ **Hidden Markov Model (HMM)** with Gaussian emissions for regime detection  
✅ **Walk-forward validation** - trained on 2012-2017, tested on 2018  
✅ **Feature engineering** with technical indicators and cross-asset correlations  
✅ **Automatic regime labeling** bridging unsupervised ML and domain knowledge  
✅ **Performance benchmarking** against buy-and-hold baseline  
✅ **Proper ML practices** - standardization, no data leakage, out-of-sample testing

---

## 🛠️ Technologies & Methods

### Machine Learning
- **Hidden Markov Models (HMM)** - Unsupervised sequence modeling
- **Gaussian Mixture Emissions** - Multi-dimensional state modeling
- **Viterbi Algorithm** - Optimal state sequence decoding
- **K-Means Initialization** - Improved convergence

### Data Processing
- **Feature Engineering** - Technical indicators (RSI, Bollinger Bands, MACD, Volatility)
- **Cross-Asset Analysis** - Systemic health scoring via correlation matrices
- **Standardization** - Feature scaling for stable HMM training
- **Time Series Manipulation** - Log returns, rolling statistics

### Software Engineering
- **Modular Architecture** - Separation of concerns (IGS vs AES)
- **Model Persistence** - Joblib serialization for trained models
- **Walk-Forward Validation** - Preventing temporal data leakage
- **Backtesting Framework** - Performance evaluation with benchmarks

### Libraries & Tools
```
yfinance      # Market data acquisition
pandas        # Data manipulation and analysis
numpy         # Numerical computing
hmmlearn      # Hidden Markov Model implementation
scikit-learn  # Preprocessing and utilities
matplotlib    # Visualization
joblib        # Model serialization
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading market data)

### Installation

```bash
# Clone or download the project
cd Final_AI_Project

# Install dependencies
pip install yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib

# Verify installation
python -c "import yfinance, pandas, numpy, hmmlearn, sklearn, matplotlib, joblib; print('✓ All packages installed successfully!')"
```

### Running the Pipeline

**⚠️ Important:** Execute modules sequentially (1 → 2 → 3). Each module depends on outputs from the previous one.

```bash
# Step 1: Feature Engineering (~30-60 seconds)
python information_generation_system_module_1.py
# Output: features_complete_2012_2018.csv

# Step 2: Regime Detection (~10-20 seconds)
python information_generation_system_module_2.py
# Outputs: regimes_train_2012_2017.csv, regimes_test_2018.csv, 
#          hmm_model.pkl, scaler.pkl, regime_labels.pkl,
#          regime_visualization_train.png

# Step 3: Backtesting (~5 seconds)
python action_execution_system.py
# Output: Performance metrics and comparison vs benchmark
```

### Expected Results

```
======================================================================
BACKTEST RESULTS (2018 Test Period)
======================================================================

[REGIME-BASED STRATEGY]
  Initial Capital:      $10,000.00
  Final Value:          $X,XXX.XX
  Profit/Loss:          $XXX.XX
  Return:               XX.XX%
  Total Trades:         XX

[BUY-AND-HOLD BENCHMARK]
  Final Value:          $X,XXX.XX
  Return:               XX.XX%

[PERFORMANCE]
  Outperformance:       +/-X.XX%

[REGIME DISTRIBUTION]
  Growth      - XXX days (XX.X%)
  Correction  - XXX days (XX.X%)
  Crisis      - XXX days (XX.X%)
======================================================================
```

---

## 📊 System Architecture

### Module 1: Feature Engineering
**Purpose:** Transform raw price data into meaningful features for regime detection

**Features Generated:**
- **Technical Indicators:** RSI (momentum), Bollinger Bands (volatility envelope), MACD (trend)
- **Volatility Metrics:** 20-day rolling standard deviation
- **Log Returns:** Stabilized price changes
- **Systemic Health Score:** Cross-asset correlation analysis (S&P 500 vs VIX, bonds, gold)

**Output:** `features_complete_2012_2018.csv` (7 years × 8 features)

### Module 2: Hidden Markov Model Training
**Purpose:** Discover hidden market regimes from observable features

**Process:**
1. **Training Phase (2012-2017):**
   - Fit 3-state HMM with Gaussian emissions
   - Initialize with K-means clustering
   - Use 100 EM iterations for convergence
   - Standardize features to prevent scale bias

2. **Automatic Regime Labeling:**
   - Analyze mean returns and volatility per state
   - Classify states: Growth (high return, low vol), Correction (negative return), Crisis (high vol)

3. **Test Phase (2018):**
   - Apply trained model to unseen data
   - No retraining to simulate real deployment

**Outputs:** 
- Labeled training/test datasets
- Serialized model artifacts (`.pkl` files)
- Visualization of regime transitions

### Module 3: Action Execution System
**Purpose:** Execute trading strategy based on detected regimes

**Decision Rules:**
```
IF Regime == "Growth"      → BUY (if not already in position)
IF Regime == "Correction"  → HOLD (maintain position)
IF Regime == "Crisis"      → SELL (exit to cash)
```

**Performance Tracking:**
- Portfolio value evolution
- Trade execution log
- Benchmark comparison (buy-and-hold S&P 500)
- Win/loss statistics

---

## 🎓 Key Learnings & Insights

### 1. Walk-Forward Validation is Critical
Training and testing on the same period creates **temporal data leakage**. This project uses strict chronological split (train: 2012-2017, test: 2018) to simulate real-world deployment where future data is unavailable.

### 2. Domain Knowledge Enhances Unsupervised Learning
Pure HMM produces unlabeled states. Automatic regime labeling (analyzing return/volatility statistics) bridges ML outputs with trading intuition, making the system interpretable and actionable.

### 3. Feature Engineering > Model Complexity
Thoughtfully designed features (RSI, Bollinger Bands, systemic health) capture market dynamics better than complex models on raw prices. The Systemic Health Score, measuring cross-asset correlations, proved especially valuable.

### 4. Standardization Prevents Bias
Without feature scaling, high-variance features (absolute prices) dominate low-variance features (RSI). Standardization ensures the HMM learns from all dimensions equally.

### 5. Model-Based Agents Enable Modularity
Separating perception (IGS) from action (AES) allows independent development and testing. The IGS can be improved (e.g., adding features, trying different models) without rewriting the trading logic.

### 6. Backtesting Reveals Deployment Challenges
Issues discovered during backtesting:
- **Price reconstruction:** Log returns require careful transformation back to prices
- **Regime transitions:** Whipsaw trades during volatile periods
- **Benchmark choice:** Buy-and-hold provides context but isn't risk-adjusted

---

## 📁 Project Structure

```
Final_AI_Project/
│
├── information_generation_system_module_1.py  # Feature engineering pipeline
├── information_generation_system_module_2.py  # HMM training and regime detection
├── action_execution_system.py                  # Trading strategy backtesting
│
├── features_complete_2012_2018.csv            # Generated features (Module 1 output)
├── regimes_train_2012_2017.csv                # Training data with regime labels
├── regimes_test_2018.csv                      # Test data with regime predictions
│
├── hmm_model.pkl                              # Trained Hidden Markov Model
├── scaler.pkl                                 # Feature standardization scaler
├── regime_labels.pkl                          # Regime label mappings
├── regime_visualization_train.png             # Regime transition visualization
│
├── README.md                                  # This file
├── QUICK_START.md                             # Simplified quick reference guide
├── Project_Description.md                     # Original project proposal
├── PROJECT_POLISH_SUMMARY.md                  # Development changelog
├── install.txt                                # Dependency installation guide
└── RoadMap.md                                 # Development timeline
```

---

## 🔬 Technical Deep Dive

### Hidden Markov Model Configuration

```python
# Key hyperparameters
n_states = 3                    # Growth, Correction, Crisis
n_iter = 100                    # EM algorithm iterations
covariance_type = "full"        # Full covariance matrices
init_params = "kmeans"          # K-means initialization
```

**Why HMM?**
- Markets exhibit **latent states** not directly observable in prices
- State persistence (regimes last days/weeks, not minutes)
- Probabilistic transitions capture uncertainty
- Gaussian emissions model continuous features

### Feature Engineering Rationale

| Feature | Purpose | Regime Signal |
|---------|---------|---------------|
| **Log Returns** | Normalized price changes | Direction & magnitude |
| **RSI** | Overbought/oversold momentum | Regime exhaustion |
| **Bollinger %B** | Price position in volatility envelope | Breakout detection |
| **MACD** | Trend strength and direction | Regime transitions |
| **Volatility** | 20-day rolling std | Risk environment |
| **Systemic Health** | Cross-asset correlations | Systemic stress |

### Decision Logic Pseudocode

```python
def execute_trade(current_regime, position):
    if current_regime == "Growth":
        if position == "cash":
            return "BUY"   # Enter long position
    elif current_regime == "Correction":
        return "HOLD"      # Maintain current position
    elif current_regime == "Crisis":
        if position == "long":
            return "SELL"  # Exit to safety
    return "HOLD"
```

---

## 🐛 Challenges & Solutions

### Challenge 1: Feature Scale Imbalance
**Problem:** Price ($3000) vs RSI (0-100) → HMM biased toward high-variance features  
**Solution:** StandardScaler normalization (mean=0, std=1) before HMM training

### Challenge 2: Data Leakage Risk
**Problem:** Training on future data inflates performance  
**Solution:** Strict temporal split + frozen model during test phase

### Challenge 3: Regime Interpretation
**Problem:** HMM produces unlabeled states (0, 1, 2)  
**Solution:** Automatic labeling via return/volatility statistics + domain knowledge

### Challenge 4: Price Reconstruction
**Problem:** Log returns don't directly reconstruct prices  
**Solution:** Exponential transformation: `price_t = price_0 * exp(sum(log_returns))`

---

## 📈 Results & Performance

### Regime Detection Quality
- **Temporal coherence:** Regimes persist for meaningful periods (weeks/months)
- **Alignment with reality:** Crisis states align with 2018 market corrections
- **Stability:** Model doesn't oscillate between states daily

### Trading Strategy Performance
- **Out-of-sample testing:** 2018 results reflect true generalization
- **Benchmark comparison:** Performance measured vs buy-and-hold
- **Risk management:** Crisis detection enables downside protection

### Visualizations
The `regime_visualization_train.png` shows:
- S&P 500 price evolution (2012-2017)
- Color-coded regime transitions
- Visual validation of regime persistence

---

## 🚧 Future Improvements

### Model Enhancements
- [ ] **Risk-adjusted metrics:** Sharpe ratio, maximum drawdown analysis
- [ ] **Transaction costs:** Include realistic trading fees
- [ ] **Position sizing:** Kelly criterion for optimal allocation
- [ ] **Multi-asset regimes:** Extend to bonds, commodities, FX

### Advanced ML Techniques
- [ ] **LSTM-HMM hybrid:** Combine sequence modeling with regime detection
- [ ] **Online learning:** Quarterly model retraining with new data
- [ ] **Ensemble methods:** Multiple HMMs with different feature sets
- [ ] **Regime confidence scores:** Probabilistic decision thresholds

### Deployment Considerations
- [ ] **Real-time data pipeline:** Live market data integration
- [ ] **API development:** RESTful service for regime predictions
- [ ] **Monitoring dashboard:** Track model drift and performance
- [ ] **Robustness testing:** Stress testing on 2008 crisis, COVID-19 crash

---

## 🎯 Skills Demonstrated

### Machine Learning
✓ Unsupervised learning (HMM clustering)  
✓ Time series analysis and forecasting  
✓ Feature engineering and domain knowledge integration  
✓ Model evaluation and validation  
✓ Hyperparameter tuning  

### Software Engineering
✓ Modular code architecture  
✓ Version control and documentation  
✓ Unit testing and debugging  
✓ Data pipeline design  
✓ Model serialization and deployment  

### Finance & Trading
✓ Quantitative strategy development  
✓ Risk management principles  
✓ Backtesting methodologies  
✓ Performance benchmarking  
✓ Market microstructure understanding  

### Data Science
✓ Pandas data manipulation  
✓ NumPy numerical computing  
✓ Matplotlib visualization  
✓ Statistical analysis  
✓ Cross-validation techniques  

---

## 📚 References & Resources

### Academic Foundations
- **Hidden Markov Models:** Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition"
- **Regime Detection:** Ang, A., & Bekaert, G. (2002). "Regime switches in interest rates"
- **Financial ML:** Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"

### Libraries & Documentation
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data acquisition
- [hmmlearn](https://hmmlearn.readthedocs.io/) - Hidden Markov Model implementation
- [scikit-learn](https://scikit-learn.org/) - Machine learning toolkit
- [pandas](https://pandas.pydata.org/) - Data analysis library

### Related Projects
- Quantitative trading strategies with ML
- Market regime classification systems
- Algorithmic trading backtesting frameworks

---

## 🤝 Contributing

This project was developed as a final project for an AI & Machine Learning course. While not actively maintained, feedback and suggestions are welcome:

- **Issues:** Report bugs or suggest enhancements
- **Pull Requests:** Improvements to documentation or code
- **Discussions:** Share your results or alternative approaches

---

## 📄 License

This project is available under the MIT License. Feel free to use the code for educational purposes, with appropriate attribution.

---

## 📧 Contact

**Nyan Linn Htun (Nathan)**  
AI & Machine Learning Final Project  
December 2025

For questions about the project methodology, implementation details, or potential collaborations, please reach out via:
- GitHub Issues (for technical questions)
- Project documentation (comprehensive troubleshooting)

---

## 🙏 Acknowledgments

- **Data Source:** Yahoo Finance for historical market data
- **ML Framework:** hmmlearn library maintainers
- **Inspiration:** Quantitative finance literature on regime detection
- **Course Instructors:** For guidance on proper ML practices and validation techniques

---

<div align="center">

**⭐ If you find this project interesting, please consider starring it! ⭐**

*Built with Python, Machine Learning, and a passion for quantitative finance*