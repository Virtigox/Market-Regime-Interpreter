# Technical Deep Dive

## Hidden Markov Model Configuration

```python
# Key hyperparameters
n_states = 3                    # Growth, Correction, Crisis
n_iter = 100                    # EM algorithm iterations
covariance_type = "full"        # Full covariance matrices
init_params = "kmeans"          # K-means initialization
```

### Why Hidden Markov Models?

Markets exhibit **latent states** not directly observable in prices. HMMs are ideal for this problem because:

- **Latent State Modeling:** Market regimes are hidden - we only observe prices, volumes, and indicators
- **State Persistence:** Regimes last days or weeks, not minutes - HMMs capture this temporal coherence
- **Probabilistic Transitions:** Markets don't switch regimes instantly; HMMs model gradual transition probabilities
- **Gaussian Emissions:** Continuous features (returns, volatility) are well-modeled by multivariate Gaussians
- **Unsupervised Learning:** No need for labeled training data - the model discovers regimes from patterns

---

## Feature Engineering Rationale

Feature selection is critical for regime detection. Each feature captures different market dynamics:

| Feature | Purpose | Regime Signal | Range |
|---------|---------|---------------|-------|
| **Log Returns** | Normalized price changes | Direction & magnitude of market movement | (-∞, +∞) |
| **RSI** | Overbought/oversold momentum | Regime exhaustion and reversal points | [0, 100] |
| **Bollinger %B** | Price position in volatility envelope | Breakout detection and volatility regime | [0, 1] |
| **MACD** | Trend strength and direction | Regime transitions and momentum shifts | (-∞, +∞) |
| **Volatility** | 20-day rolling standard deviation | Risk environment and market stress | [0, +∞) |
| **Systemic Health Score** | Cross-asset correlations | Systemic stress across markets | [-1, 1] |

### Technical Indicator Details

#### RSI (Relative Strength Index)
```python
# Momentum oscillator measuring speed and magnitude of price changes
# RSI > 70: Overbought (potential correction)
# RSI < 30: Oversold (potential recovery)
# Used to detect regime exhaustion
```

#### Bollinger Bands %B
```python
# Normalized position within Bollinger Bands
# %B > 1: Price above upper band (high volatility)
# %B < 0: Price below lower band (low volatility)
# Captures volatility regime transitions
```

#### MACD (Moving Average Convergence Divergence)
```python
# Trend-following momentum indicator
# MACD > Signal: Bullish momentum
# MACD < Signal: Bearish momentum
# Crossovers signal regime changes
```

#### Systemic Health Score
```python
# Cross-asset correlation analysis
# Measures: S&P 500 correlations with VIX, Treasury bonds, Gold
# High correlation with VIX → Market stress (Crisis regime)
# Low correlation → Stable environment (Growth regime)
```

### Feature Standardization

**Why it matters:**
- Raw features have vastly different scales (Price: $3000, RSI: 0-100)
- HMM uses Euclidean distances - large-scale features dominate
- StandardScaler transforms all features to mean=0, std=1

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_train)

# Critical: Fit on training data only, transform both train and test
# Prevents data leakage
```

---

## Decision Logic & Trading Rules

### Rule-Based Strategy

The Action Execution System implements a model-based reflex agent with simple, interpretable rules:

```python
def execute_trade(current_regime, position):
    """
    Regime-based trading logic
    
    Args:
        current_regime: "Growth", "Correction", or "Crisis"
        position: "cash" or "long"
    
    Returns:
        action: "BUY", "SELL", or "HOLD"
    """
    if current_regime == "Growth":
        if position == "cash":
            return "BUY"   # Enter long position in favorable regime
        else:
            return "HOLD"  # Maintain position
    
    elif current_regime == "Correction":
        return "HOLD"      # Wait it out - short-term pullback
    
    elif current_regime == "Crisis":
        if position == "long":
            return "SELL"  # Exit to safety during systemic risk
        else:
            return "HOLD"  # Stay in cash
    
    return "HOLD"
```

### Regime Definitions

| Regime | Characteristics | Trading Action | Rationale |
|--------|----------------|----------------|-----------|
| **Growth** | High returns, Low volatility | **BUY** | Favorable risk/reward - capture upside |
| **Correction** | Negative returns, Moderate volatility | **HOLD** | Temporary pullback - don't panic sell |
| **Crisis** | High volatility, Systemic stress | **SELL** | Protect capital during market crashes |

### Automatic Regime Labeling

The HMM produces unlabeled states (0, 1, 2). We automatically assign meaningful names:

```python
def label_regimes(states, returns, volatility):
    """
    Assign semantic labels to HMM states
    
    Strategy:
    1. Calculate mean return and volatility for each state
    2. State with highest return + lowest vol → Growth
    3. State with negative return → Correction
    4. State with highest volatility → Crisis
    """
    state_stats = {}
    for state in [0, 1, 2]:
        mask = (states == state)
        state_stats[state] = {
            'mean_return': returns[mask].mean(),
            'volatility': volatility[mask].mean()
        }
    
    # Labeling logic
    labels = {}
    for state, stats in state_stats.items():
        if stats['mean_return'] > 0 and stats['volatility'] < threshold:
            labels[state] = "Growth"
        elif stats['mean_return'] < 0:
            labels[state] = "Correction"
        else:
            labels[state] = "Crisis"
    
    return labels
```

---

## Walk-Forward Validation

### Why Temporal Split Matters

**The Problem:** Time series data has temporal dependencies
- Training on 2018, testing on 2017 → Uses "future information"
- K-fold cross-validation → Shuffles data, breaks temporal order
- Both approaches create **data leakage** - artificially inflate performance

**The Solution:** Walk-forward validation
```
├─────────────────┬─────────┤
   Train Period   │  Test   
   2012 - 2017    │  2018   
```

**Key Principles:**
1. **Chronological Split:** Train on past, test on future
2. **Frozen Model:** No retraining during test period
3. **Realistic Simulation:** Mimics real deployment where future is unknown
4. **Feature Scaling:** Fit scaler on training data only

### Implementation

```python
# Correct approach
train_data = data[data['date'] < '2018-01-01']
test_data = data[data['date'] >= '2018-01-01']

# Fit on train
scaler = StandardScaler()
model = GaussianHMM(n_components=3)

X_train_scaled = scaler.fit_transform(train_data[features])
model.fit(X_train_scaled)

# Apply to test (no refitting!)
X_test_scaled = scaler.transform(test_data[features])
test_regimes = model.predict(X_test_scaled)
```

---

## Performance Metrics

### Backtesting Framework

The system tracks:
- **Portfolio Value:** Daily mark-to-market
- **Trade Execution:** Entry/exit prices and dates
- **Benchmark Comparison:** S&P 500 buy-and-hold
- **Regime Distribution:** Time spent in each state

### Metrics Tracked

```python
class PerformanceTracker:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.position = "cash"
        self.trades = []
        self.portfolio_value = []
    
    def calculate_return(self):
        return (self.capital - self.initial_capital) / self.initial_capital
    
    def calculate_sharpe_ratio(self):
        # Risk-adjusted return
        returns = np.diff(self.portfolio_value) / self.portfolio_value[:-1]
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def max_drawdown(self):
        # Worst peak-to-trough decline
        cummax = np.maximum.accumulate(self.portfolio_value)
        drawdown = (self.portfolio_value - cummax) / cummax
        return np.min(drawdown)
```

### Benchmark Comparison

**Why Buy-and-Hold?**
- Simple baseline everyone understands
- Represents passive investing approach
- Shows value-add of regime detection

**Limitations:**
- Doesn't account for risk differences
- Ignores transaction costs
- Single benchmark may not capture strategy complexity

**Future Work:**
- Sharpe ratio comparison (risk-adjusted returns)
- Maximum drawdown analysis
- Multiple benchmarks (60/40 portfolio, momentum strategies)

---

## Data Processing Pipeline

### Module 1: Feature Engineering

```
Raw Price Data (Yahoo Finance)
    ↓
Calculate Technical Indicators
    ↓
Compute Systemic Health Score
    ↓
Handle Missing Values (forward fill)
    ↓
Save: features_complete_2012_2018.csv
```

### Module 2: Regime Detection

```
Load Features
    ↓
Train/Test Split (2012-2017 / 2018)
    ↓
Standardize Features (fit on train)
    ↓
Train HMM (3 states, 100 iterations)
    ↓
Predict Regimes (Viterbi algorithm)
    ↓
Automatic Regime Labeling
    ↓
Save: Models (.pkl) + Labeled Data (.csv)
```

### Module 3: Backtesting

```
Load Test Regime Predictions
    ↓
Initialize Portfolio ($10,000)
    ↓
For each trading day:
    - Check regime
    - Apply decision rules
    - Execute trades
    - Update portfolio value
    ↓
Calculate Performance Metrics
    ↓
Compare vs Benchmark
    ↓
Display Results
```

---

## Model Artifacts

The system saves trained components for reproducibility:

| File | Purpose | Size |
|------|---------|------|
| `hmm_model.pkl` | Trained 3-state HMM | ~10 KB |
| `scaler.pkl` | StandardScaler (mean/std) | ~1 KB |
| `regime_labels.pkl` | State → Regime mapping | <1 KB |

### Loading Models

```python
import joblib

# Load trained artifacts
model = joblib.load('hmm_model.pkl')
scaler = joblib.load('scaler.pkl')
regime_labels = joblib.load('regime_labels.pkl')

# Use for inference
new_features_scaled = scaler.transform(new_features)
predicted_states = model.predict(new_features_scaled)
predicted_regimes = [regime_labels[state] for state in predicted_states]
```

---

## Technologies & Libraries

### Core Dependencies

```python
# Data acquisition
import yfinance as yf

# Data processing
import pandas as pd
import numpy as np

# Machine learning
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt

# Serialization
import joblib
```

### Version Requirements

```
Python >= 3.8
yfinance >= 0.1.70
pandas >= 1.3.0
numpy >= 1.21.0
hmmlearn >= 0.2.7
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
joblib >= 1.1.0
```

### Installation

```bash
# Using pip
pip install yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib

# Using conda
conda install -c conda-forge yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib

# From requirements.txt
pip install -r requirements.txt
```

---

## Computational Complexity

### Training Phase

- **Feature Engineering:** O(n) where n = number of days
- **HMM Training:** O(k² × n × d × i) where:
  - k = number of states (3)
  - n = training samples (~1500 days)
  - d = feature dimensions (6-8)
  - i = EM iterations (100)
- **Typical Runtime:** 10-20 seconds on modern CPU

### Inference Phase

- **Viterbi Algorithm:** O(k² × n) - very efficient
- **Regime Prediction:** <1 second for 1 year of data
- **Real-time Capable:** Can process daily updates in milliseconds

---

## Visualization

### Regime Transition Plot

The `regime_visualization_train.png` shows:
- **X-axis:** Time (2012-2017)
- **Y-axis:** S&P 500 price
- **Color Coding:**
  - 🟢 Green: Growth regime
  - 🟡 Yellow: Correction regime
  - 🔴 Red: Crisis regime

**Interpretation:**
- Visual validation of regime persistence
- Alignment with known market events
- Smooth transitions vs erratic switching

### Creating Custom Visualizations

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(dates, prices, label='S&P 500', color='black', alpha=0.7)

# Color background by regime
for i, regime in enumerate(regimes):
    color = {'Growth': 'green', 'Correction': 'yellow', 'Crisis': 'red'}[regime]
    plt.axvspan(dates[i], dates[i+1], alpha=0.3, color=color)

plt.title('Market Regimes Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('regime_visualization.png', dpi=300)
```

---

*For system architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)*  
*For project learnings and insights, see [LEARNINGS.md](LEARNINGS.md)*  
*For future improvements, see [FUTURE_WORK.md](FUTURE_WORK.md)*
