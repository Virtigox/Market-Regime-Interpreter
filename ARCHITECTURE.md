# System Architecture

## Overview

The Market Regime Interpreter follows a **dual-system design** that separates information processing from decision making:

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET DATA (Yahoo Finance)              │
│                        S&P 500, VIX, Bonds, Gold            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          INFORMATION GENERATION SYSTEM (IGS)                │
│                  Unsupervised ML Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│  Module 1: Feature Engineering                             │
│    - Technical Indicators (RSI, Bollinger, MACD)           │
│    - Volatility Metrics                                    │
│    - Systemic Health Score                                 │
│                                                             │
│  Module 2: Regime Detection                                │
│    - Hidden Markov Model Training                          │
│    - Automatic Regime Labeling                             │
│    - State Sequence Prediction                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ↓
                    REGIME PREDICTIONS
              (Growth / Correction / Crisis)
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          ACTION EXECUTION SYSTEM (AES)                      │
│              Model-Based Reflex Agent                       │
├─────────────────────────────────────────────────────────────┤
│  Module 3: Trading Strategy                                │
│    - Regime-Based Decision Rules                           │
│    - Portfolio Management                                  │
│    - Performance Tracking                                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ↓
                    TRADING ACTIONS
                  (BUY / SELL / HOLD)
```

---

## Module 1: Feature Engineering

### Purpose
Transform raw price data into meaningful features that capture market dynamics for regime detection.

### Input
- **Data Source:** Yahoo Finance API
- **Assets:** S&P 500 (^GSPC), VIX (^VIX), 10Y Treasury (^TNX), Gold (GC=F)
- **Period:** 2012-01-01 to 2018-12-31 (7 years)
- **Frequency:** Daily

### Process

#### 1. Data Acquisition
```python
import yfinance as yf

# Download historical data
sp500 = yf.download('^GSPC', start='2012-01-01', end='2018-12-31')
vix = yf.download('^VIX', start='2012-01-01', end='2018-12-31')
bonds = yf.download('^TNX', start='2012-01-01', end='2018-12-31')
gold = yf.download('GC=F', start='2012-01-01', end='2018-12-31')
```

#### 2. Technical Indicator Calculation

**RSI (Relative Strength Index)**
- **Purpose:** Measure momentum and overbought/oversold conditions
- **Period:** 14 days
- **Range:** 0-100
- **Interpretation:** >70 overbought, <30 oversold

**Bollinger Bands %B**
- **Purpose:** Measure price position within volatility envelope
- **Components:** 20-day SMA ± 2 standard deviations
- **Range:** 0-1 (can exceed during breakouts)
- **Interpretation:** >1 above upper band, <0 below lower band

**MACD (Moving Average Convergence Divergence)**
- **Purpose:** Track trend changes and momentum
- **Components:** 12-day EMA - 26-day EMA
- **Signal Line:** 9-day EMA of MACD
- **Interpretation:** MACD > Signal is bullish

**Volatility**
- **Calculation:** 20-day rolling standard deviation of returns
- **Purpose:** Measure market uncertainty and risk
- **Higher values:** Crisis or correction regimes

#### 3. Systemic Health Score

Novel feature measuring cross-asset correlations:

```python
def calculate_systemic_health(sp500, vix, bonds, gold, window=20):
    """
    Measures systemic market health via correlations
    
    Logic:
    - High VIX correlation → Market stress (Crisis)
    - High bond correlation → Flight to safety (Correction/Crisis)
    - Low correlations → Stable environment (Growth)
    
    Returns: Composite score [-1, 1]
    """
    corr_vix = sp500.rolling(window).corr(vix)
    corr_bonds = sp500.rolling(window).corr(bonds)
    corr_gold = sp500.rolling(window).corr(gold)
    
    # Weight negative correlations (flight to safety)
    health_score = -(abs(corr_vix) + abs(corr_bonds) + abs(corr_gold)) / 3
    return health_score
```

#### 4. Data Cleaning
- **Missing Values:** Forward fill for minor gaps
- **Alignment:** Ensure all assets have matching dates
- **Outlier Handling:** Retain extremes (they signal regime changes)

### Output

**File:** `features_complete_2012_2018.csv`

**Columns:**
- `Date`: Trading date
- `Close`: S&P 500 closing price
- `Log_Return`: Log-transformed daily returns
- `RSI`: Relative Strength Index
- `Bollinger_B`: Bollinger %B indicator
- `MACD`: MACD histogram
- `Volatility`: 20-day rolling std
- `Systemic_Health`: Cross-asset correlation score

**Shape:** ~1,760 rows × 8 columns (7 years of trading days)

### Runtime
- **Typical:** 30-60 seconds
- **Network Dependent:** Yahoo Finance API download speed
- **Memory:** <50 MB

---

## Module 2: Regime Detection

### Purpose
Discover hidden market regimes using unsupervised machine learning and assign semantic labels.

### Input
- **File:** `features_complete_2012_2018.csv`
- **Features Used:** All except `Date` and `Close`
- **Training Period:** 2012-2017 (1,510 days)
- **Testing Period:** 2018 (251 days)

### Process

#### 1. Train/Test Split (Walk-Forward Validation)

```python
# Chronological split - no shuffling!
train_data = data[data['Date'] < '2018-01-01']
test_data = data[data['Date'] >= '2018-01-01']
```

**Critical:** This prevents temporal data leakage. The model cannot "peek" at 2018 data during training.

#### 2. Feature Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[feature_columns])
X_test_scaled = scaler.transform(test_data[feature_columns])  # Use train statistics
```

**Why?** Prevents high-variance features (price) from dominating low-variance features (RSI).

#### 3. Hidden Markov Model Training

```python
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(
    n_components=3,              # 3 hidden states
    covariance_type='full',      # Full covariance matrices
    n_iter=100,                  # EM algorithm iterations
    init_params='kmeans',        # K-means initialization
    random_state=42              # Reproducibility
)

model.fit(X_train_scaled)
```

**Training Algorithm:**
1. **Initialization:** K-means clustering to set initial state means
2. **Expectation-Maximization (EM):**
   - E-step: Estimate state probabilities given current parameters
   - M-step: Update parameters to maximize likelihood
   - Repeat 100 iterations until convergence
3. **Convergence:** Monitor log-likelihood improvements

#### 4. State Prediction (Viterbi Algorithm)

```python
# Find most likely state sequence
train_states = model.predict(X_train_scaled)
test_states = model.predict(X_test_scaled)
```

**Viterbi Algorithm:**
- Dynamic programming approach
- Finds globally optimal state sequence
- More accurate than local decoding

#### 5. Automatic Regime Labeling

The HMM produces numerical states (0, 1, 2). We automatically assign meaningful labels:

```python
def label_regimes_automatically(states, returns, volatility):
    """
    Assign Growth/Correction/Crisis labels based on statistics
    
    Rules:
    1. Calculate mean return and volatility for each state
    2. State with highest return + lowest vol → Growth
    3. State with negative return → Correction
    4. State with highest volatility → Crisis
    """
    state_profiles = {}
    
    for state in [0, 1, 2]:
        mask = (states == state)
        state_profiles[state] = {
            'mean_return': returns[mask].mean(),
            'volatility': volatility[mask].mean(),
            'days': mask.sum()
        }
    
    # Labeling logic
    labels = {}
    for state, profile in state_profiles.items():
        if profile['mean_return'] > 0 and profile['volatility'] < median_vol:
            labels[state] = "Growth"
        elif profile['mean_return'] < 0:
            labels[state] = "Correction"
        else:
            labels[state] = "Crisis"
    
    return labels
```

**Example Output:**
```
State 0: Growth      (mean_return: +0.08%, volatility: 0.7%)
State 1: Correction  (mean_return: -0.12%, volatility: 1.1%)
State 2: Crisis      (mean_return: -0.05%, volatility: 1.8%)
```

#### 6. Visualization

Generate regime transition plot:

```python
plt.figure(figsize=(15, 6))
plt.plot(dates, prices, 'k-', alpha=0.7)

# Color background by regime
for i in range(len(regimes)-1):
    color_map = {'Growth': 'green', 'Correction': 'yellow', 'Crisis': 'red'}
    plt.axvspan(dates[i], dates[i+1], alpha=0.3, color=color_map[regimes[i]])

plt.title('S&P 500 Market Regimes (2012-2017)')
plt.savefig('regime_visualization_train.png')
```

### Output

**Files Generated:**
1. `regimes_train_2012_2017.csv` - Training data with regime labels
2. `regimes_test_2018.csv` - Test data with regime predictions
3. `hmm_model.pkl` - Trained HMM (for reproducibility)
4. `scaler.pkl` - Feature scaler (for inference)
5. `regime_labels.pkl` - State→Regime mapping
6. `regime_visualization_train.png` - Visual validation plot

### Runtime
- **Training:** 10-20 seconds
- **Prediction:** <1 second
- **Total:** ~15 seconds

---

## Module 3: Action Execution System

### Purpose
Execute trading strategy based on detected regimes and evaluate performance.

### Input
- **File:** `regimes_test_2018.csv`
- **Columns:** `Date`, `Close`, `Regime`
- **Period:** 2018 (out-of-sample test)

### Trading Strategy

#### Decision Rules

```python
class RegimeBasedStrategy:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.position = "cash"  # or "long"
        self.shares = 0
        self.trades = []
    
    def execute(self, date, price, regime):
        if regime == "Growth":
            if self.position == "cash":
                # BUY: Enter long position
                self.shares = self.capital / price
                self.position = "long"
                self.trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': self.shares
                })
        
        elif regime == "Correction":
            # HOLD: Wait it out
            pass
        
        elif regime == "Crisis":
            if self.position == "long":
                # SELL: Exit to safety
                self.capital = self.shares * price
                self.position = "cash"
                self.shares = 0
                self.trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'value': self.capital
                })
    
    def get_portfolio_value(self, current_price):
        if self.position == "long":
            return self.shares * current_price
        else:
            return self.capital
```

#### Rationale

| Regime | Action | Reasoning |
|--------|--------|-----------|
| **Growth** | BUY (if in cash) | Favorable risk/reward - capture upside with low volatility |
| **Correction** | HOLD | Short-term pullback - avoid whipsaw trades |
| **Crisis** | SELL (if long) | Protect capital during systemic risk events |

### Benchmark Strategy

**Buy-and-Hold:**
```python
class BuyAndHold:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.shares = None
    
    def execute(self, initial_price):
        # Buy at start, hold forever
        self.shares = self.capital / initial_price
    
    def get_value(self, current_price):
        return self.shares * current_price
```

### Performance Tracking

#### Metrics Calculated

1. **Total Return**
   ```python
   total_return = (final_value - initial_capital) / initial_capital * 100
   ```

2. **Absolute Profit/Loss**
   ```python
   profit_loss = final_value - initial_capital
   ```

3. **Number of Trades**
   - Counts BUY and SELL transactions
   - Lower is better (transaction costs)

4. **Regime Distribution**
   ```python
   regime_days = {
       'Growth': len(data[data['Regime'] == 'Growth']),
       'Correction': len(data[data['Regime'] == 'Correction']),
       'Crisis': len(data[data['Regime'] == 'Crisis'])
   }
   ```

5. **Outperformance vs Benchmark**
   ```python
   outperformance = strategy_return - benchmark_return
   ```

### Output

**Console Display:**
```
======================================================================
BACKTEST RESULTS (2018 Test Period)
======================================================================

[REGIME-BASED STRATEGY]
  Initial Capital:      $10,000.00
  Final Value:          $10,450.23
  Profit/Loss:          $450.23
  Return:               4.50%
  Total Trades:         6

[BUY-AND-HOLD BENCHMARK]
  Final Value:          $9,856.12
  Return:               -1.44%

[PERFORMANCE]
  Outperformance:       +5.94%

[REGIME DISTRIBUTION]
  Growth      - 145 days (57.8%)
  Correction  - 78 days (31.1%)
  Crisis      - 28 days (11.2%)
======================================================================
```

### Runtime
- **Backtesting:** ~5 seconds
- **I/O:** <1 second
- **Total:** <10 seconds

---

## Data Flow

### Complete Pipeline

```
1. FEATURE ENGINEERING (Module 1)
   Input:  Yahoo Finance API
   Output: features_complete_2012_2018.csv
   Time:   30-60 sec

2. REGIME DETECTION (Module 2)
   Input:  features_complete_2012_2018.csv
   Output: regimes_train_2012_2017.csv
           regimes_test_2018.csv
           hmm_model.pkl
           scaler.pkl
           regime_labels.pkl
   Time:   10-20 sec

3. BACKTESTING (Module 3)
   Input:  regimes_test_2018.csv
   Output: Performance metrics (console)
   Time:   5 sec

TOTAL RUNTIME: ~1-2 minutes
```

### File Dependencies

```
information_generation_system_module_1.py
    └── features_complete_2012_2018.csv
            ↓
information_generation_system_module_2.py
    ├── regimes_train_2012_2017.csv
    ├── regimes_test_2018.csv
    ├── hmm_model.pkl
    ├── scaler.pkl
    └── regime_labels.pkl
            ↓
action_execution_system.py
    └── Performance results
```

---

## Design Principles

### 1. Separation of Concerns

**Information Generation (IGS):** Perception layer
- Processes raw data into actionable insights
- Uses unsupervised ML (no labels required)
- Independent of trading logic

**Action Execution (AES):** Action layer
- Makes trading decisions based on regimes
- Rule-based and interpretable
- Independent of regime detection method

**Benefits:**
- Can improve regime detection without changing trading rules
- Can test different strategies with same regime predictions
- Easier testing and debugging

### 2. Walk-Forward Validation

**Temporal Integrity:**
- Train on 2012-2017, test on 2018
- No shuffling or k-fold cross-validation
- Simulates real deployment scenario

**No Retraining:**
- Model frozen during test period
- Prevents look-ahead bias
- Realistic performance estimates

### 3. Reproducibility

**Fixed Random Seeds:**
```python
np.random.seed(42)
model = GaussianHMM(random_state=42)
```

**Model Serialization:**
```python
import joblib
joblib.dump(model, 'hmm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

**Benefits:**
- Same results every run
- Can share and compare models
- Debugging is easier

### 4. Modularity

Each module is self-contained:
- Can run independently (after dependencies)
- Clear input/output contracts
- Easy to extend or replace

---

## Scalability Considerations

### Current System
- **Data Volume:** 7 years, daily frequency (~1,760 samples)
- **Feature Dimensions:** 6-8 features
- **Model Complexity:** 3-state HMM
- **Runtime:** <2 minutes total

### Scaling Strategies

**More Data:**
- Add decades of historical data
- System scales linearly O(n)
- HMM training is efficient

**More Features:**
- Add fundamental data (P/E, earnings)
- Sentiment indicators (news, social media)
- Alternative data sources

**More Assets:**
- Multi-asset regime detection
- Sector-specific regimes
- Global market regimes

**Higher Frequency:**
- Intraday data (hourly, minute)
- More samples but same approach
- May need online learning

**Real-Time Deployment:**
- Streaming data pipeline
- Daily model updates
- API for regime predictions

---

## Error Handling

### Data Validation

```python
def validate_features(df):
    """Check for common data issues"""
    
    # Missing values
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")
    
    # Infinite values
    if np.isinf(df.values).any():
        raise ValueError("Dataset contains infinite values")
    
    # Date monotonicity
    if not df['Date'].is_monotonic_increasing:
        raise ValueError("Dates are not in chronological order")
    
    return True
```

### Model Validation

```python
def validate_hmm(model, data):
    """Ensure HMM converged properly"""
    
    # Check convergence
    if not model.monitor_.converged:
        warnings.warn("HMM did not converge - consider more iterations")
    
    # Check state usage
    states = model.predict(data)
    unique_states = np.unique(states)
    if len(unique_states) < model.n_components:
        warnings.warn(f"Only {len(unique_states)} states used (expected {model.n_components})")
    
    return True
```

---

*For technical implementation details, see [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)*  
*For project learnings and insights, see [LEARNINGS.md](LEARNINGS.md)*  
*For future improvements, see [FUTURE_WORK.md](FUTURE_WORK.md)*
