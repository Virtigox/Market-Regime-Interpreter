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
- **Assets:** 
  - S&P 500 Index (^GSPC)
  - 5 Sector ETFs: XLK (Tech), XLF (Finance), XLV (Healthcare), XLE (Energy), XLI (Industrials)
- **Period:** 2012-01-01 to 2018-12-31 (7 years)
- **Frequency:** Daily

### Process

#### 1. Data Acquisition
```python
import yfinance as yf

# Download S&P 500 and sector ETFs
index_ticker = "^GSPC"
sector_tickers = ["XLK", "XLF", "XLV", "XLE", "XLI"]
all_tickers = [index_ticker] + sector_tickers

raw_data = yf.download(all_tickers, start="2012-01-01", end="2018-12-31")
data = raw_data["Adj Close"]  # Use adjusted close prices
```

#### 2. Feature Calculation

**Index Returns (Log Returns)**
- **Purpose:** Capture directional movement and magnitude of market changes
- **Calculation:** `log(price_t / price_{t-1})`
- **Properties:** Time-additive, approximately normal distribution
- **Range:** (-∞, +∞)
- **Interpretation:** Positive = gains, negative = losses

**Index Volatility (21-Day Rolling Standard Deviation)**
- **Purpose:** Measure market risk and uncertainty
- **Window:** 21 trading days (≈1 month)
- **Calculation:** Rolling standard deviation of log returns
- **Range:** [0, +∞)
- **Interpretation:** Higher values indicate crisis/correction regimes

**Systemic Health Score (60-Day Sector Correlation)**
- **Purpose:** Detect market coupling and systemic fragility
- **Window:** 60 trading days (≈3 months)
- **Calculation:** Average pairwise correlation between 5 sector ETFs
- **Range:** [-1, 1] (typically 0.2 to 0.9)
- **Interpretation:**
  - Low correlation (0.2-0.4) = Healthy, decoupled sectors (Growth)
  - High correlation (0.8-1.0) = Systemic stress, lockstep movement (Crisis)

#### 3. Systemic Health Score Implementation Implementation

Measures cross-sector correlations to detect ecosystem-level behavior:

```python
def get_avg_sector_correlation(returns_slice):
    """
    Calculate average pairwise correlation between sectors
    
    Logic:
    - Uses upper triangle of correlation matrix to avoid duplicates
    - Excludes self-correlation (diagonal = 1.0)
    - Returns mean correlation across all sector pairs
    
    Interpretation:
    - Low correlation (0.2-0.4): Sectors moving independently (healthy)
    - High correlation (0.8-1.0): All sectors coupled (systemic stress)
    
    When correlation spikes to 1.0:
    - Signals liquidity crisis or market-wide panic
    - Everything becomes a "risk-on/risk-off" trade
    - Powerful early warning signal for regime shifts
    """
    corr_matrix = returns_slice.corr()
    # Take only upper triangle to avoid duplicates and diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    return corr_matrix.where(mask).mean().mean()

# Apply over 60-day rolling window
for i in range(len(sector_returns)):
    if i < 59:  # Not enough data for 60-day window
        systemic_scores.append(np.nan)
    else:
        window_data = sector_returns.iloc[i-59:i+1]
        systemic_scores.append(get_avg_sector_correlation(window_data))
```

#### 4. Data Cleaning
- **Missing Values:** Forward fill for minor gaps
- **Alignment:** Ensure all assets have matching dates
- **Outlier Handling:** Retain extremes (they signal regime changes)

### Output

**File:** `features_complete_2012_2018.csv`

**Columns:**
- `Date`: Trading date (index)
- `Index_Returns`: S&P 500 log returns
- `Index_Volatility`: 21-day rolling standard deviation
- `Systemic_Health_Score`: 60-day average sector correlation

**Shape:** ~1,500 rows × 3 feature columns (after dropping NaN from rolling windows)

**Data Quality:**
- First ~60 rows have NaN due to rolling window requirements
- Dropped via `dropna()` before saving
- Final dataset: 2012-04-02 to 2018-12-31

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

**Why?** Prevents high-variance features from dominating low-variance features. All three features contribute equally to regime detection.

#### 3. Hidden Markov Model Training

```python
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(
    n_components=3,              # 3 hidden states
    covariance_type='full',      # Full covariance matrices
    n_iter=1000,                 # EM algorithm iterations (increased for convergence)
    random_state=42              # Reproducibility
)

model.fit(X_train_scaled)
```

**Training Algorithm:**
1. **Initialization:** Random initialization of state parameters
2. **Expectation-Maximization (EM):**
   - E-step: Estimate state probabilities given current parameters
   - M-step: Update parameters to maximize likelihood
   - Repeat 1000 iterations until convergence
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
def label_regimes_automatically(states_summary):
    """
    Assign Growth/Correction/Crisis labels based on statistics
    
    Rules:
    1. Calculate mean return and volatility for each state
    2. Low vol + positive returns → Growth
    3. High vol (>75th percentile) → Crisis
    4. Everything else → Correction
    """
    regime_labels = {}
    
    for regime_id in range(3):
        vol = states_summary.loc[regime_id, 'Index_Volatility']
        ret = states_summary.loc[regime_id, 'Index_Returns']
        
        if vol < states_summary['Index_Volatility'].median() and ret > 0:
            regime_labels[regime_id] = 'Growth'
        elif vol > states_summary['Index_Volatility'].quantile(0.75):
            regime_labels[regime_id] = 'Crisis'
        else:
            regime_labels[regime_id] = 'Correction'
    
    return regime_labels
```

**Example Output:**
```
State 0: Growth      (mean_return: +0.0008, volatility: 0.0065)
State 1: Correction  (mean_return: -0.0003, volatility: 0.0095)
State 2: Crisis      (mean_return: -0.0001, volatility: 0.0180)
```

#### 6. Visualization

Generate cumulative returns plot colored by regime:

```python
# Calculate cumulative returns for visualization
train_df['Cumulative_Returns'] = (1 + train_df['Index_Returns']).cumprod() - 1

plt.figure(figsize=(15, 8))
for i in range(model.n_components):
    mask = train_df['Regime'] == i
    label = regime_labels.get(i, f'Regime {i}')
    plt.plot(train_df.index[mask], train_df["Cumulative_Returns"][mask], '.', 
             label=f'{label} (State {i})', alpha=0.7)

plt.title("S&P 500 Cumulative Returns by Market Regime (Training: 2012-2017)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig('regime_visualization_train.png', dpi=300)
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
- **Columns:** `Date`, `Index_Returns`, `Index_Volatility`, `Systemic_Health_Score`, `Regime`, `Regime_Label`
- **Period:** 2018 (out-of-sample test)

### Trading Strategy

#### Decision Rules

```python
def get_action_regime_based(current_regime_label):
    """
    Maps regime labels to trading actions.
    Growth → BUY (Aggressive)
    Correction → HOLD (Cautious)
    Crisis → SELL (Defensive)
    """
    if current_regime_label == 'Growth':
        return 'BUY'
    elif current_regime_label == 'Crisis':
        return 'SELL'
    else:  # Correction or unknown
        return 'HOLD'

def execute_trade(agent, action, price, weight=1):
    """
    Executes trades based on percentage of available cash.
    BUY: Invest 'weight' percentage of current cash
    SELL: Sell 50% of current holdings to de-risk
    """
    if action == 'BUY':
        amount_to_invest = agent['cash'] * weight
        shares_to_buy = amount_to_invest / price
        if agent['cash'] >= amount_to_invest:
            agent['cash'] -= amount_to_invest
            agent['portfolio']['S&P_500'] += shares_to_buy
            agent['trade_history'].append(('BUY', price, shares_to_buy))
            return True

    elif action == 'SELL':
        shares_to_sell = agent['portfolio']['S&P_500'] * 0.5  # Sell 50%
        if shares_to_sell > 0:
            agent['cash'] += shares_to_sell * price
            agent['portfolio']['S&P_500'] -= shares_to_sell
            agent['trade_history'].append(('SELL', price, shares_to_sell))
            return True
            
    return False
```

#### Rationale

| Regime | Action | Reasoning |
|--------|--------|-----------|
| **Growth** | BUY (weight% of cash) | Favorable risk/reward - capture upside with low volatility |
| **Correction** | HOLD | Short-term pullback - avoid whipsaw trades, wait for clarity |
| **Crisis** | SELL (50% of holdings) | Protect capital during systemic risk, gradual de-risking |

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
