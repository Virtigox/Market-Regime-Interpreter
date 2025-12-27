# Quick Start Guide - Market Regime Interpreter

## Installation (One-Time Setup)

```bash
# Install all required packages
pip install yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib
```

## Running the Project (3 Simple Steps)

### Step 1: Feature Engineering
```bash
python information_generation_system_module_1.py
```
**What it does:** Downloads S&P 500 data and calculates market indicators  
**Runtime:** ~30-60 seconds  
**Output:** `features_complete_2012_2018.csv`

---

### Step 2: Regime Detection
```bash
python information_generation_system_module_2.py
```
**What it does:** Trains HMM and identifies market regimes  
**Runtime:** ~10-20 seconds  
**Outputs:** 
- `regimes_train_2012_2017.csv`
- `regimes_test_2018.csv`
- `hmm_model.pkl`, `scaler.pkl`, `regime_labels.pkl`
- `regime_visualization_train.png`

---

### Step 3: Backtesting
```bash
python action_execution_system.py
```
**What it does:** Runs regime-based trading strategy and shows results  
**Runtime:** ~5 seconds  
**Output:** Performance metrics printed to console

---

## Expected Final Output

You should see something like:

```
======================================================================
BACKTEST RESULTS
======================================================================

[REGIME-BASED STRATEGY]
  Initial Capital:      $10,000.00
  Final Value:          $XXXXX.XX
  Profit/Loss:          $X,XXX.XX
  Return:               XX.XX%
  Total Trades:         XXX

[BUY-AND-HOLD BENCHMARK]
  Initial Capital:      $10,000.00
  Final Value:          $XXXXX.XX
  Profit/Loss:          $X,XXX.XX
  Return:               XX.XX%

[COMPARISON]
  ✓/✗ Outperformance:   X.XX%

[REGIME DISTRIBUTION]
  Growth       - XXX days (XX.X%)
  Correction   - XXX days (XX.X%)
  Crisis       - XXX days (XX.X%)
```

---

## Troubleshooting

**Problem:** "FileNotFoundError"  
**Solution:** Run modules in order (1 → 2 → 3)

**Problem:** "ModuleNotFoundError"  
**Solution:** Install missing package: `pip install [package_name]`

**Problem:** Yahoo Finance download fails  
**Solution:** Check internet connection and try again

---

**Author:** Nyan Linn Htun (Nathan)  
**Date:** December 2025
