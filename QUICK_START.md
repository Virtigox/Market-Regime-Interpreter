# Quick Start Guide

Get the Market Regime Interpreter running in under 5 minutes.

---

## Installation

```bash
pip install yfinance pandas numpy hmmlearn scikit-learn matplotlib joblib
```

---

## Execution (3 Steps)

### Step 1: Feature Engineering
```bash
python information_generation_system_module_1.py
```
- **Input:** Downloads S&P 500 + sector data from Yahoo Finance
- **Output:** [features_complete_2012_2018.csv](features_complete_2012_2018.csv)
- **Runtime:** ~30-60 seconds

### Step 2: Regime Detection
```bash
python information_generation_system_module_2.py
```
- **Input:** Feature data from Step 1
- **Output:** Regime classifications, trained model files (.pkl), visualization
- **Runtime:** ~10-20 seconds

### Step 3: Backtesting
```bash
python action_execution_system.py
```
- **Input:** Test regimes from Step 2
- **Output:** Performance metrics (console)
- **Runtime:** ~5 seconds

---

## Expected Output

```
======================================================================
BACKTEST RESULTS (2018 Out-of-Sample)
======================================================================

[REGIME-BASED STRATEGY]
  Final Value:          $XX,XXX.XX
  Return:               XX.XX%
  Total Trades:         XX

[BUY-AND-HOLD BENCHMARK]
  Final Value:          $XX,XXX.XX
  Return:               XX.XX%

[REGIME DISTRIBUTION]
  Growth       - XXX days (XX.X%)
  Correction   - XXX days (XX.X%)
  Crisis       - XXX days (XX.X%)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Run steps in order (1 → 2 → 3) |
| `ModuleNotFoundError` | Install missing package: `pip install <package>` |
| Yahoo Finance timeout | Check internet connection, retry |

---

**Nyan Linn Htun (Nathan)** | December 2025
