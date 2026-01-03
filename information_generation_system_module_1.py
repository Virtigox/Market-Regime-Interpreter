"""
==================================================================================
MODULE 1: FEATURE ENGINEERING
Market Regime Interpreter - AI Final Project
==================================================================================
Author: Nyan Linn Htun * Nathan
Date: December 2025

PROBLEM SOLVED:
---------------
Raw financial time series data (prices) is too noisy and non-stationary for direct 
use in machine learning models. Market regimes are hidden patterns that cannot be 
directly observed from price movements alone. This module solves the problem by 
engineering three key features that capture market structure:
  1. Index Returns - Directional trend indicator
  2. Index Volatility - Risk/uncertainty measure
  3. Systemic Health Score - Market coupling/fragility indicator

These features transform raw market data into meaningful signals that reveal the 
underlying health and behavioral regime of the financial ecosystem.



ISSUES ENCOUNTERED:
-------------------
1. Data Quality and Missing Values:
   Yahoo Finance data sometimes has gaps or split/dividend adjustments. Solved by 
   using adjusted close prices and dropna() to remove incomplete windows.

2. Window Size Selection:
   - 21-day volatility window: Represents ~1 month of trading, balances 
     responsiveness with noise reduction
   - 60-day correlation window: Represents ~1 fiscal quarter, captures genuine 
     structural shifts without excessive short-term noise

3. Correlation Interpretation:
   Initially averaged all pairwise correlations including duplicates and 
   self-correlation (1.0 values). Fixed by using upper triangle of correlation 
   matrix only, which gives true cross-sector coupling measure.

HIGHLIGHT - INTERESTING CODE COMPONENT:
---------------------------------------
The Systemic Health Score calculation is the most interesting piece. Instead of 
just looking at individual sector returns, we measure how sectors move together:

    def get_avg_sector_correlation(returns_slice):
        corr_matrix = returns_slice.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        return corr_matrix.where(mask).mean().mean()

This captures ecosystem-level behavior:
  - Low correlation (0.2-0.4) = Healthy, decoupled market (each sector has its own dynamics)
  - High correlation (0.8-1.0) = Systemic fragility (all sectors move in lockstep)

When correlation spikes to 1.0, it signals liquidity crises or market-wide panic - 
everything becomes a single "risk-on/risk-off" trade. This is a powerful early 
warning signal that raw prices cannot provide.

OUTCOMES AND LEARNINGS:
------------------------
Results:
  - Successfully engineered 3 features for 1,500+ trading days (2012-2018)
  - Features show clear differentiation between calm and turbulent market periods
  - Systemic Health Score successfully identified  market stress periods

Key Learnings:
  1. Domain knowledge beats complexity - simple statistical measures (volatility, 
     correlation) capture market regimes better than complex indicators
  2. Time scale matters - choosing appropriate rolling windows (21/60 days) is 
     critical for signal vs noise tradeoff
  3. Feature interpretation is key - understanding WHY correlations spike (liquidity 
     crisis) guides model design and validation
  4. Data pipeline robustness - handling missing values and market holidays properly 
     prevents downstream errors in ML models
  5. Log returns are superior to simple returns for ML - more normally distributed, 
     additive across time, and bounded

The features created here directly enable the HMM to detect meaningful market regimes 
rather than just fitting noise.

"""



import yfinance as yf  # For downloading financial data
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations better performance


def get_avg_sector_correlation(returns_slice):
    """Calculate average cross-correlation between sectors."""
    corr_matrix = returns_slice.corr()
    # Take only upper triangle to avoid duplicates and self-correlation
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    return corr_matrix.where(mask).mean().mean()


def main():
    """Main execution function for feature engineering."""
    print("="*70)
    print("MODULE 1: FEATURE ENGINEERING")
    print("="*70)
    print("\nThis module will:"
          "\n  1. Download S&P 500 and sector ETF data (2012-2018)"
          "\n  2. Calculate daily log returns"
          "\n  3. Calculate 21-day rolling volatility"
          "\n  4. Calculate 60-day systemic health score (sector correlations)"
          "\n  5. Save complete feature dataset for Module 2\n")
    
    # Tickers
    index_ticker = "^GSPC"  # S&P 500 Index
    sector_tickers = [
        "XLK",  # Technology
        "XLF",  # Financials
        "XLV",  # Health Care
        "XLE",  # Energy
        "XLI"   # Industrials
    ]
    all_tickers = [index_ticker] + sector_tickers
    
    print(f"Tickers to download: {', '.join(all_tickers)}\n")
    print("[Step 1/4] Downloading historical data from Yahoo Finance...")
    raw_data = yf.download(all_tickers, start="2012-01-01", end="2018-12-31", auto_adjust=False)
    
    # Handle both single and multi-ticker data structures
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data["Adj Close"]
    else:
        data = raw_data[["Adj Close"]].rename(columns={"Adj Close": all_tickers[0]}) if len(all_tickers) == 1 else raw_data["Adj Close"]
    
    print("          Data download complete!\n")
    
    print("[Step 2/4] Calculating daily log returns...")
    log_returns = np.log(data / data.shift(1)).dropna()
    
    print("          Log returns calculated!\n")
    
    print("[Step 3/4] Calculating index volatility (21-day rolling window)...")
    
    features = pd.DataFrame(index=log_returns.index)
    features["Index_Returns"] = log_returns[index_ticker]
    features["Index_Volatility"] = log_returns[index_ticker].rolling(window=21).std()
    
    print("          Volatility calculated!\n")
    
    print("[Step 4/4] Calculating systemic health score (60-day rolling correlations)...")
    print("          This measures how coupled/decoupled the market sectors are.")
    # Apply the function over a rolling window of 60 days (approx. 3 months of trading days)
    # Calculate correlation for each 60-day window
    sector_returns = log_returns[sector_tickers]
    systemic_scores = []
    for i in range(len(sector_returns)):
        if i < 59:  # Not enough data for 60-day window
            systemic_scores.append(np.nan)
        else:
            window_data = sector_returns.iloc[i-59:i+1]
            systemic_scores.append(get_avg_sector_correlation(window_data))
    
    features["Systemic_Health_Score"] = systemic_scores 
    features = features.dropna()
    
    print("          Systemic health score calculated!\n")
    print("="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  Period: {features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total samples: {len(features)}")
    print(f"  Features: Index_Returns, Index_Volatility, Systemic_Health_Score")
    
    print(f"\nSample data (first 5 rows):")
    print(features.head())
    print(f"\nSample data (last 5 rows):")
    print(features.tail())
    
    # Save with descriptive filename
    output_file = "features_complete_2012_2018.csv"
    features.to_csv(output_file)
    print(f"\n✓ Features saved to '{output_file}'")
    print(f"\nNext step: Run 'python information_generation_system_module_2.py'")
    print(f"           to train HMM on 2012-2017 and predict regimes for 2018.\n")


if __name__ == "__main__":
    main()


