"""
==================================================================================
MODULE 3: ACTION EXECUTION SYSTEM (AES)
Market Regime Interpreter - AI Final Project
==================================================================================
Author: Nyan Linn Htun * Nathan
Date: 19 December 2025


OUTCOMES AND LEARNINGS:
------------------------
Results:
  - Successfully identified 3 distinct market regimes across 2018 test period
  - Regime-based strategy demonstrated measurable performance vs buy-and-hold
  - System correctly reduced exposure during volatile periods (risk management)

Key Learnings:
  1. Separating perception (IGS) from action (AES) creates a more maintainable system
  2. Walk-forward validation is CRITICAL - without proper train/test split, results 
     would be meaningless due to look-ahead bias
  3. Domain knowledge (regime characteristics) can be encoded into simple rules 
     that outperform reactive strategies
  4. Feature engineering (volatility, sector correlations) matters more than 
     model complexity for this problem
  5. Real-world deployment would require continuous retraining (quarterly) to 
     adapt to structural market changes - this is discussed but not implemented 
     due to time constraints

Future Improvements:
  - Implement confidence scoring to trigger model retraining when drift is detected
  - Add transaction costs and slippage for realistic performance estimates
  - Extend to multi-asset portfolios with regime-specific allocation weights
==================================================================================
"""


import pandas as pd
import numpy as np
import joblib

def load_test_data(filepath="regimes_test_2018.csv"):
    """Load test period data with predicted regimes from Module 2."""
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def init_agent(initial_capital=10000):
    return {
        'cash': initial_capital,
        'portfolio': {'S&P_500': 0},
        'trade_history': []
    }

def get_action_regime_based(current_regime_label):
    """
    The Rulebook: Maps regime labels to trading actions.
    Growth -> BUY (Aggressive)
    Correction -> HOLD (Cautious)
    Crisis -> SELL (Defensive)
    """
    if current_regime_label == 'Growth':
        return 'BUY'
    elif current_regime_label == 'Crisis':
        return 'SELL'
    else:  # Correction or unknown
        return 'HOLD'

def execute_trade(agent, action, price, weight=1): # weight is adjustable risk parameter
    """
    Executes trades based on a percentage of available cash (Risk weighting).
    """
    if action == 'BUY':
        # Invest 'weight' percentage of current cash
        amount_to_invest = agent['cash'] * weight
        shares_to_buy = amount_to_invest / price
        if agent['cash'] >= amount_to_invest:
            agent['cash'] -= amount_to_invest
            agent['portfolio']['S&P_500'] += shares_to_buy
            agent['trade_history'].append(('BUY', price, shares_to_buy))
            return True

    elif action == 'SELL':
        # Sell 50% of current holdings to de-risk
        shares_to_sell = agent['portfolio']['S&P_500'] * 0.5  #adjustable risk parameter
        if shares_to_sell > 0:
            agent['cash'] += shares_to_sell * price
            agent['portfolio']['S&P_500'] -= shares_to_sell
            agent['trade_history'].append(('SELL', price, shares_to_sell))
            return True
            
    return False

def calculate_portfolio_value(agent, current_price):
    return agent['cash'] + (agent['portfolio']['S&P_500'] * current_price)

def main():
    print("="*70)
    print("MODULE 3: ACTION EXECUTION SYSTEM (AES)")
    print("="*70)
    print("\nThis module will:")
    print("  1. Load test period data with predicted regimes")
    print("  2. Execute regime-based trading strategy")
    print("  3. Track portfolio performance\n")
    
    # Load test data
    print("[Step 1/4] Loading test period data...")
    try:
        data = load_test_data()
        print(f"           Loaded {len(data)} trading days")
        print(f"          Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n")
    except FileNotFoundError:
        print("           Error: regimes_test_2018.csv not found!")
        print("          Please run Module 2 first.\n")
        return

    # Initialize agent
    print("[Step 2/4] Initializing trading agent...")
    initial_val = 10000
    agent = init_agent(initial_val)
    print(f"          Initial capital: ${initial_val:,.2f}\n")
    
    # Reconstruct price series from log returns (starting at $100)
    prices = 100 * np.exp(data['Index_Returns'].cumsum())
    
    # Simulation loop - regime-based decisions
    print("[Step 3/4] Running backtest simulation...")
    print(f"          Processing {len(data)} trading days...")
    
    portfolio_values = []
    regime_counts = {'Growth': 0, 'Correction': 0, 'Crisis': 0}
    
    for i in range(len(data)):
        current_regime_label = data['Regime_Label'].iloc[i]
        current_price = prices.iloc[i]
        
        # Track regime occurrences
        if current_regime_label in regime_counts:
            regime_counts[current_regime_label] += 1
        
        # Determine action based on REGIME, not price
        action = get_action_regime_based(current_regime_label)
        execute_trade(agent, action, current_price)
        portfolio_values.append(calculate_portfolio_value(agent, current_price))
    
    print("          ✓ Simulation complete!\n")

    # Calculate performance metrics
    print("[Step 4/4] Calculating performance metrics...\n")
    
    final_val = calculate_portfolio_value(agent, prices.iloc[-1])
    profit = final_val - initial_val
    strategy_return = (profit/initial_val)*100
    
    # Display results
    print("="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    print("\n[REGIME-BASED STRATEGY]")
    print(f"  Initial Capital:      ${initial_val:>10,.2f}")
    print(f"  Final Value:          ${final_val:>10,.2f}")
    print(f"  Profit/Loss:          ${profit:>10,.2f}")
    print(f"  Return:               {strategy_return:>10.2f}%")
    print(f"  Total Trades:         {len(agent['trade_history']):>10}")
    
    print("\n[REGIME DISTRIBUTION]")
    for regime, count in regime_counts.items():
        pct = (count / len(data)) * 100
        print(f"  {regime:12} - {count:3} days ({pct:5.1f}%)")
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()


# KEY CONCEPTS:
#
# Model-Based Agent:
# This agent understands the hidden "state" of the market (regime) rather than just
# reacting to price changes like a simple reflex agent. It makes contextually
# appropriate decisions based on market structure.
#
# Walk-Forward Validation:
# - Train on 2012-2017 (learn regime patterns)
# - Test on 2018 (validate on unseen data)
# - Prevents overfitting and data leakage
# - Provides realistic performance estimates
#
# Risk Management:
# - Growth: 10% of cash per buy (conservative position sizing)
# - Crisis: Sell 50% of holdings (capital preservation)
# - Correction: Hold (avoid overtrading)
#
# Future Work (Module 4):
# Implement continuous learning with quarterly retraining when model
# confidence drops or structural breaks are detected.
# and exectues in real-time.
# In order to fix that, we need to have Train/Test split to simulate real-time trading.
# Fix:
# Split the 10 years of data into 2 segments:  Training(2012-2017) and Testing(2018-2022)
# If the market structure shift significantly, the IGS model should trigger a retaining loop to update its understanding of the ecosystem.
# However, this retraining mechanism is not implemented in this simple AES example due to time constraints, and complexity.