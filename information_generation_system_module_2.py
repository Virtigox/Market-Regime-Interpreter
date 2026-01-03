"""
==================================================================================
MODULE 2: REGIME DETECTION & CLASSIFICATION
Market Regime Interpreter - AI Final Project
==================================================================================
Author: Nyan Linn Htun * Nathan
Date: December 2025

PROBLEM SOLVED:
---------------
Financial markets operate in hidden behavioral regimes that are not directly 
observable from prices alone. A trader sees price movements but cannot immediately 
tell if we're in a "healthy growth" period or approaching a "crisis" regime. This 
module solves the unsupervised learning problem of discovering and classifying 
these hidden market states using only observable features (returns, volatility, 
correlations).

The system identifies three distinct regimes:
  - Growth: Low volatility, low sector coupling, positive returns
  - Correction: Average volatility, rising coupling, flat/negative returns  
  - Crisis: High volatility, high sector coupling, negative returns

AI TECHNIQUE USED:
------------------
- Primary Algorithm: Hidden Markov Model (HMM) with Gaussian emissions
- Decoding Algorithm: Viterbi (for finding most probable state sequence)
- Preprocessing: StandardScaler (feature normalization)

Hidden Markov Model Details:
  - 3 hidden states (market regimes - not directly observable)
  - 3 observable features per timestep (returns, volatility, health score)
  - Gaussian emission probabilities (features assumed normally distributed)
  - Full covariance matrix (captures feature interactions)
  - 1000 EM iterations for parameter estimation

The HMM learns TWO things:
  1. Emission Probabilities: P(features | regime) - what each regime "looks like"
  2. Transition Probabilities: P(regime_t | regime_{t-1}) - how regimes evolve
  
This is superior to simple clustering because it considers temporal structure - 
a single day of high volatility doesn't mean crisis if we were just in growth regime.

ISSUES ENCOUNTERED:
-------------------
1. Feature Scale Imbalance:
   Raw features have vastly different scales (returns ~0.001, volatility ~0.01-0.1, 
   correlations ~0.5). Without standardization, HMM would be biased toward features 
   with larger numerical values. Solved with StandardScaler (mean=0, std=1) so all 
   features contribute equally to regime classification.

2. Arbitrary State Labels:
   HMM assigns arbitrary state numbers (0, 1, 2) with no inherent meaning. We don't 
   know which state is "Growth" vs "Crisis". Solved by automatically analyzing 
   feature statistics per state and labeling based on domain knowledge:
     - Low vol + positive returns → Growth
     - High vol + negative returns → Crisis
     - Everything else → Correction

3. Random Initialization Sensitivity:
   HMM uses random initialization, so different runs can converge to different 
   local optima. Fixed by setting random_state=42 for reproducibility. In production,
   we'd run multiple initializations and select best log-likelihood.

4. Walk-Forward Validation:
   Initially trained on all data (2012-2018), which caused data leakage - model 
   "knew the future" during backtesting. Fixed by proper train/test split:
   Train 2012-2017, Test 2018 (out-of-sample). This simulates real trading where 
   you only know the past.

HIGHLIGHT - INTERESTING CODE COMPONENT:
---------------------------------------
The automatic regime labeling logic is particularly interesting because it bridges 
the gap between unsupervised learning (HMM discovers patterns) and domain knowledge 
(we know what characteristics define Growth vs Crisis):

    for regime_id in range(n_states):
        vol = states_summary.loc[regime_id, 'Index_Volatility']
        ret = states_summary.loc[regime_id, 'Index_Returns']
        
        if vol < states_summary['Index_Volatility'].median() and ret > 0:
            regime_labels[regime_id] = 'Growth'
        elif vol > states_summary['Index_Volatility'].quantile(0.75):
            regime_labels[regime_id] = 'Crisis'
        else:
            regime_labels[regime_id] = 'Correction'

This makes the system interpretable - we're not just trusting a black box, we're 
verifying that discovered states align with financial theory. It's a hybrid approach: 
let ML discover patterns, then validate they make economic sense.

OUTCOMES AND LEARNINGS:
------------------------
Results:
  - Successfully trained HMM on 1,200+ training samples (2012-2017)
  - Identified 3 distinct regimes with clear statistical separation
  - Out-of-sample predictions (2018) show regime consistency without overfitting
  - Model correctly identified late-2018 volatility spike as Crisis regime

Key Learnings:
  1. Proper train/test split is NON-NEGOTIABLE in time series - data leakage 
     completely invalidates results. Always use walk-forward validation.
     
  2. Feature standardization matters more than I expected - without it, the model 
     basically ignored returns and only looked at volatility/correlation.
     
  3. HMM is well-suited for regime detection because it considers sequence context - 
     a single volatile day doesn't trigger crisis classification unless preceded by 
     deteriorating conditions.
     
  4. Unsupervised learning requires domain validation - just because HMM finds 3 
     clusters doesn't mean they're meaningful. We must verify they align with 
     financial theory (volatility patterns, correlation behavior).
     
  5. Model persistence (joblib) is crucial for production - we train once on 
     historical data, then apply to new data without retraining daily (unless 
     performance degrades).

The regime classifications from this module become the "world model" that drives 
decision-making in the Action Execution System (Module 3).
==================================================================================
"""



import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    """Main execution function for regime detection and classification."""
    print("="*70)
    print("MODULE 2: REGIME DETECTION & CLASSIFICATION")
    print("="*70)
    print("\nThis module will:")
    print("  1. Load features from Module 1")
    print("  2. Split into train (2012-2017) and test (2018)")
    print("  3. Train HMM on training data")
    print("  4. Identify and label market regimes")
    print("  5. Apply trained model to test data")
    print("  6. Save regime-labeled datasets for AES\n")
    
    # Load complete dataset from Module 1
    print("[Step 1/6] Loading feature dataset...")
    try:
        df_full = pd.read_csv("features_complete_2012_2018.csv", index_col=0, parse_dates=True)
        print(f"          ✓ Loaded {len(df_full)} samples\n")
    except FileNotFoundError:
        print("          ✗ Error: features_complete_2012_2018.csv not found!")
        print("          Please run Module 1 first.\n")
        exit(1)
    
    # Split into training and testing sets
    print("[Step 2/6] Splitting data into train/test periods...")
    train_df = df_full[df_full.index.year <= 2017]  # 2012-2017
    test_df = df_full[df_full.index.year == 2018]   # 2018
    
    print(f"          Train: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')} ({len(train_df)} samples)")
    print(f"          Test:  {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')} ({len(test_df)} samples)\n")
    
    # Prepare features for HMM
    print("[Step 3/6] Preparing features and training HMM...")
    features = train_df[["Index_Returns", "Index_Volatility", "Systemic_Health_Score"]].values
    
    # Standardize features (critical for HMM performance)
    print("          Standardizing features (mean=0, std=1)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features)
    
    # Train HMM with 3 hidden states
    print("          Training Gaussian HMM (3 states, 1000 iterations)...")
    n_states = 3
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_train)
    print("          HMM training complete!\n")
    
    # Predict regimes on training data using Viterbi algorithm
    print("[Step 4/6] Identifying market regimes on training data...")
    states_train = model.predict(X_train)
    train_df['Regime'] = states_train
    
    # Analyze characteristics of each regime
    states_summary = train_df.groupby('Regime').mean()
    print("\n          Regime Characteristics:")
    print("          " + "="*60)
    for i in range(n_states):
        vol = states_summary.loc[i, 'Index_Volatility']
        ret = states_summary.loc[i, 'Index_Returns']
        corr = states_summary.loc[i, 'Systemic_Health_Score']
        print(f"          Regime {i}: Vol={vol:.4f} | Ret={ret:.6f} | Corr={corr:.3f}")
    print("          " + "="*60)
    
    # Automatically label regimes based on characteristics
    print("\n          Assigning human-readable labels...")
    regime_labels = {}
    for regime_id in range(n_states):
        vol = states_summary.loc[regime_id, 'Index_Volatility']
        ret = states_summary.loc[regime_id, 'Index_Returns']
        
        if vol < states_summary['Index_Volatility'].median() and ret > 0:
            regime_labels[regime_id] = 'Growth'
        elif vol > states_summary['Index_Volatility'].quantile(0.75):
            regime_labels[regime_id] = 'Crisis'
        else:
            regime_labels[regime_id] = 'Correction'
    
    for regime_id, label in regime_labels.items():
        print(f"          Regime {regime_id} → '{label}'")
    
    train_df['Regime_Label'] = train_df['Regime'].map(regime_labels)
    joblib.dump(regime_labels, 'regime_labels.pkl')
    print("          ✓ Regime labeling complete!\n")
    
    # Apply trained model to TEST data (out-of-sample)
    print("[Step 5/6] Applying trained model to test period (2018)...")
    X_test = scaler.transform(test_df[["Index_Returns", "Index_Volatility", "Systemic_Health_Score"]].values)
    states_test = model.predict(X_test)
    test_df['Regime'] = states_test
    test_df['Regime_Label'] = test_df['Regime'].map(regime_labels)
    
    print("\n          Test Period Regime Distribution:")
    for label, count in test_df['Regime_Label'].value_counts().items():
        pct = (count / len(test_df)) * 100
        print(f"          {label:12} - {count:3} days ({pct:5.1f}%)")
    print("          ✓ Test predictions complete!\n")
    
    # Visualize regimes on training period
    print("[Step 6/6] Generating visualization and saving outputs...")
    train_df['Cumulative_Returns'] = (1 + train_df['Index_Returns']).cumprod() - 1
    
    plt.figure(figsize=(15, 8))
    for i in range(model.n_components):
        mask = train_df['Regime'] == i
        label = regime_labels.get(i, f'Regime {i}')
        plt.plot(train_df.index[mask], train_df["Cumulative_Returns"][mask], '.', 
                 label=f'{label} (State {i})', alpha=0.7)
    
    plt.title("S&P 500 Cumulative Returns by Market Regime (Training: 2012-2017)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Returns", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regime_visualization_train.png', dpi=300)
    print("          Visualization saved as 'regime_visualization_train.png'")
    
    # Save trained model and scaler
    joblib.dump(model, 'hmm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("          Model saved as 'hmm_model.pkl'")
    print("          Scaler saved as 'scaler.pkl'")
    
    # Save regime-labeled datasets with descriptive names
    train_output = "regimes_train_2012_2017.csv"
    test_output = "regimes_test_2018.csv"
    
    train_df.to_csv(train_output)
    test_df.to_csv(test_output)
    print(f"           Training data saved as '{train_output}'")
    print(f"          Test data saved as '{test_output}'")
    
    print("\n" + "="*70)
    print("REGIME DETECTION COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  ✓ HMM trained on {len(train_df)} samples (2012-2017)")
    print(f"  ✓ Regimes predicted for {len(test_df)} test samples (2018)")
    print(f"  ✓ Identified {n_states} market regimes: {', '.join(regime_labels.values())}")
    print(f"  ✓ All models and datasets saved")
    
    print(f"\nNext step: Run 'python action_execution_system.py'")
    print(f"           to backtest the regime-based trading strategy.\n")


if __name__ == "__main__":
    main()
