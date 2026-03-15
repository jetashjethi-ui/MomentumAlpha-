#!/usr/bin/env python
# coding: utf-8

# # MomentumAlpha — ML-Driven Weekly Stock Selector
# ## QuantQuest Challenge | ESummit'26
# 
# **Objective:** Predict which 2 out of 10 stocks will have positive returns next week, then construct an equal-weight long-only portfolio rebalanced weekly with 0.1% transaction costs.
# 
# **Pipeline:** Data Download → Feature Engineering → Model Training → Walk-Forward Backtest → Performance Analysis

# ## 1. Setup & Imports

# In[ ]:


import sys, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score, f1_score)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from features import compute_all_features, prepare_weekly_dataset, compute_lagged_target_features
from backtest import (walk_forward_backtest, compute_benchmark_returns,
                       compute_metrics, bootstrap_sharpe_ci, monte_carlo_test,
                       compute_information_coefficient)
from visualizations import (plot_cumulative_returns, plot_drawdown, plot_returns_distribution,
                            plot_rolling_sharpe, plot_stock_selection_heatmap, plot_stock_selection_frequency,
                            plot_monthly_returns_heatmap, plot_ic_over_time, plot_cumulative_alpha,
                            plot_annual_returns_comparison, plot_feature_importance)

np.random.seed(42)
print("All imports loaded successfully.")


# ## 2. Configuration

# In[ ]:


TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
START_DATE = '2017-01-01'
END_DATE = '2025-03-14'
INITIAL_TRAIN_END = '2022-12-31'
TOP_N = 2
TX_COST_BPS = 10
RETRAIN_FREQ = 13  # weeks
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Universe: {TICKERS}")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Train: {START_DATE} to {INITIAL_TRAIN_END}")
print(f"Test: {INITIAL_TRAIN_END} onwards")
print(f"Top-N stocks: {TOP_N}, TX cost: {TX_COST_BPS} bps each way")


# ## 3. Data Download & Cleaning

# In[ ]:


# Download stock data
CACHE_FILE = 'cached_data.pkl'
if os.path.exists(CACHE_FILE):
    print("Loading cached data...")
    cached = pd.read_pickle(CACHE_FILE)
    stock_data = cached['stock_data']
    spy_data = cached['spy_data']
    vix_data = cached['vix_data']
else:
    print("Downloading from Yahoo Finance...")
    stock_data = {}
    for ticker in TICKERS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        stock_data[ticker] = data
        print(f"  {ticker}: {len(data)} days ({data.index[0].date()} to {data.index[-1].date()})")

    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    for df in [spy_data, vix_data]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Clean data
    for ticker in TICKERS:
        df = stock_data[ticker]
        df = df.ffill()
        df = df[df['Volume'] > 0]
        stock_data[ticker] = df

    pd.to_pickle({'stock_data': stock_data, 'spy_data': spy_data, 'vix_data': vix_data}, CACHE_FILE)
    print(f"Data cached to {CACHE_FILE}")

print(f"\nStocks loaded: {len(stock_data)}")
print(f"SPY: {len(spy_data)} days, VIX: {len(vix_data)} days")


# ## 4. Exploratory Data Analysis

# In[ ]:


# Normalized price chart
fig, ax = plt.subplots(figsize=(14, 7))
for ticker in TICKERS:
    prices = stock_data[ticker]['Close']
    normalised = prices / prices.iloc[0] * 100
    ax.plot(normalised.index, normalised.values, label=ticker, linewidth=1.5)
ax.set_title('Normalised Stock Prices (Base = 100)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Normalised Price')
ax.legend(ncol=5, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/eda_prices.png')
plt.show()


# In[ ]:


# Return statistics
stats = {}
for ticker in TICKERS:
    ret = stock_data[ticker]['Close'].pct_change().dropna()
    stats[ticker] = {
        'Ann. Return': f"{(1+ret.mean())**252 - 1:.1%}",
        'Ann. Vol': f"{ret.std()*np.sqrt(252):.1%}",
        'Sharpe': f"{ret.mean()/ret.std()*np.sqrt(252):.2f}",
        'Max DD': f"{((1+ret).cumprod() / (1+ret).cumprod().cummax() - 1).min():.1%}",
        'Skewness': f"{ret.skew():.2f}",
    }
print("Daily Return Statistics:")
pd.DataFrame(stats).T


# ## 5. Feature Engineering
# Computing 100+ features across 9 categories:
# - Momentum & Trend (returns, SMA ratios, MACD, ADX)
# - Volatility (Garman-Klass, ATR, Bollinger Bands)
# - Volume (OBV, MFI, CMF, Force Index)
# - Oscillators (RSI, Stochastic, CCI, Williams %R)
# - Candlestick patterns
# - Calendar & seasonal effects
# - Market regime (SPY, VIX, Beta, Correlation)
# - Cross-sectional rankings (peer comparisons)
# - Lagged target features (past win rates)

# In[ ]:


all_features = compute_all_features(stock_data, spy_data, vix_data, TICKERS)
print(f"Features per stock: {len([c for c in all_features[TICKERS[0]].columns if c != 'ticker'])}")


# ## 6. Weekly Dataset & Target Variable
# Target: binary classification — 1 if forward 1-week return > 0, else 0.

# In[ ]:


weekly_data = prepare_weekly_dataset(all_features, stock_data, TICKERS)
weekly_data = compute_lagged_target_features(weekly_data, TICKERS)

print(f"Weekly dataset shape: {weekly_data.shape}")
print(f"Date range: {weekly_data.index.min().date()} to {weekly_data.index.max().date()}")
print(f"Target distribution:")
print(f"  Positive weeks: {(weekly_data['target']==1).sum()} ({(weekly_data['target']==1).mean():.1%})")
print(f"  Negative weeks: {(weekly_data['target']==0).sum()} ({(weekly_data['target']==0).mean():.1%})")

# Feature selection
exclude_cols = ['ticker', 'forward_return', 'target', 'close']
feature_cols = [c for c in weekly_data.columns if c not in exclude_cols]

# Remove high-NaN features
nan_pct = weekly_data[feature_cols].isnull().mean()
bad_feats = nan_pct[nan_pct > 0.3].index.tolist()
if bad_feats:
    print(f"Removing {len(bad_feats)} features with >30% NaN")
    feature_cols = [f for f in feature_cols if f not in bad_feats]

# Remove highly correlated features
corr_matrix = weekly_data[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
if to_drop:
    print(f"Removing {len(to_drop)} correlated features (|corr|>0.95)")
    feature_cols = [f for f in feature_cols if f not in to_drop]

print(f"Final features: {len(feature_cols)}")


# ## 7. Model Training
# Using 7 diverse classifiers with weighted soft voting ensemble + regression blend.

# In[ ]:


models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=5,
        num_leaves=24, subsample=0.75, colsample_bytree=0.7,
        min_child_samples=30, reg_alpha=0.5, reg_lambda=1.0,
        class_weight='balanced', verbose=-1, random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=4,
        subsample=0.75, colsample_bytree=0.7,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=800, learning_rate=0.03, depth=5,
        l2_leaf_reg=5.0, subsample=0.75,
        auto_class_weights='Balanced', verbose=0, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=800, max_depth=8, min_samples_leaf=30,
        max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=800, max_depth=8, min_samples_leaf=25,
        max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'GBM': GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.75, min_samples_leaf=30, random_state=42
    ),
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=1000,
                                    class_weight='balanced', random_state=42))
    ]),
}

print(f"Models ({len(models)}): {list(models.keys())}")
print("Training via walk-forward retraining (every 13 weeks).")


# ## 8. Walk-Forward Backtesting
# Expanding-window walk-forward validation with quarterly retraining.
# Enhanced with turnover penalty, regression blend, and VIX regime filter.

# In[ ]:


results = walk_forward_backtest(
    weekly_data, feature_cols, models, TICKERS,
    initial_train_end=INITIAL_TRAIN_END,
    retrain_freq=RETRAIN_FREQ,
    top_n=TOP_N,
    transaction_cost_bps=TX_COST_BPS,
    turnover_penalty=True,
    regime_filter=True,
    use_regression_blend=True,
    confidence_threshold=0.0,
    rebalance_freq=1,
)

predictions = results['predictions']
returns_before = results['returns_before_costs']
returns_after = results['returns_after_costs']

print(f"Backtest complete: {len(returns_after)} weeks")
print(f"Predictions: {len(predictions)} stock-weeks")


# ## 9. Benchmark Computation

# In[ ]:


benchmarks = compute_benchmark_returns(weekly_data, TICKERS, spy_data)
print(f"Benchmarks computed: {list(benchmarks.keys())}")


# ## 10. Performance Metrics (Before & After Transaction Costs)

# In[ ]:


# Compute metrics for all strategies
metrics_before = compute_metrics(returns_before)
metrics_after = compute_metrics(returns_after)
metrics_ew = compute_metrics(benchmarks['equal_weight'])
metrics_spy = compute_metrics(benchmarks['spy'])
metrics_mom = compute_metrics(benchmarks['raw_momentum'])

# Performance comparison table
comparison = pd.DataFrame({
    'Strategy (Before TC)': metrics_before,
    'Strategy (After TC)': metrics_after,
    'Equal Weight': metrics_ew,
    'SPY': metrics_spy,
    'Raw Momentum': metrics_mom,
})

# Format for display
display_metrics = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility',
                   'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Hit Rate',
                   'Avg Weekly Return', 'Calmar Ratio', 'VaR (95%)', 'CVaR (95%)']
comparison_display = comparison.loc[display_metrics]

# Format percentages
pct_rows = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility',
            'Max Drawdown', 'Hit Rate', 'Avg Weekly Return', 'VaR (95%)', 'CVaR (95%)']
for row in pct_rows:
    if row in comparison_display.index:
        comparison_display.loc[row] = comparison_display.loc[row].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)

ratio_rows = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
for row in ratio_rows:
    if row in comparison_display.index:
        comparison_display.loc[row] = comparison_display.loc[row].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

print("=" * 80)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 80)
comparison_display


# ## 11. Statistical Significance Testing

# In[ ]:


# Bootstrap CI for Sharpe
sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(returns_after, n_bootstrap=10000)
print(f"Sharpe Ratio 95% CI: [{sharpe_lo:.3f}, {sharpe_hi:.3f}]")

# Monte Carlo test
strategy_sharpe = metrics_after['Sharpe Ratio']
p_value, random_sharpes = monte_carlo_test(weekly_data, strategy_sharpe, TICKERS, n_sims=5000)
print(f"Monte Carlo p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  => Statistically significant (p < 0.05)")
else:
    print("  => Not statistically significant")

# Information Coefficient
ic_df = compute_information_coefficient(predictions)
print(f"Mean IC: {ic_df['ic'].mean():.4f}")
print(f"IC > 0: {(ic_df['ic'] > 0).mean():.1%} of weeks")


# ## 12. Visualisations

# In[ ]:


# Cumulative returns
common_idx = returns_after.index
bm_ew = benchmarks['equal_weight'].reindex(common_idx).dropna()
bm_spy = benchmarks['spy'].reindex(common_idx).dropna()
bm_mom = benchmarks['raw_momentum'].reindex(common_idx).dropna()
plot_cumulative_returns(returns_before, returns_after, benchmarks)

# Other plots
plot_drawdown(returns_after)
plot_returns_distribution(returns_after, bm_ew)
plot_rolling_sharpe(returns_after)
plot_stock_selection_heatmap(results['selected_stocks'], TICKERS)
plot_stock_selection_frequency(results['selected_stocks'], TICKERS)
plot_monthly_returns_heatmap(returns_after)
plot_ic_over_time(ic_df)
plot_cumulative_alpha(returns_after, bm_ew)
plot_annual_returns_comparison(returns_after, benchmarks)

# Feature importance
if results['model_importances']:
    last_importances = list(results['model_importances'].values())[-1]
    if last_importances:
        first_model_imp = list(last_importances.values())[0]
        imp_series = pd.Series(first_model_imp).sort_values(ascending=False).head(25)
        plot_feature_importance(imp_series)

print(f"All visualisations generated.")


# ## 13. SHAP Model Interpretability

# In[ ]:


try:
    # Get training data
    train_mask = weekly_data.index <= pd.Timestamp(INITIAL_TRAIN_END)
    X_train = weekly_data[train_mask][feature_cols].fillna(0)

    lgbm_model = models['LightGBM']
    explainer = shap.TreeExplainer(lgbm_model)
    sample_idx = np.random.choice(len(X_train), min(500, len(X_train)), replace=False)
    shap_values = explainer.shap_values(X_train.iloc[sample_idx])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_train.iloc[sample_idx], max_display=20, show=False)
    plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_summary.png')
    plt.close('all')
    print("SHAP analysis complete")
except Exception as e:
    print(f"SHAP analysis skipped: {e}")


# ## 14. Predictions CSV Export

# In[ ]:


# Export weekly predictions with selected stocks and weights
export_df = predictions[['week', 'ticker', 'ensemble_prob', 'actual_return',
                          'target', 'rank', 'selected', 'weight']].copy()
export_df.columns = ['week_start_date', 'ticker', 'predicted_prob', 'actual_return',
                      'actual_target', 'rank', 'selected', 'portfolio_weight']

# Add model probabilities
prob_cols = [c for c in predictions.columns if c.startswith('prob_')]
for col in prob_cols:
    export_df[col] = predictions[col].values

export_df.to_csv('predictions.csv', index=False)
print(f"Predictions exported: {len(export_df)} rows")
print(f"Columns: {list(export_df.columns)}")
export_df.head(10)


# ## 15. Stress Testing

# In[ ]:


stress_periods = {
    'COVID Crash (Feb-Mar 2020)': ('2020-02-01', '2020-04-01'),
    '2022 Bear Market': ('2022-01-01', '2022-12-31'),
    '2023 AI Rally': ('2023-01-01', '2023-12-31'),
    '2024 Bull Market': ('2024-01-01', '2024-12-31'),
}

print("\nStress Testing Results:")
print(f"{'Period':<30} {'Return':>10} {'MaxDD':>10} {'Sharpe':>10}")
print("-" * 65)
for name, (start, end) in stress_periods.items():
    mask = (returns_after.index >= start) & (returns_after.index <= end)
    period_ret = returns_after[mask]
    if len(period_ret) > 2:
        m = compute_metrics(period_ret)
        print(f"{name:<30} {m['Cumulative Return']:>10.1%} {m['Max Drawdown']:>10.1%} {m['Sharpe Ratio']:>10.2f}")


# ## 16. Executive Summary

# In[ ]:


print("=" * 70)
print("  MOMENTUMALPHA - EXECUTIVE DASHBOARD")
print("=" * 70)
print(f"  Test period: {returns_after.index[0].date()} to {returns_after.index[-1].date()}")
print(f"  Number of test weeks: {len(returns_after)}")
print()
print("  KEY METRICS (After Transaction Costs):")
print(f"    Cumulative Return:    {metrics_after['Cumulative Return']:.1%}")
print(f"    Annualized Return:    {metrics_after['Annualized Return']:.1%}")
print(f"    Annualized Volatility:{metrics_after['Annualized Volatility']:.1%}")
print(f"    Sharpe Ratio:         {metrics_after['Sharpe Ratio']:.3f}")
print(f"    Max Drawdown:         {metrics_after['Max Drawdown']:.1%}")
print(f"    Hit Rate:             {metrics_after['Hit Rate']:.1%}")
print()
print(f"  Sharpe 95% CI:          [{sharpe_lo:.3f}, {sharpe_hi:.3f}]")
print(f"  Monte Carlo p-value:    {p_value:.4f}")
print(f"  Mean IC:                {ic_df['ic'].mean():.4f}")
print("=" * 70)

