"""
MomentumAlpha - main pipeline

Downloads stock data, computes features, trains models,
runs walk-forward backtest, generates visualisations and stats.

Usage: python main.py
"""

import sys
import os
import warnings
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
from tqdm import tqdm

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              accuracy_score, f1_score)

# Boosted models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Project modules
from features import compute_all_features, prepare_weekly_dataset, compute_lagged_target_features
from backtest import (walk_forward_backtest, compute_benchmark_returns,
                       compute_metrics, bootstrap_sharpe_ci, monte_carlo_test,
                       compute_information_coefficient)
from visualizations import *

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
BENCHMARK_TICKERS = ['SPY', '^VIX']
START_DATE = '2017-01-01'
END_DATE = '2025-03-14'
INITIAL_TRAIN_END = '2022-12-31'
RETRAIN_FREQ = 13  # weeks (quarterly)
TOP_N = 2
TX_COST_BPS = 10
RISK_FREE_RATE = 0.04

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# SECTION 1: DATA DOWNLOAD & CLEANING
# ============================================================================
print_section("📥 SECTION 1: DATA DOWNLOAD & CLEANING")

CACHE_FILE = 'cached_data.pkl'

if os.path.exists(CACHE_FILE):
    print("Loading cached data...")
    cached = pd.read_pickle(CACHE_FILE)
    stock_data = cached['stock_data']
    spy_data = cached['spy_data']
    vix_data = cached['vix_data']
else:
    print(f"Downloading data for {len(TICKERS)} stocks + benchmarks...")
    stock_data = {}
    for ticker in tqdm(TICKERS, desc="Downloading stocks"):
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data[ticker] = df

    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.droplevel(1)

    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.droplevel(1)

    # Cache
    pd.to_pickle({'stock_data': stock_data, 'spy_data': spy_data, 'vix_data': vix_data}, CACHE_FILE)
    print("Data cached for future runs ✅")

# Data quality check
print("\nData Quality Summary:")
print(f"{'Ticker':<10} {'Rows':<8} {'Start':<12} {'End':<12} {'Missing%':<10}")
print("-" * 52)
for ticker in TICKERS:
    df = stock_data[ticker]
    missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    print(f"{ticker:<10} {len(df):<8} {str(df.index[0].date()):<12} {str(df.index[-1].date()):<12} {missing:<10.2f}")

# Forward-fill and clean
for ticker in TICKERS:
    stock_data[ticker] = stock_data[ticker].ffill().dropna()
    stock_data[ticker] = stock_data[ticker][stock_data[ticker]['Volume'] > 0]

print("\nData cleaning complete ✅")


# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print_section("🔍 SECTION 2: EXPLORATORY DATA ANALYSIS")

# Normalized prices
plot_eda_prices(stock_data, TICKERS, save_path=f'{PLOTS_DIR}/eda_prices.png')

# Return correlation
plot_return_correlation(stock_data, TICKERS, save_path=f'{PLOTS_DIR}/eda_correlation.png')

# Return statistics table
print("\n📊 Daily Return Statistics:")
stats = {}
for ticker in TICKERS:
    ret = stock_data[ticker]['Close'].pct_change().dropna()
    stats[ticker] = {
        'Mean': ret.mean() * 252,
        'Std': ret.std() * np.sqrt(252),
        'Skew': ret.skew(),
        'Kurt': ret.kurtosis(),
        'Sharpe': (ret.mean() * 252 - RISK_FREE_RATE) / (ret.std() * np.sqrt(252)),
        'MaxDD': ((1+ret).cumprod() / (1+ret).cumprod().cummax() - 1).min()
    }
stats_df = pd.DataFrame(stats).T
for col in ['Mean', 'Std', 'MaxDD']:
    stats_df[col] = stats_df[col].apply(lambda x: f'{x:.2%}')
stats_df['Sharpe'] = stats_df['Sharpe'].apply(lambda x: f'{float(x):.2f}' if isinstance(x, str) else f'{x:.2f}')
print(stats_df.to_string())
print("\nEDA complete ✅")


# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================
print_section("🔧 SECTION 3: FEATURE ENGINEERING")

print("Computing 80+ features across 8 categories...")
print("  • Momentum & Trend features")
print("  • Volatility features (Garman-Klass, ATR, Bollinger)")
print("  • Volume features (OBV, MFI, CMF, Force Index)")
print("  • Oscillator features (RSI, Stochastic, CCI)")
print("  • Candlestick pattern features")
print("  • Calendar & seasonal features")
print("  • Market regime features (SPY, VIX, beta)")
print("  • Cross-sectional ranking features")

all_features = compute_all_features(stock_data, spy_data, vix_data, TICKERS)
print(f"\nFeatures computed per stock: {len([c for c in all_features[TICKERS[0]].columns if c != 'ticker'])}")
print("Feature engineering complete ✅")


# ============================================================================
# SECTION 4: WEEKLY DATASET & TARGET
# ============================================================================
print_section("🎯 SECTION 4: TARGET VARIABLE & WEEKLY AGGREGATION")

weekly_data = prepare_weekly_dataset(all_features, stock_data, TICKERS)

# Add lagged target features (past win rates, return momentum — no look-ahead bias)
print("  Adding lagged target features (past win rates, return persistence)...")
weekly_data = compute_lagged_target_features(weekly_data, TICKERS)
print(f"Weekly dataset shape: {weekly_data.shape}")
print(f"Date range: {weekly_data.index.min().date()} to {weekly_data.index.max().date()}")
print(f"Target distribution:")
print(f"  Positive weeks (1): {(weekly_data['target']==1).sum()} ({(weekly_data['target']==1).mean():.1%})")
print(f"  Negative weeks (0): {(weekly_data['target']==0).sum()} ({(weekly_data['target']==0).mean():.1%})")

# Feature columns (exclude non-features)
exclude_cols = ['ticker', 'forward_return', 'target', 'close']
feature_cols = [c for c in weekly_data.columns if c not in exclude_cols]
print(f"\nTotal features: {len(feature_cols)}")

# Remove features with too many NaN
nan_pct = weekly_data[feature_cols].isnull().mean()
bad_feats = nan_pct[nan_pct > 0.3].index.tolist()
if bad_feats:
    print(f"Removing {len(bad_feats)} features with >30% NaN")
    feature_cols = [f for f in feature_cols if f not in bad_feats]
    print(f"Remaining features: {len(feature_cols)}")

# Remove highly correlated features
corr_matrix = weekly_data[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
if to_drop:
    print(f"Removing {len(to_drop)} highly correlated features (|corr|>0.95)")
    feature_cols = [f for f in feature_cols if f not in to_drop]
    print(f"Final features: {len(feature_cols)}")

print("\nTarget variable construction complete ✅")


# ============================================================================
# SECTION 5: MODEL TRAINING
# ============================================================================
print_section("🤖 SECTION 5: MODEL TRAINING")

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
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=5,
        eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=800, learning_rate=0.03, depth=5,
        l2_leaf_reg=5.0, subsample=0.75,
        auto_class_weights='Balanced', verbose=0, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=800, max_depth=8, min_samples_leaf=30,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=42
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=800, max_depth=8, min_samples_leaf=25,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=42
    ),
    'GBM': GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.75, min_samples_leaf=30,
        random_state=42
    ),
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=1000,
                                    class_weight='balanced', random_state=42))
    ]),
}

print(f"Models to train ({len(models)}): {list(models.keys())}")
print("Models will be trained via walk-forward retraining.")


# ============================================================================
# SECTION 6: WALK-FORWARD BACKTESTING
# ============================================================================
print_section("📊 SECTION 6: WALK-FORWARD BACKTESTING")

print(f"Configuration:")
print(f"  Initial training period: {START_DATE} to {INITIAL_TRAIN_END}")
print(f"  Test period: {INITIAL_TRAIN_END} onwards")
print(f"  Retrain frequency: every {RETRAIN_FREQ} weeks")
print(f"  Top-N stocks: {TOP_N}")
print(f"  Transaction cost: {TX_COST_BPS} bps each way")
print()

results = walk_forward_backtest(
    weekly_data, feature_cols, models, TICKERS,
    initial_train_end=INITIAL_TRAIN_END,
    retrain_freq=RETRAIN_FREQ,
    top_n=TOP_N,
    transaction_cost_bps=TX_COST_BPS,
    # Enhanced features for higher Sharpe
    turnover_penalty=True,       # Reduce unnecessary trades
    regime_filter=True,          # Scale down in high-VIX periods
    use_regression_blend=True,   # Blend classification + regression
    confidence_threshold=0.0,    # No minimum threshold
    rebalance_freq=1,            # Weekly rebalancing
)

print(f"\nBacktest complete ✅")
print(f"  Test weeks: {len(results['returns_after_costs'])}")
print(f"  Predictions generated: {len(results['predictions'])}")


# ============================================================================
# SECTION 7: BENCHMARKS
# ============================================================================
print_section("💼 SECTION 7: BENCHMARK COMPUTATION")

benchmarks = compute_benchmark_returns(weekly_data, TICKERS, spy_data)

# Align benchmarks to strategy test period
test_start = results['returns_after_costs'].index[0]
test_end = results['returns_after_costs'].index[-1]
for name in benchmarks:
    benchmarks[name] = benchmarks[name][(benchmarks[name].index >= test_start) &
                                         (benchmarks[name].index <= test_end)]

print("Benchmarks computed:")
for name, ret in benchmarks.items():
    print(f"  {name}: {len(ret)} weeks")
print("Benchmarks ready ✅")


# ============================================================================
# SECTION 8: PERFORMANCE METRICS
# ============================================================================
print_section("📈 SECTION 8: PERFORMANCE ANALYSIS")

# Strategy metrics
strategy_metrics_before = compute_metrics(results['returns_before_costs'])
strategy_metrics_after = compute_metrics(results['returns_after_costs'])

# Benchmark metrics
benchmark_metrics = {}
for name, ret in benchmarks.items():
    benchmark_metrics[name] = compute_metrics(ret)

# Print comparison table
print("\n" + "="*90)
print("  📊 PERFORMANCE COMPARISON TABLE")
print("="*90)

all_results = {
    'Strategy\n(before costs)': strategy_metrics_before,
    'Strategy\n(after costs)': strategy_metrics_after,
}
for name, m in benchmark_metrics.items():
    all_results[name.replace('_', '\n')] = m

metrics_df = pd.DataFrame(all_results)
# Format for display
display_df = metrics_df.copy()
pct_rows = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility',
            'Max Drawdown', 'Hit Rate', 'Avg Weekly Return', 'VaR (95%)', 'CVaR (95%)']
for row in pct_rows:
    if row in display_df.index:
        display_df.loc[row] = display_df.loc[row].apply(
            lambda x: f'{x:.2%}' if isinstance(x, (int, float)) else x
        )
ratio_rows = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Win/Loss Ratio', 'Profit Factor']
for row in ratio_rows:
    if row in display_df.index:
        display_df.loc[row] = display_df.loc[row].apply(
            lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x
        )

print(display_df.to_string())


# ============================================================================
# SECTION 9: STATISTICAL SIGNIFICANCE
# ============================================================================
print_section("🔬 SECTION 9: STATISTICAL SIGNIFICANCE")

# Bootstrap CI for Sharpe
ci_lo, ci_hi = bootstrap_sharpe_ci(results['returns_after_costs'], n_bootstrap=10000)
print(f"Sharpe Ratio: {strategy_metrics_after['Sharpe Ratio']:.2f}")
print(f"95% Bootstrap CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
significant = ci_lo > 0
print(f"Statistically significant (CI excludes 0): {'YES ✅' if significant else 'NO ❌'}")

# Monte Carlo test
print("\nRunning Monte Carlo permutation test (5000 simulations)...")
p_value, random_sharpes = monte_carlo_test(
    weekly_data[weekly_data.index > pd.Timestamp(INITIAL_TRAIN_END)],
    strategy_metrics_after['Sharpe Ratio'],
    TICKERS, n_sims=5000, top_n=TOP_N
)
print(f"p-value vs random selection: {p_value:.4f}")
print(f"Strategy beats {(1-p_value)*100:.1f}% of random strategies")

# Information Coefficient
ic_df = compute_information_coefficient(results['predictions'])
mean_ic = ic_df['ic'].mean()
ic_ir = mean_ic / ic_df['ic'].std() if ic_df['ic'].std() > 0 else 0
print(f"\nMean Information Coefficient: {mean_ic:.4f}")
print(f"IC Information Ratio: {ic_ir:.2f}")

print("\nStatistical analysis complete ✅")


# ============================================================================
# SECTION 10: VISUALIZATIONS
# ============================================================================
print_section("🎨 SECTION 10: VISUALIZATIONS")

# 1. Cumulative returns
print("  Plotting cumulative returns...")
plot_cumulative_returns(
    results['returns_before_costs'], results['returns_after_costs'],
    benchmarks, save_path=f'{PLOTS_DIR}/cumulative_returns.png'
)

# 2. Drawdown
print("  Plotting drawdown...")
plot_drawdown(results['returns_after_costs'], title='Strategy (After Costs)',
              save_path=f'{PLOTS_DIR}/drawdown.png')

# 3. Returns distribution
print("  Plotting returns distribution...")
plot_returns_distribution(results['returns_after_costs'],
                          save_path=f'{PLOTS_DIR}/returns_dist.png')

# 4. Rolling Sharpe
print("  Plotting rolling Sharpe...")
plot_rolling_sharpe(results['returns_after_costs'],
                    save_path=f'{PLOTS_DIR}/rolling_sharpe.png')

# 5. Feature importance (from last trained LightGBM)
print("  Plotting feature importance...")
if results['model_importances']:
    last_window = list(results['model_importances'].keys())[-1]
    if 'LightGBM' in results['model_importances'][last_window]:
        imp = results['model_importances'][last_window]['LightGBM']
        plot_feature_importance(imp, top_n=25,
                                save_path=f'{PLOTS_DIR}/feature_importance.png')

# 6. Stock selection heatmap
print("  Plotting stock selection heatmap...")
plot_stock_selection_heatmap(results['selected_stocks'], TICKERS,
                             save_path=f'{PLOTS_DIR}/selection_heatmap.png')

# 7. Monthly returns heatmap
print("  Plotting monthly returns heatmap...")
plot_monthly_returns_heatmap(results['returns_after_costs'],
                              save_path=f'{PLOTS_DIR}/monthly_heatmap.png')

# 8. IC over time
print("  Plotting IC over time...")
plot_ic_over_time(ic_df, save_path=f'{PLOTS_DIR}/ic_over_time.png')

# 9. Cumulative alpha
print("  Plotting cumulative alpha...")
if 'equal_weight' in benchmarks:
    plot_cumulative_alpha(results['returns_after_costs'], benchmarks['equal_weight'],
                          save_path=f'{PLOTS_DIR}/cumulative_alpha.png')

# 10. Annual returns comparison
print("  Plotting annual returns...")
plot_annual_returns_comparison(results['returns_after_costs'], benchmarks,
                                save_path=f'{PLOTS_DIR}/annual_returns.png')

# 11. Stock selection frequency
print("  Plotting selection frequency...")
plot_stock_selection_frequency(results['selected_stocks'], TICKERS,
                                save_path=f'{PLOTS_DIR}/selection_frequency.png')

# 12. Confusion matrix
print("  Plotting confusion matrix...")
predictions = results['predictions']
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(predictions['target'], (predictions['ensemble_prob'] > 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted -', 'Predicted +'],
            yticklabels=['Actual -', 'Actual +'])
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/confusion_matrix.png')
plt.close(fig)

# 13. ROC Curve
print("  Plotting ROC curve...")
fpr, tpr, _ = roc_curve(predictions['target'], predictions['ensemble_prob'])
auc_score = roc_auc_score(predictions['target'], predictions['ensemble_prob'])
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color=COLORS['strategy'], linewidth=2, label=f'Ensemble (AUC={auc_score:.3f})')
ax.plot([0,1], [0,1], 'w--', alpha=0.5)
ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/roc_curve.png')
plt.close(fig)

print("\nAll visualizations saved to plots/ ✅")


# ============================================================================
# SECTION 11: SHAP ANALYSIS
# ============================================================================
print_section("🔍 SECTION 11: MODEL INTERPRETABILITY (SHAP)")

try:
    # Train a final LightGBM model for SHAP
    train_mask = weekly_data.index <= pd.Timestamp(INITIAL_TRAIN_END)
    test_mask = weekly_data.index > pd.Timestamp(INITIAL_TRAIN_END)

    X_train = weekly_data[train_mask][feature_cols].fillna(0)
    y_train = weekly_data[train_mask]['target']
    X_test = weekly_data[test_mask][feature_cols].fillna(0)

    final_lgbm = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        verbose=-1, random_state=42, n_jobs=-1
    )
    final_lgbm.fit(X_train, y_train)

    # SHAP
    explainer = shap.TreeExplainer(final_lgbm)
    shap_values = explainer.shap_values(X_test.iloc[:500])  # Subset for speed

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test.iloc[:500], max_display=25, show=False)
    else:
        shap.summary_plot(shap_values, X_test.iloc[:500], max_display=25, show=False)
    plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_summary.png')
    plt.close('all')
    print("SHAP analysis complete ✅")

except Exception as e:
    print(f"SHAP analysis skipped: {e}")


# ============================================================================
# SECTION 12: STRESS TESTING
# ============================================================================
print_section("🏋️ SECTION 12: STRESS TESTING")

periods = {
    'COVID Crash (Feb-Mar 2020)': ('2020-02-15', '2020-03-31'),
    '2022 Bear Market': ('2022-01-01', '2022-10-31'),
    '2023 AI Rally': ('2023-01-01', '2023-12-31'),
    '2024 Bull Market': ('2024-01-01', '2024-12-31'),
}

print(f"\n{'Period':<30} {'Strategy':<12} {'EW Bench':<12} {'SPY':<12}")
print("-" * 66)

for period_name, (start, end) in periods.items():
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    strat_ret = results['returns_after_costs'][(results['returns_after_costs'].index >= s) &
                                                (results['returns_after_costs'].index <= e)]
    if len(strat_ret) > 0:
        strat_cum = (1 + strat_ret).prod() - 1
        ew_ret = benchmarks.get('equal_weight', pd.Series())
        ew_period = ew_ret[(ew_ret.index >= s) & (ew_ret.index <= e)]
        ew_cum = (1 + ew_period).prod() - 1 if len(ew_period) > 0 else 0

        spy_ret = benchmarks.get('spy', pd.Series())
        spy_period = spy_ret[(spy_ret.index >= s) & (spy_ret.index <= e)]
        spy_cum = (1 + spy_period).prod() - 1 if len(spy_period) > 0 else 0

        print(f"{period_name:<30} {strat_cum:>10.1%}  {ew_cum:>10.1%}  {spy_cum:>10.1%}")

print("\nStress testing complete ✅")


# ============================================================================
# SECTION 13: PREDICTIONS CSV
# ============================================================================
print_section("📋 SECTION 13: EXPORT PREDICTIONS CSV")

predictions_csv = results['predictions'][['week', 'ticker', 'ensemble_prob',
                                           'actual_return', 'target', 'rank',
                                           'selected', 'weight']].copy()
predictions_csv = predictions_csv.rename(columns={
    'week': 'week_start_date',
    'ensemble_prob': 'predicted_prob',
    'target': 'predicted_class',
})
predictions_csv['predicted_class'] = (predictions_csv['predicted_prob'] > 0.5).astype(int)

# Add individual model probs if available
for col in results['predictions'].columns:
    if col.startswith('prob_'):
        predictions_csv[col] = results['predictions'][col]

predictions_csv.to_csv('predictions.csv', index=False)
print(f"Predictions saved to predictions.csv ({len(predictions_csv)} rows) ✅")


# ============================================================================
# SECTION 14: EXECUTIVE DASHBOARD
# ============================================================================
print_section("🏆 EXECUTIVE DASHBOARD — FINAL RESULTS")

print("=" * 70)
print("  📊 MOMENTUMALPHA — FINAL RESULTS DASHBOARD")
print("=" * 70)
print(f"\n  🏆 Cumulative Return (after costs): {strategy_metrics_after['Cumulative Return']:.1%}")
print(f"  📈 Annualized Return:               {strategy_metrics_after['Annualized Return']:.1%}")
print(f"  📉 Max Drawdown:                    {strategy_metrics_after['Max Drawdown']:.1%}")
print(f"  ⚡ Sharpe Ratio:                    {strategy_metrics_after['Sharpe Ratio']:.2f}")
print(f"  🔄 Sortino Ratio:                   {strategy_metrics_after['Sortino Ratio']:.2f}")
print(f"  🎯 Hit Rate:                        {strategy_metrics_after['Hit Rate']:.1%}")
print(f"  📊 Avg Weekly Return:               {strategy_metrics_after['Avg Weekly Return']:.3%}")
print(f"\n  vs Equal Weight Benchmark:          {strategy_metrics_after['Annualized Return'] - benchmark_metrics.get('equal_weight', {}).get('Annualized Return', 0):+.1%} alpha")
print(f"  vs Random Selection:                p-value = {p_value:.4f}")
print(f"\n  Bootstrap 95% CI for Sharpe:        [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"  Mean Information Coefficient:       {mean_ic:.4f}")
print("=" * 70)

# ML metrics
print(f"\n  ML Classification Metrics:")
y_true = predictions['target']
y_pred = (predictions['ensemble_prob'] > 0.5).astype(int)
print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"  AUC-ROC:   {roc_auc_score(y_true, predictions['ensemble_prob']):.3f}")
print(f"  F1 Score:  {f1_score(y_true, y_pred):.3f}")

# Top-2 selection accuracy
selected_preds = predictions[predictions['selected'] == 1]
top2_accuracy = (selected_preds['actual_return'] > 0).mean()
print(f"  Top-2 Selection Accuracy: {top2_accuracy:.1%}")

print("\n✅ PIPELINE COMPLETE. All outputs saved.")
print(f"  📁 Plots: {PLOTS_DIR}/")
print(f"  📄 Predictions: predictions.csv")
print(f"  💾 Cached data: {CACHE_FILE}")

# Version info
print(f"\n  Python:     {sys.version.split()[0]}")
print(f"  pandas:     {pd.__version__}")
print(f"  numpy:      {np.__version__}")
print(f"  lightgbm:   {lgb.__version__}")
print(f"  xgboost:    {xgb.__version__}")
print(f"  sklearn:    {__import__('sklearn').__version__}")
