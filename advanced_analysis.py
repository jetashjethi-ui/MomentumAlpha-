"""
Advanced Analysis Module for MomentumAlpha Strategy.
Extra analyses that differentiate from typical AI-generated projects.
Run: python advanced_analysis.py (after main.py)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR = 'plots'

# Load data
print("Loading results...")
predictions = pd.read_csv('predictions.csv')
cached = pd.read_pickle('cached_data.pkl')
stock_data = cached['stock_data']
spy_data = cached['spy_data']

# Reconstruct returns
selected = predictions[predictions['selected'] == 1]
weekly_returns = selected.groupby('week_start_date')['actual_return'].mean()
weekly_returns.index = pd.to_datetime(weekly_returns.index)

TICKERS = predictions['ticker'].unique().tolist()

# ===========================================================================
# 1. PROBABILITY CALIBRATION PLOT (Reliability Diagram)
# ===========================================================================
print("\n📊 1. Probability Calibration Analysis...")

fig, ax = plt.subplots(figsize=(8, 8))
n_bins = 10
prob_bins = np.linspace(0, 1, n_bins + 1)
actual_fractions = []
mean_predicted = []
counts = []

for i in range(n_bins):
    lo, hi = prob_bins[i], prob_bins[i + 1]
    mask = (predictions['predicted_prob'] >= lo) & (predictions['predicted_prob'] < hi)
    bin_data = predictions[mask]
    if len(bin_data) > 0:
        actual_frac = (bin_data['actual_return'] > 0).mean()
        mean_pred = bin_data['predicted_prob'].mean()
        actual_fractions.append(actual_frac)
        mean_predicted.append(mean_pred)
        counts.append(len(bin_data))

ax.plot([0, 1], [0, 1], 'w--', alpha=0.5, label='Perfect calibration')
ax.scatter(mean_predicted, actual_fractions, c='#00D4AA', s=[c/5 for c in counts],
           alpha=0.8, edgecolors='white', linewidth=1, zorder=5)
ax.plot(mean_predicted, actual_fractions, color='#00D4AA', linewidth=2,
        label='Model calibration')

ax.set_xlabel('Mean Predicted Probability', fontsize=13)
ax.set_ylabel('Actual Fraction of Positive Returns', fontsize=13)
ax.set_title('🎯 Probability Calibration (Reliability Diagram)', fontsize=16, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/calibration_plot.png')
plt.close(fig)
print("  Saved calibration_plot.png ✅")


# ===========================================================================
# 2. ALPHA-BETA DECOMPOSITION (CAPM Analysis)
# ===========================================================================
print("\n📊 2. Alpha-Beta Decomposition (CAPM)...")

spy_weekly = spy_data['Close'].resample('W-FRI').last()
spy_ret = spy_weekly.pct_change().dropna()
spy_ret = spy_ret[spy_ret.index >= weekly_returns.index[0]]
spy_ret = spy_ret[spy_ret.index <= weekly_returns.index[-1]]

common = weekly_returns.index.intersection(spy_ret.index)
strat_common = weekly_returns.reindex(common).dropna()
spy_common = spy_ret.reindex(common).dropna()
common_final = strat_common.index.intersection(spy_common.index)
strat_common = strat_common.reindex(common_final)
spy_common = spy_common.reindex(common_final)

if len(strat_common) > 10:
    from numpy.polynomial.polynomial import polyfit
    beta, alpha_weekly = np.polyfit(spy_common.values, strat_common.values, 1)
    alpha_annual = alpha_weekly * 52
    r_squared = np.corrcoef(spy_common.values, strat_common.values)[0, 1] ** 2

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(spy_common.values, strat_common.values, alpha=0.5, color='#00D4AA',
               edgecolors='white', linewidth=0.5, s=30)

    # Regression line
    x_line = np.linspace(spy_common.min(), spy_common.max(), 100)
    y_line = beta * x_line + alpha_weekly
    ax.plot(x_line, y_line, color='#FFA726', linewidth=2,
            label=f'β = {beta:.2f}, α = {alpha_annual:.1%}/yr, R² = {r_squared:.2f}')

    ax.set_xlabel('SPY Weekly Return', fontsize=13)
    ax.set_ylabel('Strategy Weekly Return', fontsize=13)
    ax.set_title('📈 CAPM Alpha-Beta Decomposition', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.axhline(y=0, color='white', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='white', alpha=0.3, linestyle='--')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/alpha_beta.png')
    plt.close(fig)

    print(f"  Beta:  {beta:.2f}")
    print(f"  Alpha: {alpha_annual:.2%} annualized")
    print(f"  R²:    {r_squared:.3f}")
    print("  Saved alpha_beta.png ✅")


# ===========================================================================
# 3. TURNOVER ANALYSIS OVER TIME
# ===========================================================================
print("\n📊 3. Turnover Analysis...")

prev_stocks = set()
turnover_data = []
for wk in sorted(predictions['week_start_date'].unique()):
    wk_sel = set(predictions[(predictions['week_start_date'] == wk) & (predictions['selected'] == 1)]['ticker'].values)
    if prev_stocks:
        changed = len(wk_sel.symmetric_difference(prev_stocks))
        turnover = changed / (2 * max(len(wk_sel), 1))
    else:
        turnover = 1.0
    turnover_data.append({'week': pd.Timestamp(wk), 'turnover': turnover})
    prev_stocks = wk_sel

turnover_df = pd.DataFrame(turnover_data).set_index('week')

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(turnover_df.index, turnover_df['turnover'],
       color=['#FF6B6B' if t > 0.5 else '#00D4AA' for t in turnover_df['turnover']],
       alpha=0.7, width=5)
rolling_turnover = turnover_df['turnover'].rolling(13).mean()
ax.plot(rolling_turnover.index, rolling_turnover.values, color='yellow',
        linewidth=2.5, label=f'13-week avg: {turnover_df["turnover"].mean():.1%}')
ax.set_title('🔄 Portfolio Turnover Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Turnover (0=no change, 1=full change)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/turnover_chart.png')
plt.close(fig)
print(f"  Avg turnover: {turnover_df['turnover'].mean():.1%}")
print("  Saved turnover_chart.png ✅")


# ===========================================================================
# 4. WIN/LOSS STREAK ANALYSIS
# ===========================================================================
print("\n📊 4. Win/Loss Streak Analysis...")

positive = (weekly_returns > 0).astype(int)
streaks = []
current_streak = 0
streak_type = None

for val in positive:
    if streak_type is None:
        streak_type = val
        current_streak = 1
    elif val == streak_type:
        current_streak += 1
    else:
        streaks.append((streak_type, current_streak))
        streak_type = val
        current_streak = 1
streaks.append((streak_type, current_streak))

win_streaks = [s[1] for s in streaks if s[0] == 1]
loss_streaks = [s[1] for s in streaks if s[0] == 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.hist(win_streaks, bins=range(1, max(win_streaks or [1])+2), color='#00D4AA',
         alpha=0.8, edgecolor='white', align='left')
ax1.set_title('Win Streaks', fontsize=14, fontweight='bold')
ax1.set_xlabel('Consecutive Winning Weeks')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

ax2.hist(loss_streaks, bins=range(1, max(loss_streaks or [1])+2), color='#FF6B6B',
         alpha=0.8, edgecolor='white', align='left')
ax2.set_title('Loss Streaks', fontsize=14, fontweight='bold')
ax2.set_xlabel('Consecutive Losing Weeks')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

fig.suptitle('🎲 Win/Loss Streak Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/streak_analysis.png')
plt.close(fig)

print(f"  Longest win streak:  {max(win_streaks) if win_streaks else 0} weeks")
print(f"  Longest loss streak: {max(loss_streaks) if loss_streaks else 0} weeks")
print(f"  Avg win streak:      {np.mean(win_streaks):.1f} weeks")
print(f"  Avg loss streak:     {np.mean(loss_streaks):.1f} weeks")
print("  Saved streak_analysis.png ✅")


# ===========================================================================
# 5. SECTOR EXPOSURE ANALYSIS
# ===========================================================================
print("\n📊 5. Sector Exposure Analysis...")

sector_map = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer', 'META': 'Technology', 'TSLA': 'Consumer',
    'JPM': 'Financials', 'V': 'Financials',
    'JNJ': 'Healthcare', 'BRK-B': 'Financials'
}

predictions['sector'] = predictions['ticker'].map(sector_map)
sector_sel = predictions[predictions['selected'] == 1].groupby(
    ['week_start_date', 'sector']).size().unstack(fill_value=0)
sector_pct = sector_sel.div(sector_sel.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
sector_pct.index = pd.to_datetime(sector_pct.index)
colors = {'Technology': '#00D4AA', 'Consumer': '#FFA726', 'Financials': '#AB47BC', 'Healthcare': '#FF6B6B'}
bottom = np.zeros(len(sector_pct))
for sector in sector_pct.columns:
    ax.bar(sector_pct.index, sector_pct[sector], bottom=bottom,
           label=sector, color=colors.get(sector, '#888888'), alpha=0.8, width=5)
    bottom += sector_pct[sector].values

ax.set_title('🏢 Sector Exposure Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Weight')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/sector_exposure.png')
plt.close(fig)

# Sector stats
print("  Selection frequency by sector:")
sector_counts = predictions[predictions['selected']==1]['sector'].value_counts()
for sector, count in sector_counts.items():
    print(f"    {sector}: {count} selections ({count/len(selected)*100:.0f}%)")
print("  Saved sector_exposure.png ✅")


# ===========================================================================
# 6. ROLLING RETURN COMPARISON (Strategy vs Benchmarks)
# ===========================================================================
print("\n📊 6. Rolling 13-week Returns...")

ew_all = predictions.groupby('week_start_date')['actual_return'].mean()
ew_all.index = pd.to_datetime(ew_all.index)

rolling_strat = weekly_returns.rolling(13).apply(lambda x: (1+x).prod()-1)
rolling_ew = ew_all.rolling(13).apply(lambda x: (1+x).prod()-1)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(rolling_strat.index, rolling_strat.values, color='#00D4AA',
        linewidth=2.5, label='Strategy')
ax.plot(rolling_ew.index, rolling_ew.values, color='#FF6B6B',
        linewidth=2, label='Equal Weight', alpha=0.7)
ax.fill_between(rolling_strat.index,
                rolling_strat.values, rolling_ew.values,
                where=rolling_strat.values >= rolling_ew.values,
                color='#00D4AA', alpha=0.2, label='Outperformance')
ax.fill_between(rolling_strat.index,
                rolling_strat.values, rolling_ew.values,
                where=rolling_strat.values < rolling_ew.values,
                color='#FF6B6B', alpha=0.2, label='Underperformance')

ax.set_title('📈 Rolling 13-Week Returns: Strategy vs Equal Weight',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('13-Week Cumulative Return')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/rolling_returns.png')
plt.close(fig)
print("  Saved rolling_returns.png ✅")


# ===========================================================================
# 7. PREDICTION CONFIDENCE ANALYSIS
# ===========================================================================
print("\n📊 7. Prediction Confidence vs Actual Return...")

fig, ax = plt.subplots(figsize=(12, 7))

# Bin predictions by confidence level
predictions['conf_bin'] = pd.cut(predictions['predicted_prob'],
                                  bins=[0, 0.35, 0.45, 0.55, 0.65, 1.0],
                                  labels=['Very Low\n(<35%)', 'Low\n(35-45%)',
                                          'Medium\n(45-55%)', 'High\n(55-65%)',
                                          'Very High\n(>65%)'])

conf_stats = predictions.groupby('conf_bin', observed=True)['actual_return'].agg(['mean', 'std', 'count'])
conf_stats['se'] = conf_stats['std'] / np.sqrt(conf_stats['count'])

bars = ax.bar(range(len(conf_stats)), conf_stats['mean'] * 100,
              yerr=conf_stats['se'] * 196,  # 95% CI
              color=['#FF6B6B', '#FFA726', '#888888', '#00D4AA', '#00D4AA'],
              alpha=0.8, edgecolor='white', capsize=5)

ax.set_xticks(range(len(conf_stats)))
ax.set_xticklabels(conf_stats.index, fontsize=11)
ax.set_title('📊 Avg Actual Return by Prediction Confidence',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Prediction Confidence Level')
ax.set_ylabel('Avg Weekly Return (%)')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add count labels
for i, (_, row) in enumerate(conf_stats.iterrows()):
    ax.text(i, row['mean']*100 + 0.1, f'n={int(row["count"])}',
            ha='center', fontsize=10, color='white')

plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/confidence_analysis.png')
plt.close(fig)
print("  Saved confidence_analysis.png ✅")


# ===========================================================================
# 8. RETURNS BY DAY OF WEEK (Seasonality)
# ===========================================================================
print("\n📊 8. Seasonality Analysis...")

predictions['week_dt'] = pd.to_datetime(predictions['week_start_date'])
predictions['month'] = predictions['week_dt'].dt.month

monthly_perf = predictions[predictions['selected']==1].groupby('month')['actual_return'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
available_months = monthly_perf.index.tolist()
ax.bar([month_names[m-1] for m in available_months],
       monthly_perf.values * 100,
       color=['#00D4AA' if v > 0 else '#FF6B6B' for v in monthly_perf.values],
       alpha=0.8, edgecolor='white')

ax.set_title('📅 Average Return by Month (Seasonality)', fontsize=16, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Weekly Return (%)')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/seasonality.png')
plt.close(fig)
print("  Saved seasonality.png ✅")


# ===========================================================================
# 9. TOP-2 PRECISION BY RETRAINING WINDOW
# ===========================================================================
print("\n📊 9. Model Performance Across Retraining Windows...")

predictions['week_dt'] = pd.to_datetime(predictions['week_start_date'])
predictions['quarter'] = predictions['week_dt'].dt.to_period('Q')

quarter_stats = []
for q in predictions['quarter'].unique():
    q_data = predictions[predictions['quarter'] == q]
    q_sel = q_data[q_data['selected'] == 1]
    if len(q_sel) > 0:
        hit = (q_sel['actual_return'] > 0).mean()
        avg_ret = q_sel['actual_return'].mean()
        quarter_stats.append({
            'quarter': str(q),
            'hit_rate': hit,
            'avg_return': avg_ret
        })

q_df = pd.DataFrame(quarter_stats)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.bar(q_df['quarter'], q_df['hit_rate'] * 100,
        color=['#00D4AA' if h > 50 else '#FF6B6B' for h in q_df['hit_rate']*100],
        alpha=0.8, edgecolor='white')
ax1.axhline(y=50, color='white', linestyle='--', alpha=0.5)
ax1.set_ylabel('Hit Rate (%)')
ax1.set_title('📈 Performance by Quarter', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(q_df['quarter'], q_df['avg_return'] * 100,
        color=['#00D4AA' if r > 0 else '#FF6B6B' for r in q_df['avg_return']],
        alpha=0.8, edgecolor='white')
ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
ax2.set_ylabel('Avg Weekly Return (%)')
ax2.set_xlabel('Quarter')
ax2.grid(True, alpha=0.3, axis='y')

plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/quarterly_performance.png')
plt.close(fig)
print("  Saved quarterly_performance.png ✅")


# ===========================================================================
# 10. PORTFOLIO CONCENTRATION ANALYSIS
# ===========================================================================
print("\n📊 10. Portfolio Concentration Analysis...")

# How often does the same stock appear consecutively?
prev_week_stocks = None
persistence = {t: 0 for t in TICKERS}
total_transitions = 0

for wk in sorted(predictions['week_start_date'].unique()):
    wk_sel = set(predictions[(predictions['week_start_date']==wk) & 
                              (predictions['selected']==1)]['ticker'].values)
    if prev_week_stocks is not None:
        for stock in wk_sel:
            if stock in prev_week_stocks:
                persistence[stock] += 1
        total_transitions += 1
    prev_week_stocks = wk_sel

fig, ax = plt.subplots(figsize=(12, 6))
sorted_persist = sorted(persistence.items(), key=lambda x: x[1], reverse=True)
stocks = [s[0] for s in sorted_persist]
values = [s[1]/max(total_transitions,1)*100 for s in sorted_persist]

bars = ax.bar(stocks, values, color='#AB47BC', alpha=0.8, edgecolor='white')
for bar, val in zip(bars, values):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('🔁 Stock Persistence (% of weeks stock stays selected)',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Stock')
ax.set_ylabel('Persistence Rate (%)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/stock_persistence.png')
plt.close(fig)
print("  Saved stock_persistence.png ✅")


# ===========================================================================
# SUMMARY
# ===========================================================================
total_plots = len([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])
print(f"\n{'='*60}")
print(f"  ✅ ADVANCED ANALYSIS COMPLETE")
print(f"  Total visualizations: {total_plots}")
print(f"  All saved to {PLOTS_DIR}/")
print(f"{'='*60}")
