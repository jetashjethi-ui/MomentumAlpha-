"""
Visualization Module for MomentumAlpha Strategy.
Professional dark-theme charts for hackathon presentation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set professional dark theme
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

COLORS = {
    'strategy': '#00D4AA',
    'benchmark': '#FF6B6B',
    'spy': '#FFA726',
    'momentum': '#AB47BC',
    'positive': '#00D4AA',
    'negative': '#FF6B6B',
}


def plot_cumulative_returns(strategy_before, strategy_after, benchmarks, save_path=None):
    """Plot cumulative returns: strategy vs all benchmarks."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Align all series to common index
    common_idx = strategy_after.index

    cum_before = (1 + strategy_before.reindex(common_idx)).cumprod()
    cum_after = (1 + strategy_after.reindex(common_idx)).cumprod()

    ax.plot(cum_before.index, cum_before.values, color=COLORS['strategy'],
            linewidth=2.5, label='Strategy (before costs)', alpha=0.5, linestyle='--')
    ax.plot(cum_after.index, cum_after.values, color=COLORS['strategy'],
            linewidth=2.5, label='Strategy (after costs)')

    colors = [COLORS['benchmark'], COLORS['spy'], COLORS['momentum']]
    for (name, ret), color in zip(benchmarks.items(), colors):
        ret_aligned = ret.reindex(common_idx).dropna()
        cum = (1 + ret_aligned).cumprod()
        ax.plot(cum.index, cum.values, color=color, linewidth=1.5,
                label=name.replace('_', ' ').title(), alpha=0.8)

    ax.set_title('📈 Cumulative Returns: Strategy vs Benchmarks', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='white', linestyle=':', alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_drawdown(returns, title='Strategy', save_path=None):
    """Plot underwater/drawdown chart."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color=COLORS['negative'], alpha=0.5)
    ax.plot(drawdown.index, drawdown.values, color=COLORS['negative'], linewidth=1)

    ax.set_title(f'📉 Drawdown Chart — {title}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.annotate(f'Max DD: {max_dd_val:.1%}',
                xy=(max_dd_idx, max_dd_val),
                xytext=(30, 30), textcoords='offset points',
                fontsize=12, color=COLORS['negative'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['negative']))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_returns_distribution(returns, save_path=None):
    """Plot weekly returns distribution histogram."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(returns.values, bins=40, color=COLORS['strategy'], alpha=0.7,
            edgecolor='white', linewidth=0.5, density=True)

    # Add KDE
    from scipy.stats import norm
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, norm.pdf(x, returns.mean(), returns.std()),
            color=COLORS['spy'], linewidth=2, label='Normal fit')

    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.axvline(x=returns.mean(), color=COLORS['positive'], linestyle='-',
               alpha=0.8, label=f'Mean: {returns.mean():.3%}')

    ax.set_title('📊 Weekly Returns Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Weekly Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_rolling_sharpe(returns, window=26, save_path=None):
    """Plot rolling Sharpe ratio."""
    rolling_ret = returns.rolling(window).mean()
    rolling_vol = returns.rolling(window).std()
    rolling_sharpe = (rolling_ret / rolling_vol) * np.sqrt(52)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=COLORS['strategy'], linewidth=2)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values >= 0,
                    color=COLORS['positive'], alpha=0.3)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values < 0,
                    color=COLORS['negative'], alpha=0.3)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)

    ax.set_title(f'⚡ Rolling Sharpe Ratio ({window}-week window)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_feature_importance(importances, top_n=25, save_path=None):
    """Plot feature importance bar chart."""
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n])

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(list(reversed(sorted_imp.keys())),
                   list(reversed(sorted_imp.values())),
                   color=COLORS['strategy'], alpha=0.8, edgecolor='white', linewidth=0.5)

    ax.set_title(f'🔍 Top {top_n} Feature Importance (LightGBM)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_stock_selection_heatmap(selected_stocks, tickers, save_path=None):
    """Heatmap of stock selection frequency over time."""
    records = []
    for entry in selected_stocks:
        week = entry['week']
        for t in tickers:
            records.append({'week': week, 'ticker': t,
                           'selected': 1 if t in entry['stocks'] else 0})

    df = pd.DataFrame(records)
    pivot = df.pivot_table(values='selected', index='ticker', columns='week', fill_value=0)

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Selected'},
                linewidths=0.1, ax=ax)

    ax.set_title('🎯 Stock Selection Heatmap Over Time',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Week')
    ax.set_ylabel('Stock')

    # Reduce x-axis labels
    n_labels = 15
    step = max(1, len(pivot.columns) // n_labels)
    ax.set_xticks(range(0, len(pivot.columns), step))
    ax.set_xticklabels([str(d.date()) for d in pivot.columns[::step]], rotation=45)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_monthly_returns_heatmap(returns, save_path=None):
    """Calendar-style monthly returns heatmap."""
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    pivot = monthly_df.pivot_table(values='return', index='year', columns='month')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0,
                linewidths=1, ax=ax, cbar_kws={'label': 'Monthly Return'})

    ax.set_title('📅 Monthly Returns Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_model_comparison(model_metrics, save_path=None):
    """Bar chart comparing model performance."""
    df = pd.DataFrame(model_metrics).T
    metrics_to_show = ['Sharpe Ratio', 'Hit Rate', 'Max Drawdown', 'Annualized Return']
    available = [m for m in metrics_to_show if m in df.columns]

    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    palette = sns.color_palette('viridis', len(df))

    for i, metric in enumerate(available):
        ax = axes[i]
        bars = ax.bar(df.index, df[metric], color=palette, alpha=0.8)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('🤖 Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_ic_over_time(ic_df, save_path=None):
    """Plot Information Coefficient over time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(ic_df.index, ic_df['ic'],
           color=[COLORS['positive'] if v > 0 else COLORS['negative'] for v in ic_df['ic']],
           alpha=0.6, width=5)

    # Rolling IC
    rolling_ic = ic_df['ic'].rolling(13).mean()
    ax.plot(rolling_ic.index, rolling_ic.values, color='yellow',
            linewidth=2.5, label=f'13-week Rolling IC (avg={ic_df["ic"].mean():.3f})')

    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.set_title('📊 Information Coefficient Over Time',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('IC (Spearman Correlation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_cumulative_alpha(strategy_returns, benchmark_returns, save_path=None):
    """Plot cumulative alpha (strategy - benchmark) over time."""
    common = strategy_returns.index.intersection(benchmark_returns.index)
    alpha = strategy_returns.reindex(common) - benchmark_returns.reindex(common)
    cum_alpha = alpha.cumsum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(cum_alpha.index, cum_alpha.values, 0,
                    where=cum_alpha.values >= 0, color=COLORS['positive'], alpha=0.4)
    ax.fill_between(cum_alpha.index, cum_alpha.values, 0,
                    where=cum_alpha.values < 0, color=COLORS['negative'], alpha=0.4)
    ax.plot(cum_alpha.index, cum_alpha.values, color='white', linewidth=2)

    ax.set_title('🏆 Cumulative Alpha vs Equal-Weight Benchmark',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Alpha')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_annual_returns_comparison(strategy, benchmarks, save_path=None):
    """Bar chart of annual returns: strategy vs benchmarks."""
    annual_strategy = strategy.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    years = annual_strategy.index.year
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(years))
    width = 0.2
    ax.bar(x - width, annual_strategy.values, width, label='Strategy',
           color=COLORS['strategy'], alpha=0.8)

    for i, (name, ret) in enumerate(benchmarks.items()):
        annual_bm = ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        annual_bm = annual_bm.reindex(annual_strategy.index, fill_value=0)
        colors = [COLORS['benchmark'], COLORS['spy'], COLORS['momentum']]
        ax.bar(x + width * i, annual_bm.values, width,
               label=name.replace('_', ' ').title(),
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_title('📊 Annual Returns Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Return')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_stock_selection_frequency(selected_stocks, tickers, save_path=None):
    """Bar chart of how often each stock was selected."""
    counts = {t: 0 for t in tickers}
    total_weeks = len(selected_stocks)
    for entry in selected_stocks:
        for stock in entry['stocks']:
            if stock in counts:
                counts[stock] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    bars = ax.bar(sorted_counts.keys(), sorted_counts.values(),
                  color=COLORS['strategy'], alpha=0.8, edgecolor='white')

    # Add percentage labels
    for bar, val in zip(bars, sorted_counts.values()):
        pct = val / total_weeks * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{pct:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('🎯 Stock Selection Frequency', fontsize=16, fontweight='bold')
    ax.set_xlabel('Stock')
    ax.set_ylabel('Number of Weeks Selected')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_eda_prices(stock_data, tickers, save_path=None):
    """EDA: Normalized price chart of all stocks."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker in tickers:
        prices = stock_data[ticker]['Close']
        normalized = prices / prices.iloc[0] * 100
        ax.plot(normalized.index, normalized.values, linewidth=1.5, label=ticker)

    ax.set_title('📈 Normalized Stock Prices (Base = 100)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    ax.legend(loc='upper left', ncol=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_return_correlation(stock_data, tickers, save_path=None):
    """EDA: Correlation matrix of stock returns."""
    returns = pd.DataFrame({t: stock_data[t]['Close'].pct_change() for t in tickers}).dropna()

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)

    ax.set_title('🔗 Return Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def create_metrics_table(strategy_metrics, benchmark_metrics_dict):
    """Create a styled metrics comparison table."""
    all_metrics = {'Strategy (after costs)': strategy_metrics}
    all_metrics.update(benchmark_metrics_dict)

    df = pd.DataFrame(all_metrics)

    # Format
    pct_metrics = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility',
                   'Max Drawdown', 'Hit Rate', 'Avg Weekly Return', 'VaR (95%)', 'CVaR (95%)']
    for metric in pct_metrics:
        if metric in df.index:
            df.loc[metric] = df.loc[metric].apply(lambda x: f'{x:.2%}' if isinstance(x, (int, float)) else x)

    ratio_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                     'Win/Loss Ratio', 'Profit Factor']
    for metric in ratio_metrics:
        if metric in df.index:
            df.loc[metric] = df.loc[metric].apply(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)

    return df
