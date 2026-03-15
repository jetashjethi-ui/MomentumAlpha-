"""
Backtesting engine - walk-forward retraining loop, portfolio construction,
transaction cost modelling, and benchmark comparisons.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_1samp, ttest_ind


def walk_forward_backtest(weekly_data, feature_cols, models, tickers,
                          initial_train_end='2022-12-31',
                          retrain_freq=13,  # weeks
                          top_n=2,
                          transaction_cost_bps=10,
                          # --- NEW PARAMETERS ---
                          turnover_penalty=True,
                          regime_filter=True,
                          use_regression_blend=True,
                          confidence_threshold=0.0,
                          rebalance_freq=1,  # 1=weekly, 2=biweekly
                          ):
    """
    Walk-forward backtesting with periodic retraining.
    Enhanced with turnover penalty, regime filter, and regression blend.
    """
    tc = transaction_cost_bps / 10000

    # Get unique weeks
    all_weeks = weekly_data.index.unique().sort_values()
    train_end = pd.Timestamp(initial_train_end)
    test_weeks = all_weeks[all_weeks > train_end]

    if len(test_weeks) == 0:
        raise ValueError("No test weeks found after initial_train_end")

    predictions_list = []
    portfolio_returns_before = []
    portfolio_returns_after = []
    selected_stocks_history = []
    model_importances = {}
    weeks_since_retrain = 0
    trained_models = {}
    regression_models = {}

    prev_selected = set()
    prev_probs = {}  # Store previous probabilities for turnover penalty

    for i, week in enumerate(test_weeks):
        # Check if we need to retrain
        if i == 0 or weeks_since_retrain >= retrain_freq:
            # Training data: all data up to this week
            train_mask = weekly_data.index <= week - pd.Timedelta(days=1)
            train_data = weekly_data[train_mask].dropna(subset=feature_cols + ['target'])

            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['target']
            y_train_reg = train_data['forward_return']  # For regression

            # Train all classification models
            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    trained_models[name] = model

                    # Store feature importances if available
                    if hasattr(model, 'feature_importances_'):
                        if week not in model_importances:
                            model_importances[week] = {}
                        model_importances[week][name] = dict(
                            zip(feature_cols, model.feature_importances_)
                        )
                except Exception as e:
                    print(f"  Warning: {name} training failed: {e}")

            # Train regression models for return prediction (blend)
            if use_regression_blend:
                from sklearn.ensemble import GradientBoostingRegressor
                from lightgbm import LGBMRegressor

                regression_models = {}
                try:
                    reg_lgbm = LGBMRegressor(
                        n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        verbose=-1, random_state=42, n_jobs=-1
                    )
                    # Cap extreme returns to reduce noise
                    y_reg_capped = y_train_reg.clip(
                        y_train_reg.quantile(0.02), y_train_reg.quantile(0.98)
                    )
                    reg_lgbm.fit(X_train, y_reg_capped)
                    regression_models['LGBMReg'] = reg_lgbm
                except:
                    pass

            weeks_since_retrain = 0

        weeks_since_retrain += 1

        # Skip non-rebalance weeks
        if rebalance_freq > 1 and i % rebalance_freq != 0:
            # Hold previous positions
            if selected_stocks_history:
                last_stocks = selected_stocks_history[-1]['stocks']
                week_data = weekly_data[weekly_data.index == week]
                if len(week_data) >= len(tickers):
                    held = week_data[week_data['ticker'].isin(last_stocks)]
                    if len(held) > 0:
                        port_ret = held['forward_return'].mean()
                        portfolio_returns_before.append({'week': week, 'return': port_ret})
                        portfolio_returns_after.append({
                            'week': week, 'return': port_ret,  # No costs on hold
                            'turnover': 0, 'n_new': 0, 'n_closed': 0
                        })
                        selected_stocks_history.append({
                            'week': week, 'stocks': last_stocks
                        })
            continue

        # Get this week's data for all stocks
        week_data = weekly_data[weekly_data.index == week]

        if len(week_data) < len(tickers):
            continue

        X_test = week_data[feature_cols].fillna(0)

        # === MARKET REGIME FILTER ===
        regime_scale = 1.0
        if regime_filter and 'vix_level' in feature_cols:
            avg_vix = week_data['vix_level'].mean()
            # If VIX is extremely high (>35), reduce conviction
            if avg_vix > 35:
                regime_scale = 0.5  # Half conviction in extreme fear
            elif avg_vix > 28:
                regime_scale = 0.75

        # Get predictions from all classification models
        model_probs = {}
        for name, model in trained_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)[:, 1]
                else:
                    probs = model.predict(X_test)
                model_probs[name] = probs
            except:
                pass

        if not model_probs:
            continue

        # Classification ensemble
        weights = _get_model_weights(list(model_probs.keys()))
        clf_prob = np.zeros(len(X_test))
        total_weight = 0
        for name, probs in model_probs.items():
            w = weights.get(name, 1.0)
            clf_prob += w * probs
            total_weight += w
        clf_prob /= total_weight

        # === REGRESSION BLEND ===
        if use_regression_blend and regression_models:
            reg_preds = np.zeros(len(X_test))
            n_reg = 0
            for name, reg_model in regression_models.items():
                try:
                    reg_preds += reg_model.predict(X_test)
                    n_reg += 1
                except:
                    pass

            if n_reg > 0:
                reg_preds /= n_reg
                # Convert regression predictions to rank scores (0-1)
                reg_rank = pd.Series(reg_preds).rank(pct=True).values

                # Blend: 60% classification + 40% regression rank
                ensemble_prob = 0.6 * clf_prob + 0.4 * reg_rank
            else:
                ensemble_prob = clf_prob
        else:
            ensemble_prob = clf_prob

        # === TURNOVER PENALTY ===
        if turnover_penalty and prev_probs:
            week_tickers_list = list(week_data['ticker'].values)
            for j, ticker in enumerate(week_tickers_list):
                if ticker in prev_selected:
                    # Strong stickiness bonus — only switch if replacement is much better
                    # This dramatically reduces turnover and transaction costs
                    ensemble_prob[j] += 0.08

        # Create prediction record
        week_tickers = week_data['ticker'].values
        week_returns = week_data['forward_return'].values

        pred_df = pd.DataFrame({
            'week': week,
            'ticker': week_tickers,
            'ensemble_prob': ensemble_prob,
            'actual_return': week_returns,
            'target': week_data['target'].values,
        })

        # Add individual model probabilities
        for name, probs in model_probs.items():
            pred_df[f'prob_{name}'] = probs

        # Rank and select top N
        pred_df['rank'] = pred_df['ensemble_prob'].rank(ascending=False).astype(int)

        # === CONFIDENCE THRESHOLD ===
        if confidence_threshold > 0:
            # Only select if probability exceeds threshold
            eligible = pred_df[pred_df['ensemble_prob'] >= confidence_threshold]
            if len(eligible) >= top_n:
                pred_df['selected'] = ((pred_df['rank'] <= top_n) &
                                       (pred_df['ensemble_prob'] >= confidence_threshold)).astype(int)
            else:
                pred_df['selected'] = (pred_df['rank'] <= top_n).astype(int)
        else:
            pred_df['selected'] = (pred_df['rank'] <= top_n).astype(int)

        n_selected = pred_df['selected'].sum()
        pred_df['weight'] = pred_df['selected'] / max(n_selected, 1)

        predictions_list.append(pred_df)

        # Portfolio return (before costs)
        selected = pred_df[pred_df['selected'] == 1]
        port_ret_before = selected['actual_return'].mean()

        # Apply regime scaling (reduce exposure in high-vol regimes)
        if regime_scale < 1.0:
            # Partial investment: rest in "cash" (0% return)
            port_ret_before = port_ret_before * regime_scale

        portfolio_returns_before.append({'week': week, 'return': port_ret_before})

        # Transaction costs
        current_selected = set(selected['ticker'].values)
        new_positions = current_selected - prev_selected
        closed_positions = prev_selected - current_selected

        # Cost = entry cost for new + exit cost for closed
        entry_cost = tc * len(new_positions) / max(n_selected, 1)
        exit_cost = tc * len(closed_positions) / max(len(prev_selected), 1)
        total_cost = entry_cost + exit_cost

        port_ret_after = port_ret_before - total_cost
        portfolio_returns_after.append({
            'week': week,
            'return': port_ret_after,
            'turnover': (len(new_positions) + len(closed_positions)) / (2 * max(n_selected, 1)),
            'n_new': len(new_positions),
            'n_closed': len(closed_positions)
        })

        selected_stocks_history.append({
            'week': week,
            'stocks': list(current_selected)
        })

        # Store for next iteration
        prev_selected = current_selected
        prev_probs = dict(zip(week_tickers, ensemble_prob))

    # Combine results
    predictions = pd.concat(predictions_list, ignore_index=True)
    ret_before = pd.DataFrame(portfolio_returns_before).set_index('week')['return']
    ret_after = pd.DataFrame(portfolio_returns_after).set_index('week')

    return {
        'predictions': predictions,
        'returns_before_costs': ret_before,
        'returns_after_costs': ret_after['return'],
        'turnover': ret_after['turnover'],
        'selected_stocks': selected_stocks_history,
        'model_importances': model_importances,
        'test_weeks': test_weeks,
    }


def _get_model_weights(model_names):
    """Default ensemble weights — boosted models get more weight."""
    default_weights = {
        'LightGBM': 0.22,
        'XGBoost': 0.20,
        'CatBoost': 0.20,
        'RandomForest': 0.10,
        'ExtraTrees': 0.10,
        'GBM': 0.10,
        'LogisticRegression': 0.08,
    }
    weights = {}
    for name in model_names:
        weights[name] = default_weights.get(name, 0.10)
    return weights


def compute_benchmark_returns(weekly_data, tickers, spy_data):
    """Compute benchmark portfolio returns."""
    benchmarks = {}

    # 1. Equal-weight all 10 stocks
    all_weeks = weekly_data.index.unique().sort_values()
    ew_returns = []
    for week in all_weeks:
        week_data = weekly_data[weekly_data.index == week]
        if len(week_data) >= len(tickers):
            ew_ret = week_data['forward_return'].mean()
            ew_returns.append({'week': week, 'return': ew_ret})
    benchmarks['equal_weight'] = pd.DataFrame(ew_returns).set_index('week')['return']

    # 2. SPY buy-and-hold
    spy_weekly = spy_data['Close'].resample('W-FRI').last()
    spy_ret = spy_weekly.pct_change().shift(-1).dropna()
    benchmarks['spy'] = spy_ret

    # 3. Raw momentum baseline (top 2 by past 4-week return, no ML)
    mom_returns = []
    for week in all_weeks:
        week_data = weekly_data[weekly_data.index == week]
        if len(week_data) >= len(tickers) and 'return_20d' in week_data.columns:
            top2 = week_data.nlargest(2, 'return_20d')
            mom_ret = top2['forward_return'].mean()
            mom_returns.append({'week': week, 'return': mom_ret})
    benchmarks['raw_momentum'] = pd.DataFrame(mom_returns).set_index('week')['return']

    return benchmarks


def compute_metrics(returns, risk_free_rate=0.04, periods_per_year=52):
    """Compute comprehensive performance metrics."""
    returns = returns.dropna()
    n_periods = len(returns)
    if n_periods == 0:
        return {}

    # Cumulative return
    cum_return = (1 + returns).prod() - 1

    # Annualized return
    years = n_periods / periods_per_year
    ann_return = (1 + cum_return) ** (1 / years) - 1 if years > 0 else 0

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else 0

    # Sortino ratio
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year)
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Max drawdown
    cum_curve = (1 + returns).cumprod()
    rolling_max = cum_curve.cummax()
    drawdown = cum_curve / rolling_max - 1
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = (returns > 0).mean()

    # Average weekly return
    avg_return = returns.mean()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win/Loss ratio
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_loss = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0

    # Profit factor
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0

    # VaR and CVaR (95%)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    return {
        'Cumulative Return': cum_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Hit Rate': hit_rate,
        'Avg Weekly Return': avg_return,
        'Calmar Ratio': calmar,
        'Win/Loss Ratio': win_loss,
        'Profit Factor': profit_factor,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Num Weeks': n_periods,
    }


def bootstrap_sharpe_ci(returns, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    returns = returns.dropna().values
    sharpes = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        if sample.std() > 0:
            s = sample.mean() / sample.std() * np.sqrt(52)
            sharpes.append(s)
    alpha = (1 - ci) / 2
    return np.percentile(sharpes, alpha * 100), np.percentile(sharpes, (1 - alpha) * 100)


def monte_carlo_test(weekly_data, strategy_sharpe, tickers, n_sims=5000, top_n=2):
    """Monte Carlo permutation test: is strategy better than random selection?"""
    all_weeks = weekly_data.index.unique().sort_values()
    # Build return matrix: weeks x stocks
    return_matrix = weekly_data.pivot_table(
        values='forward_return', index=weekly_data.index, columns='ticker'
    )
    return_matrix = return_matrix.dropna()

    random_sharpes = []
    for _ in range(n_sims):
        random_returns = []
        for _, row in return_matrix.iterrows():
            picks = np.random.choice(len(tickers), top_n, replace=False)
            random_returns.append(row.iloc[picks].mean())
        random_returns = np.array(random_returns)
        if random_returns.std() > 0:
            rs = random_returns.mean() / random_returns.std() * np.sqrt(52)
            random_sharpes.append(rs)

    p_value = np.mean(np.array(random_sharpes) >= strategy_sharpe)
    return p_value, random_sharpes


def compute_information_coefficient(predictions):
    """Compute weekly Information Coefficient (Spearman correlation)."""
    weekly_ics = []
    for week in predictions['week'].unique():
        week_data = predictions[predictions['week'] == week]
        if len(week_data) >= 5:
            ic, _ = spearmanr(week_data['ensemble_prob'], week_data['actual_return'])
            if not np.isnan(ic):
                weekly_ics.append({'week': week, 'ic': ic})
    ic_df = pd.DataFrame(weekly_ics).set_index('week')
    return ic_df
