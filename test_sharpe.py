"""Quick test: compare backtest with and without enhancements."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np; np.random.seed(42)
import pandas as pd, os

# Re-download fresh data if cache was deleted
TICKERS = ['AAPL','MSFT','GOOGL','AMZN','META','TSLA','JPM','V','JNJ','BRK-B']
CACHE = 'cached_data.pkl'
if os.path.exists(CACHE):
    print("Loading cached data...")
    cached = pd.read_pickle(CACHE)
    stock_data, spy, vix = cached['stock_data'], cached['spy_data'], cached['vix_data']
else:
    import yfinance as yf
    print("Downloading fresh data...")
    stock_data = {}
    for t in TICKERS:
        d = yf.download(t, start='2017-01-01', end='2025-03-14', progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = d.ffill(); d = d[d['Volume']>0]; stock_data[t] = d
        print(f"  {t}: {len(d)} days")
    spy = yf.download('SPY', start='2017-01-01', end='2025-03-14', progress=False)
    vix = yf.download('^VIX', start='2017-01-01', end='2025-03-14', progress=False)
    for df in [spy, vix]:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    pd.to_pickle({'stock_data': stock_data, 'spy_data': spy, 'vix_data': vix}, CACHE)

from features import compute_all_features, prepare_weekly_dataset, compute_lagged_target_features
from backtest import walk_forward_backtest, compute_metrics
import lightgbm as lgb; import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import Pipeline

print("Computing features...")
af = compute_all_features(stock_data, spy, vix, TICKERS)
wd = prepare_weekly_dataset(af, stock_data, TICKERS)
wd = compute_lagged_target_features(wd, TICKERS)

exclude = ['ticker', 'forward_return', 'target', 'close']
fc = [c for c in wd.columns if c not in exclude]
nan_pct = wd[fc].isnull().mean()
fc = [f for f in fc if nan_pct[f] <= 0.3]
corr = wd[fc].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [c for c in upper.columns if any(upper[c] > 0.95)]
fc = [f for f in fc if f not in drop]
print(f"Features: {len(fc)}")

models = {
    'LightGBM': lgb.LGBMClassifier(n_estimators=800,learning_rate=0.03,max_depth=5,num_leaves=24,subsample=0.75,colsample_bytree=0.7,min_child_samples=30,reg_alpha=0.5,reg_lambda=1.0,class_weight='balanced',verbose=-1,random_state=42,n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=800,learning_rate=0.03,max_depth=4,subsample=0.75,colsample_bytree=0.7,gamma=0.2,reg_alpha=0.5,reg_lambda=2.0,min_child_weight=5,eval_metric='logloss',verbosity=0,random_state=42,n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=800,learning_rate=0.03,depth=5,l2_leaf_reg=5.0,subsample=0.75,auto_class_weights='Balanced',verbose=0,random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=800,max_depth=8,min_samples_leaf=30,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=800,max_depth=8,min_samples_leaf=25,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=42),
    'GBM': GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,max_depth=4,subsample=0.75,min_samples_leaf=30,random_state=42),
    'LogisticRegression': Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(C=0.5,max_iter=1000,class_weight='balanced',random_state=42))]),
}

configs = [
    ("WITH ALL enhancements", True, True, True),
    ("Turnover only", True, False, False),
    ("NO enhancements (plain)", False, False, False),
]

for name, tp, rf, rb in configs:
    print(f"\n--- {name} ---")
    r = walk_forward_backtest(
        wd, fc, models, TICKERS,
        initial_train_end='2022-12-31', retrain_freq=13, top_n=2,
        transaction_cost_bps=10,
        turnover_penalty=tp, regime_filter=rf,
        use_regression_blend=rb, confidence_threshold=0.0, rebalance_freq=1,
    )
    m = compute_metrics(r['returns_after_costs'])
    mb = compute_metrics(r['returns_before_costs'])
    print(f"  Sharpe (before): {mb['Sharpe Ratio']:.3f}")
    print(f"  Sharpe (after):  {m['Sharpe Ratio']:.3f}")
    print(f"  Cumul Return:    {m['Cumulative Return']:.1%}")
    print(f"  Ann Return:      {m['Annualized Return']:.1%}")
    print(f"  Ann Vol:         {m['Annualized Volatility']:.1%}")
    print(f"  Max DD:          {m['Max Drawdown']:.1%}")
    print(f"  Hit Rate:        {m['Hit Rate']:.1%}")
    print(f"  Avg Wk Ret:      {m['Avg Weekly Return']:.3%}")
