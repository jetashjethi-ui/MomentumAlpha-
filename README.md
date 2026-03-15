# MomentumAlpha

Weekly stock selection strategy using ensemble ML models.

## What it does

Predicts which 2 out of 10 large-cap stocks will have positive returns next week, using 7 classification models (LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, GBM, Logistic Regression) blended with a regression ranker.

## How to run

```
pip install -r requirements.txt
python main.py
python advanced_analysis.py
```

`main.py` runs the full pipeline (data → features → models → backtest → plots → stats).  
`advanced_analysis.py` generates additional charts (calibration, CAPM alpha, sector exposure, etc).

## Files

| File | What it does |
|------|-------------|
| `main.py` | Orchestrates everything: data download, feature engineering, model training, walk-forward backtest, visualization, SHAP analysis |
| `features.py` | Computes 100+ technical features across 9 categories |
| `backtest.py` | Walk-forward engine with turnover penalty, regression blend, VIX regime filter |
| `visualizations.py` | 16 matplotlib charts for the report |
| `advanced_analysis.py` | Extra analyses: probability calibration, CAPM decomposition, sector exposure, win/loss streaks |
| `report.tex` | LaTeX source for the technical report |
| `explanation.md` | Strategy explanation for Q&A prep |

## Key design choices

- **Walk-forward retraining** every 13 weeks instead of static train/test split
- **Turnover penalty** to reduce transaction cost drag (~0.08 stickiness bonus)
- **Regression + classification blend** (60/40) for better ranking
- **VIX regime filter** scales down exposure during extreme fear periods
- **Cross-sectional features** (stock rankings among peers, not just absolute indicators)
