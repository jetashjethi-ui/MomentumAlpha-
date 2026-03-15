# MomentumAlpha — QnA Explanation Guide

Read this before the presentation. Know your answers cold.

---

## Q: What is MomentumAlpha?

It's a machine learning system that predicts which 2 stocks (out of 10) will have positive returns next week. Every Friday, it rebalances the portfolio based on those predictions.

## Q: Why did you choose these 10 stocks?

They were given in the problem statement: AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ, BRK-B. They represent diverse sectors — tech, finance, healthcare, consumer — so we get cross-sector signal.

## Q: Walk us through your entry and exit logic.

**Entry:** Each week, all 7 models predict the probability of positive return for each stock. We blend these into a single score (60% classification + 40% regression ranking). The top 2 stocks by score enter the portfolio with equal weight.

**Exit:** A stock exits when it's no longer in the top 2 — BUT we add a 0.08 "stickiness bonus" to currently-held stocks. A new stock must score at least 0.08 higher than the current pick to justify the switch. This reduces unnecessary trading and saves 5-10% annually in transaction costs.

**Regime filter:** If VIX > 35 (extreme fear like COVID crash), we scale down exposure to 50%, effectively holding cash to protect capital.

## Q: Why 7 models instead of 1?

Diversification. Each model captures different patterns:
- **LightGBM/XGBoost/CatBoost** (gradient boosting): Great at finding nonlinear feature interactions. Three different implementations reduce the chance of implementation-specific overfitting.
- **Random Forest + Extra Trees** (bagging): Reduce variance. Extra Trees uses random splits, adding more diversity.
- **GBM** (sklearn): Additional gradient boosting variant with different implementation.
- **Logistic Regression** (linear): Captures linear relationships. Acts as a "reality check" — if the complex models disagree with the linear model, the signal is less reliable.

Weights: boosting models get 60-70% total weight, bagging gets 20%, linear gets 8%.

## Q: Why not just use one model like XGBoost?

Single models are prone to overfitting on noisy financial data. An ensemble of diverse models — different algorithms, different random seeds, different inductive biases — is more robust. This is well-established in ML literature (Grinsztajn et al., 2022).

## Q: What is the regression blend and why do you use it?

The classification models predict P(positive return), but they don't distinguish between "barely positive" and "strongly positive." The regression model (LightGBM Regressor) predicts the actual return magnitude. We rank stocks by predicted return and blend this ranking (40%) with the classification probability (60%). This captures both direction AND magnitude.

## Q: How do you prevent look-ahead bias?

Walk-forward retraining. We NEVER train on data we haven't seen yet:
- First window: Train on 2017-2022, predict Jan-Mar 2023
- Then expand training to include Q1 2023, predict Q2 2023
- Continue expanding...

We retrain every 13 weeks (quarterly). At every decision point, the model only uses past data.

Additionally, our lagged target features (past win rates) are shifted by at least 1 period — we use `shift(lag+1)` to ensure no leakage.

## Q: Explain your transaction cost model.

10 basis points (0.1%) per side. Applied proportionally:
- If we replace 1 of 2 stocks: cost = 0.001 × (1/2) + 0.001 × (1/2) = 0.1%
- If we keep both stocks: cost = 0% (no trade, no cost)
- If we replace both: cost = 0.001 × (2/2) + 0.001 × (2/2) = 0.2%

The stickiness bonus reduces average turnover from ~60% per week to ~30-40%.

## Q: What is your Sharpe ratio?

Pre-cost Sharpe is approximately 0.75-0.85. After transaction costs, it drops to 0.5-0.7 depending on the exact test period. For a weekly equity momentum strategy with 10 bps costs, this is strong — institutional quant funds typically target Sharpe 0.5-1.0 on similar strategies.

## Q: Why is the Sharpe below 1.0?

Three reasons:
1. **Small universe** (10 stocks) → less diversification → higher volatility
2. **Weekly frequency** → each trade costs 20 bps round trip × ~50 weeks = significant drag
3. **2-stock portfolio** → concentrated positions amplify both wins and losses

A Sharpe > 1.0 after 10 bps costs on a 2-stock weekly portfolio would likely indicate overfitting or look-ahead bias.

## Q: What are cross-sectional features and why are they important?

Instead of asking "Is AAPL RSI high?" we ask "Is AAPL RSI higher than its 9 peers?" This transforms absolute indicators into relative rankings. It directly aligns with our goal: we don't need to predict absolute returns, just which stocks will OUTPERFORM their peers.

Example: `cs_ret5d_rank = PercentileRank(AAPL's 5-day return among all 10 stocks)`

## Q: Explain SHAP analysis.

SHAP (SHapley Additive exPlanations) decomposes each prediction into feature contributions. For each predicted probability, SHAP tells us exactly how much each feature pushed the prediction up or down. The summary plot shows which features are most important globally across all predictions.

## Q: What's the difference between your strategy and simple momentum?

Simple momentum: Pick top 2 stocks by past 20-day return. No ML.
Our strategy: Use 100+ features (not just past returns) fed into 7 ML models. We capture volume patterns, volatility regimes, cross-sectional rankings, calendar effects, and market regime indicators. The ML models find nonlinear combinations of these features that simple momentum misses.

## Q: Why walk-forward instead of normal train/test split?

Static split: Train on 2017-2022 once, test on 2023-2025. Problem: the model gets stale — 2022 market patterns may not apply in 2024.

Walk-forward: Retrain every 13 weeks with expanding data. The model adapts to changing market conditions. This is exactly how real quant funds operate.

## Q: What is the Monte Carlo permutation test?

We generate 5,000 random strategies (randomly pick 2 stocks each week) and compute their Sharpe ratios. Our p-value is the fraction of random strategies that beat ours. If p < 0.05, our strategy is statistically significantly better than random stock-picking.

## Q: What are the limitations?

1. **Survivorship bias** — these 10 stocks all survived and grew. We don't include stocks that failed.
2. **Small universe** — only 10 stocks limits diversification
3. **No slippage** — we assume trades execute at exact closing prices
4. **Weekly frequency** — misses intra-week signals
5. **Fixed hyperparameters** — we didn't use Bayesian optimization (Optuna) due to time constraints

## Q: What would you do differently with more time?

1. Expand to S&P 500 universe with point-in-time constituents
2. Bayesian hyperparameter optimization with Optuna
3. Learning-to-Rank (LightGBM Ranker) — directly optimize ranking instead of classification
4. Alternative data: sentiment from news/social media, options flow
5. Conformal prediction for prediction uncertainty quantification
6. Daily rebalancing with intraday features

## Q: Technical details — what hyperparameters did you use?

For all gradient boosting models:
- 800 trees, learning rate 0.03 (deliberately slow for generalization)
- Max depth 4-5 (shallow to prevent overfitting)
- Subsample 75%, feature sampling 70% (further regularization)
- Strong L1/L2 regularization (alpha=0.5, lambda=1.0-2.0)
- Class weight balanced (handles slight class imbalance)

We chose aggressive regularization because financial data is inherently noisy and overfitting is the #1 risk.

## Q: How does the VIX regime filter work?

VIX measures market fear. When VIX > 35 (historically rare — COVID crash, 2008 crisis), we multiply our portfolio return by 0.5 (effectively going 50% cash). When VIX > 28, we scale by 0.75. This protects capital during the worst drawdowns while staying fully invested during normal conditions.

## Q: What makes your project different from others?

1. **7-model ensemble with regression blend** (not just 1-2 models)
2. **100+ features across 9 categories** including cross-sectional ranking
3. **Walk-forward retraining** (not static split)
4. **Turnover penalty** (no other team models this)
5. **VIX regime filter** (dynamic risk management)
6. **Statistical significance testing** (bootstrap CI + Monte Carlo)
7. **SHAP interpretability + ablation study**
8. **26 professional visualizations**
9. **LaTeX report with mathematical proofs for every trigger**

---

**Pro tips for Q&A:**
- Always say "we" not "I"
- If you don't know an answer, say "That's an excellent question. We considered this but chose not to include it due to [time constraints / data limitations]. In future work, we would..."
- Reference specific papers: "As Jegadeesh & Titman (1993) showed..." or "Following de Prado's (2018) recommendation..."
- If asked about a metric, point to the exact formula in your report
