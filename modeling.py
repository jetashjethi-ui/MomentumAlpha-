"""Model definitions shared by the walk-forward training pipeline."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier_models(*, random_state: int = 42, n_jobs: int = -1) -> dict[str, object]:
    """Create a compact, diverse ensemble for cross-sectional ranking.

    The ensemble deliberately uses a small fixed set of models. Hyperparameter
    tuning on the final walk-forward period would turn that period into a
    validation set and make the reported performance optimistic.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.025,
            max_depth=5,
            num_leaves=20,
            min_child_samples=25,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=1.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.025,
            depth=5,
            l2_leaf_reg=6.0,
            auto_class_weights="Balanced",
            random_seed=random_state,
            thread_count=n_jobs,
            allow_writing_files=False,
            verbose=False,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=12,
            max_features=0.75,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=12,
            max_features=0.75,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.25,
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
