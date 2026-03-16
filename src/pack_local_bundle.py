# pack_local_bundle.py
# Local packaging script (Windows/macOS/Linux)
# - Loads CSV
# - Builds ML models 
# - Runs StratifiedKFold CV for all ML models
# - Runs a model-based posterior scoring baseline (separately, not part of model selection)
# - Runs ablation (simple + drop-one-feature) for the best ML model
# - Saves EVERYTHING into a single .pkl bundle:
#   - models (ML)
#   - baseline_models (posterior scoring baseline)
#   - label_encoder, feature_cols, cv_config, etc.

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


# Optional: XGBoost / LightGBM
XGB_AVAILABLE = True
LGB_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGB_AVAILABLE = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(data_path: str, id_col: str, target_col: str):
    df = pd.read_csv(data_path).dropna().copy()
    feature_cols = [c for c in df.columns if c not in [id_col, target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return df, X, y, feature_cols


def build_models(random_state: int, n_jobs: int):
    """ML models only (discriminative/predictive classifiers)."""
    models = {}

    # SVM family
    models["SVM_LinearSVC"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, class_weight="balanced",
                          random_state=random_state, max_iter=8000)),
    ])

    models["SVM_linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=2000, class_weight="balanced")),
    ])

    models["SVM_polynomial"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", C=1.0, degree=3, class_weight="balanced")),
    ])

    models["SVM_RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1000, gamma=0.1, class_weight="balanced")),
    ])

    models["SVM_sigmoid"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="sigmoid", C=1.0, gamma="scale", class_weight="balanced")),
    ])

    # Naive Bayes / Logistic Regression
    models["NaiveBayes"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ])

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    # Decision Tree / Random Forest
    models["DecisionTree"] = Pipeline([
        ("clf", DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced"
        )),
    ])

    models["RandomForest"] = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=600,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
            max_features="sqrt",
        )),
    ])

    # Gradient Boosting
    models["GradientBoosting"] = Pipeline([
        ("clf", GradientBoostingClassifier(random_state=random_state)),
    ])

    # KNN
    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
    ])

    # Optional: XGBoost / LightGBM
    if XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=n_jobs
            )),
        ])

    if LGB_AVAILABLE:
        models["LightGBM"] = Pipeline([
            ("clf", LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multiclass",
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1,  #  silence warnings/logs
                min_data_in_leaf=50,
                min_gain_to_split=0.0
            )),
        ])

    return models


class LPAScoringClassifier(BaseEstimator, ClassifierMixin):
    """
    Model-based posterior scoring baseline.
    Approximates posterior assignment by:
    - Estimating class priors π_k
    - Estimating class-conditional Gaussian parameters (μ_k, σ_k^2)
    - Assigning cases by argmax posterior probability
    """

    def __init__(self, var_smoothing=1e-6):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        D = X.shape[1]

        self.class_log_prior_ = np.zeros(K)
        self.mean_ = np.zeros((K, D))
        self.var_ = np.zeros((K, D))

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_log_prior_[idx] = np.log(Xc.shape[0] / X.shape[0])
            self.mean_[idx] = Xc.mean(axis=0)
            self.var_[idx] = Xc.var(axis=0) + self.var_smoothing

        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = []

        for k in range(len(self.classes_)):
            mu = self.mean_[k]
            var = self.var_[k]
            ll = -0.5 * (
                np.log(2.0 * np.pi * var) +
                ((X - mu) ** 2) / var
            ).sum(axis=1)
            log_prob.append(self.class_log_prior_[k] + ll)

        return np.vstack(log_prob).T  # (n_samples, K)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll = jll - jll.max(axis=1, keepdims=True)
        prob = np.exp(jll)
        return prob / prob.sum(axis=1, keepdims=True)


def cv_metrics(model, X, y, n_splits: int, random_state: int, n_jobs: int):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "acc": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0) ,
        "prec_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "rec_macro": make_scorer(recall_score, average="macro", zero_division=0),
    }
    res = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score=np.nan
    )

    out = {}
    for k in scoring:
        out[f"{k}_mean"] = float(np.nanmean(res[f"test_{k}"]))
        out[f"{k}_sd"] = float(np.nanstd(res[f"test_{k}"], ddof=1))
    return out


def make_ablation_sets_simple(X: pd.DataFrame):
    vals = X.values
    mean_ = np.mean(vals, axis=1)
    top1 = np.max(vals, axis=1)
    top2 = np.sort(vals, axis=1)[:, -2:]  # two largest

    X_full = X.copy()
    X_mean = pd.DataFrame({"MeanRIASEC": mean_}, index=X.index)
    X_top1 = pd.DataFrame({"Top1RIASEC": top1}, index=X.index)
    X_top2 = pd.DataFrame({"Top2_1": top2[:, 0], "Top2_2": top2[:, 1]}, index=X.index)

    return {
        "Full6": X_full,
        "MeanOnly": X_mean,
        "Top1Score": X_top1,
        "Top2Scores": X_top2
    }


def make_drop_one_feature_sets(X: pd.DataFrame):
    sets = {"Full6": X.copy()}
    for col in X.columns:
        sets[f"Drop_{col}"] = X.drop(columns=[col]).copy()
    return sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file (e.g., Career_LPA.csv)")
    parser.add_argument("--out", default="./ml_outputs", help="Output directory")
    parser.add_argument("--id_col", default="No", help="ID column name")
    parser.add_argument("--target_col", default="Group", help="Target column name")
    parser.add_argument("--splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--jobs", type=int, default=-1, help="n_jobs for CV/training where supported")
    args = parser.parse_args()

    ensure_dir(args.out)

    # Load
    df, X, y_raw, feature_cols = load_data(args.data, args.id_col, args.target_col)

    # Encode y to 0..K-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("y classes:", len(np.unique(y)), "min/max:", int(y.min()), int(y.max()))
    print("rows:", X.shape[0], "features:", X.shape[1])

    # Build ML models
    models = build_models(random_state=args.seed, n_jobs=args.jobs)
    print("ML models:", list(models.keys()))

    # CV all ML models
    rows = []
    for name, model in models.items():
        m = cv_metrics(model, X, y, n_splits=args.splits, random_state=args.seed, n_jobs=args.jobs)
        rows.append({"Model": name, **m})
        print(f"[CV done] {name} -> acc={m['acc_mean']:.6f} ± {m['acc_sd']:.6f}")

    cv_table = pd.DataFrame(rows).sort_values(["acc_mean", "acc_sd"], ascending=[False, True])
    cv_table.insert(0, "Rank", range(1, len(cv_table) + 1))
    cv_path = os.path.join(args.out, "cv_model_comparison.csv")
    cv_table.to_csv(cv_path, index=False)

    best_model_name = str(cv_table.iloc[0]["Model"])
    print("\nBest ML model:", best_model_name)
    print("Saved ML CV table:", cv_path)

    # -------------------------
    # Baseline (posterior scoring) - evaluated separately
    # -------------------------
    baseline_est = LPAScoringClassifier(var_smoothing=1e-6)
    baseline_cv = cv_metrics(baseline_est, X, y, n_splits=args.splits, random_state=args.seed, n_jobs=args.jobs)

    baseline_table = pd.DataFrame([{"Model": "LPA_posterior_scoring_baseline", **baseline_cv}])
    baseline_path = os.path.join(args.out, "baseline_posterior_scoring_cv.csv")
    baseline_table.to_csv(baseline_path, index=False)

    print("\nBaseline (posterior scoring):")
    print(f"acc={baseline_cv['acc_mean']:.6f} ± {baseline_cv['acc_sd']:.6f}")
    print("Saved baseline table:", baseline_path)

    # -------------------------
    # Ablation (using best ML model only)
    # -------------------------
    primary_model = models[best_model_name]

    # Ablation A
    simple_sets = make_ablation_sets_simple(X)
    simple_rows = []
    for fset_name, Xsub in simple_sets.items():
        m = cv_metrics(primary_model, Xsub, y, n_splits=args.splits, random_state=args.seed, n_jobs=args.jobs)
        simple_rows.append({"FeatureSet": fset_name, **m})
        print(f"[Ablation A] {fset_name} -> acc={m['acc_mean']:.6f} ± {m['acc_sd']:.6f}")

    simple_table = pd.DataFrame(simple_rows).sort_values(["acc_mean", "acc_sd"], ascending=[False, True])
    simple_table.insert(0, "Rank", range(1, len(simple_table) + 1))
    simple_path = os.path.join(args.out, f"ablationA_simple_{best_model_name}.csv")
    simple_table.to_csv(simple_path, index=False)

    # Ablation B
    drop_sets = make_drop_one_feature_sets(X)
    drop_rows = []
    for fset_name, Xsub in drop_sets.items():
        m = cv_metrics(primary_model, Xsub, y, n_splits=args.splits, random_state=args.seed, n_jobs=args.jobs)
        drop_rows.append({"FeatureSet": fset_name, **m})
        print(f"[Ablation B] {fset_name} -> acc={m['acc_mean']:.6f} ± {m['acc_sd']:.6f}")

    drop_table = pd.DataFrame(drop_rows)
    full_acc = float(drop_table.loc[drop_table["FeatureSet"] == "Full6", "acc_mean"].values[0])
    drop_table["acc_drop_from_full"] = full_acc - drop_table["acc_mean"]
    drop_table = drop_table.sort_values(["acc_drop_from_full"], ascending=[False])
    drop_table.insert(0, "Rank_by_drop", range(1, len(drop_table) + 1))

    drop_path = os.path.join(args.out, f"ablationB_drop_one_feature_{best_model_name}.csv")
    drop_table.to_csv(drop_path, index=False)

    # -------------------------
    # Bundle everything (ML models + baseline)
    # -------------------------
    bundle = {
        "created_at": datetime.now().isoformat(),

        # ML models (used for model comparison / selection)
        "models": models,
        "best_model_name": best_model_name,

        # Baseline models (NOT part of model selection)
        "baseline_models": {
            "LPA_posterior_scoring_baseline": {
                "estimator": LPAScoringClassifier(var_smoothing=1e-6),
                "cv_result": baseline_cv,
                "description": "Posterior assignment using class priors and Gaussian class-conditional densities."
            }
        },

        # Preprocessing + data definition
        "label_encoder": le,
        "feature_cols": feature_cols,

        # CV protocol (explicitly stored)
        "cv_config": {
            "cv_type": "StratifiedKFold",
            "n_splits": args.splits,
            "shuffle": True,
            "random_state": args.seed,
            "n_jobs": args.jobs,
        },

        # Metrics protocol
        "metrics": {
            "scoring": ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        },

        # Data config
        "data_config": {
            "id_col": args.id_col,
            "target_col": args.target_col,
            "dropna": True,
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(le.classes_)),
        },

        "svm_fixed_params": {
            "SVM_linear": {"kernel": "linear", "C": 2000, "class_weight": "balanced"},
            "SVM_RBF": {"kernel": "rbf", "C": 1000, "gamma": 0.1, "class_weight": "balanced"},
        }
    }

    bundle_path = os.path.join(args.out, "all_models_bundle.pkl")
    joblib.dump(bundle, bundle_path)

    # Human-readable JSON snapshot (for transparency)
    config_path = os.path.join(args.out, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== Saved outputs ===")
    print("Bundle:", bundle_path)
    print("Config:", config_path)
    print("ML CV:", cv_path)
    print("Baseline CV:", baseline_path)
    print("Ablation A:", simple_path)
    print("Ablation B:", drop_path)


if __name__ == "__main__":
    main()
