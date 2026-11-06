"""Churn prediction with a monthly cross-validator and a pipeline."""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RegistrationSplit(BaseCrossValidator):
    """Train on month M and test on M+1 when both classes are present."""

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of valid monthly folds."""
        return sum(1 for _ in self.split(X, y, groups))

    def split(self, X, y, groups=None):
        """Yield positional indices for consecutive calendar months."""
        if "registration_month" not in X.columns:
            raise ValueError("X must contain registration_month.")

        if not is_datetime64_any_dtype(X["registration_month"]):
            raise ValueError("registration_month must have datetime dtype.")

        if y is None:
            raise ValueError("y is required to check both classes.")

        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X):
            raise ValueError("y must be one-dimensional and match X length.")

        months = X["registration_month"].dt.to_period("M")

        for month in sorted(months.dropna().unique()):
            train_idx = np.flatnonzero((months == month).to_numpy())
            test_idx = np.flatnonzero((months == month + 1).to_numpy())

            if (
                np.isin([0, 1], target[train_idx]).all()
                and np.isin([0, 1], target[test_idx]).all()
            ):
                yield train_idx, test_idx


def get_estimator():
    """Return an unfitted churn pipeline excluding cancellation leakage."""
    categorical_columns = ["gender", "level"]
    numerical_columns = [
        "total_events",
        "rate_songs",
        "rate_thumbs_up",
        "rate_thumbs_down",
        "rate_errors",
        "n_sessions",
        "has_downgraded",
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])


if __name__ == "__main__":
    train = pd.read_csv(
        "data/churn_train.csv",
        parse_dates=["registration_month"],
    )
    X = train.drop(columns="churn")
    y = train["churn"]

    print("\n=== RegistrationSplit folds ===")
    cv = RegistrationSplit()

    for i, (tr, te) in enumerate(cv.split(X, y)):
        m_tr = X.iloc[tr]["registration_month"].dt.to_period("M").unique()
        m_te = X.iloc[te]["registration_month"].dt.to_period("M").unique()

        print(
            f"Fold {i}: train={sorted(m_tr)} "
            f"(n={len(tr)}, churn={y.iloc[tr].mean():.1%})"
            f" -> test={sorted(m_te)} "
            f"(n={len(te)}, churn={y.iloc[te].mean():.1%})"
        )

    print("\n=== Cross-validation with RegistrationSplit ===")
    scores = cross_validate(
        get_estimator(),
        X,
        y,
        cv=cv,
        scoring="roc_auc",
    )
    print(pd.DataFrame(scores).round(3))
    print(f"Mean CV AUC: {scores['test_score'].mean():.3f}")
