
"""Assignment - a sklearn Pipeline and CV splitter for churn prediction."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RegistrationSplit(BaseCrossValidator):
    """Cross-validator using consecutive registration months."""

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of valid consecutive-month folds."""
        if y is None:
            raise ValueError("Target y is required to validate folds.")

        return sum(1 for _ in self.split(X, y, groups))

    def split(self, X, y, groups=None):
        """Yield valid train/test indices for consecutive months."""
        if "registration_month" not in X.columns:
            raise ValueError("Missing registration_month column.")

        dates = X["registration_month"]

        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError("registration_month must be datetime.")

        if y is None:
            raise ValueError("Target y is required.")

        months = dates.dt.to_period("M")
        unique_months = sorted(months.dropna().unique())
        target = np.asarray(y)

        if len(target) != len(X):
            raise ValueError("X and y must have the same length.")

        for month in unique_months:
            next_month = month + 1

            if next_month not in unique_months:
                continue

            train_idx = np.flatnonzero(
                (months == month).to_numpy()
            )
            test_idx = np.flatnonzero(
                (months == next_month).to_numpy()
            )

            if (
                np.unique(target[train_idx]).size < 2
                or np.unique(target[test_idx]).size < 2
            ):
                continue

            yield train_idx, test_idx


def get_estimator():
    """Return an unfitted pipeline for churn prediction."""
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

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False)),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numerical", numerical_pipeline, numerical_columns),
        ],
        remainder="drop",
    )

    estimator = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    random_state=42,
                ),
            ),
        ]
    )

    return estimator


if __name__ == "__main__":
    from sklearn.model_selection import cross_validate

    train = pd.read_csv(
        "data/churn_train.csv",
        parse_dates=["registration_month"],
    )

    X = train.drop(columns=["churn"])
    y = train["churn"]

    print("\n=== RegistrationSplit folds ===")

    cv = RegistrationSplit()

    for i, (tr, te) in enumerate(cv.split(X, y)):
        m_tr = train.iloc[tr]["registration_month"].dt.to_period("M").unique()
        m_te = train.iloc[te]["registration_month"].dt.to_period("M").unique()

        print(
            f"Fold {i}: train={sorted(m_tr)} "
            f"test={sorted(m_te)}"
        )

    print("\n=== Cross-validation ===")

    scores = cross_validate(
        get_estimator(),
        X,
        y,
        cv=cv,
        scoring="roc_auc",
    )

    print(pd.DataFrame(scores).round(3))
    print(f"Mean CV AUC: {scores['test_score'].mean():.3f}")
