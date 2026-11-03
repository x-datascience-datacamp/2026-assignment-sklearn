"""Assignment - a sklearn Pipeline and CV splitter for churn prediction.

The cross-validator compares consecutive monthly registration cohorts.
The estimator excludes identifiers and information unavailable at prediction
 time, notably the cancellation-page indicator.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Question 1
# ============================================================

class RegistrationSplit(BaseCrossValidator):
    """Cross-validator based on consecutive registration months.

    Each fold trains on month M and tests on month M+1. A fold is
    generated only when both cohorts contain both target classes.
    """

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of valid consecutive-month folds.

        Parameters
        ----------
        X : pandas.DataFrame
            Data containing the ``registration_month`` datetime column.
        y : array-like
            Binary churn targets used to filter invalid folds.
        groups : array-like, optional
            Ignored; included for scikit-learn compatibility.

        Returns
        -------
        int
            Number of folds with two classes in each cohort.
        """
        return sum(1 for _ in self.split(X, y, groups))

    def split(self, X, y, groups=None):
        """Yield positional train and test indices for valid monthly folds.

        Parameters
        ----------
        X : pandas.DataFrame
            Data containing a datetime ``registration_month`` column.
        y : array-like
            Binary churn targets.
        groups : array-like, optional
            Ignored; included for scikit-learn compatibility.

        Yields
        ------
        train_idx : numpy.ndarray
            Integer positions of users registered in month M.
        test_idx : numpy.ndarray
            Integer positions of users registered in month M+1.

        Raises
        ------
        ValueError
            If ``registration_month`` is not datetime or ``y`` is missing.
        """
        if not pd.api.types.is_datetime64_any_dtype(
            X["registration_month"]
        ):
            raise ValueError("registration_month must have datetime dtype")
        if y is None:
            raise ValueError("Target y is required to validate the folds")

        months = X["registration_month"].dt.to_period("M")
        targets = np.asarray(y)

        for month in sorted(months.dropna().unique()):
            train_idx = np.flatnonzero((months == month).to_numpy())
            test_idx = np.flatnonzero((months == month + 1).to_numpy())

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if np.unique(targets[train_idx]).size < 2:
                continue
            if np.unique(targets[test_idx]).size < 2:
                continue

            yield train_idx, test_idx


# ============================================================
# Question 2
# ============================================================

def get_estimator():
    """Return an unfitted churn-prediction scikit-learn Pipeline.

    ``cancel_initiated`` is intentionally excluded: it reveals the target
    in the training dataset but is not available at prediction time.
    Identifiers and registration dates are also excluded.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Preprocessing and logistic-regression classifier.
    """
    categorical_features = ["gender", "level"]
    numerical_features = [
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
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numerical", numerical_pipeline, numerical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


# ============================================================
# Script to test the objects implemented in this file.
# ============================================================

if __name__ == "__main__":
    from sklearn.model_selection import cross_validate

    train = pd.read_csv(
        "data/churn_train.csv", parse_dates=["registration_month"]
    )
    X = train.drop(columns=["churn"])
    y = train["churn"]

    print("\n=== RegistrationSplit folds ===")
    cv = RegistrationSplit()
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m_tr = train.iloc[tr]["registration_month"].dt.to_period("M").unique()
        m_te = train.iloc[te]["registration_month"].dt.to_period("M").unique()
        churn_tr = y.iloc[tr].mean()
        churn_te = y.iloc[te].mean()
        print(
            f"Fold {i}: train={sorted(m_tr)} "
            f"(n={len(tr)}, churn={churn_tr:.1%})"
            f"  →  test={sorted(m_te)} "
            f"(n={len(te)}, churn={churn_te:.1%})"
        )

    print("\n=== Cross-validation with RegistrationSplit ===")
    scores = cross_validate(get_estimator(), X, y, cv=cv, scoring="roc_auc")
    scores = pd.DataFrame(scores)
    print(f"CV results : {scores.round(3)}")
    print(f"Mean CV AUC   : {scores['test_score'].mean():.3f}")
