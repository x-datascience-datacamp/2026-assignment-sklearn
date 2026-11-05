"""Assignment - a sklearn Pipeline and CV splitter for churn prediction.

Context
-------
You are working with event-log data from a music streaming platform. Each row
in the raw log is one user action (page visited, song played, ...). The goal
is to predict which users will cancel their subscription (churn).

The file ``data/churn_train.csv`` contains one row per user with aggregated
features derived from the raw log:

+-------------------+--------------------------------------------+
| Column            | Description                                |
+===================+============================================+
| user_id           | Unique user identifier                     |
+-------------------+--------------------------------------------+
| gender            | M / F (categorical)                        |
+-------------------+--------------------------------------------+
| level             | Subscription tier: "free" / "paid"         |
+-------------------+--------------------------------------------+
| total_events      | log(1 + total number of events)            |
+-------------------+--------------------------------------------+
| rate_songs        | Fraction of events that are song plays     |
+-------------------+--------------------------------------------+
| rate_thumbs_up    | Fraction of events that are Thumbs-Up      |
+-------------------+--------------------------------------------+
| rate_thumbs_down  | Fraction of events that are Thumbs-Down    |
+-------------------+--------------------------------------------+
| rate_errors       | Fraction of events that are errors         |
+-------------------+--------------------------------------------+
| n_sessions        | Number of distinct sessions                |
+-------------------+--------------------------------------------+
| has_downgraded    | 1 if user ever requested a downgrade       |
+-------------------+--------------------------------------------+
| cancel_initiated  | 1 if user visited the cancellation page    |
+-------------------+--------------------------------------------+
| registration_month| Month the user registered (datetime)      |
+-------------------+--------------------------------------------+
| churn             | Target: 1 = churned, 0 = retained          |
+-------------------+--------------------------------------------+

Question 1 - RegistrationSplit
-------------------------------
Implement ``RegistrationSplit``, a scikit-learn cross-validator that splits
users by their ``registration_month`` column.

Each fold trains on users who registered in month M and evaluates on users
who registered in month M+1, simulating the production setting: you train on
your existing user base and predict on the cohort that joined next.

Only folds where *both* the training set and the test set contain at least one
churner and one non-churner are generated (otherwise the fold is useless for
evaluating churn prediction).

We also ask you to respect the PEP 8 convention: https://pep8.org. This will
be enforced with ``ruff``. You can check that there are no errors by calling
``ruff check`` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class.


Question 2 - get_estimator
--------------------------
Implement ``get_estimator()`` so that it returns a scikit-learn ``Pipeline``
that can be trained on the feature columns of ``churn_train.csv`` and predict
churn on unseen users.

Requirements:
- The pipeline must contain a ``ColumnTransformer`` that handles categorical
  columns (``gender``, ``level``) and numerical columns separately.
- Handle missing values in numerical features (``rate_errors`` contains NaN).
- End with a classifier of your choice.
- Achieve a ROC-AUC of at least 0.72 on the held-out test set.

Important: not every column should necessarily be used as a feature.
Think carefully about which columns are truly available at prediction time.

Hints
-----
- Q1 - ``pd.Series.dt.to_period('M')`` converts datetimes to year-month
  periods that are easy to sort and compare.

- Q1 - Remember to validate that ``registration_month`` is of datetime dtype
  and raise a ``ValueError`` if not.

- Q2 - Use ``sklearn.compose.ColumnTransformer`` together with
  ``sklearn.preprocessing.OneHotEncoder`` (for categoricals) and
  ``sklearn.impute.SimpleImputer`` (for numericals with NaN).

- Q2 - Explore the data first with ``pandas`` before writing the pipeline.
  Any feature that might be problematic/too correlated with the data?
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.utils.validation import column_or_1d

# ============================================================
# Question 1
# ============================================================


class RegistrationSplit(BaseCrossValidator):
    """Train on one registration month and test on the next calendar month.

    Both cohorts must contain churners and non-churners. Rows with a
    missing registration date are excluded from every fold.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Count valid folds for the DataFrame X and its binary target y."""
        return sum(1 for _ in self.split(X, y, groups))

    def split(self, X, y=None, groups=None):
        """Yield positional train and test indices in chronological order.

        X must contain a datetime registration_month column. The binary
        target y must follow the row order of X. Groups are ignored.
        """
        if not isinstance(X, pd.DataFrame) or "registration_month" not in X:
            raise ValueError("X must be a DataFrame with registration_month.")

        dates = X["registration_month"]
        if not is_datetime64_any_dtype(dates):
            raise ValueError("registration_month must have a datetime dtype.")
        if y is None:
            raise ValueError("y is required to check cohort labels.")

        labels = column_or_1d(y)
        if len(labels) != len(X):
            raise ValueError("X and y must have the same number of rows.")

        months = dates.dt.tz_localize(None).dt.to_period("M")
        cohorts = months.groupby(months, sort=True).indices
        eligible = {
            month: indices
            for month, indices in cohorts.items()
            if {0, 1}.issubset(labels[indices])
        }

        for month, train_indices in eligible.items():
            test_indices = eligible.get(month + 1)
            if test_indices is not None:
                yield train_indices, test_indices


def _engagement_features(X):
    """Return log-scaled session activity and non-song share, plus rates."""
    events = np.expm1(X["total_events"])
    events_per_session = events / X["n_sessions"].clip(lower=1)
    non_song_percent = 100 * (1 - X["rate_songs"])
    return X[["rate_thumbs_up", "rate_errors"]].assign(
        session_activity=np.log1p(events_per_session),
        non_song_activity=np.log1p(non_song_percent),
    )


# ============================================================
# Question 2
# ============================================================


def get_estimator():
    """Return a scikit-learn Pipeline for churn prediction.

    The pipeline is trained on the feature columns of ``churn_train.csv``
    (all columns except ``user_id``, ``registration_month``, and ``churn``).

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
        An unfitted pipeline ready to be used with ``fit`` / ``predict``.
    """
    categorical = ["gender", "level"]
    numerical = [
        "total_events",
        "rate_songs",
        "rate_thumbs_up",
        "rate_errors",
        "n_sessions",
    ]

    numerical_pipeline = Pipeline(
        [
            ("engagement", FunctionTransformer(_engagement_features)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    encoder = OneHotEncoder(handle_unknown="ignore")
    preprocessing = ColumnTransformer(
        [
            ("categorical", encoder, categorical),
            ("numerical", numerical_pipeline, numerical),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


# ============================================================
# Script to test the objects implemented in this file.
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path("data"),
        help="Directory containing churn_train.csv and churn_test.csv.",
    )
    args = parser.parse_args()

    train = pd.read_csv(
        args.data_dir / "churn_train.csv", parse_dates=["registration_month"]
    )
    X = train.drop(columns=["churn"])
    y = train["churn"]
    cv = RegistrationSplit()

    print("\n=== RegistrationSplit folds ===")
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m_tr = X.iloc[tr]["registration_month"].dt.to_period("M").iloc[0]
        m_te = X.iloc[te]["registration_month"].dt.to_period("M").iloc[0]
        print(
            f"Fold {i}: train={m_tr} "
            f"(n={len(tr)}, churn={y.iloc[tr].mean():.1%})"
            f" → test={m_te} (n={len(te)}, churn={y.iloc[te].mean():.1%})"
        )

    print("\n=== Cross-validation with RegistrationSplit ===")
    scores = cross_validate(
        get_estimator(), X, y, cv=cv, scoring="roc_auc", error_score="raise"
    )
    print(pd.DataFrame(scores).round(3))
    print(f"Mean CV ROC-AUC: {scores['test_score'].mean():.4f}")

    test = pd.read_csv(args.data_dir / "churn_test.csv")
    model = get_estimator().fit(X, y)
    probabilities = model.predict_proba(test.drop(columns="churn"))[:, 1]
    test_auc = roc_auc_score(test["churn"], probabilities)
    print(f"Held-out ROC-AUC: {test_auc:.4f}")
