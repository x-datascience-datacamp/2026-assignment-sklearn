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
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import BaseCrossValidator


# ============================================================
# Question 1
# ============================================================

class RegistrationSplit(BaseCrossValidator):
    """Cross-validator that splits users by registration month.

    Expects the input DataFrame to contain a ``registration_month`` column of
    datetime dtype.  Each fold trains on users who registered in month M and
    evaluates on users who registered in the following month M+1.

    Only folds where both the training set and the test set contain at least
    one churner **and** one non-churner are generated.
    """

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of valid splitting iterations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, must contain a ``registration_month`` column.
        y : array-like of shape (n_samples,), optional
            Target values.  Used to check that both classes are present.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of valid splits.
        """
        user_months = X["registration_month"].dt.to_period("M")
        months = sorted(user_months.dropna().unique())

        n_splits = 0

        for month_train, month_test in zip(months[:-1], months[1:]):

            # Check that the months are consecutive.
            if month_test != month_train + 1:
                continue

            train_idx = np.flatnonzero(
                (user_months == month_train).to_numpy()
            )

            test_idx = np.flatnonzero(
                (user_months == month_test).to_numpy()
            )

            # Both classes must be present in the training set.
            if np.unique(y[train_idx]).size != 2:
                continue

            # Both classes must be present in the test set.
            if np.unique(y[test_idx]).size != 2:
                continue

            n_splits += 1

        return n_splits


    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, must contain a ``registration_month`` column.
        y : array-like of shape (n_samples,)
            Target values.  Used to filter out folds that do not contain
            both classes.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """

        if not pd.api.types.is_datetime64_any_dtype(
            X["registration_month"]
        ):
            raise ValueError(
                "registration_month must have datetime dtype."
            )

        user_months = X["registration_month"].dt.to_period("M")
        months = sorted(user_months.dropna().unique())

        for month_train, month_test in zip(months[:-1], months[1:]):

            # Check that the months are consecutive.
            if month_test != month_train + 1:
                continue

            train_idx = np.flatnonzero(
                (user_months == month_train).to_numpy()
            )

            test_idx = np.flatnonzero(
                (user_months == month_test).to_numpy()
            )

            # Check that both classes are present in both sets.
            if np.unique(y[train_idx]).size != 2:
                continue

            if np.unique(y[test_idx]).size != 2:
                continue

            yield train_idx, test_idx



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
    categorical_features = [
        "gender",
        "level",
    ]

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
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced"
    )

    estimator = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return estimator


# ============================================================
# Script to test the objects implemented in this file.
# ============================================================



if __name__ == "__main__":
    from sklearn.model_selection import cross_validate

    train = pd.read_csv(
        "data/churn_train.csv", parse_dates=["registration_month"]
    )
    # Pass the full DataFrame (including registration_month) to the splitter.
    X = train.drop(columns=["churn"])
    y = train["churn"]

    # ---- inspect the folds produced by RegistrationSplit ----
    print("\n=== RegistrationSplit folds ===")
    cv = RegistrationSplit()
    cv.get_n_splits(X)
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

    # ---- cross-validate the estimator ----
    print("\n=== Cross-validation with RegistrationSplit ===")
    scores = cross_validate(get_estimator(), X, y, cv=cv, scoring="roc_auc")
    scores = pd.DataFrame(scores)
    print(f"CV results : {scores.round(3)}")
    print(f"Mean CV AUC   : {scores['test_score'].mean():.3f}")
