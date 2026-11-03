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
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

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

    @staticmethod
    def _get_months(X):
        """Return the registration months of ``X`` as a monthly Period array.

        Parameters
        ----------
        X : pandas.DataFrame
            Data containing a ``registration_month`` column.

        Returns
        -------
        months : pandas.Series
            Registration month of each sample, as ``Period[M]``.

        Raises
        ------
        ValueError
            If ``X`` has no ``registration_month`` column or if this column
            is not of datetime dtype.
        """
        if not hasattr(X, "columns") or "registration_month" not in X.columns:
            raise ValueError("X must contain a 'registration_month' column.")
        col = X["registration_month"]
        if not pd.api.types.is_datetime64_any_dtype(col):
            raise ValueError(
                "'registration_month' must be of datetime dtype, "
                f"got {col.dtype}."
            )
        return col.dt.to_period("M").reset_index(drop=True)

    def _iter_folds(self, X, y):
        """Yield the valid (train, test) index pairs.

        A fold is valid if month M and month M+1 both exist in the data and
        both contain at least one churner and one non-churner.

        Parameters
        ----------
        X : pandas.DataFrame
            Data containing a ``registration_month`` column.
        y : array-like of shape (n_samples,)
            Target values (0 = retained, 1 = churned).

        Yields
        ------
        idx_train : ndarray
            Indices of users registered in month M.
        idx_test : ndarray
            Indices of users registered in month M+1.
        """
        if y is None:
            raise ValueError("y is required to check that both classes exist.")
        months = self._get_months(X)
        y = np.asarray(y)
        if len(y) != len(months):
            raise ValueError("X and y must have the same number of samples.")

        for month in sorted(months.unique()):
            next_month = month + 1
            idx_train = np.flatnonzero((months == month).to_numpy())
            idx_test = np.flatnonzero((months == next_month).to_numpy())
            if len(idx_train) == 0 or len(idx_test) == 0:
                continue
            if len(np.unique(y[idx_train])) < 2:
                continue
            if len(np.unique(y[idx_test])) < 2:
                continue
            yield idx_train, idx_test

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
        return sum(1 for _ in self._iter_folds(X, y))

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
        yield from self._iter_folds(X, y)


# ============================================================
# Question 2
# ============================================================

TRAIN_PATH = "data/churn_train.csv"
CATEGORICAL_COLS = ["gender", "level"]
NUMERICAL_COLS = [
    "total_events",
    "rate_songs",
    "rate_thumbs_up",
    "rate_thumbs_down",
    "rate_errors",
    "rate_other",
    "log_events_per_session",
]
# Columns not used: user_id (identifier), registration_month (used for the
# split), cancel_initiated (leakage: identical to the target), n_sessions
# and has_downgraded (noise, no gain in cross-validation).


def add_features(X):
    """Add engineered features to the raw user table.

    Parameters
    ----------
    X : pandas.DataFrame
        Raw feature table (``total_events`` is log(1 + n_events)).

    Returns
    -------
    X_new : pandas.DataFrame
        Copy of ``X`` with two extra columns: ``rate_other`` (share of
        events that are neither songs, thumbs nor errors) and
        ``log_events_per_session`` (log of the average session length).
    """
    X = X.copy()
    n_events = np.expm1(X["total_events"])
    n_sessions = X["n_sessions"].clip(lower=1)
    X["log_events_per_session"] = np.log(n_events.clip(lower=1) / n_sessions)
    X["rate_other"] = (
        1
        - X["rate_songs"]
        - X["rate_thumbs_up"]
        - X["rate_thumbs_down"]
        - X["rate_errors"].fillna(0)
    )
    return X


def _build_pipeline():
    """Build feature engineering + preprocessing + classifier pipeline.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline with feature engineering, a ``ColumnTransformer`` (one-hot
        encoding of categorical columns, imputation and scaling of numerical
        columns) and a classifier placeholder named ``clf``.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_COLS,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_COLS,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression()),
        ]
    )


def _build_param_grid():
    """Return the grid of classifiers and hyperparameters to explore.

    Returns
    -------
    param_grid : list of dict
        One dictionary per model (ridge logistic regression, LDA, SVC,
        random forest, gradient boosting).
    """
    return [
        {
            "clf": [LogisticRegression(max_iter=1000)],  # l2 = ridge
            "clf__C": [0.01, 0.1, 1, 10],
        },
        {
            "clf": [LinearDiscriminantAnalysis(solver="lsqr")],
            "clf__shrinkage": [None, "auto", 0.1, 0.5],
        },
        {
            "clf": [SVC(probability=True, random_state=0)],
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", 0.01],
        },
        {
            "clf": [RandomForestClassifier(random_state=0, n_jobs=-1)],
            "clf__n_estimators": [200],
            "clf__max_depth": [3, 5],
            "clf__min_samples_leaf": [10, 30],
        },
        {
            "clf": [GradientBoostingClassifier(random_state=0)],
            "clf__n_estimators": [100],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [1, 2],
            "clf__subsample": [0.8],
        },
    ]


def get_estimator():
    """Run a grid search and return the best pipeline.

    The grid search is scored with ROC-AUC and uses ``RegistrationSplit``
    as cross-validator, so the selection mimics the production setting
    (train on one cohort, predict on the next one).

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
        Unfitted pipeline configured with the best hyperparameters.
    """
    df = pd.read_csv(TRAIN_PATH, parse_dates=["registration_month"])
    X = df.drop(columns=["churn"])
    y = df["churn"]

    pipeline = _build_pipeline()
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=_build_param_grid(),
        scoring="roc_auc",
        cv=RegistrationSplit(),
        n_jobs=-1,
    )
    search.fit(X, y)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return clone(pipeline).set_params(**search.best_params_)
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
