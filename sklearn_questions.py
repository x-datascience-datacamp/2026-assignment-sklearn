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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from pandas.api.types import is_datetime64_any_dtype

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

    def _check_datetime(self, months):
        """Raise a ValueError if ``months`` is not of datetime dtype."""
        if not is_datetime64_any_dtype(months):
            raise ValueError(
                "X[\"registration_month\"] should be of datetime dtype, "
                f"not {months.dtype}"
            )

    def _valid_month_pairs(self, X, y=None):
        """Return the list of (train_month, test_month) pairs that are valid.
        Also checks whether X["registration_month"] has type datetime."""

        # Type check
        months = X["registration_month"]
        self._check_datetime(months)
        # if months.dtype != "datetime64[ns]":
        #     raise ValueError(
        #         f"Type of X[\"registration_month\"] should be "
        #         f"datetime64[ns], not {months.dtype}"
        #     )

        unique_months = np.sort(months.unique())
        if y is None:
            if "churn" not in X.columns:
                raise ValueError(
                    "y is required (or X must contain a 'churn' column)."
                )
            y = X["churn"]
        y = pd.Series(np.asarray(y), index=X.index)

        def has_both_classes(month):
            labels = y[(months == month).to_numpy()]
            # churn == 0 and churn == 1 both present
            return labels.nunique() == 2

        pairs = []
        # The last month can never be a training month: it has no M+1.
        for train_month, test_month in zip(
            unique_months[:-1], unique_months[1:]
        ):
            # Only consecutive calendar months (M -> M+1) are allowed
            expected_next = pd.Timestamp(train_month) + pd.DateOffset(months=1)
            if pd.Timestamp(test_month) != expected_next:
                continue
            if has_both_classes(train_month) and has_both_classes(test_month):
                pairs.append((train_month, test_month))

        return pairs

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
        # Return the number of valid splits

        return len(self._valid_month_pairs(X, y))

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

        # Type check:  registration_month column is date_time
        months = X["registration_month"]
        self._check_datetime(months)

        # if months.dtype != "datetime64[ns]":
        #     raise ValueError(
        #         f"Type of X[\"registration_month\"] should be "
        #         f"datetime64[ns], not {months.dtype}"
        #     )

        # {Timestamp: positional indices (ndarray)}: np array of indices
        # for each month
        month_indices = months.groupby(months).indices
        # print("Month indices: ", month_indices)

        # For each valid month pairs adds the train (current M month) and
        # test (M+1 month) sets
        for train_month, test_month in self._valid_month_pairs(X, y):
            idx_train = month_indices[train_month]
            idx_test = month_indices[test_month]
            yield idx_train, idx_test


# ============================================================
# Question 2
# ============================================================
# Columns used as features. Everything else (user_id, registration_month,
# cancel_initiated, ...) is dropped by the ColumnTransformer.


# Dropped "error_rate" as it gives lower results
# Tried log(1+n) -> n transformation (exp(.)-1) on "total_events" but it did
# not change the results
CATEGORICAL_FEATURES = ["level", "gender"]
NUMERICAL_FEATURES = [
    "total_events",
    "rate_songs",
    "rate_thumbs_up",
    "rate_thumbs_down",
    "n_sessions",
    "has_downgraded",
]


def get_estimator():
    """Return a scikit-learn Pipeline for churn prediction.

    The pipeline is trained on the feature columns of ``churn_train.csv``
    (all columns except ``user_id``, ``registration_month``, and ``churn``).

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
        An unfitted pipeline ready to be used with ``fit`` / ``predict``.
    """
    # to transform level "free" "paid" into (1, 0) and (0, 1).
    # Same for gender.
    categorical_pipeline = OneHotEncoder()

    numerical_pipeline = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ],
        # drops non transformed columns: user_id, gender, registration_month,
        # cancel_initiated, rate_errors
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(random_state=42)),
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

    ######### Personnal checks ################

    # downgrade & churn proportion
    print("\nhas_downgraded = churn ?")
    print(f"proportion of equality: \
        {np.mean(X["has_downgraded"] == y)}")

    # cancel_initiated same as churn ?
    print("\ncancel_initiated same as churn ?")
    print(f"proportion of equality: \
        {np.mean(X["cancel_initiated"] == y)}")
    # nan check
    print("\n ###NaN check in train set###")
    print("Categories in X:")
    print(X.isna().mean().sort_values(ascending=False))
    print("\nNumbers of NaN in y (churn): ", np.sum(np.isnan(y)))

    ## Visualisation of test set
    print("\n### Visualisation of test set ###")
    test = pd.read_csv(
        "data/churn_test.csv", parse_dates=["registration_month"]
    )
    X_test = test.drop(columns=["churn"])
    y_test = test["churn"]
    # Features and missing values
    print(f"Features of test set: {test.columns}")
    print("\nProportion of Nans in each categories:")
    print(test.isna().mean().sort_values(ascending=False))
    # downgrade & churn proportion
    print("\nhas_downgraded = churn ?")
    print(f"proportion of equality: \
        {np.mean(X_test["has_downgraded"] == y_test)}")
    # cancel_initiated same as churn ?
    print("\ncancel_initiated same as churn ?")
    print(f"proportion of equality: \
        {np.mean(X_test["cancel_initiated"] == y_test)}")

    ######### End of the personnal checks ################

    # ---- cross-validate the estimator ----
    print("\n=== Cross-validation with RegistrationSplit ===")
    scores = cross_validate(get_estimator(), X, y, cv=cv, scoring="roc_auc")
    scores = pd.DataFrame(scores)
    print(f"CV results : {scores.round(3)}")
    print(f"Mean CV AUC   : {scores['test_score'].mean():.3f}")
