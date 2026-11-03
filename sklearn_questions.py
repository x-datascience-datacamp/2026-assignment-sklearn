import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        return sum(1 for _ in self.split(X, y, groups))

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
        if not isinstance(X, pd.DataFrame) or (
            "registration_month" not in X.columns
        ):
            raise ValueError(
                "X must be a DataFrame with a 'registration_month' column."
            )
        if not pd.api.types.is_datetime64_any_dtype(X["registration_month"]):
            raise ValueError("'registration_month' must be of datetime dtype.")

        months = X["registration_month"].dt.to_period("M").to_numpy()
        y = np.asarray(y)

        for month in sorted(set(months)):
            next_month = month + 1
            idx_train = np.where(months == month)[0]
            idx_test = np.where(months == next_month)[0]

            if len(idx_test) == 0:
                continue

            # Both sets must contain at least one churner and one non-churner
            if (
                len(np.unique(y[idx_train])) < 2
                or len(np.unique(y[idx_test])) < 2
            ):
                continue

            yield idx_train, idx_test


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
    categorical_cols = ["gender", "level"]
    numerical_cols = [
        "total_events",
        "rate_songs",
        "rate_thumbs_up",
        "rate_thumbs_down",
        "rate_errors",
        "n_sessions",
        "has_downgraded",
    ]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()),
                numerical_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
