import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RegistrationSplit(BaseCrossValidator):
    """Cross-validator that splits users by registration month.

    Expects the input DataFrame to contain a ``registration_month`` column of
    datetime dtype.  Each fold trains on users who registered in month M and
    evaluates on users who registered in the following month M+1.

    Only folds where both the training set and the test set contain at least
    one churner **and** one non-churner are generated.
    """

    @staticmethod
    def _get_periods(X):
        """Return the registration month of each user as a monthly period.

        Raises
        ------
        ValueError
            If ``registration_month`` is missing or not of datetime dtype.
        """
        if (
            not hasattr(X, "columns")
            or "registration_month" not in X.columns
        ):
            raise ValueError("X must have a 'registration_month' column.")
        col = X["registration_month"]
        if not pd.api.types.is_datetime64_any_dtype(col):
            raise ValueError(
                "'registration_month' must be of datetime dtype."
            )
        return col.dt.to_period("M")

    @staticmethod
    def _has_both_classes(y, idx):
        """Return True if ``y[idx]`` contains both classes."""
        return np.unique(y[idx]).size == 2

    def _iter_folds(self, X, y=None):
        """Yield valid (train, test) index pairs.

        If ``y`` is None, the class-balance check is skipped.
        """
        periods = self._get_periods(X)
        y = None if y is None else np.asarray(y)
        for month in sorted(periods.unique()):
            idx_train = np.flatnonzero((periods == month).to_numpy())
            idx_test = np.flatnonzero((periods == month + 1).to_numpy())
            if len(idx_train) == 0 or len(idx_test) == 0:
                continue
            if y is not None and not (
                self._has_both_classes(y, idx_train)
                and self._has_both_classes(y, idx_test)
            ):
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


def get_estimator():
    """Return a scikit-learn Pipeline for churn prediction.

    The pipeline is trained on the feature columns of ``churn_train.csv``
    (all columns except ``user_id``, ``registration_month``, and ``churn``).
    ``cancel_initiated`` is deliberately excluded: visiting the cancellation
    page is a consequence of churning, not information available beforehand
    (target leakage).

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
        "rate_thumbs_down",
        "rate_errors",
        "n_sessions",
        "has_downgraded",
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numerical,
            ),
        ],
        remainder="drop",  # drops user_id, registration_month, leaky cols
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
