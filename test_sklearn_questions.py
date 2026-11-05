# ##################################################
# YOU SHOULD NOT TOUCH THIS FILE !
# ##################################################
import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

from sklearn_questions import get_estimator
from sklearn_questions import RegistrationSplit

DATA_DIR = "data"


@pytest.fixture(scope="module")
def churn_train():
    """Load the assignment training dataset."""
    return pd.read_csv(
        f"{DATA_DIR}/churn_train.csv",
        parse_dates=["registration_month"],
    )


@pytest.fixture(scope="module")
def churn_test():
    """Load the held-out test dataset (cancel_initiated=0 for all users)."""
    return pd.read_csv(
        f"{DATA_DIR}/churn_test.csv",
        parse_dates=["registration_month"],
    )


def _split_xy(df):
    return df.drop(columns="churn"), df["churn"]


# ==================================================================
# Question 1 — RegistrationSplit
# ==================================================================

def test_registration_split_repr():
    """RegistrationSplit repr must match the expected string."""
    cv = RegistrationSplit()
    assert repr(cv) == "RegistrationSplit()"


def test_registration_split_get_n_splits(churn_train):
    """get_n_splits should return 3 for the assignment training data."""
    X = churn_train.drop(columns=["churn"])
    y = churn_train["churn"]
    cv = RegistrationSplit()
    assert cv.get_n_splits(X, y) == 3


def test_registration_split_indices_are_integers(churn_train):
    """Train and test indices must be integer arrays."""
    X = churn_train.drop(columns=["churn"])
    y = churn_train["churn"]
    cv = RegistrationSplit()
    for train_idx, test_idx in cv.split(X, y):
        assert np.asarray(train_idx).dtype.kind == "i"
        assert np.asarray(test_idx).dtype.kind == "i"


def test_registration_split_temporal_order(churn_train):
    """Training users must have registered before test users in every fold."""
    X = churn_train.drop(columns=["churn"])
    y = churn_train["churn"]
    cv = RegistrationSplit()
    for train_idx, test_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        assert (
            X_tr["registration_month"].max() < X_te["registration_month"].min()
        ), (
            "All training users must have registered before any test user."
        )


def test_registration_split_both_classes_present(churn_train):
    """Every fold must contain churners AND non-churners in both folds."""
    X = churn_train.drop(columns=["churn"])
    y = churn_train["churn"]
    cv = RegistrationSplit()
    for train_idx, test_idx in cv.split(X, y):
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]
        assert y_tr.nunique() == 2, (
            "Training fold must contain both churners and non-churners."
        )
        assert y_te.nunique() == 2, (
            "Test fold must contain both churners and non-churners."
        )


def test_registration_split_n_splits_consistent(churn_train):
    """get_n_splits must equal the number of folds produced by split."""
    X = churn_train.drop(columns=["churn"])
    y = churn_train["churn"]
    cv = RegistrationSplit()
    n_declared = cv.get_n_splits(X, y)
    n_actual = sum(1 for _ in cv.split(X, y))
    assert n_declared == n_actual


def test_registration_split_non_datetime_raises(churn_train):
    """A ValueError must be raised if registration_month is not datetime."""
    X = churn_train.drop(columns=["churn"]).copy()
    X["registration_month"] = X["registration_month"].astype(str)
    y = churn_train["churn"]
    cv = RegistrationSplit()
    with pytest.raises(ValueError, match="datetime"):
        next(cv.split(X, y))


# ==================================================================
# Question 2 — get_estimator
# ==================================================================

def test_get_estimator_returns_pipeline():
    """get_estimator() must return a sklearn Pipeline."""
    est = get_estimator()
    assert isinstance(est, Pipeline), (
        f"get_estimator() should return a Pipeline, got {type(est)}"
    )


def test_get_estimator_has_column_transformer():
    """The pipeline must contain a ColumnTransformer step."""
    est = get_estimator()
    steps_types = [type(step) for _, step in est.steps]
    assert ColumnTransformer in steps_types, (
        "The pipeline must include a ColumnTransformer step to handle "
        "categorical and numerical features separately."
    )


def test_get_estimator_performance(churn_train, churn_test):
    """Pipeline must achieve ROC-AUC >= 0.72 on the held-out test set.

    A naive pipeline that relies on all columns will see CV AUC close to 1.0
    on the training data — which should strike you as suspicious —
    then score noticeably lower here.
    """
    X_train, y_train = _split_xy(churn_train)
    X_test, y_test = _split_xy(churn_test)

    est = get_estimator()
    est.fit(X_train, y_train)

    y_proba = est.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    assert auc >= 0.72, (
        f"Expected ROC-AUC >= 0.72 on held-out test set, got {auc:.3f}. "
        "Hint: if your CV score on test was suspiciously high, check whether "
        "all features are truly available at prediction time."
    )
