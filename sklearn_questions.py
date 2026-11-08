import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
        if not isinstance(X, pd.DataFrame) or "registration_month" not in X:
            raise ValueError(
                "X must be a pandas DataFrame containing 'registration_month'."
            )

        if not pd.api.types.is_datetime64_any_dtype(X["registration_month"]):
            raise ValueError("'registration_month' must be of datetime dtype.")

        periods = X["registration_month"].dt.to_period("M")
        unique_periods = sorted(periods.unique())
        y_array = np.asarray(y)

        for period in unique_periods:
            m_train = period
            m_test = period + 1

            if m_test not in unique_periods:
                continue

            idx_train = np.where(periods == m_train)[0]
            idx_test = np.where(periods == m_test)[0]

            y_train = y_array[idx_train]
            y_test = y_array[idx_test]

            if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
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

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean"))]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    estimator = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
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
    print(f"CV results : \n{scores.round(3)}")
    print(f"Mean CV AUC   : {scores['test_score'].mean():.3f}")
