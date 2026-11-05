import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class RegistrationSplit(BaseCrossValidator):


    def get_n_splits(self, X, y=None, groups=None):
        return sum(1 for _ in self.split(X, y, groups))

    def split(self, X, y, groups=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if "registration_month" not in X.columns:
            raise ValueError("X must contain a 'registration_month' column.")
        if not pd.api.types.is_datetime64_any_dtype(X["registration_month"]):
            raise ValueError("'registration_month' must have datetime dtype.")
        if y is None:
            raise ValueError("y must be provided to check both churn classes.")

        target = pd.Series(y).reset_index(drop=True)
        if len(target) != len(X):
            raise ValueError("X and y must contain the same number of rows.")

        months = X["registration_month"].dt.to_period("M").reset_index(drop=True)
        unique_months = sorted(months.dropna().unique())

        for month in unique_months:
            next_month = month + 1
            train_idx = (months == month).to_numpy().nonzero()[0]
            test_idx = (months == next_month).to_numpy().nonzero()[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if target.iloc[train_idx].nunique() < 2:
                continue
            if target.iloc[test_idx].nunique() < 2:
                continue

            yield train_idx, test_idx


def get_estimator():

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

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                SimpleImputer(strategy="median"),
                numerical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
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

    scores = cross_validate(get_estimator(), X, y, cv=cv, scoring="roc_auc")
    scores = pd.DataFrame(scores)
    print(f"CV results : {scores.round(3)}")
    print(f"Mean CV AUC   : {scores['test_score'].mean():.3f}")
