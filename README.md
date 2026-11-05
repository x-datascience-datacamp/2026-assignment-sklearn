# Assignment for the DataCamp course X-DataScience Master - scikit-learn API: Pipeline, ColumnTransformer and CV splitter

## What we want you to check that you know how to do by doing this assignment:

- Use Git and GitHub
- Work with Python files (and not just notebooks!)
- Do a pull request on a GitHub repository
- Format your code properly using standard Python conventions
- Make your code pass tests run automatically on a continuous integration system (GitHub actions)
- Build a scikit-learn `Pipeline` with a `ColumnTransformer`
- Code a scikit-learn compatible cross-validator for data that changes over time

## How?

- Fork the repository by clicking on the `Fork` button on the upper right corner
- Clone the repository of your fork with: `git clone https://github.com/MYLOGIN/YEAR-assignment-sklearn` (replace MYLOGIN with your GitHub login and YEAR with the year in the name of the repository you forked)
- Create a branch called `MYLOGIN` using `git checkout -b MYLOGIN`
- Make the changes to complete the assignment. You have to modify the files that contain `questions` in their name. Do not modify the files that start with `test_`.
- Check locally that your solution meets the tests by running `pytest` from the root of the repo. You may need to install `pytest` using `pip` or `conda`.
- Check the code formatting of your solution with `ruff check`. You may need to install `ruff` using `pip` or `conda`.
- Open the pull request on GitHub:
  - Create a commit with `git add -u` and `git commit -m "UP my solution"`
  - Push your branch on your fork: `git push -u origin MYLOGIN`
  - Go to your repo in your browser and click the `Open a PR` button.
- Keep pushing to your branch until the continuous integration system is green.
- When it is green, put the link to the PR in the course's spreadsheet.

## Context

The data comes from the event log of a music streaming platform.
`data/churn_train.csv` contains one row per user, with features aggregated
from the raw log. The goal is to predict whether a user will cancel their
subscription (`churn = 1`). The columns are described in `sklearn_questions.py`.

# Your mission

### Question 1: `RegistrationSplit`

Implement the `RegistrationSplit` cross-validator in `sklearn_questions.py`.

It splits users by their **registration month**: train on month M, evaluate on
month M+1. Only folds where both the train and test cohorts contain at least
one churner *and* one non-churner are kept.

### Question 2: `get_estimator()`

Implement `get_estimator()` in `sklearn_questions.py`. It must return an
unfitted scikit-learn `Pipeline` that:

1. Contains a `ColumnTransformer` handling categorical (`gender`, `level`) and
   numerical columns separately.
2. Handles missing values in `rate_errors`.
3. Ends with a classifier of your choice.
4. Achieves **ROC-AUC ≥ 0.72** on the held-out test cohort.

> **Tip:** explore the data before writing the pipeline. Look at the
> relationship between `cancel_initiated` and `churn` in the training data.
> Will that relationship hold for users you have never seen before?

You can run `python sklearn_questions.py` to print the folds of your splitter
and the cross-validation scores of your pipeline.
