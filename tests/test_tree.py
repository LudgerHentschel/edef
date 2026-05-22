import importlib.util

import numpy as np
import pytest

from edef import TreeExplainer


sklearn_available = importlib.util.find_spec("sklearn") is not None
treeig_available = importlib.util.find_spec("treeig") is not None


pytestmark = pytest.mark.skipif(
    not (sklearn_available and treeig_available),
    reason="sklearn and treeig are required for tree tests",
)


if sklearn_available:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor


def make_regression_data(n=120, p=4, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (
        1.0 * X[:, 0]
        - 0.8 * X[:, 1] ** 2
        + 0.5 * X[:, 2]
        + rng.normal(scale=0.1, size=n)
    )
    return X, y


def test_tree_explainer_decision_tree_additivity():
    X, y = make_regression_data()
    baseline = X.mean(axis=0)
    X_eval = X[:30]
    y_eval = y[:30]

    model = DecisionTreeRegressor(max_depth=4, random_state=0)
    model.fit(X, y)

    explainer = TreeExplainer(
        model,
        baseline=baseline,
        feature_names=["x1", "x2", "x3", "x4"],
    )

    result = explainer(X_eval, y_eval)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-8,
    )

    assert result.observation_values.shape == X_eval.shape
    assert result.values.shape == (X_eval.shape[1],)
    assert result.standard_errors.shape == (X_eval.shape[1],)
    assert result.feature_names == ["x1", "x2", "x3", "x4"]


def test_tree_explainer_random_forest_additivity():
    X, y = make_regression_data(seed=456)
    baseline = X.mean(axis=0)
    X_eval = X[:25]
    y_eval = y[:25]

    model = RandomForestRegressor(
        n_estimators=8,
        max_depth=4,
        random_state=0,
        n_jobs=1,
    )
    model.fit(X, y)

    explainer = TreeExplainer(model, baseline=baseline)
    result = explainer(X_eval, y_eval)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-8,
    )


def test_tree_explainer_grouping_works():
    X, y = make_regression_data(seed=789)
    baseline = X.mean(axis=0)
    X_eval = X[:20]
    y_eval = y[:20]

    model = RandomForestRegressor(
        n_estimators=6,
        max_depth=3,
        random_state=0,
        n_jobs=1,
    )
    model.fit(X, y)

    explainer = TreeExplainer(
        model,
        baseline=baseline,
        feature_names=["a1", "a2", "b", "c"],
    )
    result = explainer(X_eval, y_eval)
    grouped = result.group(["a", "a", "b", "c"])

    assert grouped.feature_names == ["a", "b", "c"]

    np.testing.assert_allclose(
        grouped.values.sum(),
        result.total,
        atol=1e-8,
    )


def test_tree_explainer_rejects_bad_y_length():
    X, y = make_regression_data()
    baseline = X.mean(axis=0)

    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = TreeExplainer(model, baseline=baseline)

    with pytest.raises(ValueError, match="same number of observations"):
        explainer(X[:10], y[:9])


def test_tree_explainer_rejects_unknown_loss():
    X, y = make_regression_data()
    baseline = X.mean(axis=0)

    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)

    with pytest.raises(ValueError, match="squared_error, log_loss, and multiclass_log_loss"):
        TreeExplainer(model, baseline=baseline, loss="absolute_error")
                
def make_binary_classification_data(n=160, p=4, seed=321):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n, p))

    score = (
        1.2 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.6 * X[:, 2] * X[:, 3]
    )

    y = (score > np.median(score)).astype(float)

    return X, y


def test_tree_explainer_binary_gradient_boosting_log_loss_additivity():
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_binary_classification_data()

    baseline = X.mean(axis=0)

    model = GradientBoostingClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.07,
        random_state=0,
    )
    model.fit(X, y)

    explainer = TreeExplainer(
        model,
        baseline=baseline,
        loss="log_loss",
        target=1,
        feature_names=["x1", "x2", "x3", "x4"],
    )

    result = explainer(X[:40], y[:40])

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-8,
    )

    assert result.loss == "log_loss"
    assert result.model_type == "tree_classification"


def test_tree_explainer_binary_xgboost_log_loss_additivity():
    xgb = pytest.importorskip("xgboost")

    X, y = make_binary_classification_data(seed=654)

    baseline = X.mean(axis=0)

    model = xgb.XGBClassifier(
        n_estimators=16,
        max_depth=3,
        learning_rate=0.08,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)

    explainer = TreeExplainer(
        model,
        baseline=baseline,
        loss="log_loss",
        target=1,
    )

    result = explainer(X[:35], y[:35])

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-6,
    )


def test_tree_explainer_log_loss_rejects_non_binary_y():
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_binary_classification_data()

    model = GradientBoostingClassifier(
        n_estimators=10,
        max_depth=2,
        random_state=0,
    )
    model.fit(X, y)

    explainer = TreeExplainer(
        model,
        baseline=X.mean(axis=0),
        loss="log_loss",
        target=1,
    )

    bad_y = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="binary labels"):
        explainer(X[:3], bad_y)        