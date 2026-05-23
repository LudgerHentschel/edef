import numpy as np
import pytest

from edef import NumericalExplainer


class LinearRegressionLike:
    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef


class BinaryLogisticLike:
    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)

    def predict_proba(self, X):
        eta = np.asarray(X, dtype=float) @ self.coef
        p = 1.0 / (1.0 + np.exp(-eta))
        return np.column_stack([1.0 - p, p])


class MulticlassLogisticLike:
    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)

    def predict_proba(self, X):
        eta = np.asarray(X, dtype=float) @ self.coef.T
        eta = eta - eta.max(axis=1, keepdims=True)
        p = np.exp(eta)
        return p / p.sum(axis=1, keepdims=True)


def _binary_log_loss(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _multiclass_log_loss(y, p):
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(y.shape[0]), y.astype(int)])


def test_numerical_regression_linear_additivity():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 2.0],
        ],
        dtype=float,
    )

    y = np.array([1.0, 2.0, 3.0, 5.0], dtype=float)
    coef = np.array([0.5, 1.5], dtype=float)

    model = LinearRegressionLike(coef)

    result = NumericalExplainer(
        model,
        baseline=np.zeros(2),
        loss="squared_error",
        n_steps=16,
        step_size=1e-5,
        feature_names=["x1", "x2"],
    )(X, y)

    pred0 = np.zeros_like(y)
    pred = X @ coef

    total = np.mean((y - pred0) ** 2) - np.mean((y - pred) ** 2)

    np.testing.assert_allclose(result.values.sum(), total, atol=1e-5)
    np.testing.assert_allclose(result.total, total, atol=1e-5)
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-5)

    assert result.loss == "squared_error"
    assert result.model_type == "numerical_regression"


def test_numerical_binary_logistic_linear_additivity():
    X = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.5, 0.5],
            [1.0, 1.5],
            [1.5, -0.5],
        ],
        dtype=float,
    )

    y = np.array([0, 0, 1, 1, 1], dtype=float)
    coef = np.array([1.0, 0.5], dtype=float)

    model = BinaryLogisticLike(coef)

    result = NumericalExplainer(
        model,
        baseline=np.zeros(2),
        loss="log_loss",
        n_steps=32,
        step_size=1e-5,
        feature_names=["x1", "x2"],
    )(X, y)

    p0 = np.full_like(y, 0.5, dtype=float)

    eta = X @ coef
    p = 1.0 / (1.0 + np.exp(-eta))

    baseline_loss = np.mean(_binary_log_loss(y, p0))
    model_loss = np.mean(_binary_log_loss(y, p))

    total = baseline_loss - model_loss

    np.testing.assert_allclose(result.values.sum(), total, atol=1e-5)
    np.testing.assert_allclose(result.total, total, atol=1e-5)
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-5)

    assert result.loss == "log_loss"
    assert result.model_type == "numerical_classification"


def test_numerical_multiclass_logistic_linear_additivity():
    X = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 0.5],
            [-1.0, 1.5],
            [1.5, -0.5],
            [0.0, -1.0],
        ],
        dtype=float,
    )

    y = np.array([0, 1, 2, 1, 0, 2], dtype=int)

    coef = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, 0.25],
        ],
        dtype=float,
    )

    model = MulticlassLogisticLike(coef)

    result = NumericalExplainer(
        model,
        baseline=np.zeros(2),
        loss="multiclass_log_loss",
        n_steps=32,
        step_size=1e-5,
        feature_names=["x1", "x2"],
    )(X, y)

    p0 = np.full((X.shape[0], 3), 1.0 / 3.0)

    eta = X @ coef.T
    eta = eta - eta.max(axis=1, keepdims=True)
    p = np.exp(eta)
    p = p / p.sum(axis=1, keepdims=True)

    baseline_loss = np.mean(_multiclass_log_loss(y, p0))
    model_loss = np.mean(_multiclass_log_loss(y, p))

    total = baseline_loss - model_loss

    np.testing.assert_allclose(result.values.sum(), total, atol=1e-5)
    np.testing.assert_allclose(result.total, total, atol=1e-5)
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-5)

    assert result.loss == "multiclass_log_loss"
    assert result.model_type == "numerical_multiclass_classification"


def test_numerical_rejects_bad_loss():
    model = LinearRegressionLike([1.0, 2.0])

    with pytest.raises(ValueError, match="squared_error"):
        NumericalExplainer(
            model,
            baseline=np.zeros(2),
            loss="absolute_error",
        )


def test_numerical_binary_rejects_nonbinary_y():
    model = BinaryLogisticLike([1.0, 2.0])
    X = np.zeros((4, 2))
    y = np.array([0.0, 1.0, 2.0, 1.0])

    explainer = NumericalExplainer(
        model,
        baseline=np.zeros(2),
        loss="log_loss",
    )

    with pytest.raises(ValueError, match="binary labels"):
        explainer(X, y)


def test_numerical_multiclass_rejects_noninteger_y():
    model = MulticlassLogisticLike(np.eye(3, 2))
    X = np.zeros((4, 2))
    y = np.array([0.0, 1.0, 1.5, 2.0])

    explainer = NumericalExplainer(
        model,
        baseline=np.zeros(2),
        loss="multiclass_log_loss",
    )

    with pytest.raises(ValueError, match="integer class labels"):
        explainer(X, y)