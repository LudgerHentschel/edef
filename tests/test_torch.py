import importlib.util

import numpy as np
import pytest

from edef import linear_regression_components


torch_available = importlib.util.find_spec("torch") is not None


pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="torch is not installed",
)


if torch_available:
    import torch

    from edef import TorchExplainer


class LinearTorchModel(torch.nn.Module):
    def __init__(self, coef, intercept=0.0):
        super().__init__()
        coef = torch.as_tensor(coef, dtype=torch.float32)
        self.linear = torch.nn.Linear(coef.numel(), 1, bias=True)

        with torch.no_grad():
            self.linear.weight.copy_(coef.reshape(1, -1))
            self.linear.bias.fill_(float(intercept))

    def forward(self, X):
        return self.linear(X).reshape(-1)


def test_torch_linear_model_matches_closed_form_components():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 2.0],
        ],
        dtype=np.float32,
    )

    y = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    coef = np.array([0.5, 1.5], dtype=np.float32)

    baseline = X.mean(axis=0)
    intercept = y.mean() - baseline @ coef

    model = LinearTorchModel(coef, intercept=intercept)

    explainer = TorchExplainer(
        model,
        baseline=baseline,
        n_steps=8,
        feature_names=["x1", "x2"],
    )

    result_torch = explainer(X, y)

    components = X * coef[None, :]
    result_closed = linear_regression_components(
        y,
        components,
        feature_names=["x1", "x2"],
    )

    np.testing.assert_allclose(
        result_torch.values,
        result_closed.values,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.observation_values,
        result_closed.observation_values,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.standard_errors,
        result_closed.standard_errors,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.total,
        result_closed.total,
        atol=1e-5,
    )


def test_torch_additivity_for_nonlinear_model():
    class NonlinearModel(torch.nn.Module):
        def forward(self, X):
            return X[:, 0] ** 2 + 0.5 * X[:, 1]

    X = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [2.0, 1.5],
        ],
        dtype=np.float32,
    )

    y = np.array([1.0, 0.5, 1.0, 4.0], dtype=np.float32)

    explainer = TorchExplainer(
        NonlinearModel(),
        baseline=np.zeros(2, dtype=np.float32),
        n_steps=64,
        feature_names=["x1", "x2"],
    )

    result = explainer(X, y)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-4,
    )


def test_torch_grouping_works():
    class NonlinearModel(torch.nn.Module):
        def forward(self, X):
            return X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * X[:, 2]

    X = np.array(
        [
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.5, 2.0],
            [2.0, 1.5, 1.0],
        ],
        dtype=np.float32,
    )

    y = np.array([1.0, 0.5, 1.0, 4.0], dtype=np.float32)

    explainer = TorchExplainer(
        NonlinearModel(),
        baseline=np.zeros(3, dtype=np.float32),
        n_steps=64,
        feature_names=["x1", "x2", "x3"],
    )

    result = explainer(X, y)
    grouped = result.group(["nonlinear", "linear", "linear"])

    assert grouped.feature_names == ["nonlinear", "linear"]

    np.testing.assert_allclose(
        grouped.values.sum(),
        result.total,
        atol=1e-4,
    )


def test_torch_rejects_bad_baseline_length():
    class Model(torch.nn.Module):
        def forward(self, X):
            return X[:, 0]

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)

    explainer = TorchExplainer(
        Model(),
        baseline=np.zeros(3, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="baseline must have length"):
        explainer(X, y)


def test_torch_rejects_vector_output():
    class VectorOutputModel(torch.nn.Module):
        def forward(self, X):
            return torch.stack([X[:, 0], X[:, 1]], dim=1)

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)

    explainer = TorchExplainer(
        VectorOutputModel(),
        baseline=np.zeros(2, dtype=np.float32),
    )


    with pytest.raises(ValueError, match="requires scalar output"):
        explainer(X, y)
        
def test_torch_logistic_model_matches_closed_form_components():
    from edef import linear_logistic_components

    X = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.5, 0.5],
            [1.0, 1.5],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    coef = np.array([1.0, 0.5], dtype=np.float32)
    intercept = 0.0

    model = LinearTorchModel(coef, intercept=intercept)

    components = X * coef[None, :]
    eta = X @ coef + intercept

    result_closed = linear_logistic_components(
        y,
        components,
        eta=eta,
        include_intercept_component=False,
        feature_names=["x1", "x2"],
    )

    explainer = TorchExplainer(
        model,
        baseline=np.zeros(2, dtype=np.float32),
        loss="log_loss",
        n_steps=32,
        feature_names=["x1", "x2"],
    )

    result_torch = explainer(X, y)

    np.testing.assert_allclose(
        result_torch.values,
        result_closed.values,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.observation_values,
        result_closed.observation_values,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.total,
        result_closed.total,
        atol=1e-5,
    )


def test_torch_log_loss_rejects_nonbinary_y():
    class Model(torch.nn.Module):
        def forward(self, X):
            return X[:, 0]

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float32)

    explainer = TorchExplainer(
        Model(),
        baseline=np.zeros(2, dtype=np.float32),
        loss="log_loss",
    )

    with pytest.raises(ValueError, match="binary labels"):
        explainer(X, y)        
        
class MulticlassLinearTorchModel(torch.nn.Module):
    def __init__(self, coef, intercept=None):
        super().__init__()

        coef = torch.as_tensor(coef, dtype=torch.float32)
        n_classes, n_features = coef.shape

        self.linear = torch.nn.Linear(n_features, n_classes, bias=True)

        with torch.no_grad():
            self.linear.weight.copy_(coef)

            if intercept is None:
                self.linear.bias.zero_()
            else:
                intercept = torch.as_tensor(intercept, dtype=torch.float32)
                self.linear.bias.copy_(intercept)

    def forward(self, X):
        return self.linear(X)


def test_torch_multiclass_linear_model_matches_closed_form_components():
    from edef import linear_multiclass_components

    X = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 0.5],
            [-1.0, 1.5],
            [1.5, -0.5],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    y = np.array([0, 1, 2, 1, 0, 2], dtype=np.float32)

    coef = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, 0.25],
        ],
        dtype=np.float32,
    )

    intercept = np.zeros(3, dtype=np.float32)

    model = MulticlassLinearTorchModel(coef, intercept=intercept)

    explainer = TorchExplainer(
        model,
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
        n_steps=32,
        feature_names=["x1", "x2"],
    )

    result_torch = explainer(X, y)

    components = X[:, None, :] * coef[None, :, :]
    eta = X @ coef.T + intercept[None, :]

    result_closed = linear_multiclass_components(
        y.astype(int),
        components,
        eta=eta,
        include_intercept_component=False,
        feature_names=["x1", "x2"],
    )

    np.testing.assert_allclose(
        result_torch.values,
        result_closed.values,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.observation_values,
        result_closed.observation_values,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result_torch.total,
        result_closed.total,
        atol=1e-5,
    )

    assert result_torch.loss == "multiclass_log_loss"
    assert result_torch.model_type == "torch_multiclass_classification"


def test_torch_multiclass_nonlinear_additivity():
    class NonlinearMulticlassModel(torch.nn.Module):
        def forward(self, X):
            z0 = 1.0 * X[:, 0] ** 2
            z1 = 0.8 * X[:, 1]
            z2 = -0.5 * X[:, 0] + 0.25 * X[:, 1] ** 2
            return torch.stack([z0, z1, z2], dim=1)

    X = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [2.0, 1.5],
            [-0.5, -1.0],
            [0.75, -0.25],
        ],
        dtype=np.float32,
    )

    y = np.array([0, 1, 2, 0, 2, 1], dtype=np.float32)

    explainer = TorchExplainer(
        NonlinearMulticlassModel(),
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
        n_steps=64,
        feature_names=["x1", "x2"],
    )

    result = explainer(X, y)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-5,
    )


def test_torch_multiclass_rejects_vector_labels():
    class Model(torch.nn.Module):
        def forward(self, X):
            return torch.zeros((X.shape[0], 3), dtype=X.dtype)

    X = np.zeros((4, 2), dtype=np.float32)

    # These are not class labels.
    y = np.array([0.0, 1.0, 1.5, 2.0], dtype=np.float32)

    explainer = TorchExplainer(
        Model(),
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
    )

    with pytest.raises(ValueError, match="integer class labels"):
        explainer(X, y)


def test_torch_multiclass_rejects_scalar_output():
    class ScalarModel(torch.nn.Module):
        def forward(self, X):
            return X[:, 0]

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.float32)

    explainer = TorchExplainer(
        ScalarModel(),
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
    )

    with pytest.raises(ValueError, match="model output with shape"):
        explainer(X, y)        