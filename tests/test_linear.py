import numpy as np
import pytest

from edef import linear_regression_components, linear_logistic_components, linear_multiclass_components
from edef import LinearExplainer

def test_additivity():
    y = np.array([1.0, 2.0, 4.0, 5.0])

    components = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
        ]
    )

    result = linear_regression_components(y, components)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-12,
    )


def test_observation_values_match_formula():
    y = np.array([1.0, 2.0, 4.0, 5.0])

    components = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
        ]
    )

    result = linear_regression_components(y, components)

    y_c = y - y.mean()
    components_c = components - components.mean(axis=0)
    prediction_c = components_c.sum(axis=1)

    expected_observation_values = components_c * (2.0 * y_c - prediction_c)[:, None]

    np.testing.assert_allclose(
        result.observation_values,
        expected_observation_values,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.values,
        expected_observation_values.mean(axis=0),
        atol=1e-12,
    )


def test_total_equals_baseline_loss_minus_model_loss():
    y = np.array([1.0, 2.0, 4.0, 5.0])

    components = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
        ]
    )

    result = linear_regression_components(y, components)

    y_c = y - y.mean()
    components_c = components - components.mean(axis=0)
    prediction_c = components_c.sum(axis=1)

    expected_baseline_loss = np.mean(y_c**2)
    expected_model_loss = np.mean((y_c - prediction_c) ** 2)
    expected_total = expected_baseline_loss - expected_model_loss

    np.testing.assert_allclose(result.baseline_loss, expected_baseline_loss)
    np.testing.assert_allclose(result.model_loss, expected_model_loss)
    np.testing.assert_allclose(result.total, expected_total)


def test_standard_errors_match_sample_formula():
    y = np.array([1.0, 2.0, 4.0, 5.0])

    components = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
        ]
    )

    result = linear_regression_components(y, components)

    expected_se = result.observation_values.std(axis=0, ddof=1) / np.sqrt(len(y))

    np.testing.assert_allclose(
        result.standard_errors,
        expected_se,
        atol=1e-12,
    )


def test_zero_signal():
    y = np.array([1.0, 2.0, 4.0, 5.0])
    components = np.zeros((4, 3))

    result = linear_regression_components(y, components)

    np.testing.assert_allclose(result.values, np.zeros(3))
    np.testing.assert_allclose(result.observation_values, np.zeros((4, 3)))
    np.testing.assert_allclose(result.standard_errors, np.zeros(3))
    np.testing.assert_allclose(result.total, 0.0)
    np.testing.assert_allclose(result.proportions, np.full(3, np.nan))


def test_feature_names_are_preserved():
    y = np.array([1.0, 2.0, 4.0, 5.0])
    components = np.zeros((4, 2))

    result = linear_regression_components(
        y,
        components,
        feature_names=["a", "b"],
    )

    assert result.feature_names == ["a", "b"]


def test_default_feature_names():
    y = np.array([1.0, 2.0, 4.0, 5.0])
    components = np.zeros((4, 3))

    result = linear_regression_components(y, components)

    assert result.feature_names == ["x0", "x1", "x2"]


def test_bad_component_dimension_raises():
    y = np.array([1.0, 2.0, 3.0])
    components = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="components must have shape"):
        linear_regression_components(y, components)


def test_row_mismatch_raises():
    y = np.array([1.0, 2.0, 3.0])
    components = np.zeros((4, 2))

    with pytest.raises(ValueError, match="same number of observations"):
        linear_regression_components(y, components)


def test_feature_name_length_mismatch_raises():
    y = np.array([1.0, 2.0, 3.0])
    components = np.zeros((3, 2))

    with pytest.raises(ValueError, match="feature_names must have length"):
        linear_regression_components(
            y,
            components,
            feature_names=["a"],
        )


def test_nonfinite_y_raises():
    y = np.array([1.0, np.nan, 3.0])
    components = np.zeros((3, 2))

    with pytest.raises(ValueError, match="y must contain only finite"):
        linear_regression_components(y, components)


def test_nonfinite_components_raises():
    y = np.array([1.0, 2.0, 3.0])
    components = np.array(
        [
            [0.0, 0.0],
            [1.0, np.inf],
            [2.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="components must contain only finite"):
        linear_regression_components(y, components)


def test_at_least_two_observations_required():
    y = np.array([1.0])
    components = np.zeros((1, 2))

    with pytest.raises(ValueError, match="At least two observations"):
        linear_regression_components(y, components)
        
class DummyLinearModel:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef, dtype=float)


def test_linear_explainer_matches_component_function():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 2.0],
        ]
    )
    y = np.array([1.0, 2.0, 3.0, 5.0])
    coef = np.array([0.5, 1.5])

    model = DummyLinearModel(coef)

    explainer = LinearExplainer(model)
    result_model = explainer(X, y)

    components = X * coef[None, :]
    result_components = linear_regression_components(y, components)

    np.testing.assert_allclose(result_model.values, result_components.values)
    np.testing.assert_allclose(
        result_model.observation_values,
        result_components.observation_values,
    )
    np.testing.assert_allclose(
        result_model.standard_errors,
        result_components.standard_errors,
    )
    np.testing.assert_allclose(result_model.total, result_components.total)


def test_linear_explainer_rejects_non_linear_model():
    class BadModel:
        pass

    with pytest.raises(TypeError, match="coef_"):
        LinearExplainer(BadModel())


def test_linear_explainer_squared_error_rejects_multiclass_coef():
    model = DummyLinearModel(np.ones((2, 3)))
    explainer = LinearExplainer(model, loss="squared_error")

    X = np.zeros((4, 3))
    y = np.zeros(4)

    with pytest.raises(ValueError, match="squared_error requires"):
        explainer(X, y)
        

def test_linear_explainer_rejects_dimension_mismatch():
    model = DummyLinearModel([1.0, 2.0, 3.0])
    explainer = LinearExplainer(model)

    X = np.zeros((4, 2))
    y = np.zeros(4)

    with pytest.raises(ValueError, match="coefficient dimension"):
        explainer(X, y)


def test_linear_explainer_feature_names():
    model = DummyLinearModel([1.0, 2.0])
    explainer = LinearExplainer(model, feature_names=["a", "b"])

    X = np.zeros((4, 2))
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = explainer(X, y)

    assert result.feature_names == ["a", "b"]
    
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z):
    return np.logaddexp(0.0, z)


def binary_log_loss(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def test_linear_logistic_additivity():
    y = np.array([0.0, 0.0, 1.0, 1.0])

    components = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.5, 0.3],
            [1.0, 0.5],
        ]
    )

    result = linear_logistic_components(y, components)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-12,
    )


def test_linear_logistic_observation_values_match_formula():
    y = np.array([0.0, 0.0, 1.0, 1.0])

    components = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.5, 0.3],
            [1.0, 0.5],
        ]
    )

    result = linear_logistic_components(y, components)

    p_bar = y.mean()
    eta_bar = np.log(p_bar / (1.0 - p_bar))
    eta = eta_bar + components.sum(axis=1)

    delta = eta - eta_bar

    expected_path_weight = np.empty_like(y)
    mask = np.abs(delta) > 1e-12

    expected_path_weight[mask] = (
        y[mask]
        - (softplus(eta[mask]) - softplus(eta_bar)) / delta[mask]
    )
    expected_path_weight[~mask] = y[~mask] - sigmoid(eta_bar)

    expected_observation_values = components * expected_path_weight[:, None]

    np.testing.assert_allclose(
        result.observation_values,
        expected_observation_values,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.values,
        expected_observation_values.mean(axis=0),
        atol=1e-12,
    )


def test_linear_logistic_total_equals_baseline_loss_minus_model_loss():
    y = np.array([0.0, 0.0, 1.0, 1.0])

    components = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.5, 0.3],
            [1.0, 0.5],
        ]
    )

    result = linear_logistic_components(y, components)

    p_bar = y.mean()
    eta_bar = np.log(p_bar / (1.0 - p_bar))
    eta = eta_bar + components.sum(axis=1)
    p_hat = sigmoid(eta)

    expected_baseline_loss = np.mean(binary_log_loss(y, p_bar))
    expected_model_loss = np.mean(binary_log_loss(y, p_hat))
    expected_total = expected_baseline_loss - expected_model_loss

    np.testing.assert_allclose(result.baseline_loss, expected_baseline_loss)
    np.testing.assert_allclose(result.model_loss, expected_model_loss)
    np.testing.assert_allclose(result.total, expected_total)


def test_linear_logistic_standard_errors_match_sample_formula():
    y = np.array([0.0, 0.0, 1.0, 1.0])

    components = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.5, 0.3],
            [1.0, 0.5],
        ]
    )

    result = linear_logistic_components(y, components)

    expected_se = result.observation_values.std(axis=0, ddof=1) / np.sqrt(len(y))

    np.testing.assert_allclose(
        result.standard_errors,
        expected_se,
        atol=1e-12,
    )


def test_linear_logistic_zero_components():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    components = np.zeros((4, 3))

    result = linear_logistic_components(y, components)

    np.testing.assert_allclose(result.values, np.zeros(3))
    np.testing.assert_allclose(result.observation_values, np.zeros((4, 3)))
    np.testing.assert_allclose(result.standard_errors, np.zeros(3))
    np.testing.assert_allclose(result.total, 0.0)
    np.testing.assert_allclose(result.proportions, np.full(3, np.nan))


def test_linear_logistic_delta_zero_limit_case():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    components = np.zeros((4, 2))

    result = linear_logistic_components(y, components)

    p_bar = y.mean()
    eta_bar = np.log(p_bar / (1.0 - p_bar))
    expected_path_weight = y - sigmoid(eta_bar)

    expected_observation_values = components * expected_path_weight[:, None]

    np.testing.assert_allclose(
        result.observation_values,
        expected_observation_values,
        atol=1e-12,
    )


def test_linear_logistic_include_intercept_component():
    y = np.array([0.0, 0.0, 1.0, 1.0])

    components = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.5, 0.3],
            [1.0, 0.5],
        ]
    )

    p_bar = y.mean()
    eta_bar = np.log(p_bar / (1.0 - p_bar))

    intercept_shift = 0.3
    eta = eta_bar + components.sum(axis=1) + intercept_shift

    result = linear_logistic_components(
        y,
        components,
        eta=eta,
        include_intercept_component=True,
        feature_names=["a", "b"],
    )

    assert result.feature_names == ["a", "b", "__InterceptShift__"]

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-12,
    )

    expected_intercept_component = np.full(len(y), intercept_shift)

    # The appended observation column should equal intercept_component * path_weight.
    appended = result.observation_values[:, -1]

    delta = eta - eta_bar
    mask = np.abs(delta) > 1e-12
    expected_path_weight = np.empty_like(y)
    expected_path_weight[mask] = (
        y[mask] - (softplus(eta[mask]) - softplus(eta_bar)) / delta[mask]
    )
    expected_path_weight[~mask] = y[~mask] - sigmoid(eta_bar)

    np.testing.assert_allclose(
        appended,
        expected_intercept_component * expected_path_weight,
        atol=1e-12,
    )


def test_linear_logistic_rejects_nonbinary_y():
    y = np.array([0.0, 1.0, 2.0])
    components = np.zeros((3, 2))

    with pytest.raises(ValueError, match="binary labels"):
        linear_logistic_components(y, components)


def test_linear_logistic_rejects_eta_length_mismatch():
    y = np.array([0.0, 1.0, 1.0])
    components = np.zeros((3, 2))
    eta = np.zeros(4)

    with pytest.raises(ValueError, match="eta must have length"):
        linear_logistic_components(y, components, eta=eta)    
        
class DummyLogisticModel:
    def __init__(self, coef, intercept=0.0):
        self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=float)

    def decision_function(self, X):
        return X @ self.coef_.reshape(-1) + self.intercept_[0]


def test_linear_explainer_log_loss_matches_component_function():
    X = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.5, 0.5],
            [1.0, 1.5],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0])

    coef = np.array([1.0, 0.5])
    intercept = 0.3

    model = DummyLogisticModel(coef, intercept=intercept)

    explainer = LinearExplainer(
        model,
        loss="log_loss",
        feature_names=["a", "b"],
    )

    result_model = explainer(X, y)

    components = X * coef[None, :]
    eta = model.decision_function(X)

    result_components = linear_logistic_components(
        y,
        components,
        eta=eta,
        include_intercept_component=True,
        feature_names=["a", "b"],
    )

    np.testing.assert_allclose(result_model.values, result_components.values)
    np.testing.assert_allclose(
        result_model.observation_values,
        result_components.observation_values,
    )
    np.testing.assert_allclose(
        result_model.standard_errors,
        result_components.standard_errors,
    )
    np.testing.assert_allclose(result_model.total, result_components.total)

    assert result_model.feature_names == ["a", "b", "__InterceptShift__"]


def test_linear_explainer_log_loss_without_decision_function_uses_intercept():
    class ModelWithoutDecisionFunction:
        def __init__(self, coef, intercept):
            self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)
            self.intercept_ = np.asarray([intercept], dtype=float)

    X = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.5, 0.5],
            [1.0, 1.5],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0])

    coef = np.array([1.0, 0.5])
    intercept = -0.2

    model = ModelWithoutDecisionFunction(coef, intercept)
    explainer = LinearExplainer(model, loss="log_loss")

    result = explainer(X, y)

    components = X * coef[None, :]
    eta = X @ coef + intercept

    expected = linear_logistic_components(
        y,
        components,
        eta=eta,
        include_intercept_component=True,
    )

    np.testing.assert_allclose(result.values, expected.values)
    np.testing.assert_allclose(result.total, expected.total)


def test_linear_explainer_log_loss_requires_score_source():
    class ModelWithoutScore:
        def __init__(self, coef):
            self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)

    model = ModelWithoutScore([1.0, 2.0])
    explainer = LinearExplainer(model, loss="log_loss")

    X = np.zeros((4, 2))
    y = np.array([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(TypeError, match="decision_function or intercept_"):
        explainer(X, y)


def test_linear_explainer_rejects_unknown_loss():
    model = DummyLinearModel([1.0, 2.0])

    with pytest.raises(ValueError, match="squared_error and log_loss"):
        LinearExplainer(model, loss="absolute_error")        
        
def test_grouped_values_add_features():
    y = np.array([1.0, 2.0, 4.0, 5.0])

    components = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [1.5, 1.0, 2.0],
        ]
    )

    result = linear_regression_components(
        y,
        components,
        feature_names=["a1", "a2", "b"],
    )

    grouped = result.group(["a", "a", "b"])

    assert grouped.feature_names == ["a", "b"]

    expected_observation_values = np.column_stack(
        [
            result.observation_values[:, 0] + result.observation_values[:, 1],
            result.observation_values[:, 2],
        ]
    )

    np.testing.assert_allclose(
        grouped.observation_values,
        expected_observation_values,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        grouped.values,
        expected_observation_values.mean(axis=0),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        grouped.values.sum(),
        result.total,
        atol=1e-12,
    )


def test_grouped_standard_errors_use_grouped_observation_values():
    y = np.array([1.0, 2.0, 4.0, 5.0])

    components = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [1.5, 1.0, 2.0],
        ]
    )

    result = linear_regression_components(y, components)
    grouped = result.group(["a", "a", "b"])

    expected_se = grouped.observation_values.std(axis=0, ddof=1) / np.sqrt(result.n_obs)

    np.testing.assert_allclose(grouped.standard_errors, expected_se, atol=1e-12)


def test_group_length_mismatch_raises():
    y = np.array([1.0, 2.0, 4.0, 5.0])
    components = np.zeros((4, 3))

    result = linear_regression_components(y, components)

    with pytest.raises(ValueError, match="groups must have length"):
        result.group(["a", "b"])        
        
def test_linear_multiclass_additivity():
    y = np.array([0, 1, 2, 1, 0, 2])

    components = np.array(
        [
            [[0.2, 0.0], [0.0, 0.1], [-0.1, 0.0]],
            [[0.1, 0.2], [0.3, 0.0], [-0.2, 0.1]],
            [[0.0, 0.1], [0.1, 0.2], [0.4, 0.3]],
            [[0.2, 0.0], [0.4, 0.1], [-0.1, 0.2]],
            [[0.3, 0.1], [0.0, 0.2], [-0.2, 0.0]],
            [[0.0, 0.0], [0.2, 0.1], [0.5, 0.2]],
        ],
        dtype=float,
    )

    result = linear_multiclass_components(y, components)

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        result.additivity_error,
        0.0,
        atol=1e-10,
    )


def test_linear_multiclass_shapes_and_feature_names():
    y = np.array([0, 1, 2, 1, 0, 2])
    components = np.zeros((6, 3, 2))

    result = linear_multiclass_components(
        y,
        components,
        feature_names=["a", "b"],
    )

    assert result.values.shape == (2,)
    assert result.observation_values.shape == (6, 2)
    assert result.standard_errors.shape == (2,)
    assert result.feature_names == ["a", "b"]


def test_linear_multiclass_zero_components():
    y = np.array([0, 1, 2, 1, 0, 2])
    components = np.zeros((6, 3, 2))

    result = linear_multiclass_components(y, components)

    np.testing.assert_allclose(result.values, np.zeros(2))
    np.testing.assert_allclose(result.observation_values, np.zeros((6, 2)))
    np.testing.assert_allclose(result.standard_errors, np.zeros(2))
    np.testing.assert_allclose(result.total, 0.0)
    np.testing.assert_allclose(result.proportions, np.full(2, np.nan))


def test_linear_multiclass_total_equals_baseline_minus_model_loss():
    y = np.array([0, 1, 2, 1, 0, 2])

    components = np.array(
        [
            [[0.2, 0.0], [0.0, 0.1], [-0.1, 0.0]],
            [[0.1, 0.2], [0.3, 0.0], [-0.2, 0.1]],
            [[0.0, 0.1], [0.1, 0.2], [0.4, 0.3]],
            [[0.2, 0.0], [0.4, 0.1], [-0.1, 0.2]],
            [[0.3, 0.1], [0.0, 0.2], [-0.2, 0.0]],
            [[0.0, 0.0], [0.2, 0.1], [0.5, 0.2]],
        ],
        dtype=float,
    )

    result = linear_multiclass_components(y, components)

    n_obs = len(y)
    n_classes = components.shape[1]

    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    class_probs = class_counts / n_obs

    eta_bar = np.log(class_probs)
    eta = eta_bar.reshape(1, -1) + components.sum(axis=2)

    log_denom = np.log(np.exp(eta).sum(axis=1))
    log_probs = eta - log_denom[:, None]

    expected_baseline_loss = -np.mean(np.log(class_probs[y]))
    expected_model_loss = -np.mean(log_probs[np.arange(n_obs), y])
    expected_total = expected_baseline_loss - expected_model_loss

    np.testing.assert_allclose(result.baseline_loss, expected_baseline_loss)
    np.testing.assert_allclose(result.model_loss, expected_model_loss)
    np.testing.assert_allclose(result.total, expected_total)


def test_linear_multiclass_include_intercept_component():
    y = np.array([0, 1, 2, 1, 0, 2])

    components = np.array(
        [
            [[0.2, 0.0], [0.0, 0.1], [-0.1, 0.0]],
            [[0.1, 0.2], [0.3, 0.0], [-0.2, 0.1]],
            [[0.0, 0.1], [0.1, 0.2], [0.4, 0.3]],
            [[0.2, 0.0], [0.4, 0.1], [-0.1, 0.2]],
            [[0.3, 0.1], [0.0, 0.2], [-0.2, 0.0]],
            [[0.0, 0.0], [0.2, 0.1], [0.5, 0.2]],
        ],
        dtype=float,
    )

    n_obs, n_classes, _ = components.shape

    class_probs = np.bincount(y, minlength=n_classes).astype(float) / n_obs
    eta_bar = np.log(class_probs)

    intercept_shift = np.array([0.1, -0.2, 0.3])
    eta = (
        eta_bar.reshape(1, -1)
        + components.sum(axis=2)
        + intercept_shift.reshape(1, -1)
    )

    result = linear_multiclass_components(
        y,
        components,
        eta=eta,
        include_intercept_component=True,
        feature_names=["a", "b"],
    )

    assert result.feature_names == ["a", "b", "__InterceptShift__"]

    np.testing.assert_allclose(
        result.values.sum(),
        result.total,
        atol=1e-10,
    )


def test_linear_multiclass_grouping_works():
    y = np.array([0, 1, 2, 1, 0, 2])
    components = np.random.default_rng(123).normal(size=(6, 3, 3)) * 0.1

    result = linear_multiclass_components(
        y,
        components,
        feature_names=["a1", "a2", "b"],
    )

    grouped = result.group(["a", "a", "b"])

    assert grouped.feature_names == ["a", "b"]

    np.testing.assert_allclose(
        grouped.values.sum(),
        result.total,
        atol=1e-10,
    )


def test_linear_multiclass_rejects_bad_component_shape():
    y = np.array([0, 1, 2])
    components = np.zeros((3, 2))

    with pytest.raises(ValueError, match="components must have shape"):
        linear_multiclass_components(y, components)


def test_linear_multiclass_rejects_bad_labels():
    y = np.array([0, 1, 3])
    components = np.zeros((3, 3, 2))

    with pytest.raises(ValueError, match="class labels"):
        linear_multiclass_components(y, components)


def test_linear_multiclass_rejects_eta_shape_mismatch():
    y = np.array([0, 1, 2])
    components = np.zeros((3, 3, 2))
    eta = np.zeros((3, 2))

    with pytest.raises(ValueError, match="eta must have shape"):
        linear_multiclass_components(y, components, eta=eta)
        
class DummyMulticlassModel:
    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = np.asarray(intercept, dtype=float)

    def decision_function(self, X):
        return X @ self.coef_.T + self.intercept_[None, :]        
        
def test_linear_explainer_multiclass_matches_component_function():
    X = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 0.5],
            [-1.0, 1.5],
            [1.5, -0.5],
            [0.0, -1.0],
        ]
    )

    y = np.array([0, 1, 2, 1, 0, 2])

    coef = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, 0.25],
        ]
    )

    intercept = np.array([0.1, -0.2, 0.3])

    model = DummyMulticlassModel(coef, intercept)

    explainer = LinearExplainer(
        model,
        loss="log_loss",
        feature_names=["a", "b"],
    )

    result_model = explainer(X, y)

    components = X[:, None, :] * coef[None, :, :]
    eta = model.decision_function(X)

    result_components = linear_multiclass_components(
        y,
        components,
        eta=eta,
        include_intercept_component=True,
        feature_names=["a", "b"],
    )

    np.testing.assert_allclose(result_model.values, result_components.values)
    np.testing.assert_allclose(
        result_model.observation_values,
        result_components.observation_values,
    )
    np.testing.assert_allclose(
        result_model.standard_errors,
        result_components.standard_errors,
    )
    np.testing.assert_allclose(result_model.total, result_components.total)

    assert result_model.feature_names == ["a", "b", "__InterceptShift__"]


def test_linear_explainer_multiclass_without_decision_function_uses_intercept():
    class ModelWithoutDecisionFunction:
        def __init__(self, coef, intercept):
            self.coef_ = np.asarray(coef, dtype=float)
            self.intercept_ = np.asarray(intercept, dtype=float)

    X = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 0.5],
            [-1.0, 1.5],
            [1.5, -0.5],
            [0.0, -1.0],
        ]
    )

    y = np.array([0, 1, 2, 1, 0, 2])

    coef = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, 0.25],
        ]
    )

    intercept = np.array([0.1, -0.2, 0.3])

    model = ModelWithoutDecisionFunction(coef, intercept)

    explainer = LinearExplainer(
        model,
        loss="log_loss",
        feature_names=["a", "b"],
    )

    result_model = explainer(X, y)

    components = X[:, None, :] * coef[None, :, :]
    eta = X @ coef.T + intercept[None, :]

    result_components = linear_multiclass_components(
        y,
        components,
        eta=eta,
        include_intercept_component=True,
        feature_names=["a", "b"],
    )

    np.testing.assert_allclose(result_model.values, result_components.values)
    np.testing.assert_allclose(result_model.total, result_components.total)


def test_linear_explainer_multiclass_rejects_bad_intercept_length():
    class ModelWithoutDecisionFunction:
        def __init__(self):
            self.coef_ = np.ones((3, 2))
            self.intercept_ = np.ones(2)

    model = ModelWithoutDecisionFunction()
    explainer = LinearExplainer(model, loss="log_loss")

    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="one intercept per class"):
        explainer(X, y)


def test_linear_explainer_multiclass_rejects_bad_decision_shape():
    class BadDecisionShapeModel:
        def __init__(self):
            self.coef_ = np.ones((3, 2))
            self.intercept_ = np.zeros(3)

        def decision_function(self, X):
            return np.zeros((X.shape[0], 2))

    model = BadDecisionShapeModel()
    explainer = LinearExplainer(model, loss="log_loss")

    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="decision_function output must have shape"):
        explainer(X, y)        
        
        
def test_to_shap_explanation_basic_shape():
    shap = pytest.importorskip("shap")

    y = np.array([1.0, 2.0, 4.0, 5.0])
    components = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
        ]
    )

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 2.0],
        ]
    )

    result = linear_regression_components(
        y,
        components,
        feature_names=["a", "b"],
    )

    shap_exp = result.to_shap_explanation(data=X)

    assert isinstance(shap_exp, shap.Explanation)
    assert shap_exp.values.shape == result.observation_values.shape
    assert shap_exp.data.shape == X.shape
    assert list(shap_exp.feature_names) == ["a", "b"]

    np.testing.assert_allclose(shap_exp.values, result.observation_values)
    np.testing.assert_allclose(
        shap_exp.base_values,
        np.full(result.n_obs, result.baseline_loss),
    )


def test_to_shap_explanation_custom_base_values():
    pytest.importorskip("shap")

    y = np.array([1.0, 2.0, 4.0, 5.0])
    components = np.zeros((4, 2))

    result = linear_regression_components(
        y,
        components,
        feature_names=["a", "b"],
    )

    base_values = np.arange(result.n_obs, dtype=float)

    shap_exp = result.to_shap_explanation(base_values=base_values)

    np.testing.assert_allclose(shap_exp.base_values, base_values)        