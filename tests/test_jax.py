import importlib.util

import numpy as np
import pytest

from edef import linear_regression_components


jax_available = importlib.util.find_spec("jax") is not None
torch_available = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not jax_available,
    reason="jax is not installed",
)


if jax_available:
    import jax.numpy as jnp

    from edef import JaxExplainer


def test_jax_linear_model_matches_closed_form_components():
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

    def predict_fn(params, X):
        return jnp.dot(X, params["coef"]) + params["intercept"]

    params = {"coef": jnp.asarray(coef), "intercept": float(intercept)}

    explainer = JaxExplainer(
        predict_fn,
        params,
        baseline=baseline,
        n_steps=8,
        feature_names=["x1", "x2"],
    )

    result_jax = explainer(X, y)

    components = X * coef[None, :]
    result_closed = linear_regression_components(
        y,
        components,
        feature_names=["x1", "x2"],
    )

    np.testing.assert_allclose(result_jax.values, result_closed.values, atol=1e-5)
    np.testing.assert_allclose(
        result_jax.observation_values,
        result_closed.observation_values,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result_jax.standard_errors,
        result_closed.standard_errors,
        atol=1e-5,
    )
    np.testing.assert_allclose(result_jax.total, result_closed.total, atol=1e-5)


def test_jax_additivity_for_nonlinear_model_without_params():
    def predict_fn(X):
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

    explainer = JaxExplainer(
        predict_fn,
        baseline=np.zeros(2, dtype=np.float32),
        n_steps=64,
        feature_names=["x1", "x2"],
    )

    result = explainer(X, y)

    np.testing.assert_allclose(result.values.sum(), result.total, atol=1e-4)
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-4)


def test_jax_logistic_model_matches_closed_form_components():
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

    def predict_fn(params, X):
        return jnp.dot(X, params["coef"]) + params["intercept"]

    params = {"coef": jnp.asarray(coef), "intercept": intercept}

    components = X * coef[None, :]
    eta = X @ coef + intercept

    result_closed = linear_logistic_components(
        y,
        components,
        eta=eta,
        include_intercept_component=False,
        feature_names=["x1", "x2"],
    )

    explainer = JaxExplainer(
        predict_fn,
        params,
        baseline=np.zeros(2, dtype=np.float32),
        loss="log_loss",
        n_steps=32,
        feature_names=["x1", "x2"],
    )

    result_jax = explainer(X, y)

    np.testing.assert_allclose(result_jax.values, result_closed.values, atol=1e-5)
    np.testing.assert_allclose(
        result_jax.observation_values,
        result_closed.observation_values,
        atol=1e-5,
    )
    np.testing.assert_allclose(result_jax.total, result_closed.total, atol=1e-5)


def test_jax_multiclass_linear_model_matches_closed_form_components():
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

    def predict_fn(params, X):
        return jnp.dot(X, params["coef"].T) + params["intercept"]

    params = {"coef": jnp.asarray(coef), "intercept": jnp.asarray(intercept)}

    explainer = JaxExplainer(
        predict_fn,
        params,
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
        n_steps=32,
        feature_names=["x1", "x2"],
    )

    result_jax = explainer(X, y)

    components = X[:, None, :] * coef[None, :, :]
    eta = X @ coef.T + intercept[None, :]

    result_closed = linear_multiclass_components(
        y.astype(int),
        components,
        eta=eta,
        include_intercept_component=False,
        feature_names=["x1", "x2"],
    )

    np.testing.assert_allclose(result_jax.values, result_closed.values, atol=1e-5)
    np.testing.assert_allclose(
        result_jax.observation_values,
        result_closed.observation_values,
        atol=1e-5,
    )
    np.testing.assert_allclose(result_jax.total, result_closed.total, atol=1e-5)
    assert result_jax.model_type == "jax_multiclass_classification"


def test_jax_rejects_bad_baseline_length():
    def predict_fn(X):
        return X[:, 0]

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)

    explainer = JaxExplainer(
        predict_fn,
        baseline=np.zeros(3, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="baseline must have length"):
        explainer(X, y)


def test_jax_log_loss_rejects_nonbinary_y():
    def predict_fn(X):
        return X[:, 0]

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float32)

    explainer = JaxExplainer(
        predict_fn,
        baseline=np.zeros(2, dtype=np.float32),
        loss="log_loss",
    )

    with pytest.raises(ValueError, match="binary labels"):
        explainer(X, y)


def test_flax_explainer_duck_typed_apply_model_matches_jax_explainer():
    from edef import FlaxExplainer, JaxExplainer

    class TinyFlaxLikeModel:
        def apply(self, variables, X, scale=1.0):
            return scale * (jnp.dot(X, variables["params"]["coef"]) + variables["params"]["intercept"])

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
    variables = {
        "params": {
            "coef": jnp.asarray([0.5, 1.5], dtype=jnp.float32),
            "intercept": jnp.asarray(0.25, dtype=jnp.float32),
        }
    }

    model = TinyFlaxLikeModel()

    def predict_fn(variables, X):
        return model.apply(variables, X, scale=2.0)

    result_flax = FlaxExplainer(
        model,
        variables,
        baseline=np.zeros(2, dtype=np.float32),
        apply_kwargs={"scale": 2.0},
        n_steps=8,
    )(X, y)

    result_jax = JaxExplainer(
        predict_fn,
        variables,
        baseline=np.zeros(2, dtype=np.float32),
        n_steps=8,
    )(X, y)

    np.testing.assert_allclose(result_flax.values, result_jax.values, atol=1e-6)
    np.testing.assert_allclose(result_flax.total, result_jax.total, atol=1e-6)


def test_flax_explainer_rejects_model_without_apply():
    from edef import FlaxExplainer

    with pytest.raises(TypeError, match="callable apply"):
        FlaxExplainer(object(), {}, baseline=np.zeros(2, dtype=np.float32))


def test_nnx_explainer_batched_callable_matches_jax_explainer():
    from edef import JaxExplainer, NNXExplainer

    class TinyNNXLikeModel:
        def __init__(self):
            self.coef = jnp.asarray([0.5, 1.5], dtype=jnp.float32)
            self.intercept = jnp.asarray(0.25, dtype=jnp.float32)

        def __call__(self, X, scale=1.0):
            return scale * (jnp.dot(X, self.coef) + self.intercept)

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
    model = TinyNNXLikeModel()

    def predict_fn(model, X):
        return model(X, scale=2.0)

    result_nnx = NNXExplainer(
        model,
        baseline=np.zeros(2, dtype=np.float32),
        call_kwargs={"scale": 2.0},
        n_steps=8,
    )(X, y)

    result_jax = JaxExplainer(
        predict_fn,
        model,
        baseline=np.zeros(2, dtype=np.float32),
        n_steps=8,
    )(X, y)

    np.testing.assert_allclose(result_nnx.values, result_jax.values, atol=1e-6)
    np.testing.assert_allclose(result_nnx.total, result_jax.total, atol=1e-6)


def test_nnx_explainer_vectorizes_single_observation_callable():
    from edef import NNXExplainer

    class SingleObservationNNXLikeModel:
        def __call__(self, x):
            return x[0] ** 2 + 0.5 * x[1]

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

    result = NNXExplainer(
        SingleObservationNNXLikeModel(),
        baseline=np.zeros(2, dtype=np.float32),
        vectorize=True,
        n_steps=64,
    )(X, y)

    np.testing.assert_allclose(result.values.sum(), result.total, atol=1e-4)
    assert result.model_type == "jax_regression"


def test_nnx_explainer_rejects_noncallable_model():
    from edef import NNXExplainer

    with pytest.raises(TypeError, match="model must be callable"):
        NNXExplainer(object(), baseline=np.zeros(2, dtype=np.float32))


def test_equinox_explainer_batched_callable_matches_jax_explainer():
    from edef import EquinoxExplainer, JaxExplainer

    class TinyEquinoxLikeModel:
        def __init__(self):
            self.coef = jnp.asarray([0.5, 1.5], dtype=jnp.float32)
            self.intercept = jnp.asarray(0.25, dtype=jnp.float32)

        def __call__(self, X, scale=1.0):
            return scale * (jnp.dot(X, self.coef) + self.intercept)

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
    model = TinyEquinoxLikeModel()

    def predict_fn(model, X):
        return model(X, scale=2.0)

    result_eqx = EquinoxExplainer(
        model,
        baseline=np.zeros(2, dtype=np.float32),
        call_kwargs={"scale": 2.0},
        n_steps=8,
    )(X, y)

    result_jax = JaxExplainer(
        predict_fn,
        model,
        baseline=np.zeros(2, dtype=np.float32),
        n_steps=8,
    )(X, y)

    np.testing.assert_allclose(result_eqx.values, result_jax.values, atol=1e-6)
    np.testing.assert_allclose(result_eqx.total, result_jax.total, atol=1e-6)


def test_equinox_explainer_vectorizes_single_observation_callable():
    from edef import EquinoxExplainer

    class SingleObservationModel:
        def __call__(self, x):
            return x[0] ** 2 + 0.5 * x[1]

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

    result = EquinoxExplainer(
        SingleObservationModel(),
        baseline=np.zeros(2, dtype=np.float32),
        vectorize=True,
        n_steps=64,
    )(X, y)

    np.testing.assert_allclose(result.values.sum(), result.total, atol=1e-4)
    assert result.model_type == "jax_regression"


def test_equinox_explainer_rejects_noncallable_model():
    from edef import EquinoxExplainer

    with pytest.raises(TypeError, match="model must be callable"):
        EquinoxExplainer(object(), baseline=np.zeros(2, dtype=np.float32))

def test_jax_multiclass_rejects_label_exceeding_output_width():
    def predict_fn(X):
        return jnp.zeros((X.shape[0], 3), dtype=jnp.float32)

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.float32)

    explainer = JaxExplainer(
        predict_fn,
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
    )

    with pytest.raises(ValueError, match="less than the number of model output classes"):
        explainer(X, y)
        

@pytest.mark.skipif(not torch_available, reason="torch is not installed")
def test_jax_and_torch_explainers_agree_on_nonlinear_regression():
    """JAX and PyTorch EDEF backends decompose the same model identically.

    One tanh-MLP is built with shared weights in both frameworks; the two
    path-integral implementations should converge to the same values,
    observation-level contributions, and total loss reduction. The tolerance
    accommodates float32 and any difference in quadrature rule between the two
    backends — at large n_steps both are effectively at the exact integral.
    """
    import torch
    import torch.nn as nn

    from edef import TorchExplainer

    rng = np.random.default_rng(0)
    W1 = (rng.normal(size=(2, 4)) * 0.7).astype(np.float32)
    b1 = (rng.normal(size=(4,)) * 0.3).astype(np.float32)
    W2 = (rng.normal(size=(4, 1)) * 0.7).astype(np.float32)
    b2 = np.float32(0.1)

    X = (rng.normal(size=(64, 2)) * 0.8).astype(np.float32)
    y = (rng.normal(size=64) * 0.5).astype(np.float32)
    baseline = X.mean(axis=0)

    # JAX model
    params = {
        "W1": jnp.asarray(W1),
        "b1": jnp.asarray(b1),
        "W2": jnp.asarray(W2),
        "b2": jnp.asarray(b2),
    }

    def predict_fn(params, X):
        h = jnp.tanh(X @ params["W1"] + params["b1"])
        return (h @ params["W2"])[:, 0] + params["b2"]

    jax_result = JaxExplainer(
        predict_fn,
        params,
        baseline=baseline,
        loss="squared_error",
        n_steps=128,
        feature_names=["x1", "x2"],
    )(X, y)

    # PyTorch model with identical weights
    class TorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 4)
            self.l2 = nn.Linear(4, 1)
            with torch.no_grad():
                self.l1.weight.copy_(torch.tensor(np.ascontiguousarray(W1.T)))
                self.l1.bias.copy_(torch.tensor(b1))
                self.l2.weight.copy_(torch.tensor(np.ascontiguousarray(W2.T)))
                self.l2.bias.copy_(torch.tensor(np.array([b2], dtype=np.float32)))

        def forward(self, X):
            return self.l2(torch.tanh(self.l1(X))).squeeze(-1)

    torch_model = TorchMLP().eval()

    torch_result = TorchExplainer(
        torch_model,
        baseline=baseline,
        loss="squared_error",
        n_steps=128,
        feature_names=["x1", "x2"],
    )(X, y)

    np.testing.assert_allclose(jax_result.total, torch_result.total, atol=1e-3)
    np.testing.assert_allclose(jax_result.values, torch_result.values, atol=1e-3)
    np.testing.assert_allclose(
        jax_result.observation_values,
        torch_result.observation_values,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        jax_result.standard_errors,
        torch_result.standard_errors,
        atol=1e-3,
    )        
    
@pytest.mark.skipif(not torch_available, reason="torch is not installed")
def test_jax_and_torch_explainers_agree_on_log_loss():
    """JAX and PyTorch backends agree on binary log-loss decomposition.

    Same shared-weight tanh-MLP as the regression case, but the output is a
    scalar logit and labels are Bernoulli-sampled from the model so the total
    loss reduction is meaningfully nonzero. Exercises the log-loss gradient
    path, which is distinct code from squared error in both backends.
    """
    import torch
    import torch.nn as nn

    from edef import TorchExplainer

    rng = np.random.default_rng(1)
    W1 = (rng.normal(size=(2, 4)) * 0.7).astype(np.float32)
    b1 = (rng.normal(size=(4,)) * 0.3).astype(np.float32)
    W2 = (rng.normal(size=(4, 1)) * 0.7).astype(np.float32)
    b2 = np.float32(0.1)

    X = (rng.normal(size=(64, 2)) * 0.8).astype(np.float32)
    baseline = X.mean(axis=0)

    # Sample labels from the model's own logits so the model is a real fit.
    h_np = np.tanh(X @ W1 + b1)
    logits_np = (h_np @ W2)[:, 0] + b2
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    y = (rng.uniform(size=X.shape[0]) < probs_np).astype(np.float32)

    # JAX model
    params = {
        "W1": jnp.asarray(W1),
        "b1": jnp.asarray(b1),
        "W2": jnp.asarray(W2),
        "b2": jnp.asarray(b2),
    }

    def predict_fn(params, X):
        h = jnp.tanh(X @ params["W1"] + params["b1"])
        return (h @ params["W2"])[:, 0] + params["b2"]

    jax_result = JaxExplainer(
        predict_fn,
        params,
        baseline=baseline,
        loss="log_loss",
        n_steps=128,
        feature_names=["x1", "x2"],
    )(X, y)

    # PyTorch model with identical weights
    class TorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 4)
            self.l2 = nn.Linear(4, 1)
            with torch.no_grad():
                self.l1.weight.copy_(torch.tensor(np.ascontiguousarray(W1.T)))
                self.l1.bias.copy_(torch.tensor(b1))
                self.l2.weight.copy_(torch.tensor(np.ascontiguousarray(W2.T)))
                self.l2.bias.copy_(torch.tensor(np.array([b2], dtype=np.float32)))

        def forward(self, X):
            return self.l2(torch.tanh(self.l1(X))).squeeze(-1)

    torch_model = TorchMLP().eval()

    torch_result = TorchExplainer(
        torch_model,
        baseline=baseline,
        loss="log_loss",
        n_steps=128,
        feature_names=["x1", "x2"],
    )(X, y)

    np.testing.assert_allclose(jax_result.total, torch_result.total, atol=1e-3)
    np.testing.assert_allclose(jax_result.values, torch_result.values, atol=1e-3)
    np.testing.assert_allclose(
        jax_result.observation_values,
        torch_result.observation_values,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        jax_result.standard_errors,
        torch_result.standard_errors,
        atol=1e-3,
    )    