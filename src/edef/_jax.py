from __future__ import annotations

import numpy as np

from ._results import EDEFExplanation


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX support requires jax. Install with: pip install jax"
        ) from exc
    return jax, jnp


def _gauss_legendre_nodes_weights(n_steps: int):
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")

    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return nodes, weights


class JaxExplainer:
    """
    Path-integral EDEF explainer for differentiable JAX prediction functions.

    ``JaxExplainer`` computes an Euler Decomposition of Explained Fit (EDEF)
    by integrating loss gradients along the straight-line path from a fixed
    baseline input to each evaluation input. The resulting feature
    contributions decompose average loss reduction from the baseline
    prediction to the model prediction.

    Version 1 supports squared-error regression, binary log-loss
    classification, and multiclass log-loss classification. Binary and
    multiclass classification use logits or raw class scores, not probability
    outputs.

    Parameters
    ----------
    predict_fn : callable
        Differentiable JAX-compatible prediction function. If ``params`` is
        supplied, the function is called as ``predict_fn(params, X)``. If
        ``params`` is ``None``, the function is called as ``predict_fn(X)``.
        For ``squared_error`` and ``log_loss``, the output must have shape
        ``(n_obs,)`` or ``(n_obs, 1)``. For ``multiclass_log_loss``, the output
        must have shape ``(n_obs, n_classes)``.

    params : pytree, optional
        Model parameters passed to ``predict_fn``. Omit for closed-over or
        parameter-free prediction functions.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for each path.

    loss : {"squared_error", "log_loss", "multiclass_log_loss"}, default="squared_error"
        Loss function whose reduction is decomposed. ``"log_loss"`` expects
        binary labels and scalar logits. ``"multiclass_log_loss"`` expects
        integer class labels and class-score/logit outputs.

    n_steps : int, default=50
        Number of Gauss-Legendre quadrature nodes used to approximate the path
        integral.

    feature_names : sequence of str, optional
        Default feature names used in returned EDEF results.

    dtype : JAX dtype, optional
        Floating-point dtype used for inputs and the baseline. If omitted,
        non-floating inputs are converted to ``jnp.float32``.

    When to use
    -----------
    Use ``JaxExplainer`` for differentiable JAX models when automatic
    differentiation is available and path-integral attribution is desired.
    The interface is intentionally functional so that NNX, Flax, Haiku, Equinox,
    and hand-written JAX models can be supported through thin adapters.
    """

    def __init__(
        self,
        predict_fn,
        params=None,
        *,
        baseline,
        loss: str = "squared_error",
        n_steps: int = 50,
        feature_names=None,
        dtype=None,
    ):
        if loss not in {"squared_error", "log_loss", "multiclass_log_loss"}:
            raise ValueError(
                "JaxExplainer supports squared_error, log_loss, and "
                "multiclass_log_loss."
            )

        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable.")

        self.jax, self.jnp = _require_jax()
        self.predict_fn = predict_fn
        self.params = params
        self.baseline = baseline
        self.loss = loss
        self.n_steps = int(n_steps)
        self.feature_names = feature_names
        self.dtype = dtype
        self._nodes, self._weights = _gauss_legendre_nodes_weights(self.n_steps)
        
        if dtype is not None:
            requested = self.jnp.dtype(dtype)
            actual = self.jax.dtypes.canonicalize_dtype(requested)
            if self.jnp.issubdtype(requested, 
                                   self.jnp.floating) and actual != requested:
                import warnings
                warnings.warn(
                    f"Requested dtype {requested} was downcast to {actual} "
                    "because JAX 64-bit mode is disabled; enable it with "
                    'jax.config.update("jax_enable_x64", True) before import '
                    "to use float64.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def __call__(
            self,
            X,
            y,
            *,
            feature_names=None,
            check_additivity: bool = True,
            atol: float = 1e-5,
            rtol: float = 1e-4,
        ) -> EDEFExplanation:
        """Compute the path-integral EDEF decomposition for evaluation data."""
        jnp = self.jnp
        jax = self.jax

        X_j = self._as_array(X)
        y_j = self._as_array(y).reshape(-1)

        if X_j.ndim != 2:
            raise ValueError("X must have shape (n_obs, n_features).")

        n_obs, n_features = X_j.shape

        if y_j.shape[0] != n_obs:
            raise ValueError("y and X must have the same number of observations.")

        if n_obs < 2:
            raise ValueError("At least two observations are required.")

        if self.loss == "log_loss":
            y_float = y_j.astype(X_j.dtype)
            if not bool(jnp.all((y_float == 0.0) | (y_float == 1.0))):
                raise ValueError("y must contain only binary labels in {0, 1}.")
            y_work = y_float

        elif self.loss == "multiclass_log_loss":
            if not bool(jnp.all(y_j == jnp.round(y_j))):
                raise ValueError("y must contain integer class labels.")
            if bool(jnp.any(y_j < 0)):
                raise ValueError("y must contain nonnegative class labels.")
            y_work = y_j.astype(jnp.int32)

        else:
            y_work = y_j.astype(X_j.dtype)

        baseline_j = self._baseline_array(n_features, X_j)

        names = feature_names
        if names is None:
            names = self.feature_names
        if names is None:
            names = [f"x{i}" for i in range(n_features)]
        else:
            names = list(names)
            if len(names) != n_features:
                raise ValueError("feature_names must have length n_features.")

        X0_j = jnp.broadcast_to(baseline_j.reshape(1, -1), X_j.shape)
        delta_X = X_j - X0_j

        pred_probe = self._predict_output(X_j)

        if pred_probe.shape[0] != n_obs:
            raise ValueError(
                "Model output must have the same number of observations as X."
            )

        if self.loss == "multiclass_log_loss":
            n_classes = pred_probe.shape[1]
            y_max = int(np.asarray(jnp.max(y_work)))

            if y_max >= n_classes:
                raise ValueError(
                    "multiclass labels must be less than the number of "
                    "model output classes."
                )

        def loss_sum(X_path):
            pred = self._predict_output(X_path)
            return jnp.sum(self._loss_per_observation(y_work, pred))

        grad_loss = jax.grad(loss_sum)

        nodes_j = jnp.asarray(self._nodes, dtype=X_j.dtype)
        weights_j = jnp.asarray(self._weights, dtype=X_j.dtype)

        def _accumulate(acc, node_weight):
            node, weight = node_weight
            Xt = X0_j + node * delta_X
            grad = grad_loss(Xt)
            return acc + weight * grad, None

        weighted_grad, _ = jax.lax.scan(
            _accumulate,
            jnp.zeros_like(X_j),
            (nodes_j, weights_j),
        )

        c_j = -delta_X * weighted_grad

        pred0_one = self._predict_output(baseline_j.reshape(1, -1))
        if pred0_one.ndim == 1:
            pred0 = jnp.broadcast_to(pred0_one, (n_obs,))
        else:
            pred0 = jnp.broadcast_to(pred0_one, (n_obs, pred0_one.shape[1]))

        pred = pred_probe
        baseline_loss = jnp.mean(self._loss_per_observation(y_work, pred0))
        model_loss = jnp.mean(self._loss_per_observation(y_work, pred))
        total = baseline_loss - model_loss

        values_j = jnp.mean(c_j, axis=0)
        standard_errors_j = jnp.std(c_j, axis=0, ddof=1) / np.sqrt(n_obs)
        additivity_error = jnp.sum(values_j) - total

        values = np.asarray(values_j)
        observation_values = np.asarray(c_j)
        standard_errors = np.asarray(standard_errors_j)

        total_f = float(np.asarray(total))
        baseline_loss_f = float(np.asarray(baseline_loss))
        model_loss_f = float(np.asarray(model_loss))
        additivity_error_f = float(np.asarray(additivity_error))

        tol = atol + rtol * abs(total_f)
        if check_additivity and abs(additivity_error_f) > tol:
            raise RuntimeError(
                "EDEF contributions do not add to total loss reduction. "
                f"Additivity error: {additivity_error_f} "
                f"(tolerance: {tol})"
            )

        return EDEFExplanation(
            values=values,
            observation_values=observation_values,
            standard_errors=standard_errors,
            total=total_f,
            baseline_loss=baseline_loss_f,
            model_loss=model_loss_f,
            loss=self.loss,
            model_type=(
                "jax_regression"
                if self.loss == "squared_error"
                else "jax_classification"
                if self.loss == "log_loss"
                else "jax_multiclass_classification"
            ),
            feature_names=names,
            n_obs=n_obs,
            additivity_error=additivity_error_f,
        )

    def _as_array(self, x):
        """Convert input to a JAX array using the explainer dtype."""
        jnp = self.jnp

        out = jnp.asarray(x)

        if self.dtype is not None:
            out = out.astype(self.dtype)
        elif not jnp.issubdtype(out.dtype, jnp.floating):
            out = out.astype(jnp.float32)

        return out

    def _baseline_array(self, n_features: int, X_j):
        """Return the prepared baseline array aligned with ``X_j``."""
        jnp = self.jnp

        if isinstance(self.baseline, str):
            raise ValueError(
                "JaxExplainer currently requires a numeric baseline vector, "
                "not a string baseline rule."
            )

        b = self._as_array(self.baseline).reshape(-1)

        if b.shape[0] != n_features:
            raise ValueError("baseline must have length n_features.")

        if not bool(jnp.all(jnp.isfinite(b))):
            raise ValueError("baseline must contain only finite values.")

        return b.astype(X_j.dtype)

    def _call_predict_fn(self, X_j):
        if self.params is None:
            return self.predict_fn(X_j)
        return self.predict_fn(self.params, X_j)

    def _predict_output(self, X_j):
        """Evaluate the prediction function and validate output shape."""
        jnp = self.jnp

        pred = jnp.asarray(self._call_predict_fn(X_j))

        if self.loss in {"squared_error", "log_loss"}:
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.reshape(-1)

            if pred.ndim != 1:
                raise ValueError(
                    "JaxExplainer requires scalar output with shape "
                    "(n_obs,) or (n_obs, 1) for squared_error and log_loss."
                )

            return pred

        if self.loss == "multiclass_log_loss":
            if pred.ndim != 2:
                raise ValueError(
                    "multiclass_log_loss requires model output with shape "
                    "(n_obs, n_classes)."
                )
            if pred.shape[1] < 2:
                raise ValueError(
                    "multiclass_log_loss requires at least two classes."
                )
            return pred

        raise RuntimeError(f"Unexpected loss: {self.loss}")

    def _loss_per_observation(self, y_j, pred_j):
        """Return per-observation loss values for the selected loss."""
        jax = self.jax
        jnp = self.jnp

        if self.loss == "squared_error":
            return (y_j - pred_j) ** 2

        if self.loss == "log_loss":
            # Stable, smooth binary cross-entropy with logits.
            return jax.nn.softplus(pred_j) - y_j * pred_j

        if self.loss == "multiclass_log_loss":
            log_denom = jax.nn.logsumexp(pred_j, axis=1)
            return log_denom - pred_j[jnp.arange(pred_j.shape[0]), y_j]

        raise RuntimeError(f"Unexpected loss: {self.loss}")


class FlaxExplainer(JaxExplainer):
    """
    Thin EDEF adapter for Flax Linen-style models.

    ``FlaxExplainer`` wraps a Flax model with an ``apply`` method and delegates
    all EDEF computations to ``JaxExplainer``. The wrapper does not import Flax
    directly; it only requires that ``model.apply(variables, X, **apply_kwargs)``
    return model outputs for a batch of inputs.

    Parameters
    ----------
    model : object
        Flax Linen-style model with an ``apply`` method.

    variables : pytree
        Flax variables passed as the first argument to ``model.apply``. This is
        typically ``{"params": params}``, possibly with additional state such
        as ``batch_stats``.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for each path.

    apply_kwargs : dict, optional
        Keyword arguments passed to ``model.apply``. Common examples include
        ``{"train": False}`` or ``{"mutable": False}`` for inference.

    loss, n_steps, feature_names, dtype
        Passed through to ``JaxExplainer``.
    """

    def __init__(
        self,
        model,
        variables,
        *,
        baseline,
        apply_kwargs=None,
        loss: str = "squared_error",
        n_steps: int = 50,
        feature_names=None,
        dtype=None,
    ):
        if not hasattr(model, "apply") or not callable(model.apply):
            raise TypeError("model must have a callable apply method.")

        kwargs = {} if apply_kwargs is None else dict(apply_kwargs)

        def predict_fn(variables, X):
            return model.apply(variables, X, **kwargs)

        self.model = model
        self.variables = variables
        self.apply_kwargs = kwargs

        super().__init__(
            predict_fn,
            variables,
            baseline=baseline,
            loss=loss,
            n_steps=n_steps,
            feature_names=feature_names,
            dtype=dtype,
        )


class NNXExplainer(JaxExplainer):
    """
    Thin EDEF adapter for Flax NNX-style callable models.

    ``NNXExplainer`` wraps a Flax NNX model and delegates all EDEF
    computations to ``JaxExplainer``. The wrapper does not import Flax
    directly; it only requires that the supplied model be callable and
    differentiable with respect to its input. This matches the common NNX
    inference pattern ``model(X, **call_kwargs)``.

    Parameters
    ----------
    model : callable
        Flax NNX-style model. By default the model is called as
        ``model(X, **call_kwargs)`` and is expected to accept a batch of inputs.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for each path.

    call_kwargs : dict, optional
        Keyword arguments passed to ``model``.

    vectorize : bool, default=False
        If True, call ``jax.vmap`` around a single-observation model, so that
        the model is evaluated as ``vmap(lambda x: model(x, **call_kwargs))(X)``.
        Leave False for models that already accept batched inputs.

    loss, n_steps, feature_names, dtype
        Passed through to ``JaxExplainer``.
    """

    def __init__(
        self,
        model,
        *,
        baseline,
        call_kwargs=None,
        vectorize: bool = False,
        loss: str = "squared_error",
        n_steps: int = 50,
        feature_names=None,
        dtype=None,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")

        kwargs = {} if call_kwargs is None else dict(call_kwargs)
        use_vmap = bool(vectorize)

        if use_vmap:
            jax_module = _require_jax()[0]

            def predict_fn(model, X):
                return jax_module.vmap(lambda x: model(x, **kwargs))(X)
        else:
            def predict_fn(model, X):
                return model(X, **kwargs)

        self.model = model
        self.call_kwargs = kwargs
        self.vectorize = use_vmap

        super().__init__(
            predict_fn,
            model,
            baseline=baseline,
            loss=loss,
            n_steps=n_steps,
            feature_names=feature_names,
            dtype=dtype,
        )


class EquinoxExplainer(JaxExplainer):
    """
    Thin EDEF adapter for Equinox-style callable PyTree models.

    ``EquinoxExplainer`` wraps a callable JAX model and delegates all EDEF
    computations to ``JaxExplainer``. The wrapper does not import Equinox
    directly; it only requires that the supplied model be callable and
    differentiable with respect to its input.

    Parameters
    ----------
    model : callable pytree
        Equinox-style model. By default the model is called as
        ``model(X, **call_kwargs)`` and is expected to accept a batch of inputs.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for each path.

    call_kwargs : dict, optional
        Keyword arguments passed to ``model``.

    vectorize : bool, default=False
        If True, call ``jax.vmap`` around a single-observation model, so that
        the model is evaluated as ``vmap(lambda x: model(x, **call_kwargs))(X)``.
        Leave False for models that already accept batched inputs.

    loss, n_steps, feature_names, dtype
        Passed through to ``JaxExplainer``.
    """

    def __init__(
        self,
        model,
        *,
        baseline,
        call_kwargs=None,
        vectorize: bool = False,
        loss: str = "squared_error",
        n_steps: int = 50,
        feature_names=None,
        dtype=None,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")

        kwargs = {} if call_kwargs is None else dict(call_kwargs)
        use_vmap = bool(vectorize)

        if use_vmap:
            jax_module = _require_jax()[0]

            def predict_fn(model, X):
                return jax_module.vmap(lambda x: model(x, **kwargs))(X)
        else:
            def predict_fn(model, X):
                return model(X, **kwargs)

        self.model = model
        self.call_kwargs = kwargs
        self.vectorize = use_vmap

        super().__init__(
            predict_fn,
            model,
            baseline=baseline,
            loss=loss,
            n_steps=n_steps,
            feature_names=feature_names,
            dtype=dtype,
        )
        
class HaikuExplainer(JaxExplainer):
    """
    Thin EDEF adapter for Haiku-transformed models.

    ``HaikuExplainer`` wraps a function transformed by ``hk.transform`` and
    delegates all EDEF computations to ``JaxExplainer``. The wrapper does not
    import Haiku directly; it only requires that the supplied object expose a
    callable ``apply`` method with the stateless Haiku signature
    ``apply(params, rng, X)`` returning model outputs for a batch of inputs.

    Haiku is in maintenance mode: as of July 2023 Google DeepMind recommends
    Flax for new projects, and Haiku receives bug fixes and JAX-compatibility
    updates but no new features. This adapter exists so that EDEF can be
    applied to the substantial body of existing Haiku models without porting
    them; for new work, prefer ``FlaxExplainer`` or ``NNXExplainer``.

    Only stateless ``hk.transform`` functions are supported. Models with
    mutable state (``hk.transform_with_state``, e.g. batch normalization) have
    an ``apply`` that takes and returns state and do not fit this signature;
    wrap them in a closure that threads inference-time state and discards the
    returned state before passing the result to ``JaxExplainer`` directly.

    Parameters
    ----------
    transformed : object
        Result of ``hk.transform``. Must expose a callable ``apply`` method
        with signature ``apply(params, rng, X)``.

    params : pytree
        Haiku parameters produced by ``transformed.init`` and passed as the
        first argument to ``transformed.apply``.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for each path.

    rng : jax.Array or None, default=None
        PRNG key forwarded to ``apply`` as its second argument. ``None`` is
        appropriate for deterministic models with no stochastic operations.
        For models with stochastic ops evaluated at inference time, pass a
        fixed key so that attributions are reproducible.

    loss, n_steps, feature_names, dtype
        Passed through to ``JaxExplainer``.
    """

    def __init__(
        self,
        transformed,
        params,
        *,
        baseline,
        rng=None,
        loss: str = "squared_error",
        n_steps: int = 50,
        feature_names=None,
        dtype=None,
    ):
        if not hasattr(transformed, "apply") or not callable(transformed.apply):
            raise TypeError("transformed must have a callable apply method.")

        def predict_fn(params, X):
            return transformed.apply(params, rng, X)

        self.transformed = transformed
        self.rng = rng

        super().__init__(
            predict_fn,
            params,
            baseline=baseline,
            loss=loss,
            n_steps=n_steps,
            feature_names=feature_names,
            dtype=dtype,
        )        
