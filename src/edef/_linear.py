from __future__ import annotations

import numpy as np

from ._results import EDEFExplanation


def linear_regression_components(
    y,
    components,
    *,
    feature_names=None,
    check_additivity: bool = True,
    atol: float = 1e-10,
) -> EDEFExplanation:
    """
    Closed-form EDEF for linear regression with squared-error loss.

    Parameters
    ----------
    y : array-like, shape (n_obs,)
        Realized outcomes.

    components : array-like, shape (n_obs, n_features)
        Fitted linear signal components. For a linear model, this is typically

            components[:, j] = X[:, j] * beta[j]

        The fitted prediction, excluding intercept effects, is

            y_hat = components.sum(axis=1)

    feature_names : sequence of str, optional
        Feature names.

    check_additivity : bool, default=True
        Whether to check that feature contributions add to total fit improvement.

    atol : float, default=1e-10
        Absolute tolerance for the additivity check.

    Returns
    -------
    EDEFExplanation
        EDEF result object.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    components = np.asarray(components, dtype=float)

    if components.ndim != 2:
        raise ValueError("components must have shape (n_obs, n_features).")

    n_obs, n_features = components.shape

    if y.shape[0] != n_obs:
        raise ValueError("y and components must have the same number of observations.")

    if n_obs < 2:
        raise ValueError("At least two observations are required.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values.")

    if not np.all(np.isfinite(components)):
        raise ValueError("components must contain only finite values.")

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError("feature_names must have length n_features.")

    y_centered = y - y.mean()

    components_centered = components - components.mean(axis=0)
    prediction_centered = components_centered.sum(axis=1)

    baseline_loss = np.mean(y_centered**2)
    model_loss = np.mean((y_centered - prediction_centered) ** 2)
    total = baseline_loss - model_loss

    shared_term = 2.0 * y_centered - prediction_centered

    observation_values = components_centered * shared_term[:, None]
    values = observation_values.mean(axis=0)

    standard_errors = observation_values.std(axis=0, ddof=1) / np.sqrt(n_obs)

    additivity_error = values.sum() - total

    if check_additivity and abs(additivity_error) > atol:
        raise RuntimeError(
            "EDEF contributions do not add to total fit improvement. "
            f"Additivity error: {additivity_error}"
        )

    return EDEFExplanation(
        values=values,
        observation_values=observation_values,
        standard_errors=standard_errors,
        total=total,
        baseline_loss=baseline_loss,
        model_loss=model_loss,
        loss="squared_error",
        model_type="linear_regression_components",
        feature_names=feature_names,
        n_obs=n_obs,
        additivity_error=additivity_error,
    )
    
class LinearExplainer:
    """
    SHAP-style EDEF explainer for linear models.

    Version 1 supports fitted linear regression models with a 1D ``coef_``
    attribute. The model intercept is absorbed into the intercept-only
    baseline used by EDEF.
    """

    def __init__(
        self,
        model,
        baseline=None,
        *,
        loss: str = "squared_error",
        feature_names=None,
    ):
        if loss not in {"squared_error", "log_loss"}:
            raise ValueError("LinearExplainer supports squared_error and log_loss.")
    
        self.model = model
        self.baseline = baseline
        self.loss = loss
        self.feature_names = feature_names
    
        self.coef_ = self._get_coef(model)

    def __call__(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-10,
    ):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_obs, n_features).")

        n_features = X.shape[1]

        if self.coef_.ndim == 1:
            coef_n_features = self.coef_.shape[0]
        else:
            coef_n_features = self.coef_.shape[1]
        
        if coef_n_features != n_features:
            raise ValueError(
                "Model coefficient dimension does not match X. "
                f"coef has {coef_n_features} features, "
                f"but X has {n_features} columns."
            )

        names = feature_names
        if names is None:
            names = self.feature_names
        if names is None:
            names = self._get_feature_names(X, n_features)

        if self.loss == "squared_error":
            if self.coef_.ndim != 1:
                raise ValueError(
                    "squared_error requires a single-output linear model "
                    "with a 1D coef_ vector."
                )

            components = X * self.coef_[None, :]

            return linear_regression_components(
                y,
                components,
                feature_names=names,
                check_additivity=check_additivity,
                atol=atol,
            )

        if self.loss == "log_loss":
            eta = self._decision_function(X)

            if self.coef_.ndim == 1:
                components = X * self.coef_[None, :]

                return linear_logistic_components(
                    y,
                    components,
                    eta=eta,
                    include_intercept_component=True,
                    feature_names=names,
                    check_additivity=check_additivity,
                    atol=atol,
                )

            if self.coef_.ndim == 2:
                components = X[:, None, :] * self.coef_[None, :, :]

                return linear_multiclass_components(
                    y,
                    components,
                    eta=eta,
                    include_intercept_component=True,
                    feature_names=names,
                    check_additivity=check_additivity,
                    atol=atol,
                )

        raise RuntimeError(f"Unexpected loss: {self.loss}")


    def _decision_function(self, X) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            eta = self.model.decision_function(X)
        elif hasattr(self.model, "intercept_"):
            intercept = np.asarray(self.model.intercept_, dtype=float).reshape(-1)

            if self.coef_.ndim == 1:
                if intercept.size != 1:
                    raise ValueError(
                        "Binary classification requires a scalar intercept."
                    )
                eta = X @ self.coef_ + intercept[0]
            else:
                if intercept.size != self.coef_.shape[0]:
                    raise ValueError(
                        "Multiclass classification requires one intercept per class."
                    )
                eta = X @ self.coef_.T + intercept.reshape(1, -1)
        else:
            raise TypeError(
                "log_loss requires a model with decision_function or intercept_."
            )

        eta = np.asarray(eta, dtype=float)

        if self.coef_.ndim == 1:
            eta = eta.reshape(-1)
            if eta.shape[0] != X.shape[0]:
                raise ValueError("decision_function output must have length n_obs.")
        else:
            expected_shape = (X.shape[0], self.coef_.shape[0])
            if eta.shape != expected_shape:
                raise ValueError(
                    "decision_function output must have shape "
                    "(n_obs, n_classes)."
                )

        if not np.all(np.isfinite(eta)):
            raise ValueError("decision_function output must contain only finite values.")

        return eta

    @staticmethod
    def _get_coef(model) -> np.ndarray:
        if not hasattr(model, "coef_"):
            raise TypeError(
                "LinearExplainer requires a fitted linear model with a coef_ attribute."
            )

        coef = np.asarray(model.coef_, dtype=float)

        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef.reshape(-1)

        if coef.ndim not in {1, 2}:
            raise ValueError(
                "LinearExplainer requires coef_ to be 1D for regression/binary "
                "classification or 2D for multiclass classification."
            )

        if not np.all(np.isfinite(coef)):
            raise ValueError("model.coef_ must contain only finite values.")

        return coef
        
    @staticmethod
    def _get_feature_names(X, n_features: int) -> list[str]:
        columns = getattr(X, "columns", None)
        if columns is not None:
            return list(columns)
        return [f"x{i}" for i in range(n_features)]
        
def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _softplus(z):
    return np.logaddexp(0.0, z)


def _binary_log_loss(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def linear_logistic_components(
    y,
    components,
    *,
    eta=None,
    intercept_component=None,
    include_intercept_component: bool = False,
    feature_names=None,
    check_additivity: bool = True,
    atol: float = 1e-10,
) -> EDEFExplanation:
    """
    Closed-form EDEF for binary linear classification with log loss.

    Parameters
    ----------
    y : array-like, shape (n_obs,)
        Binary labels in {0, 1}.

    components : array-like, shape (n_obs, n_features)
        Fitted score/logit components. For logistic regression, this is typically

            components[:, j] = X[:, j] * beta[j]

    eta : array-like, shape (n_obs,), optional
        Full fitted score/logit. If omitted, the score is constructed as

            eta = eta_bar + components.sum(axis=1)

        where eta_bar is the baseline logit.

    intercept_component : array-like, shape (n_obs,), optional
        Additional score component to include, typically the difference between
        the fitted intercept and the baseline logit.

    include_intercept_component : bool, default=False
        If True, append intercept_component as an additional attribution column.

    feature_names : sequence of str, optional
        Feature names.

    check_additivity : bool, default=True
        Whether to check that feature contributions add to total fit improvement.

    atol : float, default=1e-10
        Absolute tolerance for the additivity check.

    Returns
    -------
    EDEFExplanation
        EDEF result object.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    components = np.asarray(components, dtype=float)

    if components.ndim != 2:
        raise ValueError("components must have shape (n_obs, n_features).")

    n_obs, n_features = components.shape

    if y.shape[0] != n_obs:
        raise ValueError("y and components must have the same number of observations.")

    if n_obs < 2:
        raise ValueError("At least two observations are required.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values.")

    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must contain only binary labels in {0, 1}.")

    if not np.all(np.isfinite(components)):
        raise ValueError("components must contain only finite values.")

    p_bar = float(np.clip(y.mean(), 1e-12, 1.0 - 1e-12))
    eta_bar = float(_logit(p_bar))

    if eta is None:
        eta = eta_bar + components.sum(axis=1)
    else:
        eta = np.asarray(eta, dtype=float).reshape(-1)
        if eta.shape[0] != n_obs:
            raise ValueError("eta must have length n_obs.")
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta must contain only finite values.")

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError("feature_names must have length n_features.")

    if include_intercept_component:
        if intercept_component is None:
            intercept_component = eta - eta_bar - components.sum(axis=1)
        else:
            intercept_component = np.asarray(intercept_component, dtype=float).reshape(-1)
            if intercept_component.shape[0] != n_obs:
                raise ValueError("intercept_component must have length n_obs.")
            if not np.all(np.isfinite(intercept_component)):
                raise ValueError(
                    "intercept_component must contain only finite values."
                )

        components = np.column_stack([components, intercept_component])
        feature_names = feature_names + ["__InterceptShift__"]

    p_hat = _sigmoid(eta)

    baseline_loss = float(np.mean(_binary_log_loss(y, p_bar)))
    model_loss = float(np.mean(_binary_log_loss(y, p_hat)))
    total = baseline_loss - model_loss

    delta = eta - eta_bar
    sp_eta = _softplus(eta)
    sp_eta_bar = _softplus(eta_bar)

    eps = 1e-12
    path_weight = np.empty(n_obs, dtype=float)

    mask = np.abs(delta) > eps
    path_weight[mask] = y[mask] - (sp_eta[mask] - sp_eta_bar) / delta[mask]
    path_weight[~mask] = y[~mask] - _sigmoid(eta_bar)

    observation_values = components * path_weight[:, None]
    values = observation_values.mean(axis=0)

    standard_errors = observation_values.std(axis=0, ddof=1) / np.sqrt(n_obs)

    additivity_error = values.sum() - total

    if check_additivity and abs(additivity_error) > atol:
        raise RuntimeError(
            "EDEF contributions do not add to total fit improvement. "
            f"Additivity error: {additivity_error}"
        )

    return EDEFExplanation(
        values=values,
        observation_values=observation_values,
        standard_errors=standard_errors,
        total=total,
        baseline_loss=baseline_loss,
        model_loss=model_loss,
        loss="log_loss",
        model_type="linear_logistic_components",
        feature_names=feature_names,
        n_obs=n_obs,
        additivity_error=additivity_error,
    )   
    
def _logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))

    if not keepdims:
        out = np.squeeze(out, axis=axis)

    return out


def _softmax(eta):
    log_denom = _logsumexp(eta, axis=1, keepdims=True)
    return np.exp(eta - log_denom)


def linear_multiclass_components(
    y,
    components,
    *,
    eta=None,
    intercept_component=None,
    include_intercept_component: bool = False,
    feature_names=None,
    check_additivity: bool = True,
    atol: float = 1e-10,
) -> EDEFExplanation:
    """
    Closed-form EDEF for multiclass linear classification with log loss.

    Parameters
    ----------
    y : array-like, shape (n_obs,)
        Integer class labels in {0, ..., n_classes - 1}.

    components : array-like, shape (n_obs, n_classes, n_features)
        Fitted class-score components. For multinomial logistic regression,

            components[i, k, j] = X[i, j] * beta[k, j]

    eta : array-like, shape (n_obs, n_classes), optional
        Full fitted class scores. If omitted, the score is constructed as

            eta = eta_bar + components.sum(axis=2)

        where eta_bar is the baseline class-score vector.

    intercept_component : array-like, shape (n_obs, n_classes), optional
        Additional class-score component, typically the difference between the
        fitted intercept vector and the baseline class-score vector.

    include_intercept_component : bool, default=False
        If True, append intercept_component as an additional attribution column.

    feature_names : sequence of str, optional
        Feature names.

    Returns
    -------
    EDEFExplanation
        Scalar log-loss EDEF result. Class dimensions are summed internally, so
        observation_values has shape (n_obs, n_features).
    """

    y = np.asarray(y).reshape(-1)
    components = np.asarray(components, dtype=float)

    if components.ndim != 3:
        raise ValueError(
            "components must have shape (n_obs, n_classes, n_features)."
        )

    n_obs, n_classes, n_features = components.shape

    if y.shape[0] != n_obs:
        raise ValueError("y and components must have the same number of observations.")

    if n_obs < 2:
        raise ValueError("At least two observations are required.")

    if not np.all(np.isfinite(components)):
        raise ValueError("components must contain only finite values.")

    if not np.issubdtype(y.dtype, np.integer):
        if np.all(np.equal(y, np.round(y))):
            y = y.astype(int)
        else:
            raise ValueError("y must contain integer class labels.")

    y = y.astype(int)

    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("y must contain class labels in {0, ..., n_classes - 1}.")

    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    class_probs = np.clip(class_counts / n_obs, 1e-12, 1.0)
    class_probs = class_probs / class_probs.sum()

    eta_bar = np.log(class_probs)

    if eta is None:
        eta = eta_bar.reshape(1, -1) + components.sum(axis=2)
    else:
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (n_obs, n_classes):
            raise ValueError("eta must have shape (n_obs, n_classes).")
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta must contain only finite values.")

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError("feature_names must have length n_features.")

    if include_intercept_component:
        if intercept_component is None:
            intercept_component = (
                eta
                - eta_bar.reshape(1, -1)
                - components.sum(axis=2)
            )
        else:
            intercept_component = np.asarray(intercept_component, dtype=float)
            if intercept_component.shape != (n_obs, n_classes):
                raise ValueError(
                    "intercept_component must have shape (n_obs, n_classes)."
                )
            if not np.all(np.isfinite(intercept_component)):
                raise ValueError(
                    "intercept_component must contain only finite values."
                )

        components = np.concatenate(
            [components, intercept_component[:, :, None]],
            axis=2,
        )
        feature_names = feature_names + ["__InterceptShift__"]
        n_features = n_features + 1

    baseline_loss = float(-np.mean(np.log(class_probs[y])))

    log_probs = eta - _logsumexp(eta, axis=1, keepdims=True)
    model_loss = float(-np.mean(log_probs[np.arange(n_obs), y]))

    total = baseline_loss - model_loss

    delta = eta - eta_bar.reshape(1, -1)

    # For each observation and class, compute
    #
    # integral_0^1 softmax_k(eta_bar + t * delta_i) dt
    #
    # using Gauss-Legendre quadrature. There is no simple binary-style
    # scalar softplus closed form for the multiclass softmax path.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    avg_prob = np.zeros((n_obs, n_classes), dtype=float)

    for node, weight in zip(nodes, weights):
        eta_t = eta_bar.reshape(1, -1) + float(node) * delta
        avg_prob += float(weight) * _softmax(eta_t)

    one_hot = np.zeros((n_obs, n_classes), dtype=float)
    one_hot[np.arange(n_obs), y] = 1.0

    path_weight = one_hot - avg_prob

    # components: (n_obs, n_classes, n_features)
    # path_weight: (n_obs, n_classes)
    # observation_values: sum over classes -> (n_obs, n_features)
    observation_values = np.sum(
        components * path_weight[:, :, None],
        axis=1,
    )

    values = observation_values.mean(axis=0)
    standard_errors = observation_values.std(axis=0, ddof=1) / np.sqrt(n_obs)

    additivity_error = values.sum() - total

    if check_additivity and abs(additivity_error) > atol:
        raise RuntimeError(
            "EDEF contributions do not add to total fit improvement. "
            f"Additivity error: {additivity_error}"
        )

    return EDEFExplanation(
        values=values,
        observation_values=observation_values,
        standard_errors=standard_errors,
        total=total,
        baseline_loss=baseline_loss,
        model_loss=model_loss,
        loss="log_loss",
        model_type="linear_multiclass_components",
        feature_names=feature_names,
        n_obs=n_obs,
        additivity_error=additivity_error,
    )         