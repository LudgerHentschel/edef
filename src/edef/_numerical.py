from __future__ import annotations

import numpy as np

from ._results import EDEFExplanation


def _gauss_legendre_nodes_weights(n_steps: int):
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")

    nodes, weights = np.polynomial.legendre.leggauss(n_steps)

    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    return nodes, weights


def _squared_error_loss(y, pred):
    return (y - pred) ** 2


def _binary_log_loss(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _multiclass_log_loss(y, p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)

    row_sums = p.sum(axis=1, keepdims=True)
    if not np.all(row_sums > 0):
        raise ValueError("Predicted probabilities must have positive row sums.")

    p = p / row_sums

    return -np.log(p[np.arange(y.shape[0]), y.astype(int)])


class NumericalExplainer:
    """
    Numerical EDEF explainer using finite-difference loss gradients.

    This explainer is intended for smooth black-box models that expose
    sklearn-style prediction methods.

    Supported losses:
    - squared_error: uses model.predict(X)
    - log_loss: uses model.predict_proba(X) for binary classification
    - multiclass_log_loss: uses model.predict_proba(X) for multiclass classification
    """

    def __init__(
        self,
        model,
        baseline,
        *,
        loss: str = "squared_error",
        n_steps: int = 32,
        step_size: float = 1e-4,
        feature_names=None,
    ):
        if loss not in {"squared_error", "log_loss", "multiclass_log_loss"}:
            raise ValueError(
                "NumericalExplainer supports squared_error, log_loss, "
                "and multiclass_log_loss."
            )

        if n_steps < 1:
            raise ValueError("n_steps must be positive.")

        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.model = model
        self.baseline = baseline
        self.loss = loss
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)
        self.feature_names = feature_names

    def __call__(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-4,
    ) -> EDEFExplanation:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_obs, n_features).")

        n_obs, n_features = X.shape

        if y.shape[0] != n_obs:
            raise ValueError("y and X must have the same number of observations.")

        if n_obs < 2:
            raise ValueError("At least two observations are required.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        if not np.all(np.isfinite(y.astype(float))):
            raise ValueError("y must contain only finite values.")

        if self.loss == "squared_error":
            y_work = y.astype(float)

        elif self.loss == "log_loss":
            y_work = y.astype(float)
            if not np.all((y_work == 0.0) | (y_work == 1.0)):
                raise ValueError("y must contain only binary labels in {0, 1}.")

        else:
            if not np.all(np.equal(y, np.round(y))):
                raise ValueError("y must contain integer class labels.")
            y_work = y.astype(int)
            if np.any(y_work < 0):
                raise ValueError("y must contain nonnegative class labels.")

        baseline = np.asarray(self.baseline, dtype=float).reshape(-1)

        if baseline.shape[0] != n_features:
            raise ValueError("baseline must have length n_features.")

        if not np.all(np.isfinite(baseline)):
            raise ValueError("baseline must contain only finite values.")

        names = feature_names
        if names is None:
            names = self.feature_names
        if names is None:
            names = [f"x{i}" for i in range(n_features)]
        else:
            names = list(names)
            if len(names) != n_features:
                raise ValueError("feature_names must have length n_features.")

        X0 = np.broadcast_to(baseline.reshape(1, -1), X.shape).copy()
        delta_X = X - X0

        nodes, weights = _gauss_legendre_nodes_weights(self.n_steps)

        observation_values = np.zeros((n_obs, n_features), dtype=float)

        for node, weight in zip(nodes, weights):
            Xt = X0 + float(node) * delta_X
            grad_loss = self._finite_difference_loss_gradient(Xt, y_work)

            observation_values -= float(weight) * delta_X * grad_loss

        values = observation_values.mean(axis=0)
        standard_errors = observation_values.std(axis=0, ddof=1) / np.sqrt(n_obs)

        baseline_loss = float(np.mean(self._loss_per_observation(X0, y_work)))
        model_loss = float(np.mean(self._loss_per_observation(X, y_work)))
        total = baseline_loss - model_loss

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
            loss=self.loss,
            model_type=(
                "numerical_regression"
                if self.loss == "squared_error"
                else "numerical_classification"
                if self.loss == "log_loss"
                else "numerical_multiclass_classification"
            ),
            feature_names=names,
            n_obs=n_obs,
            additivity_error=additivity_error,
        )

    def _finite_difference_loss_gradient(self, X, y):
        n_obs, n_features = X.shape
        grad = np.empty_like(X, dtype=float)
        h = self.step_size

        for j in range(n_features):
            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[:, j] += h
            X_minus[:, j] -= h

            loss_plus = self._loss_per_observation(X_plus, y)
            loss_minus = self._loss_per_observation(X_minus, y)

            grad[:, j] = (loss_plus - loss_minus) / (2.0 * h)

        return grad

    def _loss_per_observation(self, X, y):
        if self.loss == "squared_error":
            pred = self._predict_regression(X)
            return _squared_error_loss(y, pred)

        if self.loss == "log_loss":
            p = self._predict_binary_probability(X)
            return _binary_log_loss(y, p)

        if self.loss == "multiclass_log_loss":
            p = self._predict_multiclass_probability(X)
            return _multiclass_log_loss(y, p)

        raise RuntimeError(f"Unexpected loss: {self.loss}")

    def _predict_regression(self, X):
        if not hasattr(self.model, "predict"):
            raise TypeError("squared_error requires a model with predict(X).")

        pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1)

        if pred.shape[0] != X.shape[0]:
            raise ValueError("predict(X) must return one prediction per observation.")

        if not np.all(np.isfinite(pred)):
            raise ValueError("predict(X) must contain only finite values.")

        return pred

    def _predict_binary_probability(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("log_loss requires a model with predict_proba(X).")

        proba = np.asarray(self.model.predict_proba(X), dtype=float)

        if proba.ndim != 2 or proba.shape[0] != X.shape[0]:
            raise ValueError(
                "predict_proba(X) must have shape (n_obs, n_classes)."
            )

        if proba.shape[1] != 2:
            raise ValueError(
                "log_loss requires binary predict_proba output with two columns."
            )

        if not np.all(np.isfinite(proba)):
            raise ValueError("predict_proba(X) must contain only finite values.")

        return proba[:, 1]

    def _predict_multiclass_probability(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("multiclass_log_loss requires a model with predict_proba(X).")

        proba = np.asarray(self.model.predict_proba(X), dtype=float)

        if proba.ndim != 2 or proba.shape[0] != X.shape[0]:
            raise ValueError(
                "predict_proba(X) must have shape (n_obs, n_classes)."
            )

        if proba.shape[1] < 2:
            raise ValueError("multiclass_log_loss requires at least two classes.")

        if not np.all(np.isfinite(proba)):
            raise ValueError("predict_proba(X) must contain only finite values.")

        return proba