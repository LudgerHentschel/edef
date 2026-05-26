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
    Numerical EDEF explainer for smooth black-box prediction models.

    ``NumericalExplainer`` computes an Euler Decomposition of Explained Fit
    (EDEF) by approximating loss gradients with central finite differences
    along the straight-line path from a fixed baseline input to each
    evaluation input. The resulting feature contributions decompose average
    loss reduction from baseline predictions to endpoint model predictions.

    This explainer is intended for smooth or approximately smooth models that
    expose scikit-learn-style prediction methods. It is a generic fallback
    backend when analytic gradients, linear structure, or exact TreeIG traces
    are not available.

    Parameters
    ----------
    model : object
        Fitted prediction model. For ``loss="squared_error"``, the model must
        expose ``predict(X)``. For ``loss="log_loss"`` and
        ``loss="multiclass_log_loss"``, the model must expose
        ``predict_proba(X)``.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for each path.

    loss : {"squared_error", "log_loss", "multiclass_log_loss"}, default="squared_error"
        Loss function whose reduction is decomposed. ``"log_loss"`` uses the
        positive-class probability from binary ``predict_proba(X)``.
        ``"multiclass_log_loss"`` uses the full class-probability matrix from
        ``predict_proba(X)``.

    n_steps : int, default=32
        Number of Gauss-Legendre quadrature nodes used to approximate the path
        integral.

    step_size : float, default=1e-4
        Central finite-difference step size used to approximate loss
        gradients with respect to input features.

    feature_names : sequence of str, optional
        Default feature names used in returned EDEF results.

    When to use
    -----------
    Use ``NumericalExplainer`` as a fallback backend for smooth black-box
    models that expose sklearn-style prediction APIs but do not provide
    analytic gradients or exact TreeIG traces.

    Notes
    -----
    EDEF is computed for the fixed fitted model. The explainer does not refit
    the model, remove features, or evaluate counterfactual model
    specifications.

    This backend is approximate. Additivity error may reflect finite-
    difference error, quadrature error, nonsmooth model behavior, probability
    clipping in log-loss calculations, and floating-point arithmetic.
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
        """
        Initialize a numerical EDEF explainer.

        Parameters
        ----------
        model : object
            Fitted prediction model.

        baseline : array-like of shape (n_features,)
            Numeric baseline input.

        loss : {"squared_error", "log_loss", "multiclass_log_loss"}, default="squared_error"
            Loss function whose reduction is decomposed.

        n_steps : int, default=32
            Number of Gauss-Legendre quadrature nodes.

        step_size : float, default=1e-4
            Central finite-difference step size.

        feature_names : sequence of str, optional
            Default feature names for reported feature contributions.
        """
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
        self._nodes, self._weights = _gauss_legendre_nodes_weights(self.n_steps)

    def __call__(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-4,
    ) -> EDEFExplanation:
        """
        Compute the numerical EDEF decomposition for evaluation data.

        Parameters
        ----------
        X : array-like of shape (n_obs, n_features)
            Evaluation feature matrix. Values must be finite and numeric.

        y : array-like of shape (n_obs,)
            Observed outcomes or class labels. For ``loss="squared_error"``,
            values are interpreted as numeric outcomes. For ``loss="log_loss"``,
            labels must be binary values in ``{0, 1}``. For
            ``loss="multiclass_log_loss"``, labels must be nonnegative integer
            class indices.

        feature_names : sequence of str, optional
            Feature names used in the returned explanation. If omitted, names
            supplied at construction are used. If neither is supplied, names
            are generated as ``"x0"``, ``"x1"``, and so on.

        check_additivity : bool, default=True
            Whether to verify that average feature contributions reconstruct
            total average loss reduction up to numerical tolerance.

        atol : float, default=1e-4
            Absolute tolerance used for the additivity check. A looser default
            is used because this backend uses numerical finite differences and
            numerical quadrature.

        Returns
        -------
        explanation : EDEFExplanation
            Explanation object containing average feature contributions,
            observation-level feature contributions, standard errors, total
            loss reduction, baseline loss, model loss, feature names, sample
            size, and additivity error.

        Notes
        -----
        The returned ``values`` satisfy approximately,

            values.sum() = baseline_loss - model_loss

        where losses are averaged over the evaluation sample. The discrepancy
        is reported as ``additivity_error``.

        Examples
        --------
        >>> from edef import NumericalExplainer
        >>> explainer = NumericalExplainer(model, baseline=x0, loss="squared_error")
        >>> explanation = explainer(X_test, y_test)
        >>> explanation.values
        """
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

        nodes = self._nodes
        weights = self._weights

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
                "EDEF contributions do not add to total loss reduction. "
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
        """Approximate per-observation loss gradients by central differences."""
        n_obs, n_features = X.shape
        h = self.step_size

        eye = np.eye(n_features, dtype=float) * h

        X_plus = (
            X[:, None, :]
            + eye[None, :, :]
        ).reshape(n_obs * n_features, n_features)

        X_minus = (
            X[:, None, :]
            - eye[None, :, :]
        ).reshape(n_obs * n_features, n_features)

        y_rep = np.repeat(y, n_features)

        loss_plus = self._loss_per_observation(X_plus, y_rep)
        loss_minus = self._loss_per_observation(X_minus, y_rep)

        grad = (
            (loss_plus - loss_minus)
            .reshape(n_obs, n_features)
            / (2.0 * h)
        )

        return grad
        
    def _loss_per_observation(self, X, y):
        """Return per-observation losses for the selected loss."""
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
        """Return validated scalar predictions from ``model.predict``."""
        if not hasattr(self.model, "predict"):
            raise TypeError("squared_error requires a model with predict(X).")

        pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1)

        if pred.shape[0] != X.shape[0]:
            raise ValueError("predict(X) must return one prediction per observation.")

        if not np.all(np.isfinite(pred)):
            raise ValueError("predict(X) must contain only finite values.")

        return pred

    def _predict_binary_probability(self, X):
        """Return validated positive-class probabilities from ``predict_proba``."""
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
        """Return validated class-probability matrices from ``predict_proba``."""
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