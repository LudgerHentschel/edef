from __future__ import annotations

import numpy as np

from ._results import EDEFExplanation


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch support requires torch. Install with: pip install torch"
        ) from exc
    return torch


def _gauss_legendre_nodes_weights(n_steps: int):
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")

    nodes, weights = np.polynomial.legendre.leggauss(n_steps)

    # Transform from [-1, 1] to [0, 1].
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    return nodes, weights


class TorchExplainer:
    """
    Path-integral EDEF explainer for differentiable PyTorch models.

    ``TorchExplainer`` computes an Euler Decomposition of Explained Fit
    (EDEF) by integrating loss gradients along the straight-line path from a
    fixed baseline input to each evaluation input. The resulting feature
    contributions decompose average loss reduction from the baseline
    prediction to the model prediction.

    Version 1 supports squared-error regression, binary log-loss
    classification, and multiclass log-loss classification. Binary and
    multiclass classification use logits or raw class scores, not probability
    outputs.

    Parameters
    ----------
    model : torch.nn.Module
        Fitted PyTorch model. The model must map an input tensor of shape
        ``(n_obs, n_features)`` to outputs compatible with ``loss``. The
        explainer temporarily switches the model to evaluation mode during
        attribution and restores training mode afterward if needed.

    baseline : array-like or torch.Tensor of shape (n_features,)
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

    device : torch.device or str, optional
        Device to which inputs and the baseline are moved before attribution.
        If omitted, PyTorch's default behavior is used for newly created
        tensors, and existing tensors remain on their current device.

    dtype : torch.dtype, optional
        Floating-point dtype used for inputs and the baseline. If omitted,
        non-floating inputs are converted to ``torch.float32``.

    When to use
    -----------
    Use ``TorchExplainer`` for differentiable PyTorch models when automatic
    differentiation is available and path-integral attribution is desired.

    Notes
    -----
    EDEF is computed for the fixed fitted model. The explainer does not refit
    the model, remove features, or evaluate counterfactual model
    specifications.

    The path integral is approximated numerically. Increasing ``n_steps`` may
    reduce quadrature error for highly nonlinear models, at additional
    computational cost.
    """
    
    def __init__(
        self,
        model,
        baseline,
        *,
        loss: str = "squared_error",
        n_steps: int = 50,
        feature_names=None,
        device=None,
        dtype=None,
    ):
        """
        Initialize a PyTorch EDEF explainer.

        Parameters
        ----------
        model : torch.nn.Module
            Fitted PyTorch model.

        baseline : array-like or torch.Tensor of shape (n_features,)
            Numeric baseline input.

        loss : {"squared_error", "log_loss", "multiclass_log_loss"}, default="squared_error"
            Loss function whose reduction is decomposed.

        n_steps : int, default=50
            Number of Gauss-Legendre quadrature nodes.

        feature_names : sequence of str, optional
            Default feature names for reported feature contributions.

        device : torch.device or str, optional
            Device used for attribution tensors.

        dtype : torch.dtype, optional
            Floating-point dtype used for attribution tensors.
        """
        
        if loss not in {"squared_error", "log_loss", "multiclass_log_loss"}:
            raise ValueError(
                "TorchExplainer supports squared_error, log_loss, and multiclass_log_loss."
            )

        self.torch = _require_torch()

        self.model = model
        self.baseline = baseline
        self.loss = loss
        self.n_steps = n_steps
        self.feature_names = feature_names
        self.device = device
        self.dtype = dtype

        self._nodes, self._weights = _gauss_legendre_nodes_weights(self.n_steps)

    def __call__(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-6,
    ) -> EDEFExplanation:
        """
        Compute the path-integral EDEF decomposition for evaluation data.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (n_obs, n_features)
            Evaluation feature matrix.

        y : array-like or torch.Tensor of shape (n_obs,)
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

        atol : float, default=1e-6
            Absolute tolerance used for the additivity check.

        Returns
        -------
        explanation : EDEFExplanation
            Explanation object containing average feature contributions,
            observation-level feature contributions, standard errors, total
            loss reduction, baseline loss, model loss, feature names, sample
            size, and additivity error.

        Notes
        -----
        The returned ``values`` satisfy, up to quadrature and floating-point
        error,

            values.sum() = baseline_loss - model_loss

        where losses are averaged over the evaluation sample.

        During attribution, the model is evaluated in ``eval`` mode. If the
        model was in training mode before the call, that state is restored
        afterward.

        Examples
        --------
        >>> from edef import TorchExplainer
        >>> explainer = TorchExplainer(model, baseline=x0, loss="squared_error")
        >>> explanation = explainer(X_test, y_test)
        >>> explanation.values
        """
        
        torch = self.torch

        X_t = self._as_tensor(X)
        y_t = self._as_tensor(y).reshape(-1)

        if self.loss == "log_loss":
            if not torch.all((y_t == 0.0) | (y_t == 1.0)):
                raise ValueError("y must contain only binary labels in {0, 1}.")

        if self.loss == "multiclass_log_loss":
            if not torch.all(y_t == torch.round(y_t)):
                raise ValueError("y must contain integer class labels.")
            if torch.any(y_t < 0):
                raise ValueError("y must contain nonnegative class labels.")

        if X_t.ndim != 2:
            raise ValueError("X must have shape (n_obs, n_features).")

        n_obs, n_features = X_t.shape

        if y_t.shape[0] != n_obs:
            raise ValueError("y and X must have the same number of observations.")

        if n_obs < 2:
            raise ValueError("At least two observations are required.")

        baseline_t = self._baseline_tensor(n_features, X_t)

        names = feature_names
        if names is None:
            names = self.feature_names
        if names is None:
            names = [f"x{i}" for i in range(n_features)]
        else:
            names = list(names)
            if len(names) != n_features:
                raise ValueError("feature_names must have length n_features.")

        X0_t = baseline_t.reshape(1, -1).expand_as(X_t)
        delta_X = X_t - X0_t

        nodes = self._nodes
        weights = self._weights

        c_t = torch.zeros_like(X_t)

        was_training = getattr(self.model, "training", False)
        self.model.eval()

        try:
            for node, weight in zip(nodes, weights):
                t = float(node)
                a = float(weight)

                Xt = X0_t + t * delta_X
                Xt = Xt.detach().clone().requires_grad_(True)

                pred_t = self._predict_output(Xt)
                loss_i = self._loss_per_observation(y_t, pred_t)

                grad = torch.autograd.grad(
                    loss_i.sum(),
                    Xt,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                c_t = c_t - a * delta_X * grad

            with torch.no_grad():
                pred0 = self._predict_output(X0_t)
                pred = self._predict_output(X_t)
                
                baseline_loss = torch.mean(self._loss_per_observation(y_t, 
                                             pred0))
                model_loss = torch.mean(self._loss_per_observation(y_t, pred))

                total = baseline_loss - model_loss

                values_t = c_t.mean(dim=0)
                standard_errors_t = c_t.std(dim=0, unbiased=True) / np.sqrt(n_obs)

                additivity_error = values_t.sum() - total

        finally:
            if was_training:
                self.model.train()

        values = values_t.detach().cpu().numpy()
        observation_values = c_t.detach().cpu().numpy()
        standard_errors = standard_errors_t.detach().cpu().numpy()

        total_f = float(total.detach().cpu().item())
        baseline_loss_f = float(baseline_loss.detach().cpu().item())
        model_loss_f = float(model_loss.detach().cpu().item())
        additivity_error_f = float(additivity_error.detach().cpu().item())

        if check_additivity and abs(additivity_error_f) > atol:
            raise RuntimeError(
                "EDEF contributions do not add to total loss reduction. "
                f"Additivity error: {additivity_error_f}"
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
                "torch_regression"
                if self.loss == "squared_error"
                else "torch_classification"
                if self.loss == "log_loss"
                else "torch_multiclass_classification"
            ),
            feature_names=names,
            n_obs=n_obs,
            additivity_error=additivity_error_f,
        )

    def _as_tensor(self, x):
        """Convert input to a tensor using the explainer dtype and device."""
        torch = self.torch

        if isinstance(x, torch.Tensor):
            out = x
        else:
            out = torch.as_tensor(x)

        if self.dtype is not None:
            out = out.to(dtype=self.dtype)
        elif not torch.is_floating_point(out):
            out = out.to(dtype=torch.float32)

        if self.device is not None:
            out = out.to(self.device)

        return out

    def _baseline_tensor(self, n_features: int, X_t):
        """Return the prepared baseline tensor aligned with ``X_t``."""
        torch = self.torch

        baseline = self.baseline

        if isinstance(baseline, str):
            raise ValueError(
                "TorchExplainer currently requires a numeric baseline vector, "
                "not a string baseline rule."
            )

        b = self._as_tensor(baseline).reshape(-1)

        if b.shape[0] != n_features:
            raise ValueError("baseline must have length n_features.")

        if not torch.all(torch.isfinite(b)):
            raise ValueError("baseline must contain only finite values.")

        return b.to(device=X_t.device, dtype=X_t.dtype)

    def _predict_output(self, X_t):
        """Evaluate the model and validate output shape for the selected loss."""
        pred = self.model(X_t)

        if self.loss in {"squared_error", "log_loss"}:
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.reshape(-1)

            if pred.ndim != 1:
                raise ValueError(
                    "TorchExplainer requires scalar output with shape "
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
        
    def _loss_per_observation(self, y_t, pred_t):
        """Return per-observation loss values for the selected loss."""    
        torch = self.torch

        if self.loss == "squared_error":
            return (y_t - pred_t) ** 2

        if self.loss == "log_loss":
            return torch.nn.functional.binary_cross_entropy_with_logits(
                pred_t,
                y_t,
                reduction="none",
            )

        if self.loss == "multiclass_log_loss":
            y_long = y_t.to(dtype=torch.long)
            return torch.nn.functional.cross_entropy(
                pred_t,
                y_long,
                reduction="none",
            )

        raise RuntimeError(f"Unexpected loss: {self.loss}")