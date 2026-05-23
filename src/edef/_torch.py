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
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")

    nodes, weights = np.polynomial.legendre.leggauss(n_steps)

    # Transform from [-1, 1] to [0, 1].
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    return nodes, weights


class TorchExplainer:
    """
    Path-integral EDEF explainer for smooth PyTorch regression models.

    Version 1 supports:
    - scalar regression output;
    - squared-error loss;
    - fixed baseline vector;
    - Gauss-Legendre quadrature;
    - observation-level feature contributions.
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

    def __call__(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-6,
    ) -> EDEFExplanation:
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

        nodes, weights = _gauss_legendre_nodes_weights(self.n_steps)

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
                "EDEF contributions do not add to total fit improvement. "
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