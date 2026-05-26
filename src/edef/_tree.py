from __future__ import annotations

import numpy as np

from ._results import EDEFExplanation


def _require_treeig():
    try:
        from treeig import TreeIG
    except ImportError as exc:
        raise ImportError(
            "TreeExplainer requires treeig. Install with: pip install treeig"
        ) from exc
    return TreeIG


def _squared_error_loss(y, pred):
    return (y - pred) ** 2


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_log_loss(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _logsumexp_1d(z):
    z = np.asarray(z, dtype=float)
    zmax = np.max(z)
    return zmax + np.log(np.sum(np.exp(z - zmax)))


def _multiclass_log_loss_one(y_i, scores):
    return _logsumexp_1d(scores) - scores[int(y_i)]


class TreeExplainer:
    """
    EDEF explainer for supported tree models using TreeIG path traces.

    ``TreeExplainer`` computes an Euler Decomposition of Explained Fit
    (EDEF) for fitted tree-based models by using TreeIG split-crossing traces.
    The explainer decomposes average loss reduction from the baseline
    prediction to the model prediction into feature-level contributions.

    Version 1 supports squared-error tree regression, binary log-loss
    additive-score classification, and multiclass softmax log-loss
    additive-score classification. Classification is based on raw margins,
    logits, or class scores, not probability-output attributions.

    Parameters
    ----------
    model : object
        Fitted tree-based model supported by TreeIG.

    baseline : array-like of shape (n_features,)
        Numeric baseline input used as the starting point for TreeIG paths.

    loss : {"squared_error", "log_loss", "multiclass_log_loss"}, default="squared_error"
        Loss function whose reduction is decomposed. Use ``"squared_error"``
        for regression, ``"log_loss"`` for binary additive-score
        classification, and ``"multiclass_log_loss"`` for multiclass
        additive-score classification.

    feature_names : sequence of str, optional
        Default feature names used in returned EDEF results.

    target : int or None, default=None
        TreeIG target used for scalar-output attribution. For binary
        classification, this follows TreeIG's target convention. Ignored for
        ``loss="multiclass_log_loss"``, where all class-score targets are
        traced.

    time_tol : float, default=1e-10
        Tolerance used by TreeIG when ordering split-crossing times.

    n_classes : int or None, default=None
        Number of classes for multiclass log-loss attribution. If omitted,
        ``len(model.classes_)`` is used when available.

    When to use
    -----------
    Use ``TreeExplainer`` for tree ensembles supported by TreeIG, including
    supported scikit-learn, XGBoost, and LightGBM models. This backend uses
    exact TreeIG path traces rather than numerical differentiation.

    Notes
    -----
    EDEF is computed for the fixed fitted model. The explainer does not refit
    the model, remove features, or evaluate counterfactual model
    specifications.

    For supported TreeIG models, path events are computed exactly from tree
    split crossings. Additivity error should therefore mainly reflect
    floating-point arithmetic and backend prediction conventions.
    """

    def __init__(
        self,
        model,
        baseline,
        *,
        loss: str = "squared_error",
        feature_names=None,
        target=None,
        time_tol: float = 1e-10,
        n_classes=None,
    ):
        """
        Initialize a tree-based EDEF explainer.

        Parameters
        ----------
        model : object
            Fitted tree-based model supported by TreeIG.

        baseline : array-like of shape (n_features,)
            Numeric baseline input.

        loss : {"squared_error", "log_loss", "multiclass_log_loss"}, default="squared_error"
            Loss function whose reduction is decomposed.

        feature_names : sequence of str, optional
            Default feature names for reported feature contributions.

        target : int or None, default=None
            TreeIG target for scalar-output attribution. Ignored for
            multiclass log-loss attribution.

        time_tol : float, default=1e-10
            Tolerance used by TreeIG when ordering split-crossing times.

        n_classes : int or None, default=None
            Number of classes for multiclass log-loss attribution.
        """
        if loss not in {"squared_error", "log_loss", "multiclass_log_loss"}:
            raise ValueError(
                "TreeExplainer supports squared_error, log_loss, "
                "and multiclass_log_loss."
            )

        TreeIG = _require_treeig()

        self.model = model
        self.baseline = baseline
        self.loss = loss
        self.feature_names = feature_names
        self.target = target
        self.time_tol = time_tol
        self.n_classes = n_classes
        self._TreeIG = TreeIG

        self._treeig_by_target = {}

        if loss == "multiclass_log_loss":
            self._treeig = None
        else:
            self._treeig = self._get_treeig(target)

    def __call__(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-5,
    ) -> EDEFExplanation:
        """
        Compute the tree-based EDEF decomposition for evaluation data.

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

        atol : float, default=1e-5
            Absolute tolerance used for the additivity check. A looser default
            is used because several tree backends use float32 internals.

        Returns
        -------
        explanation : EDEFExplanation
            Explanation object containing average feature contributions,
            observation-level feature contributions, standard errors, total
            loss reduction, baseline loss, model loss, feature names, sample
            size, and additivity error.

        Notes
        -----
        The returned ``values`` satisfy, up to floating-point error,

            values.sum() = baseline_loss - model_loss

        where losses are averaged over the evaluation sample.

        Examples
        --------
        >>> from edef import TreeExplainer
        >>> explainer = TreeExplainer(model, baseline=x0, loss="squared_error")
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

        if self.loss == "multiclass_log_loss":
            return self._call_multiclass_log_loss(
                X,
                y,
                feature_names=feature_names,
                check_additivity=check_additivity,
                atol=atol,
            )

        y = y.astype(float)

        if self.loss == "log_loss":
            if not np.all((y == 0.0) | (y == 1.0)):
                raise ValueError("y must contain only binary labels in {0, 1}.")

        names = self._resolve_feature_names(feature_names, n_features)

        if self.loss in {"squared_error", "log_loss"} and hasattr(
            self._treeig,
            "loss_attribution",
        ):
            out = self._treeig.loss_attribution(
                X,
                y,
                loss=self.loss,
                target=self.target,
            )

            values = out["values"]
            observation_values = out["observation_values"]
            standard_errors = out["standard_errors"]
            baseline_loss = out["baseline_loss"]
            model_loss = out["model_loss"]
            total = out["total"]

            additivity_error = values.sum() - total

            if check_additivity and abs(additivity_error) > atol:
                raise RuntimeError(
                    "EDEF contributions do not add to total loss reduction. "
                    f"Additivity error: {additivity_error}"
                )

            model_type = (
                "tree_regression"
                if self.loss == "squared_error"
                else "tree_classification"
            )

            return EDEFExplanation(
                values=values,
                observation_values=observation_values,
                standard_errors=standard_errors,
                total=total,
                baseline_loss=baseline_loss,
                model_loss=model_loss,
                loss=self.loss,
                model_type=model_type,
                feature_names=names,
                n_obs=n_obs,
                additivity_error=additivity_error,
            )

        trace = self._treeig.trace(X, target=self.target)

        counts = trace["counts"]
        features = trace["features"]
        jumps = trace["jumps"]
        baseline_prediction = trace["baseline_prediction"]
        endpoint_prediction = trace["endpoint_prediction"]

        observation_values = np.zeros((n_obs, n_features), dtype=float)

        for i in range(n_obs):
            pred_before = float(baseline_prediction)
            n_events = int(counts[i])

            for k in range(n_events):
                j = int(features[i, k])
                jump = float(jumps[i, k])

                pred_after = pred_before + jump

                if self.loss == "squared_error":
                    contribution = (
                        _squared_error_loss(y[i], pred_before)
                        - _squared_error_loss(y[i], pred_after)
                    )
                else:
                    p_before = _sigmoid(pred_before)
                    p_after = _sigmoid(pred_after)
                    contribution = (
                        _binary_log_loss(y[i], p_before)
                        - _binary_log_loss(y[i], p_after)
                    )

                if j >= 0:
                    observation_values[i, j] += contribution

                pred_before = pred_after

        values = observation_values.mean(axis=0)
        standard_errors = observation_values.std(axis=0, ddof=1) / np.sqrt(n_obs)

        if self.loss == "squared_error":
            baseline_loss = float(np.mean(_squared_error_loss(y, baseline_prediction)))
            model_loss = float(np.mean(_squared_error_loss(y, endpoint_prediction)))
            model_type = "tree_regression"
        else:
            baseline_loss = float(
                np.mean(_binary_log_loss(y, _sigmoid(baseline_prediction)))
            )
            model_loss = float(
                np.mean(_binary_log_loss(y, _sigmoid(endpoint_prediction)))
            )
            model_type = "tree_classification"

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
            model_type=model_type,
            feature_names=names,
            n_obs=n_obs,
            additivity_error=additivity_error,
        )

    def _call_multiclass_log_loss(
        self,
        X,
        y,
        *,
        feature_names=None,
        check_additivity: bool = True,
        atol: float = 1e-8,
    ) -> EDEFExplanation:
        """Compute multiclass log-loss EDEF using class-score TreeIG traces."""
        y = np.asarray(y).reshape(-1)
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_obs, n_features).")

        n_obs, n_features = X.shape

        if y.shape[0] != n_obs:
            raise ValueError("y and X must have the same number of observations.")

        if not np.all(np.equal(y, np.round(y))):
            raise ValueError("y must contain integer class labels.")

        y = y.astype(int)

        if np.any(y < 0):
            raise ValueError("y must contain nonnegative class labels.")

        n_classes = self._resolve_n_classes()

        if n_classes < 2:
            raise ValueError("multiclass_log_loss requires at least two classes.")

        if np.max(y) >= n_classes:
            raise ValueError("y contains class labels outside n_classes.")

        names = self._resolve_feature_names(feature_names, n_features)

        treeig0 = self._get_treeig(0)

        if hasattr(treeig0, "multiclass_loss_attribution"):
            out = treeig0.multiclass_loss_attribution(
                X,
                y,
                n_classes=n_classes,
            )

            values = out["values"]
            observation_values = out["observation_values"]
            standard_errors = out["standard_errors"]
            baseline_loss = out["baseline_loss"]
            model_loss = out["model_loss"]
            total = out["total"]

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
                loss="multiclass_log_loss",
                model_type="tree_multiclass_classification",
                feature_names=names,
                n_obs=n_obs,
                additivity_error=additivity_error,
            )

        traces = []
        baseline_scores = np.zeros(n_classes, dtype=float)
        endpoint_scores = np.zeros((n_obs, n_classes), dtype=float)

        for k in range(n_classes):
            treeig_k = self._get_treeig(k)
            trace_k = treeig_k.trace(X, target=k)

            traces.append(trace_k)

            baseline_scores[k] = float(trace_k["baseline_prediction"])
            endpoint_scores[:, k] = np.asarray(
                trace_k["endpoint_prediction"],
                dtype=float,
            )

        observation_values = np.zeros((n_obs, n_features), dtype=float)

        for i in range(n_obs):
            current_scores = baseline_scores.copy()
            events = []

            for k in range(n_classes):
                trace_k = traces[k]
                n_events = int(trace_k["counts"][i])

                for m in range(n_events):
                    events.append(
                        (
                            float(trace_k["times"][i, m]),
                            k,
                            int(trace_k["features"][i, m]),
                            float(trace_k["jumps"][i, m]),
                        )
                    )

            events.sort(key=lambda item: (item[0], item[1], item[2]))

            for _, class_index, feature_index, jump in events:
                before_loss = _multiclass_log_loss_one(y[i], current_scores)
                current_scores[class_index] += jump
                after_loss = _multiclass_log_loss_one(y[i], current_scores)

                contribution = before_loss - after_loss

                if feature_index >= 0:
                    observation_values[i, feature_index] += contribution

            np.testing.assert_allclose(
                current_scores,
                endpoint_scores[i],
                atol=atol,
                rtol=0.0,
            )

        values = observation_values.mean(axis=0)
        standard_errors = observation_values.std(axis=0, ddof=1) / np.sqrt(n_obs)

        baseline_losses = np.array(
            [_multiclass_log_loss_one(y_i, baseline_scores) for y_i in y],
            dtype=float,
        )

        model_losses = np.array(
            [
                _multiclass_log_loss_one(y[i], endpoint_scores[i])
                for i in range(n_obs)
            ],
            dtype=float,
        )

        baseline_loss = float(np.mean(baseline_losses))
        model_loss = float(np.mean(model_losses))
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
            loss="multiclass_log_loss",
            model_type="tree_multiclass_classification",
            feature_names=names,
            n_obs=n_obs,
            additivity_error=additivity_error,
        )

    def _get_treeig(self, target):
        """Return a cached TreeIG explainer for a scalar target."""
        key = None if target is None else int(target)

        if key not in self._treeig_by_target:
            self._treeig_by_target[key] = self._TreeIG(
                self.model,
                baseline=self.baseline,
                target=key,
                time_tol=self.time_tol,
            )

        return self._treeig_by_target[key]

    def _resolve_n_classes(self) -> int:
        """Resolve the number of classes for multiclass attribution."""
        if self.n_classes is not None:
            return int(self.n_classes)

        classes = getattr(self.model, "classes_", None)

        if classes is not None:
            return int(len(classes))

        raise ValueError(
            "multiclass_log_loss requires n_classes or a model with classes_."
        )

    def _resolve_feature_names(self, feature_names, n_features: int) -> list[str]:
        """Resolve or generate feature names for the explanation output."""
        names = feature_names

        if names is None:
            names = self.feature_names

        if names is None:
            return [f"x{i}" for i in range(n_features)]

        names = list(names)

        if len(names) != n_features:
            raise ValueError("feature_names must have length n_features.")

        return names