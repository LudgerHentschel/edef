from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EDEFExplanation:
    """
    Container for EDEF feature-contribution results.

    ``EDEFExplanation`` stores the output of an Euler Decomposition of
    Explained Fit. The primary quantities are level contributions: feature
    values that sum to total average loss reduction, up to numerical error.

    Parameters
    ----------
    values : ndarray of shape (n_features,)
        Average feature contributions to loss reduction.

    observation_values : ndarray of shape (n_obs, n_features)
        Observation-level feature contributions.

    standard_errors : ndarray of shape (n_features,)
        Standard errors of ``values``, computed across observations.

    total : float
        Total average loss reduction, equal to ``baseline_loss - model_loss``.

    baseline_loss : float
        Average loss at the baseline prediction.

    model_loss : float
        Average loss at the endpoint model prediction.

    loss : str
        Loss function used in the decomposition.

    model_type : str
        Backend/model-type label.

    feature_names : list of str
        Feature names corresponding to columns of ``values`` and
        ``observation_values``.

    n_obs : int
        Number of observations used in the decomposition.

    additivity_error : float, default=0.0
        Difference between ``values.sum()`` and ``total``.

    Notes
    -----
    The level contributions in ``values`` are the main EDEF estimands.
    Proportions are derived summaries and may be unstable when ``total`` is
    close to zero.
    """
    values: np.ndarray
    observation_values: np.ndarray
    standard_errors: np.ndarray
    total: float
    baseline_loss: float
    model_loss: float
    loss: str
    model_type: str
    feature_names: list[str]
    n_obs: int
    additivity_error: float = 0.0

    @property
    def proportions(self) -> np.ndarray:
        """
        Return feature contributions divided by total loss reduction.

        Returns
        -------
        proportions : ndarray of shape (n_features,)
            ``values / total``. If ``total`` is zero, all entries are ``nan``.
        """
        if self.total == 0:
            return np.full_like(self.values, np.nan, dtype=float)
        return self.values / self.total

    @property
    def t_values(self) -> np.ndarray:
        """
        Return contribution t-values.

        Returns
        -------
        t_values : ndarray of shape (n_features,)
            ``values / standard_errors`` where standard errors are positive;
            ``nan`` otherwise.
        """
        out = np.full_like(self.values, np.nan, dtype=float)
        mask = self.standard_errors > 0
        out[mask] = self.values[mask] / self.standard_errors[mask]
        return out

    def as_dict(self) -> dict:
        """
        Return the explanation as a dictionary.

        Returns
        -------
        result : dict
            Dictionary containing stored fields together with derived
            ``proportions`` and ``t_values``.
        """
        return {
            "values": self.values,
            "observation_values": self.observation_values,
            "standard_errors": self.standard_errors,
            "total": self.total,
            "baseline_loss": self.baseline_loss,
            "model_loss": self.model_loss,
            "loss": self.loss,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "n_obs": self.n_obs,
            "additivity_error": self.additivity_error,
            "proportions": self.proportions,
            "t_values": self.t_values,
        }
        
    def group(self, groups) -> "EDEFExplanation":
        """
        Aggregate feature contributions into user-specified groups.

        Parameters
        ----------
        groups : sequence
            Group label for each feature. Must have length ``n_features``.
            Features with the same label are summed observation by observation.

        Returns
        -------
        grouped : EDEFExplanation
            New explanation whose features are the unique group labels in
            first-occurrence order.

        Notes
        -----
        Grouping is performed on ``observation_values`` before recomputing
        averages and standard errors. This preserves the covariance structure
        among features within each group.
        """
        groups = list(groups)

        if len(groups) != len(self.feature_names):
            raise ValueError("groups must have length n_features.")

        unique_groups = list(dict.fromkeys(groups))
        group_index = {g: i for i, g in enumerate(unique_groups)}

        grouped_observation_values = np.zeros(
            (self.observation_values.shape[0], len(unique_groups)),
            dtype=float,
        )

        for j, group in enumerate(groups):
            grouped_observation_values[:, group_index[group]] += (
                self.observation_values[:, j]
            )

        grouped_values = grouped_observation_values.mean(axis=0)
        grouped_standard_errors = (
            grouped_observation_values.std(axis=0, ddof=1) / np.sqrt(self.n_obs)
        )

        grouped_additivity_error = grouped_values.sum() - self.total

        return EDEFExplanation(
            values=grouped_values,
            observation_values=grouped_observation_values,
            standard_errors=grouped_standard_errors,
            total=self.total,
            baseline_loss=self.baseline_loss,
            model_loss=self.model_loss,
            loss=self.loss,
            model_type=f"{self.model_type}_grouped",
            feature_names=unique_groups,
            n_obs=self.n_obs,
            additivity_error=grouped_additivity_error,
        )        

        
    def to_frame(self, *, sort: bool = True):
        """
        Convert feature-level results to a pandas DataFrame.

        Parameters
        ----------
        sort : bool, default=True
            Whether to sort rows by absolute contribution magnitude in
            descending order.

        Returns
        -------
        frame : pandas.DataFrame
            DataFrame with columns ``feature``, ``value``,
            ``standard_error``, ``t_value``, and ``proportion``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_frame requires pandas. Install with: pip install pandas"
            ) from exc

        out = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": self.values,
                "standard_error": self.standard_errors,
                "t_value": self.t_values,
                "proportion": self.proportions,
            }
        )

        if sort:
            out = (
                out.iloc[np.argsort(np.abs(out["value"].to_numpy()))[::-1]]
                .reset_index(drop=True)
            )

        return out
        
    def plot(
        self,
        *,
        kind: str = "bar",
        alpha: float = 0.05,
        sort: bool = True,
        max_features: int | None = None,
        ax=None,
    ):
        """
        Plot feature-level EDEF contributions.

        Parameters
        ----------
        kind : {"bar"}, default="bar"
            Plot type. Only horizontal bar plots are currently supported.

        alpha : float, default=0.05
            Two-sided confidence level parameter for normal-approximation
            error bars.

        sort : bool, default=True
            Whether to sort features by absolute contribution magnitude.

        max_features : int or None, default=None
            Maximum number of features to display.

        ax : matplotlib.axes.Axes, optional
            Existing axes on which to draw the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        """
        if kind != "bar":
            raise ValueError("Only kind='bar' is currently supported.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "plot requires matplotlib. Install with: pip install matplotlib"
            ) from exc

        z = 1.959963984540054  # 95% normal critical value
        if alpha != 0.05:
            try:
                from statistics import NormalDist
                z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
            except Exception:
                raise ValueError("alpha != 0.05 requires Python statistics.NormalDist.")

        values = self.values.copy()
        se = self.standard_errors.copy()
        names = np.asarray(self.feature_names, dtype=object)

        order = np.arange(len(values))
        
        if sort:
            order = np.argsort(np.abs(values))[::-1]
        
        if max_features is not None:
            order = order[:max_features]

        values = values[order]
        se = se[order]
        names = names[order]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(values))))

        y_pos = np.arange(len(values))

        ax.barh(y_pos, values, xerr=z * se, align="center", alpha=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("EDEF contribution to loss reduction")
#        ax.set_title("EDEF feature contributions")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax


    def to_shap_explanation(self, *, data=None, base_values=None):
        """
        Convert observation-level EDEF values to a SHAP Explanation object.

        Parameters
        ----------
        data : array-like, optional
            Optional feature data to attach to the SHAP explanation.

        base_values : array-like or None, default=None
            Base values supplied to ``shap.Explanation``. If omitted, an array
            filled with ``baseline_loss`` is used.

        Returns
        -------
        explanation : shap.Explanation
            SHAP-compatible explanation containing EDEF observation-level
            contributions.

        Notes
        -----
        This conversion is for interoperability and visualization. EDEF
        values decompose loss reduction, not model-output differences, so the
        resulting object should not be interpreted as ordinary SHAP values.
        """    
        try:
            import shap
        except ImportError as exc:
            raise ImportError("to_shap_explanation requires shap. Install with: pip install shap") from exc

        if base_values is None:
            base_values = np.full(self.n_obs, self.baseline_loss, dtype=float)

        return shap.Explanation(
            values=self.observation_values,
            base_values=base_values,
            data=data,
            feature_names=self.feature_names,
        )        
        
    def __str__(self) -> str:
        lines = []

        lines.append("EDEFExplanation")
        lines.append("-" * 15)
        lines.append(f"loss          : {self.loss}")
        lines.append(f"model_type    : {self.model_type}")
        lines.append(f"n_obs         : {self.n_obs}")
        lines.append(f"baseline_loss : {self.baseline_loss:.6f}")
        lines.append(f"model_loss    : {self.model_loss:.6f}")
        lines.append(f"total         : {self.total:.6f}")
        lines.append(f"additivity_err: {self.additivity_error:.3e}")
        lines.append("")
        lines.append("Feature contributions")
        lines.append("---------------------")

        for name, value, se, tval, share in zip(
            self.feature_names,
            self.values,
            self.standard_errors,
            self.t_values,
            self.proportions,
        ):
            lines.append(
                f"{name:>18s}  "
                f"edef={value: .6f}  "
                f"se={se: .6f}  "
                f"t={tval: .3f}  "
                f"share={share: .3f}"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()        