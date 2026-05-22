from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EDEFExplanation:
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
        if self.total == 0:
            return np.full_like(self.values, np.nan, dtype=float)
        return self.values / self.total

    @property
    def t_values(self) -> np.ndarray:
        out = np.full_like(self.values, np.nan, dtype=float)
        mask = self.standard_errors > 0
        out[mask] = self.values[mask] / self.standard_errors[mask]
        return out

    def as_dict(self) -> dict:
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
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame requires pandas. Install with: pip install pandas") from exc

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
            out = out.sort_values("value", ascending=False).reset_index(drop=True)

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

        order = np.argsort(np.abs(values))[::-1]
        if sort:
            order = np.argsort(values)[::-1]

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
        ax.set_xlabel("EDEF contribution to fit")
#        ax.set_title("EDEF feature contributions")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax


    def to_shap_explanation(self, *, data=None, base_values=None):
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