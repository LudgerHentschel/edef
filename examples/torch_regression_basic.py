import numpy as np
import torch

import edef


class NonlinearRegressionModel(torch.nn.Module):
    def forward(self, X):
        return X[:, 0] ** 2 + 0.5 * X[:, 1]


def main():
    rng = np.random.default_rng(123)

    n_obs = 300

    X = rng.normal(size=(n_obs, 2)).astype(np.float32)
    y = (
        X[:, 0] ** 2
        + 0.5 * X[:, 1]
        + rng.normal(scale=0.25, size=n_obs)
    ).astype(np.float32)

    model = NonlinearRegressionModel()

    explainer = edef.TorchExplainer(
        model,
        baseline=np.zeros(2, dtype=np.float32),
        loss="squared_error",
        n_steps=64,
        feature_names=["x1_squared", "x2_linear"],
    )

    result = explainer(X, y)

    print("Feature contributions")
    print("---------------------")
    for name, value, se, tval, share in zip(
        result.feature_names,
        result.values,
        result.standard_errors,
        result.t_values,
        result.proportions,
    ):
        print(
            f"{name:>12s}  "
            f"edef={value: .6f}  "
            f"se={se: .6f}  "
            f"t={tval: .3f}  "
            f"share={share: .3f}"
        )

    print()
    print("Totals")
    print("------")
    print(f"baseline_loss    : {result.baseline_loss:.6f}")
    print(f"model_loss       : {result.model_loss:.6f}")
    print(f"explained_fit    : {result.total:.6f}")
    print(f"sum_contributions: {result.values.sum():.6f}")
    print(f"additivity_error : {result.additivity_error:.3e}")


if __name__ == "__main__":
    main()