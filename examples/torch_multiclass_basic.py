import numpy as np
import torch

import edef


class NonlinearMulticlassModel(torch.nn.Module):
    def forward(self, X):
        z0 = 1.2 * X[:, 0] ** 2 - 0.25
        z1 = 0.9 * X[:, 1] + 0.2
        z2 = -0.7 * X[:, 0] + 0.4 * X[:, 1] ** 2
        return torch.stack([z0, z1, z2], dim=1)


def main():
    rng = np.random.default_rng(123)

    n_obs = 500

    X = rng.normal(size=(n_obs, 2)).astype(np.float32)

    z0 = 1.2 * X[:, 0] ** 2 - 0.25
    z1 = 0.9 * X[:, 1] + 0.2
    z2 = -0.7 * X[:, 0] + 0.4 * X[:, 1] ** 2

    scores = np.column_stack([z0, z1, z2])
    scores = scores - scores.max(axis=1, keepdims=True)

    p = np.exp(scores)
    p = p / p.sum(axis=1, keepdims=True)

    y = np.array([rng.choice(3, p=row) for row in p], dtype=np.float32)

    model = NonlinearMulticlassModel()

    explainer = edef.TorchExplainer(
        model,
        baseline=np.zeros(2, dtype=np.float32),
        loss="multiclass_log_loss",
        n_steps=64,
        feature_names=["x1_nonlinear", "x2_nonlinear"],
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
            f"{name:>16s}  "
            f"edef={value: .6f}  "
            f"se={se: .6f}  "
            f"t={tval: .3f}  "
            f"share={share: .3f}"
        )

    print()
    print("Totals")
    print("------")
    print(f"baseline_log_loss: {result.baseline_loss:.6f}")
    print(f"model_log_loss   : {result.model_loss:.6f}")
    print(f"explained_fit    : {result.total:.6f}")
    print(f"sum_contributions: {result.values.sum():.6f}")
    print(f"additivity_error : {result.additivity_error:.3e}")


if __name__ == "__main__":
    main()