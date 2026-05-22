import numpy as np
from sklearn.linear_model import LogisticRegression

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 600
    n_features = 3
    n_classes = 3

    X = rng.normal(size=(n_obs, n_features))

    coef = np.array(
        [
            [1.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.8, -0.6, 0.0],
        ]
    )
    intercept = np.array([0.2, -0.1, -0.1])

    eta = X @ coef.T + intercept
    eta = eta - eta.max(axis=1, keepdims=True)
    p = np.exp(eta)
    p = p / p.sum(axis=1, keepdims=True)

    y = np.array([rng.choice(n_classes, p=row) for row in p])

    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(X, y)

    explainer = edef.LinearExplainer(
        model,
        loss="log_loss",
        feature_names=["x1", "x2", "x3"],
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
            f"{name:>18s}  "
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