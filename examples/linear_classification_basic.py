import numpy as np
from sklearn.linear_model import LogisticRegression

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 500
    n_features = 3

    X = rng.normal(size=(n_obs, n_features))
    beta = np.array([1.5, 0.75, 0.0])

    eta = X @ beta - 0.25
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p, size=n_obs)

    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(X, y)

    explainer = edef.LinearExplainer(
        model,
        loss="log_loss",
        feature_names=["x1", "x2", "x3"],
    )

    result = explainer(X, y)

    grouped = result.group(
        ["signal", "signal", "noise", "intercept"]
    )

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
    print("Grouped contributions")
    print("---------------------")
    for name, value, se, tval, share in zip(
        grouped.feature_names,
        grouped.values,
        grouped.standard_errors,
        grouped.t_values,
        grouped.proportions,
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