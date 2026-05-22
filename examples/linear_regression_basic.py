import numpy as np
from sklearn.linear_model import LinearRegression

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 200
    n_features = 3

    X = rng.normal(size=(n_obs, n_features))
    beta = np.array([1.0, 0.5, 0.0])
    noise = rng.normal(scale=0.5, size=n_obs)

    y = X @ beta + noise

    model = LinearRegression()
    model.fit(X, y)

    explainer = edef.LinearExplainer(
        model,
        feature_names=["x1", "x2", "x3"],
    )

    result = explainer(X, y)

    grouped = result.group(["signal", "signal", "noise"])

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
            f"{name:>8s}  "
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
            f"{name:>8s}  "
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