import numpy as np
from sklearn.ensemble import RandomForestRegressor

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 500
    n_features = 4

    X = rng.normal(size=(n_obs, n_features))

    y = (
        1.0 * X[:, 0]
        - 0.8 * X[:, 1] ** 2
        + 0.5 * X[:, 2]
        + rng.normal(scale=0.1, size=n_obs)
    )

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=4,
        random_state=0,
        n_jobs=1,
    )
    model.fit(X, y)

    baseline = X.mean(axis=0)

    explainer = edef.TreeExplainer(
        model,
        baseline=baseline,
        loss="squared_error",
        feature_names=["x1_linear", "x2_squared", "x3_linear", "x4_noise"],
    )

    result = explainer(X, y)

    grouped = result.group(
        ["signal", "signal", "signal", "noise"]
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
            f"{name:>12s}  "
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