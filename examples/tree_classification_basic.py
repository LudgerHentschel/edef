import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 500
    n_features = 4

    X = rng.normal(size=(n_obs, n_features))

    score = (
        1.2 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.6 * X[:, 2] * X[:, 3]
    )

    y = (score > np.median(score)).astype(float)

    model = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.07,
        random_state=0,
    )
    model.fit(X, y)

    explainer = edef.TreeExplainer(
        model,
        baseline=X.mean(axis=0),
        loss="log_loss",
        target=1,
        feature_names=["x1_linear", "x2_linear", "x3_interact", "x4_interact"],
    )

    result = explainer(X, y)

    grouped = result.group(
        ["main", "main", "interaction", "interaction"]
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
            f"{name:>14s}  "
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
            f"{name:>14s}  "
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