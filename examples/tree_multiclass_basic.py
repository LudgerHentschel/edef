import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 600
    n_features = 4
    n_classes = 3

    X = rng.normal(size=(n_obs, n_features))

    scores = np.column_stack(
        [
            1.2 * X[:, 0] - 0.3 * X[:, 1],
            1.0 * X[:, 1] + 0.4 * X[:, 2],
            -0.8 * X[:, 0] + 0.6 * X[:, 3],
        ]
    )

    y = np.argmax(scores + rng.normal(scale=0.75, size=(n_obs, n_classes)), axis=1)

    model = GradientBoostingClassifier(
        n_estimators=25,
        max_depth=3,
        learning_rate=0.07,
        random_state=0,
    )
    model.fit(X, y)

    explainer = edef.TreeExplainer(
        model,
        baseline=X.mean(axis=0),
        loss="multiclass_log_loss",
        feature_names=["x1", "x2", "x3", "x4"],
    )

    result = explainer(X, y)

    grouped = result.group(["main", "main", "secondary", "secondary"])

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
    print(f"baseline_log_loss: {result.baseline_loss:.6f}")
    print(f"model_log_loss   : {result.model_loss:.6f}")
    print(f"explained_fit    : {result.total:.6f}")
    print(f"sum_contributions: {result.values.sum():.6f}")
    print(f"additivity_error : {result.additivity_error:.3e}")


if __name__ == "__main__":
    main()