import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import edef


def main():
    rng = np.random.default_rng(123)

    n_obs = 400

    X = rng.normal(size=(n_obs, 3))

    y = (
        1.0 * X[:, 0]
        + 0.75 * X[:, 1] ** 2
        + rng.normal(scale=0.25, size=n_obs)
    )

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(12,),
            activation="tanh",
            alpha=1e-4,
            max_iter=2000,
            random_state=0,
        ),
    )

    model.fit(X, y)

    explainer = edef.NumericalExplainer(
        model,
        baseline=X.mean(axis=0),
        loss="squared_error",
        n_steps=32,
        step_size=1e-4,
        feature_names=["x1_linear", "x2_nonlinear", "x3_noise"],
    )

    result = explainer(X, y)

    print(result)

    grouped = result.group(["signal", "signal", "noise"])

    print()
    print("Grouped contributions")
    print("---------------------")
    print(grouped)


if __name__ == "__main__":
    main()