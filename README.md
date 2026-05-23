# EDEF

EDEF (Euler Decomposition of Explained Fit) decomposes realized predictive
performance into additive feature contributions.

For each observation, EDEF returns feature attributions $\phi_j$ satisfying

$$\sum_j \phi_j = \mathcal{L}(y,\, \hat{y}(x_0)) - \mathcal{L}(y,\, \hat{y}(x)),$$

where $\mathcal{L}$ is the prediction loss, $x_0$ is a baseline input, and $x$
is the observation. A positive contribution means the feature improved realized
predictive fit relative to the baseline.

Standard attribution methods explain predictions. EDEF explains whether those
predictions were accurate.

## Using EDEF

EDEF follows a familiar explainer pattern:

```python
explainer = edef.LinearExplainer(model, feature_names=["x1", "x2", "x3"])
result = explainer(X, y)
print(result)
```

```text
Feature contributions
---------------------
      x1  edef= 0.979  se= 0.134  t= 7.33  share= 0.741
      x2  edef= 0.343  se= 0.064  t= 5.33  share= 0.260
      x3  edef=-0.001  se= 0.002  t=-0.40  share=-0.000
```

Unlike most attribution methods, EDEF reports standard errors and t-statistics
alongside attribution values. Features that move predictions without improving
accuracy show up near zero.

## Why EDEF?

Prediction-attribution methods answer "why did the model predict this value?"
EDEF answers "which features made the model accurate here?"

These questions have different answers. A feature can strongly influence a
prediction while contributing nothing to predictive accuracy — or can
actively hurt it. This happens when a feature moves predictions in the wrong
direction, when a feature is overfit, or when a feature captures real signal
on average but adds noise on a particular evaluation sample.

Consider a model trained to predict financial returns. Feature A captures a
persistent signal; feature B was correlated with returns in the training set
but is uncorrelated in the evaluation period. Both features generate large
prediction movements. SHAP or Integrated Gradients assigns large importances to
both. EDEF assigns large importance to A and near-zero importance to B —
because B's prediction movements did not improve realized fit.

The distinction matters most where prediction accuracy is the object of
interest: in model monitoring, out-of-sample validation, feature selection,
overfit detection, and scientific settings where fit to held-out outcomes
is the standard of evidence.

## How EDEF works

EDEF applies the path-integral perspective of Integrated Gradients — but to
the loss function rather than the prediction.

Along the straight-line path

$$x(t) = x_0 + t \cdot (x - x_0), \qquad 0 \le t \le 1,$$

the loss reduction from baseline to observation is

$$\mathcal{L}(y, \hat{y}(x_0)) - \mathcal{L}(y, \hat{y}(x))
= -\int_0^1 \frac{d}{dt}\,\mathcal{L}(y, \hat{y}(x(t)))\,dt.$$

By the chain rule this integral decomposes additively across features:

$$\phi_j = (x_j - x_{0,j}) \int_0^1
\left[-\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot
\frac{\partial \hat{y}}{\partial x_j}\bigg|_{x(t)}\right] dt.$$

The integrand is the prediction gradient $\partial \hat{y}/\partial x_j$
multiplied by the loss gradient $\partial \mathcal{L}/\partial \hat{y}$.
This chain-rule factor is what distinguishes EDEF from Integrated Gradients,
which integrates only the prediction gradient. For squared-error loss,
$\partial \mathcal{L}/\partial \hat{y} = -2(y - \hat{y}(x(t)))$, so EDEF
weights the prediction gradient by how wrong the prediction is at each point
along the path. Features that move predictions toward the truth accumulate
positive contributions; features that move predictions away accumulate
negative contributions.

This integral is computed differently for each model class:

- **Linear models.** The integral has a closed form. For regression,
  $\phi_j = \beta_j(x_j - x_{0,j}) \cdot (2\bar{y} - \bar{y}_{\text{pred}})$
  in centered coordinates. No quadrature is needed.

- **Tree models.** Via TreeIG, the path trace gives the exact sequence of
  split crossings and prediction jumps along $x(t)$. At each crossing, the
  loss changes by a computable amount. EDEF assigns that loss change to the
  crossing feature. The result is exact — no quadrature, no approximation.

- **PyTorch models.** Automatic differentiation computes
  $\partial \mathcal{L}/\partial x_j$ at each interpolation point;
  Gauss-Legendre quadrature integrates over $t$.

- **Black-box sklearn models.** Finite-difference approximations to the loss
  gradient replace automatic differentiation; Gauss-Legendre quadrature
  integrates over $t$.

## Statistical inference

Because EDEF computes a contribution $\phi_j(x_i)$ for each feature and each
observation, feature importances are sample averages:

$$\bar\phi_j = \frac{1}{n} \sum_{i=1}^n \phi_j(x_i).$$

Sample averages have standard errors. EDEF reports them:

$$\widehat{\text{se}}(\bar\phi_j)
= \frac{1}{\sqrt{n}}\,\text{sd}\bigl(\phi_j(x_1), \ldots, \phi_j(x_n)\bigr).$$

Standard errors unlock inference that most attribution methods cannot provide:
t-statistics to test whether a feature's contribution is distinguishable from
zero, standard errors on grouped contributions, and uncertainty quantification
across resampled evaluation sets. In settings where prediction accuracy is
itself the quantity of scientific interest — rather than a prediction to be
explained — these inferential outputs are as important as the point estimates.

## Relation to SHAP, Integrated Gradients, and SAGE

SHAP and Integrated Gradients explain predictions. EDEF and SAGE explain
realized model fit. These are fundamentally different attribution targets.

**SHAP and Integrated Gradients** ask:
> "How much does feature $j$ contribute to the prediction?"

**EDEF and SAGE** ask:
> "How much does feature $j$ contribute to realized predictive accuracy?"

### Integrated Gradients

Integrated Gradients computes $\phi_j = (x_j - x_{0,j}) \int_0^1 \partial \hat{y}/\partial x_j\big|_{x(t)}\,dt$ — the integral of the prediction gradient along the path from $x_0$ to $x$. EDEF computes the integral of the loss gradient along the same path. They share a path, a baseline, and an integration method. They differ in exactly one thing: what is integrated.

That difference in the integrand is the full story. IG measures how much each
feature moved the prediction as we interpolate from baseline to observation.
EDEF measures how much each feature improved or worsened predictive accuracy
as we make that same interpolation. For a perfect prediction, IG and EDEF
give the same sign for every feature. For a poor prediction, features that
moved the prediction in the wrong direction get negative EDEF attribution
even if they get large positive IG attribution.

For a linear model with zero intercept and zero baseline, EDEF and IG agree in
sign but differ in magnitude, with EDEF attributions scaled by the accuracy of
the prediction. As predictions become less accurate, the two methods diverge.

### SHAP

SHAP builds attributions from discrete feature inclusion effects averaged over
coalitions of other features. It does not follow a path and does not observe
realized outcomes. A feature can receive large SHAP importance purely because
it moves predictions strongly, regardless of whether those prediction movements
correspond to actual patterns in the outcome variable.

SHAP's coalition construction is deliberately indifferent to whether predictions
are accurate. The same coalition structure, the same expected-prediction
baseline, and the same discrete inclusion/exclusion logic apply whether the
model generalizes well or poorly. This makes SHAP a precise tool for
explaining the model's behavior in input space, and an imprecise tool for
evaluating that behavior against realized outcomes.

### SAGE

SAGE is the closest existing method to EDEF in motivation. Both measure feature
contributions to realized predictive performance rather than to predictions.
They differ substantially in construction.

SAGE applies Shapley-style coalition averaging to predictive performance: it
measures how much each feature changes expected loss as it enters or leaves a
coalition, where absent features are marginalized over a background
distribution. The SAGE attribution for feature $j$ asks "how much worse would
the model perform if it could not use feature $j$?" — a global counterfactual
question about feature removal.

EDEF asks "how much did feature $j$ contribute to the loss reduction for this
observation, along the specific path from baseline to observation?" — a local
path-integral question about feature movement. The difference is analogous to
the difference between SHAP and IG: Shapley-style marginalizing out features
versus path-integral accumulation of gradient contributions.

Three practical consequences follow. First, EDEF requires only a baseline
vector; SAGE requires a background distribution from which to marginalize out
features. Second, EDEF computes observation-level contributions that aggregate
naturally to sample-average importances with standard errors; SAGE produces
global importance estimates without natural observation-level decompositions.
Third, EDEF exploits closed-form path integrals and exact tree path traces for
efficient computation; SAGE currently lacks analogous backend optimizations
and can be expensive for large models.

## Available explainers

| Explainer | Intended models | Method |
|---|---|---|
| `LinearExplainer` | Linear and generalized linear models | Closed-form exact decomposition |
| `TorchExplainer` | PyTorch neural networks | Autograd + Gauss-Legendre quadrature |
| `TreeExplainer` | Tree ensembles | Exact TreeIG path traces |
| `NumericalExplainer` | Any sklearn-style model | Finite-difference + Gauss-Legendre quadrature |

## Supported models

### Linear models

- Linear regression
- Binary logistic regression
- Multiclass logistic regression

### PyTorch models

- Regression (squared-error loss)
- Binary classification (log loss)
- Multiclass classification (softmax log loss)

### Tree models (via TreeIG)

- `sklearn.tree.DecisionTreeRegressor`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.ExtraTreesRegressor`
- `sklearn.ensemble.GradientBoostingRegressor`
- `sklearn.ensemble.GradientBoostingClassifier`
- `xgboost.XGBRegressor`, `xgboost.XGBClassifier`, `xgboost.Booster`
- `lightgbm.LGBMRegressor`, `lightgbm.LGBMClassifier`, `lightgbm.Booster`

Tree classification uses raw margins/logits rather than predicted
probabilities. Probabilities are not additive across trees.

### Numerical black-box models

Any model with `predict(X)` (regression) or `predict_proba(X)`
(classification), including sklearn pipelines and `MLPRegressor`/
`MLPClassifier`.

## Not currently supported

- probability-output tree attribution;
- missing-value tree routing;
- CatBoost.

## Installation

```bash
pip install edef
```

Optional dependencies:

```bash
pip install torch    # for TorchExplainer
pip install treeig   # for TreeExplainer
pip install shap     # for SHAP plotting compatibility
```

## Linear regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import edef

rng = np.random.default_rng(123)
X = rng.normal(size=(200, 3))
y = X @ np.array([1.0, 0.5, 0.0]) + rng.normal(scale=0.5, size=200)

model = LinearRegression().fit(X, y)
result = edef.LinearExplainer(model, feature_names=["x1", "x2", "x3"])(X, y)
print(result)
```

The decomposition is exact for any linear model: contributions sum to the
realized reduction in mean squared error relative to the sample mean.

## Binary classification

```python
from sklearn.linear_model import LogisticRegression
import edef

model = LogisticRegression().fit(X, y)
result = edef.LinearExplainer(model, loss="log_loss", feature_names=[...])(X, y)
```

## Tree regression

```python
from sklearn.ensemble import GradientBoostingRegressor
import edef

model = GradientBoostingRegressor(n_estimators=100, max_depth=3).fit(X, y)

explainer = edef.TreeExplainer(model, baseline=X.mean(axis=0), loss="squared_error")
result = explainer(X_eval, y_eval)
```

EDEF uses TreeIG to find the exact sequence of split-crossing events along
the interpolation path for each observation. Each crossing changes the
prediction, which changes the loss. EDEF assigns the loss change at each
crossing to the crossing feature. The result is exact — no quadrature,
no approximation parameters.

## Tree classification

```python
from sklearn.ensemble import GradientBoostingClassifier
import edef

model = GradientBoostingClassifier(...).fit(X, y)
explainer = edef.TreeExplainer(model, baseline=X.mean(axis=0), loss="log_loss")
result = explainer(X_eval, y_eval)
```

For multiclass models, use `loss="multiclass_log_loss"`. EDEF merges the
split-crossing sequences across all class-margin trees, applying exact softmax
log-loss changes at each event.

## PyTorch models

```python
import edef

explainer = edef.TorchExplainer(
    model,
    baseline=X_train.mean(axis=0),
    loss="squared_error",   # or "log_loss", "multiclass_log_loss"
    n_steps=50,
    feature_names=[...],
)
result = explainer(X_eval, y_eval)
```

## Black-box sklearn models

```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import edef

model = make_pipeline(StandardScaler(), MLPRegressor(...)).fit(X, y)

explainer = edef.NumericalExplainer(
    model,
    baseline=X.mean(axis=0),
    loss="squared_error",
    n_steps=32,
    feature_names=[...],
)
result = explainer(X_eval, y_eval)
```

## Grouped contributions

```python
grouped = result.group(["signal", "signal", "noise"])
```

Grouped contributions preserve exact additivity. Group labels map input
features to named groups; features sharing a label are summed. This is
useful for one-hot encoded variables, embedding blocks, factor groups, and
hierarchical feature structures.

## Statistical inference

```python
result.values            # feature contributions (point estimates)
result.standard_errors   # standard errors
result.t_values          # t-statistics: values / standard_errors
result.proportions       # share of total explained fit
result.to_frame()        # pandas DataFrame, sorted by contribution
result.plot()            # horizontal bar chart with confidence intervals
```

Standard errors are computed from the observation-level contributions
and scale correctly under grouping.

## SHAP plotting

```python
shap_exp = result.to_shap_explanation(data=X)

import shap
shap.plots.beeswarm(shap_exp)
```

The underlying values are EDEF realized-fit contributions. The SHAP plotting
interface is used for visualization only.

## Warmup

`TorchExplainer` and `TreeExplainer` use JIT-compiled kernels. Trigger
compilation before your main evaluation:

```python
explainer = edef.TreeExplainer(model, baseline=x0, loss="squared_error")
explainer.warmup(X[:3], y[:3])
result = explainer(X, y)
```

## Project status

EDEF covers the dominant regression and classification models in the Python
ecosystem with exact or high-accuracy decompositions:

- closed-form exact attribution for linear models;
- autograd path integration for PyTorch models;
- exact attribution for tree ensembles via TreeIG;
- numerical attribution for any sklearn-interface model;
- multiclass log-loss decomposition throughout;
- observation-level contributions with standard errors and t-statistics;
- grouping, SHAP-compatible plotting, and pandas output.

## References

EDEF:

- Hentschel, Ludger. 2026.
  "Feature importance for model fit: Nonlinear regression and
  classification in machine learning models."

- Hentschel, Ludger. 2026.
  "Feature importance for predictive accuracy: An Euler decomposition."

TreeIG:

- Hentschel, Ludger. 2026.
  "TreeIG: Exact Integrated Gradients for Tree-Based Models."

Integrated Gradients:

- Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 2017.
  "Axiomatic Attribution for Deep Networks."
  *International Conference on Machine Learning (ICML).*

SHAP and TreeSHAP:

- Lundberg, Scott M., and Su-In Lee. 2017.
  "A Unified Approach to Interpreting Model Predictions."
  *Advances in Neural Information Processing Systems (NeurIPS).*

- Lundberg, Scott M., Gabriel Erion, and Su-In Lee. 2020.
  "From Local Explanations to Global Understanding with Explainable AI
  for Trees."
  *Nature Machine Intelligence.*

SAGE:

- Covert, Ian, Scott Lundberg, and Su-In Lee. 2020.
  "Understanding Global Feature Contributions With Additive Importance
  Measures."
  *NeurIPS.*

## License

BSD 3-Clause License.