# EDEF

EDEF (Euler Decomposition of Explained Fit) decomposes realized predictive
performance into additive feature contributions.

For regression, EDEF decomposes reductions in realized squared-error loss.
For classification, EDEF decomposes improvements in realized log loss.

Formally, EDEF computes attributions $\phi_j$ satisfying

$$\sum_j \phi_j = \mathcal{L}(\widehat{y}(x_0)) - \mathcal{L}(\widehat{y}(x)),$$

where $\mathcal{L}(\widehat{y}(x_0)$ is the model loss at a baseline input $x_0$
and $\mathcal{L}(\widehat{y}(x))$ is the model loss at the observed input $x$.
A larger value corresponds to better realized predictive performance.

EDEF answers a different question from standard prediction-attribution methods.

Prediction attribution asks:

```text
"Why did the model predict this value?"
```

EDEF asks:

```text
"How much did this feature contribute to realized predictive performance?"
```

These questions and their answers serve different purposes. Explaining
predictions characterizes model outputs, regardless of whether those outputs
are accurate. Explaining realized fit characterizes which features actually
drove predictive accuracy, conditional on observed outcomes.

## Why EDEF?

Prediction-attribution methods explain how features influence model outputs.
They do not explain whether those prediction movements improved realized
predictive performance.

A feature can strongly affect predictions while contributing little, nothing,
or even negatively to realized fit.

For example:

- a feature may move predictions aggressively but mostly add noise;
- an overfit feature may appear highly important for predictions while
  harming out-of-sample performance;
- unstable nonlinear effects may generate large prediction attributions
  with weak realized predictive value.

EDEF measures realized fit contributions directly by decomposing changes in
loss observation-by-observation.

This distinction matters especially in:

- noisy prediction problems, such as financial forecasting;
- model monitoring and auditing;
- feature-selection validation;
- overfit or unstable models;
- scientific prediction settings where fit to held-out outcomes matters.

Because EDEF works at the observation level, it naturally supports standard
errors, t-statistics, and inference for feature importance — outputs most
prediction-attribution methods do not provide.

## Relation to SHAP, Integrated Gradients, and SAGE

SHAP and Integrated Gradients explain predictions. EDEF and SAGE explain
realized model fit. These are fundamentally different attribution targets.

**SHAP and Integrated Gradients** ask:

```text
"How much does feature j contribute to the prediction?"
```

**EDEF and SAGE** ask:

```text
"How much does feature j contribute to realized model fit?"
```

### SHAP

SHAP computes attribution values from discrete feature inclusion effects
averaged over coalitions. SHAP does not evaluate realized outcomes when
computing prediction attributions.

A feature can receive large SHAP importance if it moves predictions strongly,
even if those prediction movements do not improve realized predictive
performance.

### Integrated Gradients

Integrated Gradients accumulates prediction changes along a straight-line path
from a baseline input $x_0$ to the observed input $x$.

For smooth models, EDEF builds directly on this path-integral perspective, but
applies it to loss functions rather than predictions. For tree models, EDEF
uses exact TreeIG path traces rather than numerical quadrature.

### SAGE

SAGE is one of the few existing methods that explains model fit rather than
predictions. SAGE applies Shapley-value ideas to predictive performance by
measuring changes in loss as features are removed and marginalized out.

EDEF and SAGE share the same focus on realized predictive performance. They
differ fundamentally in construction:

- SAGE uses Shapley-style coalition averaging over feature subsets;
- EDEF uses Euler/path-integral decompositions of realized loss changes.

SAGE currently does not provide the large-scale backend optimizations that
TreeSHAP provides for SHAP. EDEF instead exploits additive path decompositions
directly, yielding exact and computationally efficient realized-fit
attributions for many important model classes.

## Supported models

### Linear models

- Linear regression
- Binary logistic regression
- Multiclass logistic regression

### PyTorch models

- Regression
- Binary classification

### Tree models (via TreeIG)

- Regression
- Binary additive-score classification
- Multiclass additive-score classification

Tree classification uses raw margins/logits rather than probabilities.

## Not currently supported

Current limitations include:

- PyTorch multiclass classification;
- probability-output tree attribution;
- missing-value tree routing;
- CatBoost;
- SHAP-compatible plotting with uncertainty visualization.

## Installation

Basic install:

```bash
pip install edef
```

Local editable install:

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install torch        # for TorchExplainer
pip install treeig       # for TreeExplainer
pip install shap         # for SHAP plotting compatibility
```

## Quickstart: Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression

import edef

rng = np.random.default_rng(123)

X = rng.normal(size=(200, 3))
beta = np.array([1.0, 0.5, 0.0])
y = X @ beta + rng.normal(scale=0.5, size=200)

model = LinearRegression()
model.fit(X, y)

explainer = edef.LinearExplainer(
    model,
    feature_names=["x1", "x2", "x3"],
)

result = explainer(X, y)
print(result)
```

Typical output:

```text
Feature contributions
---------------------
      x1  edef= 0.978775  se= 0.133580  t= 7.327  share= 0.741
      x2  edef= 0.343310  se= 0.064430  t= 5.328  share= 0.260
      x3  edef=-0.000607  se= 0.001504  t=-0.403  share=-0.000
```

Positive contributions indicate improved realized predictive fit. Feature
contributions add exactly to total explained fit:

```python
np.testing.assert_allclose(result.values.sum(), result.total)
```

## Quickstart: Binary Classification

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

import edef

rng = np.random.default_rng(123)

X = rng.normal(size=(500, 3))
beta = np.array([1.5, 0.75, 0.0])
eta = X @ beta - 0.25
p = 1.0 / (1.0 + np.exp(-eta))
y = rng.binomial(1, p, size=500)

model = LogisticRegression(C=1e6, solver="lbfgs")
model.fit(X, y)

explainer = edef.LinearExplainer(
    model,
    loss="log_loss",
    feature_names=["x1", "x2", "x3"],
)

result = explainer(X, y)
print(result)
```

For classification, EDEF decomposes realized reductions in log loss.

## Quickstart: Tree Regression

```python
from sklearn.ensemble import RandomForestRegressor

import edef

model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0)
model.fit(X, y)

explainer = edef.TreeExplainer(
    model,
    baseline=X.mean(axis=0),
    loss="squared_error",
)

result = explainer(X, y)
print(result)
```

For tree models, EDEF uses exact TreeIG split-crossing decompositions rather
than numerical gradient approximations.

## Quickstart: Tree Multiclass Classification

```python
from sklearn.ensemble import GradientBoostingClassifier

import edef

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
)

result = explainer(X, y)
print(result)
```

For multiclass tree models, EDEF computes exact softmax log-loss decompositions
using TreeIG path traces across all class margins.

## Grouped contributions

Feature contributions can be grouped after estimation.

```python
grouped = result.group(["signal", "signal", "noise"])
```

Grouped contributions preserve exact additivity:

```python
np.testing.assert_allclose(grouped.values.sum(), grouped.total)
```

This is especially useful for:

- one-hot encoded variables;
- grouped factors and embedding blocks;
- sector/style decompositions;
- hierarchical feature structures.

## Statistical inference

EDEF computes observation-level fit contributions and aggregates them across
the evaluation sample. This permits direct estimation of standard errors,
t-statistics, and grouped contribution inference — outputs most
prediction-attribution methods do not provide.

Results expose:

```python
result.values            # feature contributions
result.standard_errors   # standard errors
result.t_values          # t-statistics
result.proportions       # share of total explained fit
result.to_frame()        # pandas DataFrame
```

## SHAP plotting compatibility

EDEF can export contributions into a SHAP-compatible `Explanation` object
for visualization.

```python
shap_exp = result.to_shap_explanation(data=X)
```

This allows direct use of SHAP plotting routines:

```python
import shap
shap.plots.beeswarm(shap_exp)
```

The compatibility layer supports SHAP plotting only. The underlying values remain EDEF realized-fit contributions rather than SHAP prediction attributions.

## Example notebooks

The repository includes executable Jupyter notebooks illustrating the main
EDEF workflows and conceptual distinctions.

### Quickstart notebooks

- `01_linear_regression_quickstart.ipynb`
- `02_linear_classification_quickstart.ipynb`
- `03_tree_regression_quickstart.ipynb`
- `04_tree_classification_quickstart.ipynb`

These notebooks demonstrate:
- exact additivity of realized-fit contributions;
- standard errors and t-statistics;
- grouped contributions;
- regression and classification workflows;
- exact TreeIG-based decomposition for tree models.

### Conceptual comparison notebook

- `05_shap_vs_edef.ipynb`

This notebook compares SHAP prediction attribution with EDEF realized-fit
attribution on the same fitted models and evaluation samples.

The comparison highlights a central distinction:

- SHAP explains how features influence predictions;
- EDEF explains how features influence realized predictive performance.

In particular, the notebook demonstrates that a feature can receive large
prediction attribution while contributing little or negatively to realized
model fit.


## Core idea

For regression, EDEF decomposes the realized reduction in mean squared error
relative to an intercept-only baseline:

```text
baseline loss = E[(y - E[y])^2]
model loss    = E[(y - ŷ)^2]
```

EDEF computes additive feature contributions satisfying:

```text
sum_j contribution_j  =  baseline loss - model loss
```

The decomposition operates observation-by-observation and aggregates naturally
across evaluation samples. For linear models, contributions are computed in
closed form. For PyTorch models, EDEF integrates loss gradients along
the straight-line path from the baseline to each observation. For tree models,
EDEF uses exact TreeIG split-crossing path traces.

## Warmup for tree and PyTorch models

`TorchExplainer` and `TreeExplainer` use JIT-compiled kernels. The first call
includes compilation, which can take several seconds. Run warmup in advance
to compile before your main evaluation:

```python
explainer = edef.TreeExplainer(model, baseline=x0, loss="squared_error")
explainer.warmup(X[:3], y[:3])

result = explainer(X, y)
```

Subsequent calls on the same model are fast.

## Project status

EDEF is production-ready for exact attribution of model fit. The current
release covers the dominant regression and classification models in the Python
ecosystem:

- closed-form linear regression and binary classification;
- PyTorch gradient-based EDEF;
- exact tree-model EDEF via TreeIG;
- multiclass log-loss EDEF;
- grouped contributions;
- statistical inference and standard errors;
- SHAP-compatible plotting adapters.

Future extensions may include:

- CatBoost support, which requires customized analysis of oblivious trees
  and categorical split structure;
- alternative allocation rules for simultaneous multi-feature effects at
  coincident split crossings.

## References

EDEF:

- Hentschel, Ludger. 2026.
  "Feature importance for model fit: Nonlinear regression and classification
  in machine learning models."

- Hentschel, Ludger. 2026.
  "Feature importance for predictive accuracy: An Euler decomposition."

Integrated Gradients:

- Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 2017.
  "Axiomatic Attribution for Deep Networks."
  *International Conference on Machine Learning (ICML).*

SHAP and TreeSHAP:

- Lundberg, Scott M., and Su-In Lee. 2017.
  "A Unified Approach to Interpreting Model Predictions."
  *Advances in Neural Information Processing Systems (NeurIPS).*

- Lundberg, Scott M., Gabriel Erion, and Su-In Lee. 2020.
  "From Local Explanations to Global Understanding with Explainable AI for Trees."
  *Nature Machine Intelligence.*

SAGE:

- Covert, Ian, Scott Lundberg, and Su-In Lee. 2020.
  "Understanding Global Feature Contributions With Additive Importance Measures."
  *NeurIPS.*

TreeIG:

- Hentschel, Ludger. 2026.
  "TreeIG: Exact Integrated Gradients for Tree-Based Models."

## License

BSD 3-Clause License.