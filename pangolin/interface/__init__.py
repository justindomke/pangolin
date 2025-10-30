"""
This module defines a friendly interface for creating models in the Pangolin IR.

## Example

Take the following model:

```text
x    ~ normal(0,1)
y[i] ~ exponential(d[i])
z[i] ~ normal(x, y[i])
```

In Pangolin, you can declare this model like so:

```python
>>> from pangolin import interface as pi
>>>
>>> x = pi.normal(0,1)
>>> y = pi.vmap(pi.exponential)(pi.constant([2,3,4]))
>>> z = pi.vmap(pi.normal, [None, 0])(x, y)

```

This produces the following internal representation.

```python
>>> pi.print_upstream(z)
shape | statement
----- | ---------
()    | a = 0
()    | b = 1
()    | c ~ normal(a,b)
(3,)  | d = [2 3 4]
(3,)  | e ~ vmap(exponential, (0,), 3)(d)
(3,)  | f ~ vmap(normal, (None, 0), 3)(c,e)

```

## Reference card

The entire interface is based on methods that create `InfixRV` objects (or that create functions that create `InfixRV` objects). Here are all the functions:

| To do... | Use... |
| ---- | --------- |
| Constants | `constant` |
| Arithmetic | `add` `sub` `mul` `div` (or infix `+`, `-`, `*`, `/`) |
| Trigonometry | `arccos` `arccosh` `arcsin` `arcsinh` `arctan` `arctanh` `cos` `cosh` `sin` `sinh` `tan` `tanh` |
| Other scalar functions | `pow` (or infix `**`) `sqrt` `abs` `exp` `inv_logit` `expit` `sigmoid` `log` `loggamma` `logit` `step` `softmax` |
| Linear algebra | `matmul` (or infix `@`) `inv` |
| Other multivariate functions | `sum` |
| Scalar distributions | `normal` `normal_prec` `lognormal` `cauchy` `bernoulli` `bernoulli_logit` `beta` `binomial` `categorical` `uniform` `beta_binomial` `exponential` `gamma` `poisson` `student_t`|
| Multivariate distributions | `multi_normal` `multinomial` `dirichlet` |
| Control flow | `vmap` `composite` `autoregressive` `autoregress` |
| Indexing | `index` (or `InfixRV.__getitem__` / `[]` operator) |

"""

from .base import *

# from . import vmap
# from . import vmap
from .vmapping import vmap
from .compositing import composite
from .autoregressing import autoregressive, autoregress
from .indexing import index
from pangolin.ir import print_upstream

__all__ = [
    "InfixRV",
    "constant",
    "vmap",
    "composite",
    "autoregressive",
    "autoregress",
    "index",
    "normal",
    "normal_prec",
    "lognormal",
    "cauchy",
    "bernoulli",
    "bernoulli_logit",
    "binomial",
    "uniform",
    "beta",
    "beta_binomial",
    "exponential",
    "gamma",
    "poisson",
    "student_t",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "abs",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cos",
    "cosh",
    "exp",
    "inv_logit",
    "expit",
    "sigmoid",
    "log",
    "loggamma",
    "logit",
    "sin",
    "sinh",
    "step",
    "tan",
    "tanh",
    "multi_normal",
    "categorical",
    "multinomial",
    "dirichlet",
    "matmul",
    "inv",
    "softmax",
    "sum",
    "base",
    "vmapping",
    "compositing",
    "autoregressing",
    "indexing",
    "print_upstream",
    "ScalarBroadcasting",
]


# from . import base
# from .base import *
# from .interface import normal
# from .vmap import vmap
# from .composite import composite
# from .autoregressive import autoregressive
# from .loops import Loop, slot


# __all__ = ['base','vmap','composite','autoregressive','Loop','slot','OperatorRV']
# __all__ += for_api # pyright: ignore [reportUnsupportedDunderAll]
