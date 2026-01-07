"""
This module defines a friendly interface for creating models in the Pangolin IR.

Example
-------

Take the following model:

.. code-block:: text

    x    ~ normal(0,1)
    y[i] ~ exponential(d[i])
    z[i] ~ normal(x, y[i])

In Pangolin, you can declare this model like so:

.. code-block:: python

    >>> from pangolin import interface as pi
    >>>
    >>> x = pi.normal(0,1)
    >>> y = pi.vmap(pi.exponential)(pi.constant([2,3,4]))
    >>> z = pi.vmap(pi.normal, [None, 0])(x, y)

This produces the following internal representation.


.. code-block:: python

    >>> pi.print_upstream(z)
    shape | statement
    ----- | ---------
    ()    | a = 0
    ()    | b = 1
    ()    | c ~ normal(a,b)
    (3,)  | d = [2 3 4]
    (3,)  | e ~ vmap(exponential, [0], 3)(d)
    (3,)  | f ~ vmap(normal, [None, 0], 3)(c,e)

Reference card
--------------

The entire interface is based on methods that create :class:`InfixRV` objects (or that create functions that create :class:`InfixRV` objects).

Here are all the functions:

============================ ======
To do...                     Use...
============================ ======
Constants                    :func:`constant`
Arithmetic                   :func:`add` :func:`sub` :func:`mul` :func:`div` (or infix ``+``, ``-``, ``*``, ``/``)
Trigonometry                 :func:`arccos` :func:`arccosh` :func:`arcsin` :func:`arcsinh` :func:`arctan` :func:`arctanh` :func:`cos` :func:`cosh` :func:`sin` :func:`sinh` :func:`tan` :func:`tanh`
Other scalar functions       :func:`pow` (or infix ``**``) :func:`sqrt` :func:`abs` :func:`exp` :func:`inv_logit` :func:`expit` :func:`sigmoid` :func:`log` :func:`loggamma` :func:`logit` :func:`step`
Linear algebra               :func:`matmul` (or infix ``@``) :func:`inv`
Other multivariate functions :func:`sum` :func:`softmax`
Scalar distributions         :func:`normal` :func:`normal_prec` :func:`lognormal` :func:`cauchy` :func:`bernoulli` :func:`bernoulli_logit` :func:`beta` :func:`binomial` :func:`categorical` :func:`uniform` :func:`beta_binomial` :func:`exponential` :func:`gamma` :func:`poisson` :func:`student_t`
Multivariate distributions   :func:`multi_normal` :func:`multinomial` :func:`dirichlet` :func:`wishart`
Control flow                 :func:`vmap` :func:`composite` :func:`autoregressive` :func:`autoregress`
Indexing                     :func:`index` (or :func:`InfixRV.__getitem__` / ``[]`` operator)
============================ ======

Auto-casting
------------

Most functions take `RVLike` arguments, meaning either an `RV` or something that can
be cast to a NumPy array. In the latter case, it is implicitly cast to a constant RV.
For example, `categorical` takes an `RVLike` argument, so instead of tediously writin
this:

>>> probs_list = [0.1, 0.2, 0.7]
>>> probs_array = np.array(probs_list)
>>> probs_rv = constant(probs_array)
>>> x = categorical(probs_rv)

You can simply write:

>>> x = categorical([0.1, 0.2, 0.7])

Broadcasting
------------

Do you love broadcasting? Do you hate broadcasting? Are you lukewarm about broadcasting?
Well, good news. In this interface, you can configure how broadcasting works. You can
use "simple" broadcasting, you can use full NumPy-style broadcasting, or
you can turn broadcasting off completely. See `Broadcasting` for details.

API docs
------------------------------

"""

from __future__ import annotations
from .base import *
from .base import RVLike

# from . import vmap
# from . import vmap
from .vmapping import vmap
from .compositing import composite
from .autoregressing import Autoregressable, Autoregressed, autoregressive, autoregress
from .indexing import index
from pangolin.ir import print_upstream

RVLike = RVLike  # no-op assignment so it can be documented
"""A type class indicating either:

1. An `RV`
2. A NumPy array
3. Something that can be transformed into a NumPy array, such as a float or a list of lists of floats.

Many functions in this interface take `RVLike` arguments.
If the argument to the function is not an `RV`, then an `InfixRV` with a `Constant` op
is automatically created.
So, for example, ``cos(2.5)`` is equivalent to

``cos(InfixRV(pangolin.ir.Constant(2.5)))``.

Note that JAX arrays will typecheck as valid instances of `RVLike` (which is probably
good) but so will strings (which is probably bad).
"""


__all__ = [
    "InfixRV",
    "constant",
    "vmap",
    "composite",
    "Autoregressable",
    "Autoregressed",
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
    "wishart",
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
    "RVLike",
    "Broadcasting",
    "Config",
    "config",
    "override",
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
