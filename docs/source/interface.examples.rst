Examples
========

.. currentmodule:: pangolin.interface

.. testsetup::

   from pangolin import interface as pi

**An array-shaped model**

Broadcasting applies scalar distributions element by element. For example, consider three
Gamma rates and three corresponding Poisson observations:

.. code-block:: text

    a = (2, 4, 6)
    z[i] ~ gamma(a[i], 1)
    x[i] ~ poisson(z[i])

This can be written using broadcasting:

>>> a = pi.constant([2, 4, 6])
>>> z = pi.gamma(a, 1)
>>> x = pi.poisson(z)
>>> pi.print_upstream(x, {"a": a, "z": z, "x": x})
shape | statement
----- | ---------
(3,)  | a = [2 4 6]
()    | b = 1
(3,)  | z ~ vmap(gamma, [0, None], 3)(a, b)
(3,)  | x ~ vmap(poisson, [0], 3)(z)

Broadcasting is provided by the interface. Scalar Ops in the IR itself do not broadcast, so the
interface lowers these operations to explicit `VMap` Ops.

**A time-series model**

The `scan` transformation expresses a time series whose current state depends on its previous
state. Here the initial state is itself an RV, and the resulting latent time series is observed
with Gaussian noise:

.. code-block:: text

    z ~ uniform(-1, 1)
    x[0] ~ normal(0.8 z, 0.5)
    x[t] ~ normal(0.8 x[t-1], 0.5)
    y[t] ~ normal(x[t], 2)

Calling `scan` first creates a new interface function, analogous to `normal`:

>>> from pangolin import interface as pi
>>> time_series = pi.scan(lambda x: pi.normal(0.8 * x, 0.5), length=4)

Conceptually, this has the interface ``time_series(initial) -> time-series RV``. The rest of the
model can then be constructed by calling that function on the initial-condition RV:

>>> z = pi.uniform(-1, 1)
>>> x = time_series(z)
>>> y = pi.normal(x, 2)
>>> pi.print_upstream(y, {"z": z, "x": x, "y": y})
shape | statement
----- | ---------
()    | a = -1
()    | b = 1
()    | z ~ uniform(a, b)
(4,)  | x ~ scan(composite(1, [0.8, mul, 0.5, normal], [[], [1, 0], [], [2, 3]]), 4, [])(z)
()    | c = 2
(4,)  | y ~ vmap(normal, [0, None], 4)(x, c)

**Vectorized models**

Scalar interface functions support broadcasting, so in simple situations vectorization can be
expressed as in NumPy. For more complex vectorization and multivariate functions, Pangolin rejects
reliance on NumPy's esoteric array-manipulation tricks—manually reshaping arrays, inserting
singleton axes, and permuting dimensions—and its grotesque panoply of operation-specific rules for
axes, batching, and indexing. Instead, Pangolin provides five generic, explicit functional
combinators for building new models from ordinary Python functions: `composite`,
`vmap`, `vfor`, `sfor`, and `scan`.

Now suppose three groups have different transition scales. The complete model is:

.. code-block:: text

    z ~ uniform(-1, 1)
    s = (0.3, 0.5, 0.7)
    x[i, 0] ~ normal(0.8 z, s[i])
    x[i, t] ~ normal(0.8 x[i, t-1], s[i])
    y[i, t] ~ normal(x[i, t], 2)

Here `scan` creates a two-argument interface function. Setting ``in_axes=None`` uses the same
scale at every time step:

>>> from pangolin import interface as pi
>>> time_series = pi.scan(
...     lambda x, scale: pi.normal(0.8 * x, scale),
...     length=4,
...     in_axes=None,
... )

Conceptually, this has the interface ``time_series(initial, scale) -> time-series RV``.

The remaining RVs can now be constructed. One option is ordinary vectorization with `vmap`. The
initial state is shared across groups, while the scale is mapped:

>>> z = pi.uniform(-1, 1)
>>> s = pi.constant([0.3, 0.5, 0.7])
>>> x = pi.vmap(time_series, in_axes=[None, 0])(z, s)
>>> y = pi.normal(x, 2)
>>> y.shape
(3, 4)

Alternatively, the same complete model can be written with `vfor`, which mirrors the indexed
mathematics:

>>> from pangolin import interface as pi
>>> time_series = pi.scan(
...     lambda x, scale: pi.normal(0.8 * x, scale),
...     length=4,
...     in_axes=None,
... )
>>> z = pi.uniform(-1, 1)
>>> s = pi.constant([0.3, 0.5, 0.7])
>>> x = pi.vfor(lambda i: time_series(z, s[i]), i=3)
>>> y = pi.normal(x, 2)
>>> y.shape
(3, 4)

Both forms are valid; `vfor` is an optional convenience. Ordinary broadcasting cannot express
the recurrence over both groups and time, so an explicit mapping transformation is needed.
