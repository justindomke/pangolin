pangolin.interface.vfor
=======================

.. currentmodule:: pangolin.interface

.. py:function:: vfor(fun=None, /, **sizes)

   Vectorize a function over named integer ranges.

   ``fun`` must have a fixed positional signature. Each parameter receives a scalar `RV`
   representing an integer coordinate from zero through one less than its named size. ``fun``
   must return a PyTree of newly created `RV` objects; it cannot return a coordinate or an
   existing RV captured through a closure. With no coordinate parameters, ``fun`` is called
   directly and its RV outputs need not be newly created.

   Axis sizes are matched to function parameters by name. Output axes occur in
   function-parameter order, regardless of the order in which keyword arguments are supplied.

   ``vfor`` supports direct calls, decorator factories, and multiline
   decorator syntax.

   :param fun: Function to vectorize, or ``None`` when creating a decorator. It must have a fixed
               positional signature, accept one scalar `RV` coordinate per parameter, and return a
               PyTree of newly created RV objects. No parameter may be named ``interface``.
   :type fun: Callable[..., PyTree[RV]] | None
   :param \*\*sizes: Positive integer size for every parameter of ``fun``, supplied under the corresponding
                     parameter name.
   :type \*\*sizes: int

   :returns: **output_or_decorator** -- The vectorized output PyTree when ``fun`` is supplied, otherwise a decorator that accepts
             ``fun`` and produces that output.
   :rtype: PyTree[RV] | Callable[[Callable[..., PyTree[RV]]], PyTree[RV]]

   :raises TypeError: If ``fun`` is not callable, has no fixed positional signature, a size is not an integer, or does not return a PyTree
       of `RV` objects.
   :raises ValueError: If an axis size is missing, an unknown axis name is supplied, a size is not positive, a parameter is named ``interface``, or
       an output RV was not created by ``fun``.

   .. rubric:: Notes

   Static type checking preserves the number of coordinate arguments and exact PyTree structure
   of the output, and requires the first six arguments to be `RV` objects. Runtime validation
   still applies to every argument.

   .. rubric:: Examples

   Direct call:

   >>> from pangolin import ir
   >>> from pangolin.interface import vfor
   >>> x = vfor(lambda i, j: 10 * i + j, i=2, j=3)
   >>> x.shape
   (2, 3)

   Keyword order does not affect axis order:

   >>> y = vfor(lambda i, j: 10 * i + j, j=3, i=2)
   >>> y.shape
   (2, 3)
   >>> ir.rv_equal(x, y)
   True

   Decorator factory:

   >>> make_x = vfor(i=2, j=3)
   >>> x = make_x(lambda i, j: 10 * i + j)
   >>> x.shape
   (2, 3)

   Multiline decorator form:

   >>> @vfor(i=2, j=3)
   ... def x(i, j):
   ...     return 10 * i + j
   >>> x.shape
   (2, 3)

   Every function parameter must have a named size:

   >>> vfor(lambda i, j: i + j, i=2)
   Traceback (most recent call last):
   ...
   ValueError: vfor missing size for axis 'j'

   Unknown axis names are rejected:

   >>> vfor(lambda i, j: i + j, i=2, j=3, k=4)
   Traceback (most recent call last):
   ...
   ValueError: vfor got size for unknown axis 'k'; expected ('i', 'j')

   Python defaults are not interpreted as axis sizes:

   >>> vfor(lambda i=3: i + 1)
   Traceback (most recent call last):
   ...
   TypeError: <lambda> has no fixed positional signature: 'i' has a default
