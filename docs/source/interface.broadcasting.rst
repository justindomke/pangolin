.. _interface-broadcasting:

Broadcasting
============

.. currentmodule:: pangolin.interface

Do you love broadcasting? Do you hate broadcasting? Are you lukewarm about broadcasting?
Well, good news. In this interface, you can configure how broadcasting works. You can
turn broadcasting off, use simple broadcasting, allow rank promotion, or use full
NumPy-style broadcasting. Use ``Interface(broadcasting=value)`` to create a configured
interface or :func:`overload` for a scoped change.

Broadcasting controls how scalar functions accept array-shaped arguments. By default, Pangolin
uses simple broadcasting. Create an ``Interface(broadcasting="rank")`` to select a
persistent mode for that interface, or use ``overload`` to change it temporarily within a block.

The four modes form a hierarchy: ``OFF ⊂ SIMPLE ⊂ RANK ⊂ NUMPY``. Each mode in this hierarchy
accepts a strictly larger set of input shape combinations.

Broadcasting applies to "all-scalar" functions where all inputs and outputs are scalar. Under
``OFF``, every input must actually be scalar, and the output is scalar. Under ``SIMPLE``, scalar
inputs may be mixed with non-scalar inputs, but all non-scalar inputs must have exactly the same
shape. Under ``RANK``, shapes are aligned on the right; missing leading dimensions are allowed,
but dimensions present in multiple inputs must match exactly. Under ``NUMPY``, shapes are aligned
on the right; missing leading dimensions are allowed, and aligned dimensions must match or one
must have size one.

Suppose ``f(a,b,c)`` is a scalar function. Then here is how broadcasting behaves:

+-----------------------------------------+--------+-----------+-----------+-------------+
|                                         | Broadcasting mode                            |
+                                         +--------+-----------+-----------+-------------+
|                                         | OFF    | SIMPLE    | RANK      | NUMPY       |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``a.shape`` | ``b.shape`` | ``c.shape`` | ``f(a,b,c).shape``                           |
+=============+=============+=============+========+===========+===========+=============+
| ``()``      | ``()``      | ``()``      | ``()`` | ``()``    | ``()``    | ``()``      |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,)``    | ``()``      | ``()``      | n/a    | ``(5,)``  | ``(5,)``  | ``(5,)``    |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,)``    | ``(5,)``    | ``()``      | n/a    | ``(5,)``  | ``(5,)``  | ``(5,)``    |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,)``    | ``(5,)``    | ``(5,)``    | n/a    | ``(5,)``  | ``(5,)``  | ``(5,)``    |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,9)``   | ``()``      | ``()``      | n/a    | ``(5,9)`` | ``(5,9)`` | ``(5,9)``   |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,9)``   | ``(5,9)``   | ``()``      | n/a    | ``(5,9)`` | ``(5,9)`` | ``(5,9)``   |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,9)``   | ``(5,9)``   | ``(5,9)``   | n/a    | ``(5,9)`` | ``(5,9)`` | ``(5,9)``   |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(9,)``    | ``(5,9)``   | ``()``      | n/a    | n/a       | ``(5,9)`` | ``(5,9)``   |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(1,9)``   | ``(5,9)``   | ``()``      | n/a    | n/a       | n/a       | ``(5,9)``   |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(1,5,1)`` | ``(2,1,9)`` | ``()``      | n/a    | n/a       | n/a       | ``(2,5,9)`` |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+
| ``(5,)``    | ``(6,)``    | ``()``      | n/a    | n/a       | n/a       | n/a         |
+-------------+-------------+-------------+--------+-----------+-----------+-------------+

This interface does not (currently) support automatic broadcasting for non-scalar
functions.


Examples
~~~~~~~~

With broadcasting disabled, scalar functions accept only scalar arguments.

>>> import numpy as np
>>> from pangolin.interface import Interface
>>> pi = Interface(broadcasting="off")
>>> pi.student_t(0,1,1).shape # doctest: +ELLIPSIS
()

As expected, ``OFF`` rejects any non-scalar argument.

>>> pi.student_t(0,np.ones(5),1).shape # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
ValueError: Can't broadcast shapes () and (5,) and () with broadcasting=OFF

``SIMPLE`` additionally accepts scalar arguments mixed with non-scalar arguments, provided
that all non-scalar arguments have exactly the same shape.

>>> pi = Interface(broadcasting="simple")
>>> pi.student_t(0,1,1).shape
()
>>> pi.student_t(0,np.ones(5),1).shape
(5,)
>>> pi.student_t(0,np.ones((5,3)),1).shape
(5, 3)
>>> pi.student_t(0,np.ones((5,3)),np.ones((5,3))).shape
(5, 3)

As expected, ``SIMPLE`` rejects non-scalar arguments with different shapes.

>>> pi.student_t(0,np.ones(3),np.ones((5,3))).shape
Traceback (most recent call last):
    ...
ValueError: Can't broadcast shapes () and (3,) and (5, 3) with broadcasting=SIMPLE

``RANK`` additionally accepts shapes that differ only by missing leading dimensions.

>>> pi = Interface(broadcasting="rank")
>>> pi.student_t(0,1,1).shape
()
>>> pi.student_t(0,np.ones(5),1).shape
(5,)
>>> pi.student_t(0,np.ones((5,3)),1).shape
(5, 3)
>>> pi.student_t(0,np.ones((5,3)),np.ones((5,3))).shape
(5, 3)
>>> pi.student_t(0,np.ones(3),np.ones((5,3))).shape
(5, 3)

As expected, ``RANK`` does not broadcast dimensions of size one.

>>> pi.student_t(0,np.ones((1,3)),np.ones((5,3))).shape
Traceback (most recent call last):
    ...
ValueError: Can't broadcast shapes () and (1, 3) and (5, 3) with broadcasting=RANK

``NUMPY`` additionally broadcasts dimensions of size one.

>>> pi = Interface(broadcasting="numpy")
>>> pi.student_t(0,np.ones((1,3)),np.ones((5,3))).shape
(5, 3)
>>> pi.student_t(0,np.ones((1,5,1)),np.ones((2,1,9))).shape
(2, 5, 9)

As expected, ``NUMPY`` rejects incompatible dimensions that are both larger than one.

>>> pi.student_t(0,np.ones((2,3)),np.ones((4,3))).shape
Traceback (most recent call last):
    ...
ValueError: Can't broadcast shapes () and (2, 3) and (4, 3) with broadcasting=NUMPY

Finally, restore the default mode.

>>> pi = Interface(broadcasting="simple")

You can also change this behavior temporarily in a code block by using the
:func:`overload` context manager.

As expected, temporarily selecting ``OFF`` rejects these non-scalar arguments, but selecting
``RANK`` allows them.

>>> with pi.overload(broadcasting="off"):
...     x = pi.normal(np.zeros(3), np.ones((5,3)))
Traceback (most recent call last):
...
ValueError: Can't broadcast shapes (3,) and (5, 3) with broadcasting=OFF
>>> with pi.overload(broadcasting="rank"):
...     x = pi.normal(np.zeros(3), np.ones((5,3)))
...     x.shape
(5, 3)
