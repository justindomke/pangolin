pangolin.interface.overload
===========================

.. currentmodule:: pangolin.interface

.. py:function:: overload(*, broadcasting=None)

   Temporarily change the default interface's configuration. Unspecified options retain their
   current values. Overrides may be nested, and the previous configuration is restored on exit,
   including when an exception occurs.

   :param broadcasting: One of ``"off"``, ``"simple"``, ``"rank"``, or ``"numpy"``; ``None`` leaves
      the current mode unchanged.
   :raises TypeError: If an unknown option is supplied or the value is not a string or ``None``.
   :raises ValueError: If the mode is unsupported.

   >>> from pangolin import interface as pi
   >>> with pi.overload(broadcasting="rank"):
   ...     x = pi.normal([0, 1], [[1, 1], [2, 2]])
   ...     x.shape
   (2, 2)

