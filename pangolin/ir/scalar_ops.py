"""
This module defines some "all scalar" ops where all inputs and outputs are scalars. In each case
there is an `Op` *class* created, and then an `Op` *instance* for convenience. For example,
`Normal` represents the class of all Gaussian conditional distributions. These do not have any
parameters, because there aren't different "types" of normal distributions. For convenience,
we also provide `normal=Normal()`, which is one particular Gaussian conditional distribution.

The user is expected to interact with these (lower case) instances, e.g. by calling
```python
x = pangolin.ir.scalar_ops.normal(0,1)
```

If for some reason you didn't want to use the built-in convenience class, this is equivalent to
writing

```python
x = pangolin.ir.scalar_ops.Normal()(0,1)
```

Or, if you wanted to be even more pedantic, you could write

```python
x = pangolin.ir.rv.RV(pangolin.ir.scalar_ops.normal,0,1)
```

or even

```python
x = pangolin.ir.rv.RV(pangolin.ir.scalar_ops.Normal(),0,1)
```

But there's no reason to do that.

"""


import numpy as np
from pangolin import util, ir
from .op import Op

# def all_scalar_op_factory(num_parents, name, random):
#     """
#     "Factory" to create an "all scalar" op where all parents and inputs are scalars.

#     Parameters
#     ----------
#     num_parents: int
#         The number of parameters for the op
#     name: str
#         The name for the op (used for printing)
#     random: bool
#         Is the op a conditional distribution (True) or a deterministic function (False)

#     Returns
#     -------
#     OpClass
#         A new subtype of Op
#     """

#     camel_case_name = name[0].upper() + name[1:]

#     class AllScalarOp(Op):
#         """
#         Convenience class to create "all scalar" distributions, where all parents and
#         outputs are scalar. Most of the common ops (e.g. `normal`, `add`, `exp`) are
#         instances of this Op.
#         """

#         def __init__(self):
#             "@public"
#             super().__init__(name=name, random=random)

#         def _get_shape(self, *parents_shapes):
#             if len(parents_shapes) != num_parents:
#                 raise ValueError(f"{name} op got {len(parents_shapes)} arguments but expected"
#                                  f" {num_parents}.")
#             for shape in parents_shapes:
#                 assert shape == (), "all parents must have shape ()"
#             return ()

#         def __eq__(self,other):
#             return isinstance(other,AllScalarOp)

#         def __hash__(self):
#             return hash((self.name, self.random, num_parents))

#         def __str__(self):
#             """
#             Provides a more compact representation, e.g. `normal` instead of `Normal()`
#             """
#             return util.camel_case_to_snake_case(self.name)

#     return AllScalarOp


class ScalarOp(Op):
    """
    An `Op` expecting scalar inputs and producing a single scalar output.
    """
    def __init__(self, random, num_parents):
        ""
        self._num_parents = num_parents
        super().__init__(random=random)

    def _get_shape(self, *parents_shapes):
        if len(parents_shapes) != self._num_parents:
            raise ValueError(f"{self.name} op got {len(parents_shapes)} arguments but expected"
                                f" {self._num_parents}.")
        for shape in parents_shapes:
            assert shape == (), "all parents must have shape ()"
        return ()

    def __eq__(self, other):
        "Returns true only if the two classes are exactly the same"
        return type(self) is type(other) 

    def __hash__(self):
        return hash((self.name, self.random, self._num_parents))



class Cauchy(ScalarOp):
    """A Cauchy distribution.
    """
    def __init__(self):
        """Creates a Cauchy, parameterized by location and scale.
        >>> dist = Cauchy()
        >>> print(dist)
        cauchy
        >>> print(repr(dist))
        Cauchy()
        """
        super().__init__(True, 2)


class Normal(ScalarOp):
    """A [normal](https://en.wikipedia.org/wiki/Normal_distribution) distribution.
    Expects the first parent to be the location / mean and the second to be the *scale* / standard deviation."""
    def __init__(self):
        super().__init__(True, 2)

class NormalPrec(ScalarOp):
    """A [normal](https://en.wikipedia.org/wiki/Normal_distribution) distribution.
    Expects the first parameter to be the location / mean and the second to be the *precision* / inverse variance."""
    def __init__(self):
        super().__init__(True, 2)


class LogNormal(ScalarOp):
    """A Log-Normal distribution
    """
    def __init__(self):
        super().__init__(True, 2)


class Bernoulli(ScalarOp):
    """A Bernoulli distribution
    """
    def __init__(self):
        super().__init__(True, 1)


class BernoulliLogit(ScalarOp):
    """A Bernoulli-Logit distribution
    """
    def __init__(self):
        super().__init__(True, 1)


class Binomial(ScalarOp):
    """A Binomial distribution.
    """
    def __init__(self):
        super().__init__(True, 2)

class Uniform(ScalarOp):
    """A Uniform distribution.
    """
    def __init__(self):
        super().__init__(True, 2)

class Beta(ScalarOp):
    """A Beta distribution.
    """
    def __init__(self):
        super().__init__(True, 2)

class Exponential(ScalarOp):
    """An Exponential distribution.
    """
    def __init__(self):
        super().__init__(True, 1)

class Gamma(ScalarOp):
    """A Gamma distribution.
    """
    def __init__(self):
        super().__init__(True, 2)


class Poisson(ScalarOp):
    """A Poisson distribution.
    """
    def __init__(self):
        super().__init__(True, 1)


class BetaBinomial(ScalarOp):
    """A Beta Binomial distribution.
    """
    def __init__(self):
        super().__init__(True, 3)

class StudentT(ScalarOp):
    """A Student-T distribution.
    """
    def __init__(self):
        super().__init__(True, 3)

class Add(ScalarOp):
    """Addition of two scalars
    """
    def __init__(self):
        super().__init__(False, 2)

class Sub(ScalarOp):
    """Subtraction of two scalars
    """
    def __init__(self):
        super().__init__(False, 2)

class Mul(ScalarOp):
    """Multiplication of two scalars
    """
    def __init__(self):
        super().__init__(False, 2)

class Div(ScalarOp):
    """Division of two scalars
    """
    def __init__(self):
        super().__init__(False, 2)

class Pow(ScalarOp):
    """One scalar to a scalar power
    """
    def __init__(self):
        super().__init__(False, 2)


# all the scalar functions included in JAGS manual
# in general, try to use numpy / scipy names where possible

class Abs(ScalarOp):
    """Absolute value of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Arccos(ScalarOp):
    """Arccos of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Arccosh(ScalarOp):
    """Arccosh of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Arcsin(ScalarOp):
    """Arcsin of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Arcsinh(ScalarOp):
    """Arcsinh of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Arctan(ScalarOp):
    """Arctan of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Arctanh(ScalarOp):
    """Arctanh of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Cos(ScalarOp):
    """Cos of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Cosh(ScalarOp):
    """Cosh of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Exp(ScalarOp):
    """Exp of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class InvLogit(ScalarOp):
    """Inverse logit of a scalar. (AKA sigmoid)
    """
    def __init__(self):
        super().__init__(False, 1)


class Log(ScalarOp):
    """Natural logarithm of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)


class Loggamma(ScalarOp):
    """Log gamma function of a scalar
    """
    # TODO: do we want scipy.special.loggamma or scipy.special.gammaln? different!
    def __init__(self):
        super().__init__(False, 1)

class Logit(ScalarOp):
    """Logit of a scalar
    """
    # called Logit in JAGS / stan / scipy
    def __init__(self):
        super().__init__(False, 1)

class Sin(ScalarOp):
    """Sin of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Sinh(ScalarOp):
    """Sinh of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Step(ScalarOp):
    """Step function of a scalar
    """
    # step in JAGS / stan
    def __init__(self):
        super().__init__(False, 1)

class Tan(ScalarOp):
    """Tan of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)

class Tanh(ScalarOp):
    """Tanh of a scalar
    """
    def __init__(self):
        super().__init__(False, 1)


# # use evil meta-programming to create convenience instances
# import re
# pattern = re.compile(r'(?<!^)(?=[A-Z])') # convert CamelCase to underscore_style
# # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
#
# import inspect
# all_vars = dict(vars())
# for name in all_vars:
#     obj = all_vars[name]
#     if inspect.isclass(obj):
#         if issubclass(obj,ir.Op):
#             print(f"WE GOT A CLASS: {name}")
#             camel_case_name = name.lower()
#             underscore_name = pattern.sub('_',camel_case_name).lower()
#
#             exec(f"{name.lower()} = {name}()")


# inv_logit = InvLogit()
# expit = inv_logit  # scipy name
# "another name for inv_logit"
# sigmoid = inv_logit  # another name
# "another name for inv_logit"
