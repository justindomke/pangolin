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

__docformat__ = 'numpy'


import numpy as np
from pangolin import util, ir

def all_scalar_op_factory(num_parents, name, random):
    """
    "Factory" to create an "all scalar" op where all parents and inputs are scalars.

    Parameters
    ----------
    num_parents: int
        The number of parameters for the op
    name: str
        The name for the op (used for printing)
    random: bool
        Is the op a conditional distribution (True) or a deterministic function (False)

    Returns
    -------
    OpClass
        A new subtype of Op
    """

    camel_case_name = name[0].upper() + name[1:]

    class AllScalarOp(ir.Op):
        """
        Convenience class to create "all scalar" distributions, where all parents and
        outputs are scalar. Most of the common ops (e.g. `normal`, `add`, `exp`) are
        instances of this Op.
        """

        def __init__(self):
            super().__init__(name=name, random=random)

        def _get_shape(self, *parents_shapes):
            if len(parents_shapes) != num_parents:
                raise ValueError(f"{name} op got {len(parents_shapes)} arguments but expected"
                                 f" {num_parents}.")
            for shape in parents_shapes:
                assert shape == (), "all parents must have shape ()"
            return ()

        def __eq__(self,other):
            return isinstance(other,AllScalarOp)

        def __hash__(self):
            return hash((self.name, self.random, num_parents))

        def __str__(self):
            """
            Provides a more compact representation, e.g. `normal` instead of `Normal()`
            """
            return util.camel_case_to_snake_case(self.name)

    return AllScalarOp




Cauchy = all_scalar_op_factory(2, "Cauchy", True)
"A Cauchy parameterized by location and scale. Call as `Cauchy()`."
Normal = all_scalar_op_factory(2, "Normal", True)
NormalPrec = all_scalar_op_factory(2, "NormalPrec", True)
LogNormal = all_scalar_op_factory(2, "LogNormal", True)
Bernoulli = all_scalar_op_factory(1, "Bernoulli", True)
BernoulliLogit = all_scalar_op_factory(1, "BernoulliLogit", True)
Binomial = all_scalar_op_factory(2, "Binomial", True)
Uniform = all_scalar_op_factory(2, "Uniform", True)
Beta = all_scalar_op_factory(2, "Beta", True)
Exponential = all_scalar_op_factory(1, "Exponential", True)
Gamma = all_scalar_op_factory(2, "Gamma", True)
Poisson = all_scalar_op_factory(1, "Poisson", True)
BetaBinomial = all_scalar_op_factory(3, "BetaBinomial", True)
StudentT = all_scalar_op_factory(3, "StudentT", True)
# basic math operators, typically triggered infix operations
Add = all_scalar_op_factory(2, "Add", False)
Sub = all_scalar_op_factory(2, "Sub", False)
Mul = all_scalar_op_factory(2, "Mul", False)
Div = all_scalar_op_factory(2, "Div", False)
Pow = all_scalar_op_factory(2, "Pow", False)
# all the scalar functions included in JAGS manual
# in general, try to use numpy / scipy names where possible
Abs = all_scalar_op_factory(1, "Abs", False)
Arccos = all_scalar_op_factory(1, "Arccos", False)
Arccosh = all_scalar_op_factory(1, "Arccosh", False)
Arcsin = all_scalar_op_factory(1, "Arcsin", False)
Arcsinh = all_scalar_op_factory(1, "Arcsinh", False)
Arctan = all_scalar_op_factory(1, "arctan", False)
Arctanh = all_scalar_op_factory(1, "arctanh", False)
Cos = all_scalar_op_factory(1, "Cos", False)
Cosh = all_scalar_op_factory(1, "Cosh", False)
Exp = all_scalar_op_factory(1, "Exp", False)
InvLogit = all_scalar_op_factory(1, "InvLogit", False)
Log = all_scalar_op_factory(1, "Log", False)
Loggamma = all_scalar_op_factory(1, "Loggamma", False)
"Log gamma function. TODO: do we want scipy.special.loggamma or scipy.special.gammaln? different!"
Logit = all_scalar_op_factory(1, "Logit", False)  # logit in JAGS / stan / scipy
Sin = all_scalar_op_factory(1, "Sin", False)
Sinh = all_scalar_op_factory(1, "Sinh", False)
Step = all_scalar_op_factory(1, "Sin", False)  # step in JAGS / stan
Tan = all_scalar_op_factory(1, "Tan", False)
Tanh = all_scalar_op_factory(1, "Tanh", False)

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
