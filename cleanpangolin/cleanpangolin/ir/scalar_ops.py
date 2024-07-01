"""
This module defines some "all scalar" ops where all inputs and outputs are scalars. In each case
there is an `Op` *class* created, and then an `Op` *instance* for convenience. For example,
`Normal` represents the class of all Gaussian conditional distributions. These do not have any
parameters, because there aren't different "types" of normal distributions. For convenience,
we also provide `normal=Normal()`, which is one particular Gaussian conditional distribution.

The user is expected to interact with these (lower case) instances, e.g. by calling
```python
x = cleanpangolin.ir.scalar_ops.normal(0,1)
```

If for some reason you didn't want to use the built-in convenience class, this is equivalent to
writing

```python
x = cleanpangolin.ir.scalar_ops.Normal()(0,1)
```

Or, if you wanted to be even more pedantic, you could write

```python
x = cleanpangolin.ir.rv.RV(cleanpangolin.ir.scalar_ops.normal,0,1)
```

or even

```python
x = cleanpangolin.ir.rv.RV(cleanpangolin.ir.scalar_ops.Normal(),0,1)
```

But there's no reason to do that.


"""

__docformat__ = 'numpy'


import numpy as np
from cleanpangolin import util, ir

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
cauchy = Cauchy()
"Convenience instance of the Cauchy distribution. Call as `cauchy(loc,scale)`."
Normal = all_scalar_op_factory(2, "Normal", True)
normal = Normal()
NormalPrec = all_scalar_op_factory(2, "NormalPrec", True)
normal_prec = NormalPrec()
Bernoulli = all_scalar_op_factory(1, "Bernoulli", True)
bernoulli = Bernoulli()
BernoulliLogit = all_scalar_op_factory(1, "BernoulliLogit", True)
bernoulli_logit = BernoulliLogit()
Binomial = all_scalar_op_factory(2, "Binomial", True)
binomial = Binomial()
Uniform = all_scalar_op_factory(2, "Uniform", True)
uniform = Uniform()
"Uniform distribution. `uniform(low,high)`"
Beta = all_scalar_op_factory(2, "Beta", True)
beta = Beta()
Exponential = all_scalar_op_factory(1, "Exponential", True)
exponential = Exponential()
Gamma = all_scalar_op_factory(2, "Gamma", True)
gamma = Gamma()
Poisson = all_scalar_op_factory(1, "Poisson", True)
poisson = Poisson()
BetaBinomial = all_scalar_op_factory(3, "BetaBinomial", True)
beta_binomial = BetaBinomial()
StudentT = all_scalar_op_factory(3, "StudentT", True)
student_t = StudentT()
# basic math operators, typically triggered infix operations
Add = all_scalar_op_factory(2, "Add", False)
add = Add()
Sub = all_scalar_op_factory(2, "Sub", False)
sub = Sub()
Mul = all_scalar_op_factory(2, "Mul", False)
mul = Mul()
Div = all_scalar_op_factory(2, "Div", False)
div = Div()
Pow = all_scalar_op_factory(2, "Pow", False)
pow = Pow()
def sqrt(x):
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x,0.5)
# all the scalar functions included in JAGS manual
# in general, try to use numpy / scipy names where possible
Abs = all_scalar_op_factory(1, "Abs", False)
abs = Abs()
Arccos = all_scalar_op_factory(1, "Arccos", False)
arccos = Arccos()
Arccosh = all_scalar_op_factory(1, "Arccosh", False)
arccosh = Arccosh()
Arcsin = all_scalar_op_factory(1, "Arcsin", False)
arcsin = Arcsin()
Arcsinh = all_scalar_op_factory(1, "Arcsinh", False)
arcsinh = Arcsinh()
Arctan = all_scalar_op_factory(1, "arctan", False)
arctan = Arctan()
Arctanh = all_scalar_op_factory(1, "arctanh", False)
arctanh = Arctanh()
Cos = all_scalar_op_factory(1, "Cos", False)
cos = Cos()
Cosh = all_scalar_op_factory(1, "Cosh", False)
cosh = Cosh()
Exp = all_scalar_op_factory(1, "Exp", False)
exp = Exp()
InvLogit = all_scalar_op_factory(1, "InvLogit", False)
inv_logit = InvLogit()
expit = inv_logit
sigmoid = inv_logit
Log = all_scalar_op_factory(1, "Log", False)
log = Log()
Loggamma = all_scalar_op_factory(1, "Loggamma", False)
"Log gamma function. TODO: do we want scipy.special.loggamma or scipy.special.gammaln? different!"
log_gamma = Loggamma()
Logit = all_scalar_op_factory(1, "Logit", False)  # logit in JAGS / stan / scipy
logit = Logit()
Sin = all_scalar_op_factory(1, "Sin", False)
sin = Sin()
Sinh = all_scalar_op_factory(1, "Sinh", False)
sinh = Sinh()
Step = all_scalar_op_factory(1, "Sin", False)  # step in JAGS / stan
step = Step()
Tan = all_scalar_op_factory(1, "Tan", False)
tan = Tan()
Tanh = all_scalar_op_factory(1, "Tanh", False)
tanh = Tanh()

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
