"""
This package defines a special subtype of RV that supports operator overloading
"""

from cleanpangolin.ir import RV, Op, Constant

# from cleanpangolin.ir.scalar_ops import add, mul, sub, div, pow
import cleanpangolin

# from cleanpangolin.ir.op import SetAutoRV
from cleanpangolin import ir
from .index import index
from cleanpangolin.util import most_specific_class


class OperatorRV(RV):
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __repr__(self):
        return "Operator" + super().__repr__()

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        # convert ellipsis into slices
        num_ellipsis = len([i for i in idx if i is ...])
        if num_ellipsis > 1:
            raise ValueError("an index can only have a single ellipsis ('...')")
        elif num_ellipsis == 1:
            where = idx.index(...)
            slices_needed = self.ndim - (len(idx) - 1)  # sub out ellipsis
            if where > 0:
                idx_start = idx[:where]
            else:
                idx_start = ()
            idx_mid = (slice(None),) * slices_needed
            idx_end = idx[where + 1 :]
            idx = idx_start + idx_mid + idx_end

        if self.ndim == 0:
            raise Exception("can't index scalar RV")
        elif isinstance(idx, tuple) and len(idx) > self.ndim:
            raise Exception("RV indexed with more dimensions than exist")
        return index(self, *idx)

    # def __str__(self):
    #    return "Operator" + super().__str__()

    # def __iter__(self):
    #     for n in range(self.shape[0]):
    #         yield self[n]


rv_classes = [OperatorRV]

def current_rv_class():
    return rv_classes[-1]

class SetCurrentRV:
    def __init__(self, rv_class):
        self.rv_class = rv_class

    def __enter__(self):
        rv_classes.append(self.rv_class)

    def __exit__(self, exc_type, exc_value, exc_tb):
        assert rv_classes.pop() == self.rv_class


def makerv(x):
    "Cast something to a constant if necessary"
    if isinstance(x, RV):
        return x
    else:
        return current_rv_class()(Constant(x))

def wrap_op(op: Op, docstring: str=""):
    """
    Given an op, create a convenience function.

    Always returns a new RV of the most specific class. (Which must be unique)
    """

    def fun(*args):
        args = tuple(makerv(a) for a in args)
        rv_class = current_rv_class()
        return rv_class(op, *args)

    fun.__doc__ = docstring

    return fun

def create_rv(op, *args):
    args = tuple(makerv(a) for a in args)
    rv_class = current_rv_class()
    return rv_class(op, *args)

def normal(loc,scale):
    """Create a [normal](https://en.wikipedia.org/wiki/Normal_distribution) distributed random
    variable. Call as `normal(loc,scale)`.

    **Note:** The second parameter is the *scale* standard deviation."""
    return create_rv(ir.Normal(),loc,scale)

# scalar ops
cauchy = wrap_op(ir.Cauchy())
"Create a [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution) distributed random " \
"variable. Call as `cauchy(loc,scale)`."
#normal = wrap_op(ir.Normal())
# "Create a [normal](https://en.wikipedia.org/wiki/Normal_distribution) distributed random " \
# "variable. Call as `normal(loc,scale)`.\n\n **Note:** The second parameter is the *scale* / " \
# "standard deviation."
normal_prec = wrap_op(ir.NormalPrec())
"""Create a [normal](https://en.wikipedia.org/wiki/Normal_distribution) distributed random 
variable. Call as `normal_prec(loc,prec)`.\n\n **Note:** The second parameter is the *precision* 
inverse variance."""
bernoulli = wrap_op(ir.Bernoulli())
"""Create a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distributed random 
variable. Call as `bernoulli(theta)`, where the mean is `theta`. (`theta` must be between 0 and 
1.)"""
bernoulli_logit = wrap_op(ir.BernoulliLogit())
"""Create a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distributed random 
variable. Call as `bernoulli_logit(alpha)`, where the mean is `sigmoid(alpha)`. (`alpha` can be 
any real number.)"""
binomial = wrap_op(ir.Binomial())
"""Create a [binomial](https://en.wikipedia.org/wiki/Binomial_distribution) distributed random 
variable. Call as `binomial(n,p)`, where `n` is the number of repetions and `p` is the 
probability of success for each repetition."""
uniform = wrap_op(ir.Uniform())
"""Create a [uniformly](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) 
distributed random variable. Call as `uniform(low,high)`. `low` must be less than `high`."""
beta = wrap_op(ir.Beta())
"""Create a [beta](https://en.wikipedia.org/wiki/Beta_distribution) distributed random variable. 
Call as `beta(alpha,beta)`."""
exponential = wrap_op(ir.Exponential())
"""Create an [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) distributed 
random variable. Call as `exponential(lambda)`, where `lambda` is the scale."""
gamma = wrap_op(ir.Gamma())
"""Create an [gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distributed
random variable. Call as `exponential(lambda)`, where `lambda` is the scale."""
poisson = wrap_op(ir.Poisson())
"""Create an [poisson](https://en.wikipedia.org/wiki/Poisson_distribution) distributed
random variable. Call as `poisson(lambda)`, where `lambda` is the rate."""
student_t = wrap_op(ir.StudentT())
"""Create a [location-scale student-t](
https://en.wikipedia.org/wiki/Student\'s_t-distribution#Location-scale_t_distribution) 
distributed random variable. Call as `student_t(nu,loc,scale)`, where `nu` is the rate."""
add = wrap_op(ir.Add())
"""Add two scalar random variables. Call as `add(a,b)` (or `a+b` if `a` or `b` are `OperatorRV`)"""
sub = wrap_op(ir.Sub())
"""Subtract two scalar random variables. Call as `sub(a,b)` (or `a-b` if `a` or `b` are 
`OperatorRV`)"""
mul = wrap_op(ir.Mul())
"""Multiply two scalar random variables. Call as `mul(a,b)` (or `a*b` if `a` or `b` are 
`OperatorRV`)"""
div = wrap_op(ir.Div())
"""Divide two scalar random variables. Call as `div(a,b)` (or `a/b` if `a` or `b` are 
`OperatorRV`)"""
pow = wrap_op(ir.Pow())
"""Take one random variable to a power. Call as `pow(a,b)` (or `a**b` if `a` and `b` are 
`OperatorRV`)"""

def sqrt(x):
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x, 0.5)


abs = wrap_op(ir.Abs())
arccos = wrap_op(ir.Arccos())
arccosh = wrap_op(ir.Arccosh())
arcsin = wrap_op(ir.Arcsin())
arcsinh = wrap_op(ir.Arcsinh())
arctan = wrap_op(ir.Arctan())
arctanh = wrap_op(ir.Arctanh())
cos = wrap_op(ir.Cos())
cosh = wrap_op(ir.Cosh())
exp = wrap_op(ir.Exp())
inv_logit = wrap_op(ir.InvLogit())
expit = inv_logit
sigmoid = inv_logit
log = wrap_op(ir.Log())
log_gamma = wrap_op(ir.Loggamma())
logit = wrap_op(ir.Logit())
sin = wrap_op(ir.Sin())
sinh = wrap_op(ir.Sinh())
step = wrap_op(ir.Step())
tan = wrap_op(ir.Tan())
tanh = wrap_op(ir.Tanh())

# multivariate dists
multi_normal = wrap_op(ir.MultiNormal())
"""Create a multivariate normal distributed random variable. Call as `multi_normal(mean,cov)`"""
categorical = wrap_op(ir.Categorical())
"""Create a Categorical-distributed random variable. Call as `categorical(theta)` where `theta` 
is a vector that sums to one."""
multinomial = wrap_op(ir.Multinomial())
"""Create a [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)-distributed 
random variable. Call as `multinomial(n,p)` where `n` is the number of repetitions and `p` is a 
vector of probabilities that sums to one."""
"""Convenience instance of `Multinomial`."""
dirichlet = wrap_op(ir.Dirichlet())
"""Convenience instance of `Dirichlet`."""

# multivariate funs
matmul = wrap_op(ir.MatMul())
inv = wrap_op(ir.Inv())
softmax = wrap_op(ir.Softmax())
"""Softmax for 1-D vectors (TODO: conform to syntax of [scipy.special.softmax](
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html))"""


def sum(x: RV, axis: int):
    """Sum random variable along given axis"""
    op = ir.Sum(axis)
    return wrap_op(op)(x)



from functools import wraps
def wrap_fun(fun):
    """
    Given a function that takes some set of arguments and returns an RV, get a new one that is
    "safe" in the sense of (1) casting all inputs to RV and (2) returning an RV of the current
    type.
    """

    @wraps(fun)
    def new_fun(*args):
        args = tuple(makerv(a) for a in args)
        rv = fun(*args)
        rv_class = current_rv_class()
        return rv_class(rv.op, rv.parents)

    new_fun.__doc__ = f"{fun.__doc__};\n HI"
    return new_fun

@wrap_fun
def new_normal(loc, scale):
    """
    Get a normally distributed random variable.

    Parameters
    ----------
    loc
        location / mean (scalar)
    scale
        scale / standard deviation (positive scalar)
    """
    return RV(ir.Normal(), loc, scale)