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
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike

# type hint for RVs or things that we will implicitly cast to RVs
# this will include jax arrays (good) but will also accept strings (bad)
RV_or_array = ArrayLike | RV

for_api = []  # list of all functions to be exported for global API


def api(fun):
    """
    Decorator to include function in global API
    """
    for_api.append(fun.__name__)
    return fun


class OperatorRV(RV):
    __array_priority__ = 1000  # so x @ y works when x numpy.ndarray and y RV

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

        return index(self, *idx)

    # def __str__(self):
    #    return "Operator" + super().__str__()

    # def __iter__(self):
    #     for n in range(self.shape[0]):
    #         yield self[n]


# class RVManager(ABC):
#     @classmethod
#     @abstractmethod
#     def makerv(cls,x) -> RV:
#         pass
#
#     @classmethod
#     @abstractmethod
#     def create_rv(cls, op, *parents):
#         pass
#
#
# class OperatorRVMangager(RVManager):
#     @classmethod
#     def create_rv(cls, op, *parents):
#         return OperatorRV(op, *parents)
#
#     @classmethod
#     def makerv(cls,x) -> RV:
#         if isinstance(x, RV):
#             return x
#         else:
#             return OperatorRV(Constant(x))

def assert_not_non_operator_rv(x):
    if isinstance(x,RV):
        if not isinstance(x,OperatorRV):
            raise Exception("got non-operatorRV RV {type(x)} {x}")

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


def my_makerv(x) -> OperatorRV:
    assert_not_non_operator_rv(x)

    if isinstance(x, OperatorRV):
        return x
    else:
        return current_rv_class()(Constant(x))


makerv_funs = [my_makerv]


@api
def makerv(x) -> OperatorRV:
    "Cast something to a constant if necessary"
    out = makerv_funs[-1](x)
    if not isinstance(out, OperatorRV):
        raise Exception(
            f"Custom makerv function must return subclass of OperatorRV was " f"{type(out)}"
        )
    return out


# def wrap_op(op: Op, docstring: str = ""):
#     """
#     Given an op, create a convenience function.
#
#     Always returns a new RV of the most specific class. (Which must be unique)
#     """
#
#     def fun(*args):
#         args = tuple(makerv(a) for a in args)
#         rv_class = current_rv_class()
#         return rv_class(op, *args)
#
#     fun.__doc__ = docstring
#
#     return fun


def create_rv(op, *args):
    args = tuple(makerv(a) for a in args)
    rv_class = current_rv_class()
    return rv_class(op, *args)


# scalar ops


@api
def normal(loc: RV_or_array, scale: RV_or_array) -> OperatorRV:
    """Create a [normal](https://en.wikipedia.org/wiki/Normal_distribution) distributed random
    variable.

    **Note:** The second parameter is the *scale* / standard deviation."""
    return create_rv(ir.Normal(), loc, scale)


@api
def normal_prec(loc, prec) -> OperatorRV:
    """Create a [normal](https://en.wikipedia.org/wiki/Normal_distribution) distributed random
    variable.

    **Note:** The second parameter is the *precision* / inverse variance."""
    return create_rv(ir.NormalPrec(), loc, prec)


@api
def cauchy(loc, scale) -> OperatorRV:
    """Create a [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution) distributed random
    variable."""
    return create_rv(ir.Cauchy(), loc, scale)


@api
def bernoulli(theta) -> OperatorRV:
    """Create a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distributed random
    variable.

    Parameters
    ----------
    theta
        the mean. must be between 0 and 1
    """
    return create_rv(ir.Bernoulli(), theta)


@api
def bernoulli_logit(alpha) -> OperatorRV:
    """Create a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distributed random
    variable.

    Parameters
    ----------
    alpha
        any real number. determines the mean as mean = `sigmoid(alpha)`
    """

    return create_rv(ir.BernoulliLogit(), alpha)


@api
def binomial(n, p) -> OperatorRV:
    """Create a [binomial](https://en.wikipedia.org/wiki/Binomial_distribution) distributed random
    variable. Call as `binomial(n,p)`, where `n` is the number of repetions and `p` is the
    probability of success for each repetition."""
    return create_rv(ir.Binomial(), n, p)


@api
def uniform(low, high) -> OperatorRV:
    """Create a [uniformly](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    distributed random variable. `low` must be less than `high`."""

    return create_rv(ir.Uniform(), low, high)


@api
def beta(alpha, beta) -> OperatorRV:
    """Create a [beta](https://en.wikipedia.org/wiki/Beta_distribution) distributed random variable."""
    return create_rv(ir.Beta(), alpha, beta)


@api
def beta_binomial(n, alpha, beta) -> OperatorRV:
    """Create a
    [beta-binomial](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
    distributed random variable.

    **Note**: This follows the (n,alpha,beta) convention of
    [stan](https://mc-stan.org/docs/2_19/functions-reference/beta-binomial-distribution.html)
    (and Wikipedia). Some other systems (e.g.
    [numpyro](https://num.pyro.ai/en/stable/distributions.html#betabinomial))
    use alternate variable orderings. This is no problem for you as a user—pangolin does the
    re-ordering for you if you call the numpyro backend. But keep it in mind if translating a
    model from one system to another.
    """
    return create_rv(ir.BetaBinomial(), n, alpha, beta)


@api
def exponential(scale) -> OperatorRV:
    """Create an [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) distributed
    random variable."""

    return create_rv(ir.Exponential(), scale)


@api
def gamma(alpha, beta) -> OperatorRV:
    """Create an [gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distributed
    random variable.

    **Note:** We (like [stan](https://mc-stan.org/docs/2_21/functions-reference/gamma
    -distribution.html)) follow the "shape/rate" parameterization, *not* the "shape/scale"
    parameterization.
    """

    return create_rv(ir.Gamma(), alpha, beta)


@api
def poisson(rate) -> OperatorRV:
    """Create an [poisson](https://en.wikipedia.org/wiki/Poisson_distribution) distributed
    random variable."""

    return create_rv(ir.Poisson(), rate)


@api
def student_t(nu, loc, scale) -> OperatorRV:
    """Create a [location-scale student-t](
    https://en.wikipedia.org/wiki/Student\'s_t-distribution#Location-scale_t_distribution)
    distributed random variable. Call as `student_t(nu,loc,scale)`, where `nu` is the rate."""

    # TODO: make loc and scale optional?

    return create_rv(ir.StudentT(), nu, loc, scale)


@api
def add(a, b) -> OperatorRV:
    """Add two scalar random variables. Typically one would type `a+b` rather than `add(a,b)`."""
    return create_rv(ir.Add(), a, b)


@api
def sub(a, b) -> OperatorRV:
    """Subtract two scalar random variables. Typically one would type `a-b` rather than `sub(a,
    b)`."""
    return create_rv(ir.Sub(), a, b)


@api
def mul(a, b) -> OperatorRV:
    """Multiply two scalar random variables. Typically one would type `a*b` rather than `mul(a,b)`."""
    return create_rv(ir.Mul(), a, b)


@api
def div(a, b) -> OperatorRV:
    """Divide two scalar random variables. Typically one would type `a/b` rather than `div(a,b)`."""
    return create_rv(ir.Div(), a, b)


@api
def pow(a, b) -> OperatorRV:
    """Take one scalar to another scalar power. Typically one would type `a**b` rather than `pow(
    a,b)`."""
    return create_rv(ir.Pow(), a, b)


@api
def sqrt(x) -> OperatorRV:
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x, 0.5)


@api
def abs(a) -> OperatorRV:
    return create_rv(ir.Abs(), a)


@api
def arccos(a) -> OperatorRV:
    return create_rv(ir.Arccos(), a)


@api
def arccosh(a) -> OperatorRV:
    return create_rv(ir.Arccosh(), a)


@api
def arcsin(a) -> OperatorRV:
    return create_rv(ir.Arcsin(), a)


@api
def arcsinh(a) -> OperatorRV:
    return create_rv(ir.Arcsinh(), a)


@api
def arctan(a) -> OperatorRV:
    return create_rv(ir.Arctan(), a)


@api
def arctanh(a) -> OperatorRV:
    return create_rv(ir.Arctanh(), a)


@api
def cos(a) -> OperatorRV:
    return create_rv(ir.Cos(), a)


@api
def cosh(a) -> OperatorRV:
    return create_rv(ir.Cosh(), a)


@api
def exp(a) -> OperatorRV:
    return create_rv(ir.Exp(), a)


@api
def inv_logit(a) -> OperatorRV:
    return create_rv(ir.InvLogit(), a)


@api
def expit(a) -> OperatorRV:
    """Equivalent to `inv_logit`"""
    return create_rv(ir.InvLogit(), a)


@api
def sigmoid(a) -> OperatorRV:
    """Equivalent to `inv_logit`"""
    return create_rv(ir.InvLogit(), a)


@api
def log(a) -> OperatorRV:
    return create_rv(ir.Log(), a)


@api
def log_gamma(a) -> OperatorRV:
    """Log gamma function.

    **TODO**: do we want
    [`scipy.special.loggamma`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html)
    or
    [`scipy.special.gammaln`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html)?
    These are different!
    """
    return create_rv(ir.Loggamma(), a)


@api
def logit(a) -> OperatorRV:
    return create_rv(ir.Logit(), a)


@api
def sin(a) -> OperatorRV:
    return create_rv(ir.Sin(), a)


@api
def sinh(a) -> OperatorRV:
    return create_rv(ir.Sinh(), a)


@api
def step(a) -> OperatorRV:
    return create_rv(ir.Step(), a)


@api
def tan(a) -> OperatorRV:
    return create_rv(ir.Tan(), a)


@api
def tanh(a) -> OperatorRV:
    return create_rv(ir.Tanh(), a)


# multivariate dists
@api
def multi_normal(mean, cov) -> OperatorRV:
    """Create a multivariate normal distributed random variable. Call as `multi_normal(mean,cov)`"""
    return create_rv(ir.MultiNormal(), mean, cov)


@api
def categorical(theta) -> OperatorRV:
    """Create a [categorical](https://en.wikipedia.org/wiki/Categorical_distribution)-distributed
    where `theta` is a vector of non-negative reals that sums to one."""
    return create_rv(ir.Categorical(), theta)


@api
def multinomial(n, p) -> OperatorRV:
    """Create a [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)-distributed
    random variable. Call as `multinomial(n,p)` where `n` is the number of repetitions and `p` is a
    vector of probabilities that sums to one."""
    return create_rv(ir.Multinomial(), n, p)


@api
def dirichlet(alpha) -> OperatorRV:
    """Create a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)-distributed
    random variable. Call as `dirichlet(alpha)` where `alpha` is a 1-D vector of positive reals."""
    return create_rv(ir.Dirichlet(), alpha)


# multivariate funs


@api
def matmul(a, b) -> OperatorRV:
    """
    Matrix product of two arrays. The behavior follows that of
    [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
    except that (currently) `a` and `b` must both be 1-D or 2-D arrays. In particular:
    * If `a` and `b` are both 1-D then this represents an inner-product.
    * If `a` is 1-D and `b` is 2-D then this represents vector/matrix multiplication
    * If `a` is 2-D and `b` is 1-D then this represents matrix/vector multiplication
    * If `a` and `b` are both 2-D then this represents matrix/matrix multiplication
    """
    return create_rv(ir.MatMul(), a, b)


@api
def inv(a) -> OperatorRV:
    """
    Take the inverse of a matrix. Input must be a 2-D square (invertible) array.
    """
    return create_rv(ir.Inv(), a)


@api
def softmax(a) -> OperatorRV:
    """
    Take [softmax](https://en.wikipedia.org/wiki/Softmax_function) function. (TODO: conform to
    syntax of [scipy.special.softmax](
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html))

    Parameters
    ----------
    a
        1-D vector
    """
    return create_rv(ir.Inv(), a)


@api
def sum(x: RV, axis: int) -> OperatorRV:
    """
    Take the sum of a random variable along a given axis

    Parameters
    ----------
    x
        an RV (or something that can be cast to a Constant RV)
    axis
        a non-negative integer (cannot be a random variable)
    """
    if not isinstance(axis, int):
        raise ValueError("axis argument for sum must be an integer")
    return create_rv(ir.Sum(axis), x)


# from functools import wraps
#
#
# def wrap_fun(fun):
#     """
#     Given a function that takes some set of arguments and returns an RV, get a new one that is
#     "safe" in the sense of (1) casting all inputs to RV and (2) returning an RV of the current
#     type.
#     """
#
#     @wraps(fun)
#     def new_fun(*args):
#         args = tuple(makerv(a) for a in args)
#         rv = fun(*args)
#         rv_class = current_rv_class()
#         return rv_class(rv.op, rv.parents)
#
#     new_fun.__doc__ = f"{fun.__doc__};\n HI"
#     return new_fun
#
#
# @wrap_fun
# def new_normal(loc, scale):
#     """
#     Get a normally distributed random variable.
#
#     Parameters
#     ----------
#     loc
#         location / mean (scalar)
#     scale
#         scale / standard deviation (positive scalar)
#     """
#     return RV(ir.Normal(), loc, scale)
