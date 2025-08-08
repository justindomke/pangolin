"""
This package defines a special subtype of RV that supports operator overloading
"""
from pangolin.ir import RV, Op, Constant

# from pangolin.ir.scalar_ops import add, mul, sub, div, pow

# from pangolin.ir.op import SetAutoRV
from pangolin import ir
#from .index import index
from numpy.typing import ArrayLike

RV_or_ArrayLike = ArrayLike | RV
"""Represents either a `RV` or a value that can be cast to a NumPy array, such as a float or list of floats.
This will include JAX arrays (which is good) but also unfortunately accepts strings (which is bad)."""

####################################################################################################
# The core InfixRV class. Like an RV except has infix operations
####################################################################################################

class InfixRV(RV):
    """An Infix RV is exactly like a standard `RV` except it supports infix operations.

    Examples
    --------
    >>> a = InfixRV(Constant(2))
    >>> b = InfixRV(Constant(3))
    >>> a + b
    InfixRV(Add(), InfixRV(Constant(2)), InfixRV(Constant(3)))
    >>> a**b
    InfixRV(Pow(), InfixRV(Constant(2)), InfixRV(Constant(3)))
    >>> -a
    InfixRV(Mul(), InfixRV(Constant(2)), InfixRV(Constant(-1)))
    """

    __array_priority__ = 1000  # so x @ y works when x numpy.ndarray and y RV

    def __init__(self, op, *parents):
        super().__init__(op, *parents)

    def __neg__(self):
        return mul(self,-1)

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
        return "Infix" + super().__repr__()

    def __str__(self):
        return super().__str__()

    # def __getitem__(self, idx):
    #     if not isinstance(idx, tuple):
    #         idx = (idx,)

    #     return index(self, *idx)


####################################################################################################
# makerv
####################################################################################################

def non_infix_rv(x):
    if isinstance(x, RV):
        if not isinstance(x, InfixRV):
            return True
    return False

def makerv(x) -> RV:
    """
    If the input is `RV`, then it just returns it. Otherwise, creates an InfixRV.
    """

    if isinstance(x,RV):
        return x
    else:
        return InfixRV(Constant(x))

def make_infix_rv(x) -> InfixRV:
    """
    If the input is an `InfixRV`, then it just returns it. Otherwise, creates an InfixRV.
    Fails is input is RV but not InfixRV
    """
    assert not non_infix_rv(x)

    if isinstance(x,RV):
        assert isinstance(x, InfixRV)
        return x
    else:
        return InfixRV(Constant(x))

####################################################################################################
# Scalar ops
####################################################################################################


def create_rv(op, *args) -> InfixRV:
    args = tuple(makerv(a) for a in args)
    op.get_shape(*[a.shape for a in args]) # checks shapes
    return InfixRV(op, *args)


def normal(loc: RV_or_ArrayLike, scale: RV_or_ArrayLike) -> InfixRV:
    """Creates a [normal](https://en.wikipedia.org/wiki/Normal_distribution) distributed random variable.

    Parameters
    ----------
    loc: `RV_or_ArrayLike`
        mean for the distribution. Should be scalar.
    scale: `RV_or_ArrayLike`
        scale / standard deviation. Should be scalar.

    See also
    --------
    `pangolin.ir.scalar_ops.Normal`."""
    return create_rv(ir.Normal(), loc, scale)


def normal_prec(loc, prec) -> InfixRV:
    """Creates a normal distribution parameterized by the precision.

    Parameters
    ----------
    loc: RV_or_ArrayLike
        mean
    scale

    See `pangolin.ir.scalar_ops.NormalPrec`."""
    return create_rv(ir.NormalPrec(), loc, prec)


def lognormal(mu, sigma) -> InfixRV:
    """Create a [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) distributed random
    variable."""
    return create_rv(ir.NormalPrec(), mu, sigma)


def cauchy(loc, scale) -> InfixRV:
    """Create a [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution) distributed random
    variable."""
    return create_rv(ir.Cauchy(), loc, scale)


def bernoulli(theta) -> InfixRV:
    """Create a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distributed random
    variable.

    Parameters
    ----------
    theta
        the mean. must be between 0 and 1
    """
    return create_rv(ir.Bernoulli(), theta)


def bernoulli_logit(alpha) -> InfixRV:
    """Create a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distributed random
    variable.

    Parameters
    ----------
    alpha
        any real number. determines the mean as mean = `sigmoid(alpha)`
    """

    return create_rv(ir.BernoulliLogit(), alpha)


def binomial(n, p) -> InfixRV:
    """Create a [binomial](https://en.wikipedia.org/wiki/Binomial_distribution) distributed random
    variable. Call as `binomial(n,p)`, where `n` is the number of repetions and `p` is the
    probability of success for each repetition."""
    return create_rv(ir.Binomial(), n, p)


def uniform(low, high) -> InfixRV:
    """Create a [uniformly](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    distributed random variable. `low` must be less than `high`."""

    return create_rv(ir.Uniform(), low, high)


def beta(alpha, beta) -> InfixRV:
    """Create a [beta](https://en.wikipedia.org/wiki/Beta_distribution) distributed random variable."""
    return create_rv(ir.Beta(), alpha, beta)


def beta_binomial(n, alpha, beta) -> InfixRV:
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


def exponential(scale) -> InfixRV:
    """Create an [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) distributed
    random variable."""

    return create_rv(ir.Exponential(), scale)


def gamma(alpha, beta) -> InfixRV:
    """Create an [gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distributed
    random variable.

    **Note:** We (like [stan](https://mc-stan.org/docs/2_21/functions-reference/gamma
    -distribution.html)) follow the "shape/rate" parameterization, *not* the "shape/scale"
    parameterization.
    """

    return create_rv(ir.Gamma(), alpha, beta)


def poisson(rate) -> InfixRV:
    """Create an [poisson](https://en.wikipedia.org/wiki/Poisson_distribution) distributed
    random variable."""

    return create_rv(ir.Poisson(), rate)


def student_t(nu, loc, scale) -> InfixRV:
    """Create a [location-scale student-t](
    https://en.wikipedia.org/wiki/Student\'s_t-distribution#Location-scale_t_distribution)
    distributed random variable. Call as `student_t(nu,loc,scale)`, where `nu` is the rate."""

    # TODO: make loc and scale optional?

    return create_rv(ir.StudentT(), nu, loc, scale)


def add(a, b) -> InfixRV:
    """Add two scalar random variables. Typically one would type `a+b` rather than `add(a,b)`."""
    return create_rv(ir.Add(), a, b)

def sub(a, b) -> InfixRV:
    """Subtract two scalar random variables. Typically one would type `a-b` rather than `sub(a,
    b)`."""
    return create_rv(ir.Sub(), a, b)

def mul(a, b) -> InfixRV:
    """Multiply two scalar random variables. Typically one would type `a*b` rather than `mul(a,b)`."""
    return create_rv(ir.Mul(), a, b)

def div(a, b) -> InfixRV:
    """Divide two scalar random variables. Typically one would type `a/b` rather than `div(a,b)`."""
    return create_rv(ir.Div(), a, b)

def pow(a, b) -> InfixRV:
    """Take one scalar to another scalar power. Typically one would type `a**b` rather than `pow(
    a,b)`."""
    return create_rv(ir.Pow(), a, b)

def sqrt(x) -> InfixRV:
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x, 0.5)

def abs(a) -> InfixRV:
    return create_rv(ir.Abs(), a)


def arccos(a) -> InfixRV:
    return create_rv(ir.Arccos(), a)


def arccosh(a) -> InfixRV:
    return create_rv(ir.Arccosh(), a)


def arcsin(a) -> InfixRV:
    return create_rv(ir.Arcsin(), a)


def arcsinh(a) -> InfixRV:
    return create_rv(ir.Arcsinh(), a)


def arctan(a) -> InfixRV:
    return create_rv(ir.Arctan(), a)


def arctanh(a) -> InfixRV:
    return create_rv(ir.Arctanh(), a)


def cos(a) -> InfixRV:
    return create_rv(ir.Cos(), a)


def cosh(a) -> InfixRV:
    return create_rv(ir.Cosh(), a)


def exp(a) -> InfixRV:
    return create_rv(ir.Exp(), a)


def inv_logit(a) -> InfixRV:
    return create_rv(ir.InvLogit(), a)


def expit(a) -> InfixRV:
    """Equivalent to `inv_logit`"""
    return create_rv(ir.InvLogit(), a)


def sigmoid(a) -> InfixRV:
    """Equivalent to `inv_logit`"""
    return create_rv(ir.InvLogit(), a)


def log(a) -> InfixRV:
    return create_rv(ir.Log(), a)

def log_gamma(a) -> InfixRV:
    """Log gamma function.

    **TODO**: do we want
    [`scipy.special.loggamma`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html)
    or
    [`scipy.special.gammaln`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html)?
    These are different!
    """
    return create_rv(ir.Loggamma(), a)


def logit(a) -> InfixRV:
    return create_rv(ir.Logit(), a)


def sin(a) -> InfixRV:
    return create_rv(ir.Sin(), a)


def sinh(a) -> InfixRV:
    return create_rv(ir.Sinh(), a)


def step(a) -> InfixRV:
    return create_rv(ir.Step(), a)


def tan(a) -> InfixRV:
    return create_rv(ir.Tan(), a)


def tanh(a) -> InfixRV:
    return create_rv(ir.Tanh(), a)

####################################################################################################
# Multivariate dists
####################################################################################################

def multi_normal(mean, cov) -> InfixRV:
    """Create a multivariate normal distributed random variable. Call as `multi_normal(mean,cov)`"""
    return create_rv(ir.MultiNormal(), mean, cov)


def categorical(theta) -> InfixRV:
    """Create a [categorical](https://en.wikipedia.org/wiki/Categorical_distribution)-distributed
    where `theta` is a vector of non-negative reals that sums to one."""
    return create_rv(ir.Categorical(), theta)


def multinomial(n, p) -> InfixRV:
    """Create a [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)-distributed
    random variable. Call as `multinomial(n,p)` where `n` is the number of repetitions and `p` is a
    vector of probabilities that sums to one."""
    return create_rv(ir.Multinomial(), n, p)


def dirichlet(alpha) -> InfixRV:
    """Create a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)-distributed
    random variable. Call as `dirichlet(alpha)` where `alpha` is a 1-D vector of positive reals."""
    return create_rv(ir.Dirichlet(), alpha)


####################################################################################################
# Multivariate funs
####################################################################################################


def matmul(a, b) -> InfixRV:
    """
    Matrix product of two arrays. The behavior follows that of
    [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
    except that `a` and `b` must both be 1-D or 2-D arrays. In particular:
    * If `a` and `b` are both 1-D then this represents an inner-product.
    * If `a` is 1-D and `b` is 2-D then this represents vector/matrix multiplication
    * If `a` is 2-D and `b` is 1-D then this represents matrix/vector multiplication
    * If `a` and `b` are both 2-D then this represents matrix/matrix multiplication
    """
    return create_rv(ir.MatMul(), a, b)

def inv(a) -> InfixRV:
    """
    Take the inverse of a matrix. Input must be a 2-D square (invertible) array.
    """
    return create_rv(ir.Inv(), a)


def softmax(a) -> InfixRV:
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


def sum(x: RV, axis: int) -> InfixRV:
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


