"""
This package defines a special subtype of RV that supports operator overloading
"""

from cleanpangolin.ir import RV, Op, Constant
#from cleanpangolin.ir.scalar_ops import add, mul, sub, div, pow
import cleanpangolin
#from cleanpangolin.ir.op import SetAutoRV
from cleanpangolin import ir
from .index import index
from cleanpangolin.util import most_specific_class

def get_rv_class(*args:RV):
    return most_specific_class(*args, base_classes=(OperatorRV,))

class OperatorRV(RV):

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self,other)

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
            idx_end = idx[where + 1:]
            idx = idx_start + idx_mid + idx_end

        if self.ndim == 0:
            raise Exception("can't index scalar RV")
        elif isinstance(idx, tuple) and len(idx) > self.ndim:
            raise Exception("RV indexed with more dimensions than exist")
        return index(self, *idx)

    #def __str__(self):
    #    return "Operator" + super().__str__()

    # def __iter__(self):
    #     for n in range(self.shape[0]):
    #         yield self[n]


def makerv(x):
    if isinstance(x,RV):
        return x
    else:
        return OperatorRV(Constant(x))

# class SetAutoRV:
#     def __init__(self, rv_class):
#         self.rv_class = rv_class
#
#     def __enter__(self):
#         current_rv.append(self.rv_class)
#
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         assert current_rv.pop(1) == self.rv_class


def wrap_op(op:Op):
    """
    Given an op, create a convenience function.

    Always returns a new RV of the most specific class. (Which must be unique)
    """
    def fun(*args):
        args = tuple(makerv(a) for a in args)
        rv_class = get_rv_class(*args)
        return rv_class(op, *args)
    return fun

# scalar ops
cauchy = wrap_op(ir.Cauchy())
"Convenience instance of the Cauchy distribution. Call as `cauchy(loc,scale)`."
normal = wrap_op(ir.Normal())
normal_prec = wrap_op(ir.NormalPrec())
bernoulli = wrap_op(ir.Bernoulli())
bernoulli_logit = wrap_op(ir.BernoulliLogit())
binomial = wrap_op(ir.Binomial())
uniform = wrap_op(ir.Uniform())
"Uniform distribution. `uniform(low,high)`"
beta = wrap_op(ir.Beta())
exponential = wrap_op(ir.Exponential())
gamma = wrap_op(ir.Gamma())
poisson = wrap_op(ir.Poisson())
student_t = wrap_op(ir.StudentT())
add = wrap_op(ir.Add())
sub = wrap_op(ir.Sub())
mul = wrap_op(ir.Mul())
div = wrap_op(ir.Div())
pow = wrap_op(ir.Pow())
def sqrt(x):
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x,0.5)
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
"""Convenience wrapper for `MultiNormal`."""
categorical = wrap_op(ir.Categorical())
"""Convenience instance of `Categorical`."""
multinomial = wrap_op(ir.Multinomial())
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

# indexing
# TODO: ir.index should actually live in this module in the first place
# def index(node, *idx):
#     node = makerv(node)
#     idx = tuple(makerv(i) for i in idx)
#     rv_class = get_rv_class(node, *idx)
#     out = ir.index(node, *idx)
#     return rv_class(out.op, *out.parents)
