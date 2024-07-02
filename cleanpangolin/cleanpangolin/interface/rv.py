"""
This package defines a special subtype of RV that supports operator overloading
"""

from cleanpangolin.ir.rv import RV
#from cleanpangolin.ir.scalar_ops import add, mul, sub, div, pow
import cleanpangolin
#from cleanpangolin.ir.op import SetAutoRV
from cleanpangolin import ir

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

    #def __str__(self):
    #    return "Operator" + super().__str__()

    # def __iter__(self):
    #     for n in range(self.shape[0]):
    #         yield self[n]

current_rv = [OperatorRV]

# class SetAutoRV:
#     def __init__(self, rv_class):
#         self.rv_class = rv_class
#
#     def __enter__(self):
#         current_rv.append(self.rv_class)
#
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         assert current_rv.pop(1) == self.rv_class


def cur_rv():
    """
    A decorator that turns a "regular" function into one that always returns an OperatorRV.
    """

    return current_rv[-1]

#makerv = op_rv(ir.makerv)



cauchy = ir.Cauchy()
"Convenience instance of the Cauchy distribution. Call as `cauchy(loc,scale)`."
normal = op_rv(ir.Normal())
normal_prec = op_rv(ir.NormalPrec())
bernoulli = op_rv(ir.Bernoulli())
bernoulli_logit = op_rv(ir.BernoulliLogit())
binomial = op_rv(ir.Binomial())
uniform = op_rv(ir.Uniform())
"Uniform distribution. `uniform(low,high)`"
beta = op_rv(ir.Beta())
exponential = op_rv(ir.Exponential())
gamma = op_rv(ir.Gamma())
poisson = op_rv(ir.Poisson())
student_t = op_rv(ir.StudentT())
add = op_rv(ir.Add())
sub = op_rv(ir.Sub())
mul = op_rv(ir.Mul())
div = op_rv(ir.Div())
pow = op_rv(ir.Pow())
def sqrt(x):
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x,0.5)
abs = op_rv(ir.Abs())
arccos = op_rv(ir.Arccos())
arccosh = op_rv(ir.Arccosh())
arcsin = op_rv(ir.Arcsin())
arcsinh = op_rv(ir.Arcsinh())
arctan = op_rv(ir.Arctan())
arctanh = op_rv(ir.Arctanh())
cos = op_rv(ir.Cos())
cosh = op_rv(ir.Cosh())
exp = op_rv(ir.Exp())
inv_logit = op_rv(ir.InvLogit())
expit = inv_logit
sigmoid = inv_logit
log = op_rv(ir.Log())
log_gamma = op_rv(ir.Loggamma())
logit = op_rv(ir.Logit())
sin = op_rv(ir.Sin())
sinh = op_rv(ir.Sinh())
step = op_rv(ir.Step())
tan = op_rv(ir.Tan())
tanh = op_rv(ir.Tanh())

matmul = op_rv(ir.MatMul)