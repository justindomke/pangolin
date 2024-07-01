"""
This package defines a special subtype of RV that supports operator overloading
"""

from cleanpangolin.ir.rv import RV
#from cleanpangolin.ir.scalar_ops import add, mul, sub, div, pow
import cleanpangolin
from cleanpangolin.ir.op import SetAutoRV
from cleanpangolin import ir

def op_rv(fun):
    """
    A decorator that turns a "regular" function into one that always returns an OperatorRV.
    """

    def new_fun(*args):
        with SetAutoRV(OperatorRV):
            out = fun(*args)
        return out
    return new_fun

# create functions for infix operators
add = op_rv(ir.add)
sub = op_rv(ir.sub)
mul = op_rv(ir.mul)
div = op_rv(ir.div)
pow = op_rv(ir.div)
matmul = op_rv(ir.matmul)

makerv = op_rv(ir.makerv)

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