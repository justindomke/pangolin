import numpy as np
from collections.abc import Iterable

# TODOs:
# * Add a few numpy-esque convenience functions like `t.sum(A,axis=1)`
# * Stop `is a logical node and cannot be observed` errors before running JAGS
# * Stop `Node inconsistent with parents` errors in JAGS (basically impossible?)
# * Allow vectorized equalities / inequalities / I / ifelse
# * Cache samples (if you just did E, keep it around)


################################################################################
# Abstract DAG stuff (a user should never need this)
################################################################################

class Node:
    def __init__(self,*parents,tracer=None):
        self.parents = parents

        # a tracer function can be used
        traced_parents = tuple(filter(lambda a:hasattr(a,'tracer'),parents))
        if traced_parents:
            assert tracer is None

            for p in traced_parents:
                assert p.tracer == traced_parents[0].tracer, "tracers must be same"
            self.tracer = traced_parents[0].tracer
            #self.tracer(self) # run the tracer
        else:
            if tracer is not None:
                self.tracer = tracer


def dfs(node, upstream):
    # if node in upstream:
    if is_in(node, upstream):
        return

    if isinstance(node, Node):
        for p in node.parents:
            dfs(p, upstream)

    # upstream.add(node)
    upstream[node] = None  # use dict as ordered set


def upstream_nodes(nodes):
    # transform into a list if needed
    if isinstance(nodes, Node):
        return upstream_nodes([nodes])

    # upstream = set()
    upstream = {}
    for node in nodes:
        dfs(node, upstream)
    return upstream.keys()


def is_in(node, nodes):
    for p in nodes:
        if id(node) == id(p) or node.id == p.id:
            return True
    return False

def downstream_nodes(roots,nodes,always_include=[]):
    downstream = []
    for node in nodes:
        if is_in(node,roots):
            continue
        if is_in(node,always_include):
            downstream.append(node)
            continue
        for p in node.parents:
            if is_in(p,downstream+roots):
                downstream.append(node)
                break
    return downstream

def middle_nodes(roots,leafs):
    "all nodes that are both upstream of a leaf and downstream of a root"
    # TODO: this goes through the whole graph and then filters, could be inefficient but whatever
    nodes = upstream_nodes(leafs)
    middle = {}
    for node in nodes:
        if is_in(node,leafs):
            continue
        for p in node.parents:
            if p in middle or is_in(p,roots):
                middle[node] = None # unordered set
    return list(middle.keys())


################################################################################
# The basic random variable class
################################################################################

def makerv(a):
    """
    If a node is a RV return it, else transform into a constant.
    The purpose of this is to make it possible to do things like
    x = d.norm(0,1)
    instead of
    x = d.norm(Constant(0),Constant(1))
    """

    if isinstance(a,RV):
        return a
    else:
        return Constant(a)


class RV(Node):
    """
    Abstract random variable class. An RV does these thing:
    - It has a unique id
    - It knows it's shape
    - It offers semantic sugar for infix operators (a+b insted of t.add(a,b))
    - It remembers if it can be conditioned or not
    """

    allid = 0 # class variable used to number nodes

    # here (and in all RV constructors) the parents are the non-keyword arguments
    # these are expected to all be random variables
    # everything else is keyword only
    def __init__(self,*parents,shape,can_condition):
        # force all parents to be RVs, casting to constants if needed
        new_parents = [makerv(p) for p in parents]
        super().__init__(*new_parents)
        self.shape = shape
        self.id = RV.allid
        self.can_condition=can_condition
        RV.allid += 1

    def __add__(self,b):
        return t.add(self,b)

    __radd__ = __add__

    def __mul__(self,b):
        return t.mul(self,b)

    __rmul__ = __mul__

    def __sub__(self,b):
        return self + (-1 * b)

    def __rsub__(self,b):
        return b + (-1 * self)

    def __pow__(self,b):
        "JAGS doesn't allow do x**y if x<0 and y non-integer"
        return t.pow(self,b)

    def __rpow__(self,a):
        return t.pow(a,self)

    def __truediv__(self,b):
        return t.div(self,b)

    def __rtruediv__(self,b):
        return t.div(b,self)

    # need to still make dicts and stuff work even when we overload equality
    def __hash__(self):
        #val = str(self).__hash__()
        val = id(self)
        #print('hashing',val)
        return val

    def __eq__(self,b):
        return Equality(self,b)

    def __ge__(self,b):
        b = makerv(b)
        assert self.shape==(), "can only compare scalars"
        assert b.shape==(), "can only compare scalars"
        return Equality(dinterval(self,b),1)

    def __le__(self,b):
        b = makerv(b)
        assert self.shape==(), "can only compare scalars"
        assert b.shape==(), "can only compare scalars"
        return Equality(dinterval(b,self),1)

    def __getitem__(self,idx):
        #assert len(self.shape)==1, "can only index into 1-D vectors"
        return VectorIndex(self,idx)

    def __matmul__(self,B):
        B = makerv(B)
        return t.matmul(self,B)

    #def condition(self,value):
    #    raise Exception("Trying to condition non-conditionable value")
    def condition(self,value):
        """
        conditioning takes a value and returns a list of (node,value) pairs
        the purpose of this is to "re-assign" conditioning in cases of deterministic
        variables, etc
        """
        # default conditioning, needs to be overridden sometimes
        if self.can_condition:
            assert self.shape == np.array(value).shape
            return [(self,value)]
        else:
            if not self.can_condition:
                raise Exception("trying to condition non-conditionable variable: " + str(type(self)))

    def __repr__(self):
        typestr = str(type(self))
        typestr = typestr[11:-2]
        ret = typestr + '('
        stuff = vars(self)
        for a in stuff:
            if not a.startswith('__'):
                thing = repr(stuff[a])
                if len(thing) < 10:
                    ret += a + '=' + repr(stuff[a]) + ', '
                #ret += a + '=' + repr(self.__getattribute__(a)) + ','
                #ret += a + '=' + self.__getattribute__(a).repr() + ','
        ret = ret[:-2] + ')'
        return ret

    def var_name(self):
        # start and end with v to make sure no partial matches
        return 'v'+str(self.id)+'v'

    def ref(self,loop_indices=[],local_indices=[]):
        # a function to get the appropriate reference for a variable
        # that has a given shape and has some number of loop indices attached

        if self.shape == () and loop_indices == []:
            assert local_indices==[]
            return self.var_name()

        ret = self.var_name() + '['

        # add stuff for loop indices
        for ii in loop_indices:
            ret += ii + ','
        
        # add stuff for shape of this node itself
        for n,j in enumerate(self.shape):
            # if an index string was given, use it
            if n < len(local_indices):
                ret += local_indices[n] + ','
            else:
                # otherwise, take all
                ret += '1:' + str(j)+','
        if ret.endswith(','):
            ret = ret[:-1]
        
        ret += ']'

        return ret

    @property
    def T(self):
        "take transpose like x.T like in numpy"
        return t.t(self)

    @property
    def ndim(self):
        "allows x.ndim instead of len(x.shape)"
        return len(self.shape)

################################################################################
# A constant is a special type of random variable
################################################################################

class Constant(RV):
    """
    An RV for a fixed constant
    (No parents allowed)
    """

    def __init__(self,value):
        self.value = np.array(value)
        assert (self.value.dtype == np.float64) or (self.value.dtype == np.int64)
        super().__init__(shape=self.value.shape,can_condition=False) # no parents
    
    def getcode(self,*args,**vargs):
        raise Exception("should not generate code for Constant nodes")

    def __repr__(self):
        return "Constant(" + str(self.value) + ")"

################################################################################
# An operator is used to define the basic distributions and transformations
################################################################################

def slice_string(shape,varname):
    # given a shape, get the string to represent index i in first dimension and entire rest of array
    if shape == ():
        return '' # no subsetting needed!
    ret = '[' + varname + ','
    for j in shape[1:]:
        ret += '1:' + str(j)+','
    ret = ret[:-1]+']'
    return ret    
     
class Operator(RV):
    op_str = None # this is an abstract class, shouldn't be directly created

    def __init__(self, *parents, fun_str, shape, can_condition):
        self.fun_str = fun_str
        super().__init__(*parents,shape=shape,can_condition=can_condition)

    # in order to generate code, an operator needs to know four things:
    # 1. What variable name to use to put the output
    # 2. What variables names to use to look up the parents
    # 3. What loop indices should be applied to the output
    # 4. What loop indices should be applied to each of the parents
    # all of these are just strings

    def getcode(self,loop_indices=None,par_loop_indices=None):
        assert self.op_str is not None, "shouldn't directly instantiate Operator"

        if loop_indices is None:
            loop_indices = []
        if par_loop_indices is None:
            par_loop_indices = [[] for i in self.parents]
        assert len(par_loop_indices) == len(self.parents)
        code = self.ref(loop_indices)
        code += self.op_str
        code += self.fun_str
        code += '('
        
        for n in range(len(self.parents)):
            code += self.parents[n].ref(par_loop_indices[n])
            if n < len(self.parents)-1:
                code += ','
        code += ');\n'

        return code

# almost the same except generate code slightly differently
class InfixOperator(Operator):
    def getcode(self,loop_indices=None,par_loop_indices=None):
        if loop_indices is None:
            loop_indices = []
        if par_loop_indices is None:
            par_loop_indices = [[] for i in self.parents]
        assert len(par_loop_indices) == len(self.parents)
        code = self.ref(loop_indices)
        code += self.op_str
        code += '('
        code += self.parents[0].ref(par_loop_indices[0])
        code += ')'
        code += self.fun_str
        code += '('
        code += self.parents[1].ref(par_loop_indices[1])
        code += ');\n'
        return code

################################################################################
# Generic Distribution types
################################################################################

class Distribution(Operator):
    op_str = '~'

    # initializer just serves to choose can_condition
    def __init__(self, *parents, fun_str,shape):
        super().__init__(*parents,fun_str=fun_str,shape=shape,can_condition=True)

class MultivariateDistribution(Distribution):
    # a special class for native JAGS multivariate distributions
    # We only need this to make sure user doesn't try to index the distribution
    # and then condition it (which isn't allowed)
    pass

class AllScalarDistribution(Distribution):
    "convenience class where all parents are assumed to be scalar"
    def __init__(self, *parents, fun_str,has_quantiles=False):
        parents = [makerv(p) for p in parents]
        #parents_shapes = [()]*len(parents)
        shape = ()
        self.has_quantiles = has_quantiles
        super().__init__(*parents,fun_str=fun_str,shape=shape)

    def quantile(self,val):
        "quantile (inverse cdf). only exists for a somewhat random set of scalar dists"
        assert self.shape == (), "quantiles only for scalar dists"
        # replace d with q
        return ScalarDeterministic(val,*self.parents,fun_str='q' + self.fun_str[1:])

    def cdf(self,val):
        "cdf. only exists for a somewhat random set of scalar dists"
        assert self.shape == (), "cdfs only for scalar dists"
        # replace d with p
        return ScalarDeterministic(val,*self.parents,fun_str='p' + self.fun_str[1:])

################################################################################
# Actual distributions
################################################################################

# class as a hack for a namespace
class d:
    # TODO: maybe someday random variables could have a known domain and we could check that they match the inputs
    # TODO: better docstrings

    ################################################################################
    # Contiunous univariate (sec 9.2.1 of JAGS user manual)
    ################################################################################

    # a somewhat random subset of these provide quantiles and CDFs
    # (table 6.2 of JAGS user manual)

    def beta(a,b):
        "beta"
        return AllScalarDistribution(a,b,fun_str='dbeta',has_quantiles=True)
    def chisqr(a):
        return AllScalarDistribution(a,fun_str='dchisqr',has_quantiles=True)
    def dexp(a,b):
        "double eponential (or laplace)"
        return AllScalarDistribution(a,b,fun_str='ddexp',has_quantiles=True)
    def exp(a):
        return AllScalarDistribution(a,fun_str='dexp',has_quantiles=True)
    def f(a,b):
        return AllScalarDistribution(a,b,fun_str='df',has_quantiles=True)
    def gamma(a,b):
        return AllScalarDistribution(a,b,fun_str='dgamma',has_quantiles=True)
    def gen_gamma(a,b,c):
        return AllScalarDistribution(a,b,c,fun_str='dgen.gamma',has_quantiles=True)
    def logis(a,b):
        "logistic distribution"
        return AllScalarDistribution(a,b,fun_str='dlogis',has_quantiles=True)
    def lnorm(a,b):
        "log normal"
        return AllScalarDistribution(a,b,fun_str='dlnorm',has_quantiles=True)
    def nchisqr(a,b):
        "Non-central chi-squared"
        return AllScalarDistribution(a,b,fun_str='dnchisqr',has_quantiles=True)
    def nt(a,b):
        "Non-central t"
        return AllScalarDistribution(a,b,fun_str='dnt')
    def norm(a,b):
        "Normal distribution HEY HI HELLO JAGS USES PRECISION **NOT VARIANCE**"
        return AllScalarDistribution(a,b,fun_str='dnorm',has_quantiles=True)
    def par(a,b):
        "pareto"
        return AllScalarDistribution(a,b,fun_str='dpar',has_quantiles=True)
    def t(a,b,c):
        "student t"
        return AllScalarDistribution(a,b,c,fun_str='dt',has_quantiles=True)
    def unif(a,b):
        "uniform"
        return AllScalarDistribution(a,b,fun_str='dunif')
    def web(a,b):
        "Weibull"
        return AllScalarDistribution(a,b,fun_str='dweib',has_quantiles=True)

    # convenience distributions not in JAGS

    def cauchy(a,b):
        "cauchy distribution (just calls t)"
        return d.t(a,b,1)

    ################################################################################
    # Discrete univariate (sec 9.2.2 of JAGS user manual)
    ################################################################################

    def bern(a):
        "Bernoulli"
        return AllScalarDistribution(a,fun_str='dbern',has_quantiles=True)

    def bin(a,b):
        "Binomial"
        return AllScalarDistribution(a,b,fun_str='dbin',has_quantiles=True)
    
    def cat(a):
        "Categorical"
        # because JAGS is 1-indexed but python is zero-indexed we must subtract 1
        # unfortunately this breaks Vmap
        a = makerv(a)
        assert len(a.shape)==1, "categorical takes 1-D vector input"
        return Distribution(a,fun_str='dcat',shape=())-1

    def hyper(a,b,c,d):
        "noncenteral hypergeometric: first three inputs should be intergers, last continuous"
        return AllScalarDistribution(a,b,c,d,fun_str='dhyper',has_quantiles=True)

    def negbin(a,b):
        "negative binomial"
        return AllScalarDistribution(a,b,fun_str='dnegbin',has_quantiles=True)
    
    def pois(a):
        "Poisson"
        return AllScalarDistribution(a,fun_str='dpois',has_quantiles=True)

    ################################################################################
    # Multivariate (sec 9.2.3 of JAGS user manual)
    ################################################################################

    def dirch(a):
        "Dirichlet"
        a = makerv(a)
        assert len(a.shape)==1, "dirichlet takes 1-D vector input"
        return MultivariateDistribution(a,fun_str='ddirch',shape=a.shape) # output has same shape

    def mnorm(a,b):
        "mutivariate normal - NOTE: b is a PRECISION MATRIX"
        a = makerv(a) # RV.__init__ does this anyway but want to fool with shapes
        b = makerv(b) 
        assert len(a.shape)==1
        N = a.shape[0]
        assert b.shape == (N,N)
        return MultivariateDistribution(a,b,fun_str='dmnorm',shape=(N,))

    def mnorm_vcov(a,b):
        "multivariate normal with covariance parameterization"
        a = makerv(a)
        b = makerv(b)
        assert len(a.shape)==1
        N = a.shape[0]
        assert b.shape == (N,N)
        return MultivariateDistribution(a,b,fun_str='dmnorm.vcov',shape=(N,))
    
    def mt(a,b,c):
        "Multivariate t"
        a = makerv(a) # only so we can fool with sizes
        b = makerv(b)
        c = makerv(c)
        assert len(a.shape)==1
        N = a.shape[0]
        assert b.shape == (N,N)
        assert c.shape == ()
        return MultivariateDistribution(a,b,c,fun_str='dmt',shape=(N,))

    def multi(a,b):
        "Multinomial"
        a = makerv(a)
        b = makerv(b)
        assert len(a.shape)==1
        N = a.shape[0]
        assert N>0
        assert b.shape==()
        return MultivariateDistribution(a,b,fun_str='dmulti',shape=(N,))

    def sample(a,b):
        "sampling without replacement" # JAGS user manual has wrong title
        a = makerv(a)
        b = makerv(b)
        assert len(a.shape)==1
        N = a.shape[0]
        assert N>0
        assert b.shape==()
        return MultivariateDistribution(a,b,fun_str='dsample',shape=(N,))

    def wish(a,b):
        "Wishart"
        a = makerv(a)
        b = makerv(b)
        assert len(a.shape)==2
        assert a.shape[0] == a.shape[0], "Wishart takes PSD input"
        assert b.shape == (), "Wishart takes scalar second input"
        N = a.shape[0]
        return MultivariateDistribution(a,b,fun_str='dwish',shape=(N,N))


    # Special distribution (own module)
    def normmix(a,b,c):
        "mixture of 1-D normals"
        a = makerv(a)
        b = makerv(b)
        c = makerv(c)
        assert len(a.shape)==1
        assert a.shape==b.shape
        assert a.shape==c.shape
        return Distribution(a,b,c,fun_str='dnormmix',shape=())


################################################################################
# Special fake "distribution" (users could but shouldn't touch this)
################################################################################

# TODO: allow these to be any shape
# TODO: make sure _geq_ and _leq_ work accordingly
# TODO: and make sure I works accordingly
def dinterval(a,b):
    assert a.shape == ()
    assert b.shape == ()
    return Distribution(a,b,fun_str='dinterval',shape=())


################################################################################
# Generic determnistic transformation stuff
################################################################################

class Deterministic(Operator):
    op_str = '<-'
    
    # initializer just serves to choose can_condition
    def __init__(self,*parents,fun_str,shape):
        # TODO: special cases where you can condition (e.g. sum)
        super().__init__(*parents,fun_str=fun_str,shape=shape, can_condition=False)

def vectorize_parents(parents):
    """
    Check that a set of nodes are mutually vectorizable. This means that either:
    - All are scalar
    - All are vectors of the same chape
    - Some are scalars and the rest are vectors of the same shape
    If successful, gives the new shape
    You could save efficiency by returning the parents in rv form
    """
    # JAGS user manual:
    # "Scalar functions taking scalar arguments are automatically vectorized.
    # They can also be called when the arguments are arrays with conforming
    # dimensions, or scalars."

    # turn parents into rvs (so we can check shapes)
    parents = [makerv(p) for p in parents]

    shape = ()
    for p in parents:
        if p.shape==():
            continue # always OK
        elif p.shape == shape:
            continue # already match existing shape
        elif p.shape != () and shape == ():
            shape = p.shape # update existing shape to match
            continue
        else:
            raise Exception("all parents must be scalars or vectors of same size")
    return shape


class ScalarDeterministic(Deterministic):
    def __init__(self, *parents, fun_str):
        shape = vectorize_parents(parents)

        super().__init__(*parents,fun_str=fun_str,shape=shape)

class InfixScalarDeterministic(InfixOperator,ScalarDeterministic):
    pass

# Basically a class for matrix multiplication
class InfixDeterministic(InfixOperator,Deterministic):
    pass


################################################################################
# Concrete transformations
################################################################################

# evil hack function to programatically generate all the scalar function code below...
def generate_scalar_funs():
    scalar_fun_names = ['abs','arccos','arccosh','arcsin','arcsinh','arctan','arctanh','cos','cosh','cloglog','exp','icloglog','ilogit','log','logfact','loggam','logit','phi','probit','round','sin','sinh','sqrt','step','tan','tanh','trunc']
    for name in scalar_fun_names:
        print("    def "+name+"(a):")
        print("        return ScalarDeterministic('"+name+"',a)")

class t:

    def sum(a):
        a = makerv(a)
        assert len(a.shape)==1, "can only sum 1-D vectors"
        return Deterministic(a,fun_str='sum',shape=())

    def pow(a,b):
        return ScalarDeterministic(a,b,fun_str='pow')
    def equals(a,b):
        return ScalarDeterministic(a,b,fun_str='equals')
    def ifelse(x,a,b):
        assert isinstance(x,Equality)
        return ScalarDeterministic(I(x),a,b,fun_str='ifelse')

    ################################################################################
    # Simple scalar->scalar transformations
    ################################################################################

    # all the functions from table 9.1 in JAGS user manual except 'equals' 'ifelse' and 'pow' (above)

    def abs(a):
        return ScalarDeterministic(a,fun_str='abs')
    def arccos(a):
        return ScalarDeterministic(a,fun_str='arccos')
    def arccosh(a):
        return ScalarDeterministic(a,fun_str='arccosh')
    def arcsin(a):
        return ScalarDeterministic(a,fun_str='arcsin')
    def arcsinh(a):
        return ScalarDeterministic(a,fun_str='arcsinh')
    def arctan(a):
        return ScalarDeterministic(a,fun_str='arctan')
    def arctanh(a):
        return ScalarDeterministic(a,fun_str='arctanh')
    def cos(a):
        return ScalarDeterministic(a,fun_str='cos')
    def cosh(a):
        return ScalarDeterministic(a,fun_str='cosh')
    def cloglog(a):
        return ScalarDeterministic(a,fun_str='cloglog')
    def exp(a):
        return ScalarDeterministic(a,fun_str='exp')
    def icloglog(a):
        return ScalarDeterministic(a,fun_str='icloglog')
    def ilogit(a):
        return ScalarDeterministic(a,fun_str='ilogit')
    def log(a):
        return ScalarDeterministic(a,fun_str='log')
    def logfact(a):
        return ScalarDeterministic(a,fun_str='logfact')
    def loggam(a):
        return ScalarDeterministic(a,fun_str='loggam')
    def logit(a):
        return ScalarDeterministic(a,fun_str='logit')
    def phi(a):
        return ScalarDeterministic(a,fun_str='phi')
    def probit(a):
        return ScalarDeterministic(a,fun_str='probit')
    def round(a):
        return ScalarDeterministic(a,fun_str='round')
    def sin(a):
        return ScalarDeterministic(a,fun_str='sin')
    def sinh(a):
        return ScalarDeterministic(a,fun_str='sinh')
    def sqrt(a):
        return ScalarDeterministic(a,fun_str='sqrt')
    def step(a):
        return ScalarDeterministic(a,fun_str='step')
    def tan(a):
        return ScalarDeterministic(a,fun_str='tan')
    def tanh(a):
        return ScalarDeterministic(a,fun_str='tanh')
    def trunc(a):
        return ScalarDeterministic(a,fun_str='trunc')

    ################################################################################
    # Special case needed for infix operators
    ################################################################################

    def add(a,b):
        return InfixScalarDeterministic(a,b,fun_str='+')
        #return ScalarDeterministic('sum',a,b)
        #return ScalarDeterministic('sum',a,b)

        # you could try to make a special observable function for add (like below)
        # but in my experience it only works in very special cases
        # better to just forget about it

        # # VERY SPECIAL transformation: an "observable" function
        # a = makerv(a)
        # b = makerv(b)
        # shape = vectorize_parents([a,b])
        # return Distribution('sum',a,b,shape=shape)

    def div(a,b):
        return InfixScalarDeterministic(a,b,fun_str='/')

    def mul(a,b):
        return InfixScalarDeterministic(a,b,fun_str='*')

    ################################################################################
    # Matrix functions
    ################################################################################

    def matmul(a,b):
        a = makerv(a) # for shapes
        b = makerv(b)
        if len(a.shape)==1 and len(b.shape)==1:
            assert a.shape[0]==b.shape[0], "shapes dont match"
            newshape = ()
        elif len(a.shape)==2 and len(b.shape)==1:
            assert a.shape[1]==b.shape[0], "shapes dont match"
            newshape = (a.shape[0],)
        elif len(a.shape)==1 and len(b.shape)==2:
            assert a.shape[0]==b.shape[0], "shapes dont match"
            newshape = (b.shape[1],)
        elif len(a.shape)==2 and len(b.shape)==2:
            assert a.shape[1]==b.shape[0], "shapes dont match"
            newshape = (a.shape[0],b.shape[1])
        else:
            print("FAIL")
            raise Exception("sizes don't make sense for matmul")
        
        return InfixDeterministic(a,b,fun_str='%*%',shape=newshape)    

    def t(a):
        "transpose"
        a = makerv(a)
        assert len(a.shape)==2, "can only transpose 2-d RVs"
        N,M = a.shape
        shape = (M,N)
        return Deterministic(a,fun_str='t',shape=shape)


################################################################################
# Special operator to take an equality (perhaps coming from and inequality, and
# transform it back into a random variable)
################################################################################

def I(eq):
    return t.equals(eq.dist,eq.const)

################################################################################
# Indexing
################################################################################

# TODO: N-D indexing

def slice_to_const(idx,N):
    return Constant(range(N)[idx])

class VectorIndex(RV):
    def __init__(self,parent,idx):
        # idx should be a scalar - it IS ALLOWED to be a random variable
        assert len(parent.shape)==1, "can only index vectors (for now)"

        # you can index using
        # 1: A slice (must be fixed integers)
        # 2: An array (can be RVs)
        # 3: An integer (can be RVs)

        if isinstance(idx,slice):
            # slices must be FIXED INTEGERS
            #idx = Constant(slice_to_range(idx,parent.shape[0]))
            #idx = Constant(range(parent.shape[0])[idx])
            idx = slice_to_const(idx,parent.shape[0])

        idx = makerv(idx)

        ## TODO: process negatives (this code should work)
        # if isinstance(idx,Constant):
        #     if idx.value < 0:
        #         idx = makerv(parent.shape[0]+idx.value)

        if idx.shape==():
            pass
            # can use whatever as a scalar index
        elif len(idx.shape)==1:
            assert isinstance(idx,Constant), "vector indices must be constant"
        else:
            raise Exception("Indices must be scalars or vectors")

        # TODO: this ain't right. Should inherit can_condition
        # Conditioning is complicated
        # - is parent multivariate
        # - is parent a distribution
        # - is parent deterministic

        if isinstance(parent,MultivariateDistribution):
            can_condition=False
        else:
            can_condition=parent.can_condition

        shape = idx.shape
        super().__init__(parent,idx,shape=shape,can_condition=can_condition)
        # even though idx might be constant we include it as a parent so it will be found in the DAG

    def getcode(self,loop_indices=None,par_loop_indices=None):        
        #print("Vmapper | loop_indices    ",self.id,self.parents[0].id,loop_indices)
        #print("Vmapper | par_loop_indices",self.id,self.parents[1].id,par_loop_indices)

        parent = self.parents[0]
        idx = self.parents[1]

        if loop_indices is None:
            loop_indices = []
        if par_loop_indices is None:
            par_loop_indices = [[] for i in self.parents]

        assert len(par_loop_indices)==2

        code = self.ref(loop_indices,[])
        code += '<-'
        local_indices = [idx.ref(par_loop_indices[1])+'+1'] # JAGS 1 indexed
        code += parent.ref(par_loop_indices[0],local_indices)
        code += ';\n'
        #print('Vmapper | ' + code)
        return code

    def condition(self, value):
        """
        conditioning takes a value and returns a list of (node,value) pairs
        the purpose of this is to "re-assign" conditioning in cases of deterministic
        variables, etc
        """

        pardist = self.parents[0]
        idx = self.parents[1]

        if not self.can_condition:
            raise Exception("trying to condition non-conditionable variable: " + str(type(self)))
        # if isinstance(pardist,MultivariateDistribution):
        #     raise Exception("can't condition on index of a native JAGS multivariate RV")

        # JAGS doesn't allow you to observe deterministic nodes.
        # So we observe parent with hidden entries
        # - not allowed for native Multivariate dists)
        # later combine these into a single array
        # - this isn't super efficient)

        parvalue = np.zeros(self.parents[0].shape) + np.nan
        assert isinstance(idx,Constant), "can't condition value of random index"
        parvalue[idx.value] = value
        return [(pardist,parvalue)]

################################################################################
# Stuff for constraints
################################################################################

class Equality:
    def __init__(self,dist,const):

        self.dist = makerv(dist)
        self.const = makerv(const)
        if self.dist.shape != self.const.shape:
            raise Exception("Size mis-match: " + str(self.dist.shape) + " vs " + str(self.const.shape))

class Given:
    def __init__(self,*eqs):
        for e in eqs:
            assert isinstance(e,Equality), "All arguments to Given must be Equality objects"
            assert isinstance(e.const,Constant), "Can only do Given(a==b) when b is a Constant"

        self.eqs = eqs

################################################################################
# Things to create "vectorized map" random variables (really just JAGS loops)
################################################################################

# fundamentally here's that this code does:
# - It creates "fake" parents corresponding to the sizes of a non-vectorized input
# - It runs the given function on those fake parents and traces the computation
# - It generates code for a for loop and injects indices to generate vectorized outputs

class Vmapper(RV):
    def __init__(self,*parents,fake_parent,vec_pars,N,looper=None):
        shape = (N,) + fake_parent.shape
        can_condition = fake_parent.can_condition
        super().__init__(*parents,shape=shape,can_condition=can_condition)

        if isinstance(fake_parent,(Vmapper,Scanner)):
            self.recursions = fake_parent.recursions+1
        else:
            self.recursions = 0
        self.fake_parent = fake_parent
        self.vec_pars = vec_pars

    def getcode(self,loop_indices=[],par_loop_indices=None):
        # indent is just to indent the code so it looks more pretty
        ii = 'i' + str(self.recursions)

        if par_loop_indices is None:
            par_loop_indices = [[] for l in self.parents]
        
        loop_indices2 = loop_indices + [ii]
        
        par_loop_indices2 = []
        for l,vec_par in zip(par_loop_indices,self.vec_pars):
            if vec_par:
                par_loop_indices2.append(l + [ii])
            else:
                par_loop_indices2.append(l)

        N = self.shape[0]
        # start for loop
        code = 'for ('+ii+' in 1:'+str(N)+'){\n'
        # actually generate the code
        mycode = self.fake_parent.getcode(loop_indices2,par_loop_indices2)
        # evil string replacement to change variable names
        mycode = mycode.replace(self.fake_parent.var_name(),self.var_name())
        for p1,p2 in zip(self.fake_parent.parents,self.parents):
            mycode = mycode.replace(p1.var_name(),p2.var_name())
        # add indentation just so JAGS code looks pretty
        mycode = '  ' + mycode.replace('\n','\n  ')[:-2]
        # add code
        code += mycode
        code += '}\n'
        return code

# special case for mapping a constant
def choose_mapper(*parents,fake_parent,vec_pars,N):
    if isinstance(fake_parent,Constant):
        # Rather than modifying the code we just reshape the array
        assert len(parents)==0
        assert vec_pars==[]
        X = fake_parent.value
        X = np.tile(X,[N,*([1]*X.ndim)])
        return makerv(X)
    else:
        return Vmapper(*parents,fake_parent=fake_parent,vec_pars=vec_pars,N=N)

def IID(node,N):
    node = makerv(node)
    "Make IID copies of a RV. This works for Deterministic and Constant nodes as well."
    return choose_mapper(*node.parents,fake_parent=node,vec_pars=[False]*len(node.parents),N=N)


class vmap:
    def __init__(self,f,vec_pars=None):
        self.f = f
        self.vec_pars=vec_pars
    
    def __call__(self,*parents):
        #print('self.vec_pars',self.vec_pars)
        if self.vec_pars is None:
            vec_pars = [True]*len(parents)
        else:
            vec_pars = self.vec_pars
        assert len(parents)==len(vec_pars), "number of arguments doesn't match"
        return trace_vmap(self.f,*parents,vec_pars=vec_pars)


def trace_vmap(f,*parents,vec_pars):
    parents = [makerv(p) for p in parents]

    assert len(vec_pars)==len(parents)

    # create fake parents (for those that get vectorized)
    fake_parents = []
    N = None # vectorization size
    for (p,vec_par) in zip(parents,vec_pars):
        if vec_par:
            if N is None:
                N = p.shape[0]
            else:
                assert p.shape[0]==N, "all vectorized parents must have same first dim"
            fake_parents.append(makerv(np.zeros(p.shape[1:])))
        else:
            fake_parents.append(p)

    # get the vectorization size
    if N is None:
        raise Exception("need at least one vectorized input")

    # run the function on fake parents
    fake_outputs = f(*fake_parents)

    # transform outputs into RVs (in case function returned constants)
    if isinstance(fake_outputs,Iterable):
        fake_outputs = [makerv(p) for p in fake_outputs]
        single_output = False
    else:
        fake_outputs = [makerv(fake_outputs)]
        single_output = True

    # get the nodes we need to process.
    fake_downstream = middle_nodes(roots=fake_parents,leafs=fake_outputs) + fake_outputs
    
    # now we want to go through all the downstream nodes and convert them to vectorized versions
    # but there is one caveat:
    # some of the PARENTS of these downstream nodes will be vectorized already and some won't

    alread_vectorized = {} # set of nodes that have already been vectorized
    real_nodes = {}        # map unvectorized nodes to vectorized ones

    # first of all, mark the parents as vectorized (or not)
    for (p,fp,vec_par) in zip(parents,fake_parents,vec_pars):
        real_nodes[fp] = p
        if vec_par:
            alread_vectorized[fp] = None # set

    # convert downstream nodes to (vectorized) versions
    for fake_node in fake_downstream:
        my_parents = []
        for fp in fake_node.parents:
            if fp in alread_vectorized:
               my_parents.append(real_nodes[fp]) # swap out this parent
            else:
               my_parents.append(fp)             # leave parent unvectorized
            
        my_vec_pars = [fp in alread_vectorized for fp in fake_node.parents]

        node = choose_mapper(*my_parents,fake_parent=fake_node,vec_pars=my_vec_pars,N=N)

        real_nodes[fake_node] = node
        alread_vectorized[fake_node] = None # set

    # return outputs
    new_nodes = [real_nodes[p] for p in fake_outputs]

    if single_output:
        return new_nodes[0]
    else:
        return new_nodes

################################################################################
# Things to create "scan" random variables (really just JAGS loops)
################################################################################

# the fundamental idea of this is similar to vmap: run the function on fake "slices",
# trace the computation, and then output a for loop and inject indices to make everything work

class ItemScanner(RV):
    # a random variable that will be scanned
    # unlike most cases, this doesn't generate it's own code; that is done by the parent Scan object
    # so this is sort of a dummy object; the only thing it has is a shape and an ID
    def __init__(self,*parents,fake_parent,vec_pars,N,tracer_avoid):
        shape = (N,) + fake_parent.shape
        # TODO: This is complicated... e.g. what if you scan over multivariate random variables
        # and then later take an index
        can_condition = fake_parent.can_condition
        super().__init__(*parents,shape=shape,can_condition=can_condition)
        self.fake_parent = fake_parent
        self.vec_pars = vec_pars

        for p in parents:
            #assert not hasattr(p,'tracer'), "should never have traced parents!"
            if hasattr(p,'tracer'):
                if p.tracer == tracer_avoid:
                    raise Exception('invalid tracer encountered')

class scan: # not a RV
    def __init__(self,f):
        self.f = f
    def __call__(self,*args):
        assert len(args)>=2, "scan call should take at least two arguments"
        return Scanner(*args,f=self.f)

# TODO: allow multiple outputs (would require significant re-doing since then has no simple shape
class Scanner(RV):

    def __init__(self,init,*parents,f):
        init = makerv(init)
        parents = [makerv(p) for p in parents]

        fake_old_carry = makerv(np.zeros(init.shape))
        fake_parents    = [makerv(np.zeros(p.shape[1:])) for p in parents]

        tracer = np.random.rand()
        fake_old_carry.tracer = tracer
        #fake_parent.tracer    = tracer
        for p in fake_parents:
            p.tracer = tracer

        # TODO: make sure all parents have same first dim
        
        # get vectorization size
        N = parents[0].shape[0]

        # run the function on fake inits
        fake_carry, fake_output = f(fake_old_carry,*fake_parents)
        
        fake_carry = makerv(fake_carry)
        fake_output = makerv(fake_output)

        if isinstance(fake_carry,Constant) and fake_carry.value.shape == (0,):
            raise Exception("must produce some carry value (can be ignored)")

        if fake_carry.shape != init.shape:
            raise Exception("init value must match shape of carry")

        # ugly hack to deal with a special case (make sure it can get vectorized)
        # TODO: remove hack
        if id(fake_output) == id(fake_old_carry):
            fake_output = fake_output + 0
        
        # system gets confused when output is a parent of carry or vice versa
        if id(fake_output) in [id(p) for p in fake_carry.parents]:
            fake_output = fake_output + 0
        #if id(fake_carry) in [id(p) for p in fake_output.parents]:
        #    fake_carry = fake_carry + 0

        # system also gets confused when carry is same as output
        if id(fake_carry) == id(fake_output):
            fake_output = fake_output + 0

        # need to create vectorized variables for all intermediates
        fake_downstream = middle_nodes(roots=fake_parents+[fake_old_carry],leafs=[fake_carry,fake_output]) + [fake_carry,fake_output]

        # tricky thing: need an "old carry" to generate the nodes
        # but we need the references to refer to the "real" carry at the end
        old_carry = ItemScanner(*init.parents,fake_parent=init,N=N,vec_pars=[True,True],tracer_avoid=tracer)

        real_nodes = {}
        # don't put in input
        real_nodes[fake_old_carry] = old_carry
        #real_nodes[fake_parent] = parent
        for fp,p in zip(fake_parents,parents):
            real_nodes[fp] = p

        extra_parents = {} # variables used in the function that aren't explicitly passed

        # create real nodes - the notion of "parents" is wierd here because they are interwoven
        # but we maintain them ignoring that
        for fake_node in fake_downstream:
            my_parents = []
            my_vec_pars = []
            for fp in fake_node.parents:
                #my_parents.append(real_nodes[fp])
                if fp in real_nodes:
                    my_parents.append(real_nodes[fp])
                    my_vec_pars.append(True)
                else:
                    # these parents need to be reflected in DAG somehow...
                    my_parents.append(fp)
                    extra_parents[fp] = None # set
                    my_vec_pars.append(False)

            # carry node is one larger
            if id(fake_node) == id(fake_carry):
                myN = N+1
            else:
                myN = N

            real_nodes[fake_node] = ItemScanner(*my_parents,fake_parent=fake_node,vec_pars=my_vec_pars,N=myN,tracer_avoid=tracer)

            if id(fake_node)==id(fake_carry):
                carry = real_nodes[fake_node]
            if id(fake_node)==id(fake_output):
                output = real_nodes[fake_node]

        # it could be that fake_output is just fake_old_carry with no changes
        # this seems to break things

        # consistency with JAX would say that we should return the final carry value
        # but to make our abstractions work, we don't

        shape = (N,) + fake_output.shape

        # TODO: better can_condition
        #super().__init__(init,parent,*list(extra_parents.keys()),shape=shape,can_condition=True)
        super().__init__(init,*parents,*list(extra_parents.keys()),shape=shape,can_condition=True)

        # hard to figure out recusions...
        self.recursions = 0
        for fake_node in real_nodes: # TODO: real_nodes is a bad name...
            if isinstance(fake_node,(Vmapper,Scanner)):
                self.recursions = max(self.recursions,fake_node.recursions+1)

        self.output          = output
        self.carry           = carry
        self.old_carry       = old_carry

        self.fake_old_carry  = fake_old_carry
        #self.fake_parent     = fake_parent
        self.fake_parents    = fake_parents
        self.fake_carry      = fake_carry
        self.fake_output     = fake_output

        #self.fake_downstream = fake_downstream
        self.real_nodes      = real_nodes
        self.N = N
    
    def getcode(self,loop_indices=None,par_loop_indices=None):
        #if loop_indices is not None:
        #    raise NotImplementedError
        if loop_indices is None:
            loop_indices = []

        if par_loop_indices is None:
            par_loop_indices = [[] for l in self.parents]

        # what do we need to do with par_loop_indices?
        # it's pretty simple: when those nodes are referenced, add in the loop indices
        
        par_loop_indices_dict = {}
        for p,pli in zip(self.parents,par_loop_indices):
            par_loop_indices_dict[p] = pli
        #print('par_loop_indices',par_loop_indices, [[]]*len(self.parents))
        #print('parent ids:',[p.id for p in self.parents])

        init = self.parents[0]
        #parent = self.parents[1]
        #parents = self.parents[1]

        # TODO: better variable name?
        ii = 'i' + str(self.recursions)

        # initialize carry
        init_loop_indices = par_loop_indices[0]
        code = self.carry.ref(local_indices=['1'],loop_indices=loop_indices) + ' <- ' + init.ref(loop_indices=init_loop_indices) + ';\n'

        # start for loop
        code += "for("+ii+" in 1:"+str(self.N)+"){\n"
        
        # compute the loop indices for all nodes
        all_loop_indices = {}
        # always loop into old_carry, new_carry, input, and output
        all_loop_indices[self.fake_old_carry] = loop_indices + [ii]

        # real_nodes is a mapping from all the fake nodes (that need to be computed)
        # to all the real nodes

        for fake_node in self.real_nodes:

            node = self.real_nodes[fake_node]
            #assert isinstance(node,ItemScanner)

            if isinstance(fake_node,Constant):
                continue # don't generate code for constant nodes

            # when generating carry, generate for next iteration
            if id(fake_node)==id(self.fake_carry):
                loop_indices2 = loop_indices + [ii+'+1']
            else:
                loop_indices2 = loop_indices + [ii]

            all_loop_indices[fake_node] = loop_indices2

            par_loop_indices2 = []
            assert len(fake_node.parents) == len(node.vec_pars)
            for (p,vec_par) in zip(fake_node.parents,node.vec_pars):

                my_loop_indices = []

                if p in par_loop_indices_dict:
                    if par_loop_indices_dict[p] != []:
                        #print('this happens',p.id)
                        my_loop_indices += par_loop_indices_dict[p]
                    
                if p in self.real_nodes:
                    real_p = self.real_nodes[p]
                    if real_p in par_loop_indices_dict:
                        #print('this also happens',p.id)
                        my_loop_indices += par_loop_indices_dict[real_p]

                        if p in par_loop_indices_dict:
                            if par_loop_indices_dict[p] != []:
                                raise Exception("this shouldn't happen")

                if p in all_loop_indices:
                    my_loop_indices += all_loop_indices[p]
                elif vec_par:
                    my_loop_indices += [ii]
                else:
                    pass

                par_loop_indices2.append(my_loop_indices)
            
            mycode = fake_node.getcode(loop_indices2,par_loop_indices2)

            #print('fake node id', fake_node.id, ' parent ids',[p.id for p in fake_node.parents])
            
            # evil string replacement
            mycode = swap_ids(mycode,fake_node,node,'fake->node')
            
            assert len(fake_node.parents) == len(node.parents)
            for p1,p2 in zip(fake_node.parents,node.parents):
                mycode = swap_ids(mycode,p1,p2,'p1->p2')

            mycode = swap_ids(mycode,self.old_carry,self.carry,'old_carry->carry')

            mycode = swap_ids(mycode,self.output,self,'output->self')

            # add indentation just so JAGS code looks pretty
            mycode = '  ' + mycode.replace('\n','\n  ')[:-2]

            code += mycode

        code += '}\n'

        return code

def swap_ids(code,p1,p2,label=''):
    "take come code, and replace the ids for one node with another"
    if p1.var_name() in code:
        #print('swapping ', p1.var_name(), ' -> ', p2.var_name() , label)
        return code.replace(p1.var_name(),p2.var_name())
    else:
        return code

# convenience class for a scan when carry and output are same
class recurse: # not a RV
    def __init__(self,f):
        def myf(*args):
            out = f(*args)
            return out, out # return two copies of the same thing
        self.f = myf
    def __call__(self,*args):
        return scan(self.f)(*args)
    
################################################################################
# Things to do inference
################################################################################

def makestring(i,node):
    return 'v' + str(i)

def is_blank(code):
    return code == "model{\n}\n"

# TODO: make inference routines call this
def jags_code(var0):
    if isinstance(var0,Iterable):
        var0 = [makerv(v) for v in var0]
        #singleinput = False
        var = var0
    else:
        var0 = makerv(var0)
        #singleinput = True
        var = [var0]        

    nodes = upstream_nodes(var)

    # create mapping from nodes to variable name strings
    #var_str = {node: node.var_name() for i,node in enumerate(nodes)}

    #evidence = {}
    code = "model{\n"
    for node in nodes:
        if not isinstance(node,Constant):
            code += node.getcode()
    code += "}\n"
    return code


def sample(var0,given=None,*,init=None,niter=1000):

    if isinstance(var0,Iterable):
        var0 = [makerv(v) for v in var0]
        singleinput = False
        var = var0
    else:
        var0 = makerv(var0)
        singleinput = True
        var = [var0]        

    if given is None:
        given = Given()
    assert isinstance(given,Given)

    nodes = upstream_nodes(var + [e.dist for e in given.eqs])

    # create mapping from nodes to variable name strings
    var_str = {node: node.var_name() for i,node in enumerate(nodes)}

    evidence = {}
    code = "model{\n"
    for node in nodes:
        #if isinstance(node,Constant) or isinstance(node,Operator):
        if isinstance(node,Constant):
            evidence[node.var_name()] = node.value
        else:
            code += node.getcode()
    code += "}\n"

    for e in given.eqs:
        # equality is a list of equality objects
        # each equality object has e.dist and e.const
        cond = e.dist.condition(e.const.value)
        for node,val_i in cond:
            # need to check/merge multiple observations
            if var_str[node] in evidence:
                # TODO: assert shape matches expectations? 
                # multiple observations are OK, provided they are for a vector
                # and the non-hidden parts don't overlap
                # we need to do the merging here
                cur = evidence[var_str[node]]
                assert cur.shape != (), "redundant/inconsistent observations"
                assert np.all(np.isnan(cur) + np.isnan(val_i)), "overlapping observations"
                good = ~np.isnan(val_i)
                evidence[node.var_name()][good] = val_i[good]
            else:
                evidence[node.var_name()]=val_i

    var_names = [v.var_name() for v in var]

    if init is not None:
        inits = {var_str[e]:init[e] for e in init}
    else:
        inits = init

    # special case for blank code (JAGS doesn't know what to do with it)
    if is_blank(code):
        print('warning: no model code - (OK in principle...)')
        # replace code with dummy code
        code = "model{\ndummy<-1;\n}\n"
    
    samps = jags(code,var_names,inits=inits,niter=niter,**evidence)

    if singleinput:
        return samps[0]
    else:
        return samps

def E(var,given=None,**vargs):
    if isinstance(var,Iterable):
        #for v in var:
        #    assert isinstance(v,RV), "can only compute E for random variables"
        [var is makerv(v) for v in var]
        samps = sample(var,given,**vargs)
        return [np.mean(s,axis=0) for s in samps]
    else:
        var = makerv(var)
        assert isinstance(var,RV), "can only compute E for random variables"
        samps = sample(var,given,**vargs)
        return np.mean(samps,axis=0)        

def P(var,given=None,**vargs):
    # only handles single things
    assert isinstance(var,Equality)
    return E(I(var),given,**vargs)


def var(var,given=None,**vargs):
    if isinstance(var,Iterable):
        #for v in var:
        #    assert isinstance(v,RV), "can only compute E for random variables"
        samps = sample(var,given,**vargs)
        return [np.mean(s**2,axis=0) - np.mean(s,axis=0)**2 for s in samps]
    else:
        assert isinstance(var,RV), "can only compute E for random variables"
        samps = sample(var,given,**vargs)
        return np.mean(samps**2,axis=0) - np.mean(samps,axis=0)**2

    # TODO: surely better to work with raw samples

    # only handles scalars
    # assert isinstance(node,RV)
    # assert node.shape==(), "we only have variance for scalars"
    # u = E(node,given,**vargs)
    # return E(node**2,given,**vargs)-u**2

    # # this is working code to support iterable lists of nodes, but is it good?
    # if isinstance(node,Iterable):
    #     node_squared = [makerv(n)**2 for n in node]        
    #     # do inference together to make sure positive
    #     u_all = E(node + node_squared,given,**vargs)
    #     u = u_all[:len(u_all)//2]
    #     u2 = u_all[len(u_all)//2:]
    #     return [u2i - ui**2 for (u2i,ui) in zip(u2,u)]
    # else:
    #     #assert isinstance(node,RV)
    #     node = makerv(node)
    #     u,u2 = E([node,node**2],given,**vargs)
    #     return u2-u**2


# def var(node,given=None,**vargs):
#     # TODO: surely better to work with raw samples

#     # only handles scalars
#     # assert isinstance(node,RV)
#     # assert node.shape==(), "we only have variance for scalars"
#     # u = E(node,given,**vargs)
#     # return E(node**2,given,**vargs)-u**2

#     # this is working code to support iterable lists of nodes, but is it good?
#     if isinstance(node,Iterable):
#         node_squared = [makerv(n)**2 for n in node]        
#         # do inference together to make sure positive
#         u_all = E(node + node_squared,given,**vargs)
#         u = u_all[:len(u_all)//2]
#         u2 = u_all[len(u_all)//2:]
#         return [u2i - ui**2 for (u2i,ui) in zip(u2,u)]
#     else:
#         #assert isinstance(node,RV)
#         node = makerv(node)
#         u,u2 = E([node,node**2],given,**vargs)
#         return u2-u**2

def std(node,given=None,**vargs):
    # TODO: surely better to work with raw samples
    v = var(node,given,**vargs)
    if isinstance(v,list):
        return [vi**0.5 for vi in v]
    else:
        return v**0.5

def cov(node1,node2,given=None,**vargs):
    # TODO: surely better to work with raw samples
    #assert isinstance(node1,RV)
    #assert isinstance(node2,RV)
    node1 = makerv(node1)
    node2 = makerv(node2)
    assert node1.shape == (), "cov only for scalars"
    assert node2.shape == (), "cov only for scalars"
    #u1 = E(node1,given,**vargs)
    #u2 = E(node2,given,**vargs)
    #return E(node1*node2,given,**vargs)-u1*u2
    u1,u2,u12 = E([node1,node2,node1*node2],given,**vargs)
    return u12-u1*u2

# TODO: make this work
# def cov_matrix(node,given=None,**vargs):
#     assert isinstance(node,RV)
#     assert len(node.shape=1), "cov_matrix only for vectors"
#     N = node.shape[0]
#     outer = ###


def corr(node_a,node_b,given=None,**vargs):
    # TODO: surely better to work with raw samples
    #assert isinstance(node_a,RV)
    #assert isinstance(node_b,RV)
    node_a = makerv(node_a)
    node_b = makerv(node_b)
    assert node_a.shape == (), "corr only for scalars"
    assert node_b.shape == (), "corr only for scalars"
    # this is nice and simple but can give you nonsense
    #σ1 = std(node1,given,**vargs)
    #σ2 = std(node2,given,**vargs)
    #return cov(node1,node2,given,**vargs)/(σ1*σ2)
    ua,ub,uaua,uaub,ubub = E([node_a,node_b,node_a*node_a,node_a*node_b,node_b*node_b],given,**vargs)
    σa = (uaua - ua**2)**0.5
    σb = (ubub - ub**2)**0.5
    mycov = uaub - ua*ub
    return mycov / (σa*σb)

################################################################################
# Small standalone interface to call JAGS by writing files and
# calling the command line program
################################################################################

def read_coda(nchains):
    # first get all the variables

    f = open("CODAindex.txt")
    lines = f.readlines()
    var_start = {}
    var_end = {}
    nsamps = None
    for line in lines:
        varname, start, end = line.split(' ')
        var_start[varname] = int(start)-1
        var_end[varname] = int(end)

        my_nsamps = var_end[varname] - var_start[varname]
        if nsamps is not None:
            assert nsamps == my_nsamps, "assume # samps is same for all variables"
        nsamps = my_nsamps
    f.close()

    # now, collect things according to arrays
    scalar_vars = []
    vector_lb   = {}
    vector_ub   = {}
    for var in var_start:
        if '[' in var:
            varname, rest = var.split('[')

            num_str = var.split('[')[1].split(']')[0]
            nums = [int(s) for s in num_str.split(',')]
            if varname not in vector_lb:
                vector_lb[varname] = np.iinfo(np.int32).max + np.zeros(len(nums),dtype=int)
                vector_ub[varname] = np.iinfo(np.int32).min + np.zeros(len(nums),dtype=int)

            for i,num in enumerate(nums):
                vector_lb[varname][i] = min(vector_lb[varname][i],num)
                vector_ub[varname][i] = max(vector_lb[varname][i],num)
        else:
            scalar_vars.append(var)
    
    # vector_lb and vector_ub are now arrays with the lowest and highest index seen in each dim
    # this is so we can figure out the size

    data = [np.loadtxt("CODAchain" + str(chain+1) + ".txt") for chain in range(nchains)]
    def read_range(start,end):
        return np.concatenate([data[chain][start:end,1] for chain in range(nchains)])

    outs = {}
    for var in scalar_vars:
        outs[var] = read_range(var_start[var],var_end[var])
    for var in vector_lb:
        # get names including array indices
        shape = vector_ub[var]-vector_lb[var]+1
        outs[var] = np.zeros([nsamps,*shape])

        if len(shape)==1:
            for i in range(shape[0]):
                full_name = var + '[' + str(i+1) + ']'
                # likely inefficient to keep re-reading all the data...
                stuff = read_range(var_start[full_name],var_end[full_name])
                outs[var][:,i] = stuff
        elif len(shape)==2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # TODO: I thiiiinnnkk row-major vs column-major doesn't matter
                    full_name = var + '[' + str(i+1) + ',' + str(j+1) + ']'
                    stuff = read_range(var_start[full_name],var_end[full_name])
                    outs[var][:,i,j] = stuff
        elif len(shape)==3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        full_name = var + '[' + str(i+1) + ',' + str(j+1) + ',' + str(k+1) + ']'
                        stuff = read_range(var_start[full_name],var_end[full_name])
                        outs[var][:,i,j,k] = stuff
        else:
            assert False, "Only implemented scalar, vector, and 2-d matrix reads (but easy!)"

    return outs

#read_coda(4)

def jags(code,monitor_vars,*,inits=None,niter=10000,nchains=1,**evidence):
    assert hasattr(monitor_vars,'__iter__')
    
    # write the model to a file
    f = open("model.bug","w")
    f.write(code)
    f.close()

    # TODO: avoid repetition below and deal with more than 2 dimensions

    # write data
    f = open("data.R","w")
    for var in evidence:
        #assert var.shape==()
        data = np.array(evidence[var])
        if data.shape == ():
            f.write(var + " <- " + str(evidence[var]) + "\n")
        elif len(data.shape)==1:
            f.write(var + " <- " + vecstring(data) + '\n')
        elif len(data.shape)>=2:
            # TODO: can't be right: transpose won't work in ND
            f.write('`' + var + '`' + " <- structure(" + vecstring(data.T.ravel()) + ', .Dim=' + vecstring(data.shape) + ')\n')
        else:
            raise NotImplementedError("VAR" + str(data.shape))
    f.close()

    if True:#inits is not None:
        # write inits
        f = open("inits.R","w")
        f.write(".RNG.seed <- " + str(np.random.randint(0,10000000)) + "\n")
        f.write('.RNG.name <- "base::Wichmann-Hill"\n')
        if inits is not None:
            #for var in inits:
            #assert var.shape==()
            data = np.array(inits[var])
            if data.shape == ():
                f.write(var + " <- " + str(inits[var]) + "\n")
            elif len(data.shape)==1:
                f.write(var + " <- " + vecstring(data) + '\n')
            elif len(data.shape)>=2:
                f.write('`' + var + '`' + " <- structure(" + vecstring(data.T.ravel()) + ', .Dim=' + vecstring(data.shape) + ')\n')
            else:
                raise NotImplementedError("VAR" + str(data.shape))
        f.close()

    # write a script
    script = 'model in "model.bug"\n'
    if "dnormmix" in code: # load mixture module if needed
        script += "load mix\n"
    if evidence != {}:
        script += 'data in "data.R"\n'
    script += "compile, nchains(" + str(nchains) + ")\n"
    if True:#inits:
        script += 'parameters in "inits.R"\n'
    script += "initialize\n"
    script += "update " + str(niter) + "\n"
    for var in monitor_vars:
        script += "monitor " + var + "\n"
    script += "update " + str(niter) + "\n"
    script += "coda *\n"
    f = open("script.txt","w")
    f.write(script)
    f.close()

    # actually run it
    import subprocess
    try:
        output = subprocess.check_output(['jags', 'script.txt'], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        if "is a logical node and cannot be observed" in output:
            raise Exception("JAGS gave 'is a logical node and cannot be observed' error. Usually this is the result of trying to condition in ways that aren't supported.") from None

        if "Node inconsistent with parents" in output:
            raise Exception("JAGS gave 'Node inconsistent with parents' error. Often this is resolved by providing initial values") from None

        print('JAGS ERROR!')
        print('---------------------------------')
        print(output)
        print('---------------------------------')
        print('JAGS CODE')
        print('---------------------------------')
        print(code)
        print('---------------------------------')

        raise Exception("JAGS error (this is likely triggered by a bug in Pangolin)")
        #return [None for v in monitor_vars]

    # read in variable information
    f = open("CODAindex.txt")
    lines = f.readlines()
    var_start = {}
    var_end = {}
    for line in lines:
        varname, start, end = line.split(' ')
        var_start[varname] = int(start)-1
        var_end[varname] = int(end)
    f.close()

    outs = read_coda(nchains) # gets a dict
    #if len(monitor_vars)==1:
    #    return outs[monitor_vars[0]]
    #else:
    #    return [outs[v] for v in monitor_vars]
    return [outs[v] for v in monitor_vars]

################################################################################
# Little utility functions
################################################################################

def vecstring(a):
    """
    Given an input of a=np.array([1,2,3,4])
    returns "c(1,2,3,4)"
    only works for 1-d vectors
    """
    a = np.array(a)
    assert a.ndim==1
    ret = 'c('
    for i,ai in enumerate(a):
        if np.isnan(ai):
            ret += 'NA'
        else:
            ret += str(ai)
        if i < a.shape[0]-1:
            ret += ','
    ret += ')'
    return ret
