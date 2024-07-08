from . import dag, util

# import dag
import numpy as np
import jax
from .ir import *
from typing import Sequence

# valid unicode start characters: https://www.asmeurer.com/python-unicode-variable-names/start-characters.html

Ø = None
_ = None
º = None


def normal(loc, scale=None, prec=None):
    "1-d normals with multiple possible parameterizations"
    match (scale, prec):
        case (scale, None):
            return normal_scale(loc, scale)
        case (None, prec):
            return normal_prec(loc, prec)
        case _:
            raise Exception("must provide scale or prec but not both")


def sum(x, axis=None):
    x = makerv(x)
    sum_op = Sum(axis)
    return sum_op(x)


# Remember, transforms don't have any functionality!


# class Transform:
#     def get_shape(self, base_shape):
#         pass
#
#     def __call__(self, cond_dist):
#         print(f"{cond_dist=}")
#         assert isinstance(
#             cond_dist, CondDist
#         ), "transforms can only be called on CondDist objects"
#         return TransformedCondDist(cond_dist, self)
#
#
# class TransformedCondDist(CondDist):
#     "given base_dist(x|y) and transform(x) represent transform(x)|y"
#
#     def __init__(self, base_cond_dist, transform):
#         super().__init__("transformed")
#         self.base_cond_dist = base_cond_dist
#         self.transform = transform
#
#     def get_shape(self, *parent_shapes):
#         base_shape = self.base_cond_dist.get_shape(*parent_shapes)
#         new_shape = self.transform.get_shape(base_shape)
#         return new_shape
#
#     @property
#     def is_random(self):
#         return self.base_cond_dist.is_random
#
#     def __repr__(self):
#         return (
#             "transformed("
#             + repr(self.base_cond_dist)
#             + ","
#             + repr(self.transform)
#             + ")"
#         )
#
#     def __str__(self):
#         return str(self.transform) + " ∘ " + str(self.base_cond_dist)
#
#
# class ScalarTransform(Transform):
#     def __init__(self, name):
#         self.name = name
#
#     def get_shape(self, base_shape):
#         assert base_shape == ()
#         return ()
#
#     def __repr__(self):
#         return self.name
#
#     def __str__(self):
#         return self.name
#
#
# inverse_softplus = ScalarTransform("inverse_softplus")
# softplus = ScalarTransform("softplus")

################################################################################
# log prob
################################################################################


class InvalidAncestorQuery(Exception):
    pass


def log_prob(vars, given_vars=None):
    """
    Given some set of RVs, get a new RV that represents the conditional
    log-probability of those RVs. (Evaluated in simple "ancestor order")

    CURRENTLY EXPERIMENTAL, HAS VERY POOR BACKEND SUPPORT, DO NOT USE
    """

    # if any vars are upstream of given, then can't do

    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_given_vars, given_vars_treedef = jax.tree_util.tree_flatten(given_vars)

    upstream_of_given = dag.upstream_nodes(flat_given_vars)
    if any(n in upstream_of_given and not n in flat_given_vars for n in flat_vars):
        raise InvalidAncestorQuery("evaluated node upstream of given")

    nodes = dag.upstream_nodes(flat_vars, block_condition=lambda p: p in flat_given_vars)

    l = None

    for node in nodes:
        if node.cond_dist.random:
            # if found a node not in vars or given, then bad query
            if not (node in flat_vars or node in flat_given_vars):
                raise InvalidAncestorQuery("unexpected random node")

            my_l = LogProb(node.cond_dist)(node, *node.parents)
            if l is None:
                l = my_l
            else:
                l += my_l

        # evaluated[node] = None
    return l

################################################################################
# convenience functions to create Composite dists
################################################################################

def make_composite(fun,*input_shapes):
    """
    Inputs:
    - fun: A function that takes RVs as inputs and returns a single RV
    - *input_shapes: shapes for each explicit input to fun
    Outputs:
    - op: a Composite op representing the function
    - consts: a list of constants that fun captures as a closure
    The final op expects as inputs *first* all the elements of consts and then the explicit inputs
    """

    # TODO: take pytree of arguments
    dummy_args = [AbstractRV(shape) for shape in input_shapes]

    f = lambda *args: [fun(*args)] # vmap_generated_nodes runs on "flat" functoins
    all_vars, [out] = vmap_generated_nodes(f, *dummy_args)
    assert isinstance(out, RV), "output of function must be a single RV"

    print(f"{all_vars=}")
    print(f"{out=}")

    cond_dists = []
    par_nums = []
    linear_order = {}

    for var in dummy_args:
        assert isinstance(var, AbstractRV)
        linear_order[var] = dummy_args.index(var)

    current_position = len(dummy_args)

    consts = []
    for var in all_vars:
        for p in var.parents:
            if p not in all_vars and p not in linear_order:
                linear_order[p] = current_position
                current_position += 1
                consts.append(p)

    num_inputs = current_position

    for var in all_vars:
        my_cond_dist = var.cond_dist
        my_par_nums = [linear_order[p] for p in var.parents]
        cond_dists.append(my_cond_dist)
        par_nums.append(my_par_nums)
        linear_order[var] = current_position
        current_position += 1

    return Composite(num_inputs, cond_dists, par_nums), consts

def composite(fun):
    def myfun(*inputs):
        input_shapes = [x.shape for x in inputs]
        op, consts = make_composite(fun,*input_shapes)
        return op(*consts, *inputs)
    return myfun

################################################################################
# Autoregressive convenience function
################################################################################

def autoregressive(fun,length=None):
    """
    Convenience function for creating Autoregressive(Composite) RVs
    fun - function where 1st argument is recursive variable and other arguments are whatever.
          it is OK if fun implicitly uses existing random variables as a closure.
    length - length to be mapped over (optional unless there are no argments)
    """
    def myfun(init, *args):
        if length is None:
            if args == ():
                raise ValueError("autoregressive needs length if there are no mapped args")
            else:
                my_length = args[0].shape[0]
                for a in args:
                    assert a.shape[0] == my_length, "all args must have matching first dim"
        else:
            for a in args:
                assert a.shape[0] == length, "all args must have first dim matching length"
            my_length = length

        args_shapes = tuple(a.shape[1:] for a in args)
        input_shapes = (init.shape,) + args_shapes
        base_op, consts = make_composite(fun, *input_shapes)
        print(f"{consts=}")
        num_constants = len(consts)
        op = Autoregressive(base_op, length=my_length, num_constants=num_constants)
        return op(init,*consts,*args)
    return myfun



################################################################################
# Full-blown VMap
################################################################################


def vmap_dummy_args(in_axes, axis_size, *args):
    """
    Given a "full" arguments, get a list of "dummy" (AbstractRV) arguments
    """
    assert len(in_axes) == len(args)
    dummy_args = []
    for i, a in zip(in_axes, args):
        new_shape, new_axis_size = split_shape(a.shape, i)
        d = a.cond_dist
        if isinstance(d, VMapDist) and i == 0:
            my_dummy = AbstractRVWithDist(d.base_cond_dist, new_shape)
        else:
            my_dummy = AbstractRV(new_shape)
        dummy_args.append(my_dummy)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    return dummy_args, axis_size


def vmap_generated_nodes(f, *dummy_args):
    """
    Get all the nodes generated by a function
    This is tricky because that function might include captured closure variables.
    To deal with this we run the function twice on different copies of the arguments.
    Then, anything that's included in one but not the other is new.
    """

    for p in dummy_args:
        assert isinstance(p, AbstractRV)

    # f is assumed to be flat!
    dummy_output1 = f(*dummy_args)
    assert isinstance(dummy_output1, list), "vmap_eval must be called on flat functions"

    # TODO: This will search the WHOLE upstream DAG, should try to make more efficient
    dummy_nodes1 = dag.upstream_nodes(dummy_output1)

    dummy_output2 = f(*dummy_args)
    assert isinstance(dummy_output2, list), "vmap_eval must be called on flat functions"
    assert len(dummy_output1) == len(dummy_output2)
    for d1, d2 in zip(dummy_output1, dummy_output2):
        if d1 is None and d2 is None:
            continue
        assert d1.shape == d2.shape

    # def excluded_node(node):
    #    return node in dummy_nodes1

    # need to use ids instead because of new notion of RV equality
    dummy_ids1 = [id(x) for x in dummy_nodes1]

    def excluded_node(node):
        return id(node) in dummy_ids1

    dummy_nodes2 = dag.upstream_nodes(dummy_output2, block_condition=excluded_node)
    return tuple(dummy_nodes2), dummy_output2


def vmap_eval(f, in_axes, axis_size, *args):
    """
    actually evaluate a vmap.
    This function (but not vmap itself) works on "flat" function f, meaning that each
    argument of the function is just a RV. And the function must return
    a list of arguments which again are each just a RV.
    """

    args = list(makerv(a) for a in args)

    dummy_args, axis_size = vmap_dummy_args(in_axes, axis_size, *args)

    dummy_nodes, dummy_outputs = vmap_generated_nodes(f, *dummy_args)

    # TODO: disabled for now — figure out why!
    # I think the problem is that if inputs are passed directly to outputs then
    # they don't count as generated
    # for dummy_output in dummy_outputs:
    #    print(f"{dummy_output=}")
    #    assert dummy_output in dummy_nodes, f"{dummy_output} not in dummy_nodes"

    for dummy_arg in dummy_args:
        assert dummy_arg not in dummy_nodes

    dummy_to_real = util.WriteOnceDict()
    dummy_mapped_axis = util.WriteOnceDict()
    for dummy_arg, i, arg in zip(dummy_args, in_axes, args):
        dummy_to_real[dummy_arg] = arg
        dummy_mapped_axis[dummy_arg] = i

    # trouble: what about non-dummy arguments!

    for dummy_node in dummy_nodes:
        dummy_parents = dummy_node.parents
        # replace dummies with real nodes
        parents = [dummy_to_real[p] if p in dummy_to_real else p for p in dummy_parents]
        # lookup mapped axes or just use None
        my_in_axes = [
            dummy_mapped_axis[p] if p in dummy_mapped_axis else None
            for p in dummy_parents
        ]

        # slight optimization: don't vmap constant nodes (could disable)
        if isinstance(dummy_node.cond_dist, Constant) and not dummy_node in dummy_outputs:
            assert my_in_axes == []
            real_node = dummy_node
            dummy_to_real[dummy_node] = real_node
            dummy_mapped_axis[dummy_node] = None
        elif (
                all(axis is None for axis in my_in_axes)
                and not dummy_node.cond_dist.random
                and not dummy_node in dummy_outputs
        ):
            assert not isinstance(dummy_node, AbstractRV)
            real_node = RV(dummy_node.cond_dist, *parents)
            dummy_to_real[dummy_node] = real_node
            dummy_mapped_axis[dummy_node] = None
        else:
            cond_dist = VMapDist(
                dummy_node.cond_dist, in_axes=my_in_axes, axis_size=axis_size
            )
            real_node = RV(cond_dist, *parents)
            dummy_to_real[dummy_node] = real_node
            dummy_mapped_axis[dummy_node] = 0

    output = [dummy_to_real[dummy] for dummy in dummy_outputs]
    return output


class vmap:
    def __init__(self, f, in_axes=0, axis_size=None):
        self.f = f
        self.in_axes = in_axes
        self.axis_size = axis_size

    def __call__(self, *args):
        # no greedy casting because this leads to ambiguity
        # if the user sends [(1,2),(3,4)] is that a list of two
        # arrays?
        args = jax.tree_util.tree_map(makerv, args)

        # if isinstance(d, VMapDist) and i == 0:
        #     my_dummy = AbstractRVWithDist(d.base_cond_dist, new_shape)
        # else:
        #     my_dummy = AbstractRV(new_shape)

        def get_dummy(i, x):
            if i is None:
                new_shape = x.shape
            else:
                lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1:])
                new_shape = lo + hi

            # return AbstractRV(new_shape)
            if isinstance(x.cond_dist, VMapDist):
                return AbstractRVWithDist(x.cond_dist.base_cond_dist, new_shape)
            else:
                return AbstractRV(new_shape)

        dummy_args = util.tree_map_recurse_at_leaf(
            get_dummy, self.in_axes, args, is_leaf=util.is_leaf_with_none
        )
        new_in_axes = util.tree_map_recurse_at_leaf(
            lambda i, x: i, self.in_axes, dummy_args, is_leaf=util.is_leaf_with_none
        )

        tree1 = jax.tree_util.tree_structure(args, is_leaf=util.is_leaf_with_none)
        tree2 = jax.tree_util.tree_structure(dummy_args, is_leaf=util.is_leaf_with_none)
        tree3 = jax.tree_util.tree_structure(new_in_axes, is_leaf=util.is_leaf_with_none)
        assert tree1 == tree2
        assert tree1 == tree3

        flat_in_axes, axes_treedef = jax.tree_util.tree_flatten(
            new_in_axes, is_leaf=util.is_leaf_with_none
        )
        flat_f, flatten_inputs, unflatten_output = util.flatten_fun(
            self.f, *dummy_args, is_leaf=util.is_leaf_with_none
        )
        flat_args = flatten_inputs(*args)
        flat_output = vmap_eval(flat_f, flat_in_axes, self.axis_size, *flat_args)
        output = unflatten_output(flat_output)
        return output


def mixture(mixing_rv, fun):
    if mixing_rv.cond_dist == bernoulli:
        nvals = 2
    elif mixing_rv.cond_dist == categorical:
        nvals = mixing_rv.parents[0].shape[0]
    else:
        raise NotImplementedError(
            "currently can only handle bernoulli and categorical as mixing dist"
        )
    vmap_rv = vmap(fun, 0)(np.arange(nvals))
    mixing_dist = mixing_rv.cond_dist
    num_mixing_args = len(mixing_rv.parents)
    vmap_dist = vmap_rv.cond_dist
    mixture_dist = Mixture(mixing_dist, num_mixing_args, vmap_dist)
    mixture_rv = mixture_dist(*mixing_rv.parents, *vmap_rv.parents)
    return mixture_rv


class AbstractRV(RV):
    def __init__(self, shape):
        super().__init__(AbstractCondDist(shape))

    def __repr__(self):
        return "Abstract" + str(self.shape)

    def __str__(self):
        return str(self.shape)

    def __eq__(self, other):
        # DON'T use RV equality
        return dag.Node.__eq__(self, other)

    def __hash__(self):
        # DON'T use RV equality
        return dag.Node.__hash__(self)


# TODO: make this the only abstractRV type
class AbstractRVWithDist(AbstractRV):
    """
    Like a RV (has a cond_dist and shape) but no parents
    """

    def __init__(self, cond_dist, shape):  # noqa
        self._shape = shape
        self.cond_dist = cond_dist
        self.parents = []

    def __repr__(self):
        return f"AbstractDistRV({self.shape},{self.cond_dist})"

    def __str__(self):
        return repr(self)


class plate:
    def __init__(self, *args, N=None, in_axes=-1):
        self.in_axes = in_axes
        self.size = N
        self.args = tuple(makerv(a) for a in args)

    def __call__(self, f):
        assert hasattr(f, "__call__"), "must be callable"

        if self.in_axes == -1:
            if self.args == ():
                in_axes = None
            else:
                in_axes = 0
        else:
            in_axes = self.in_axes
        return vmap(f, in_axes, axis_size=self.size)(*self.args)


################################################################################
# printing stuff
################################################################################


def print_upstream_old(*vars):
    vars = jax.tree_util.tree_leaves(vars)
    nodes = dag.upstream_nodes(vars)

    if vars == []:
        print("[empty vars, nothing to print]")
        return

    # get maximum # parents
    max_pars = 0
    max_shape = 0
    for node in nodes:
        max_pars = max(max_pars, len(node.parents))
        max_shape = max(max_shape, len(str(node.shape)))

    digits = 1 + int(np.log10(len(nodes) - 1))
    par_str_len = (digits + 1) * max_pars - 1

    id = 0
    node_to_id = {}
    for node in nodes:
        par_ids = [node_to_id[p] for p in node.parents]

        par_id_str = util.comma_separated(par_ids, util.num2str, False)
        par_id_str = par_id_str + " " * (par_str_len - len(par_id_str))

        shape_str = str(node.shape)
        shape_str += " " * (max_shape - len(shape_str))

        print(
            util.num2str(id)
            + ": "
            + shape_str
            + " ["
            + par_id_str
            + "] "
            + str(node.cond_dist)
        )
        node_to_id[node] = id
        id += 1


def print_upstream(*vars):
    vars = jax.tree_util.tree_leaves(vars)
    nodes = dag.upstream_nodes(vars)

    if vars == []:
        print("[empty vars, nothing to print]")
        return

    # get maximum # parents
    max_pars = 0
    max_shape = 5
    for node in nodes:
        max_pars = max(max_pars, len(node.parents))
        max_shape = max(max_shape, len(str(node.shape)))

    if len(nodes) > 1:
        digits = 1 + int(np.log10(len(nodes) - 1))
        par_str_len = (digits + 1) * max_pars - 1
    else:
        par_str_len = 0

    id = 0
    node_to_id = {}  # type: ignore
    print(f"shape{' ' * (max_shape - 5)} | statement")
    print(f"{'-' * max_shape} | ---------")
    for node in nodes:
        assert isinstance(node,RV)

        par_ids = [node_to_id[p] for p in node.parents]

        par_id_str = util.comma_separated(par_ids, util.num2str, False)
        # par_id_str = par_id_str + " " * (par_str_len - len(par_id_str))

        shape_str = str(node.shape)
        shape_str += " " * (max_shape - len(shape_str))

        op = "~" if node.cond_dist.random else "="

        line = f"{shape_str} | {util.num2str(id)} {op} {str(node.cond_dist)}"
        if node.parents:
            line += "(" + par_id_str + ")"

        print(line)

        node_to_id[node] = id
        id += 1


def label_fn(observed_vars=(), labels=None):
    if labels is None:
        labels = {}

    def fn(node):
        style = "filled" if node in observed_vars else None

        if node in labels:
            label = labels[node] + ": " + str(node.cond_dist)
        else:
            label = str(node.cond_dist)

        if node.cond_dist.random:
            shape = "oval"
        else:
            shape = "plaintext"

        return dict(style=style, label=label, shape=shape)

    return fn


def viz_generic(vars, label_fn):
    if isinstance(vars, dict):  # hack to handle replacements. Probably bad
        vars = tuple(vars.values())

    vars = jax.tree_util.tree_leaves(vars)
    nodes = dag.upstream_nodes(vars)

    import graphviz  # doctest: +NO_EXE

    graphviz.set_jupyter_format("png")

    dot = graphviz.Digraph()

    id = 0
    node_to_id = {}
    for node in nodes:
        dot.node(str(id), **label_fn(node))

        for p in node.parents:
            dot.edge(str(node_to_id[p]), str(id))

        node_to_id[node] = id
        id += 1
    return dot


def viz(vars, observed_vars=(), labels=None):
    fn = label_fn(observed_vars=observed_vars, labels=labels)
    return viz_generic(vars, fn)


viz_upstream = viz  # TODO: delete after changing calls

# def retrieve_name(var):
#     import inspect
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is var][-1]

def retrieve_name(var):
    """
    Use evil inspection tricks to try to print IR using original variable names
    """
    import inspect
    frames = inspect.stack()
    for frame_info in frames[1:]:  # Start from the caller's frame
        frame = frame_info.frame
        if frame.f_back is None:
            return None
        stuff = frame.f_back.f_locals.items()
        for var_name, var_val in stuff:
            if var_val is var:
                return var_name
    return None

def print_ir(vars,name_space=6,include_shapes=False):
    vars = jax.tree_util.tree_leaves(vars)
    nodes = dag.upstream_nodes(vars)
    names = {}
    i = 0
    for n in nodes:
        assert isinstance(n,RV)
        name = retrieve_name(n)
        if name is None:
            name = f"tmp{i}"
            i += 1
        names[n] = name
        s = names[n]
        if len(s)<name_space:
            s += ' '*(name_space-len(s))
        s += '= RV('
        s += repr(n.cond_dist)
        if len(n.parents):
            s += ','
        s += util.comma_separated([names[p] for p in n.parents],lambda s: ' '+str(s), parens=False)
        s += ')'
        if include_shapes:
            if len(s) < 70:
                s += ' '*(70-len(s))
            s += f" # shape={n.shape}"
        print(s)




# TODO: Fix to remove new_infer
# def viz_samples(vars, precision=2):
#     vars = jax.tree_util.tree_leaves(vars)
#     nodes = dag.upstream_nodes(vars)
#
#     from . import new_infer as infer
#
#     samps = infer.sample(nodes, niter=1)
#
#     def fn(node):
#         i = nodes.index(node)
#         s = np.array(samps[i][0])
#         label = np.array_str(s, precision=precision)
#         label = np.array2string(
#             s,
#             precision=precision,
#             formatter={"float_kind": lambda x: f"%.{precision}f" % x},
#         )
#         return dict(label=label)
#
#     return viz_generic(nodes, fn)
#
#
# def viz_samples_live(vars, reps=25, wait=1, precision=2):
#     from IPython.display import clear_output, display, Image, SVG
#     import IPython.display
#     import time
#
#     for i in range(reps):
#         clear_output(wait=True)
#         graph = viz_samples(vars, precision=precision)
#         IPython.display.display_png(graph)
#         time.sleep(wait)


_all_objects = vars()


def list_all_cond_dists():
    from inspect import isclass

    """Convenience function to print out all available CondDists"""
    # print(_all_objects)
    print("List of all CondDist OBJECTS with random=True:")
    for name, item in _all_objects.items():
        if isinstance(item, CondDist) and item.random:
            print(f"  {name}")
    print("List of all CondDist OBJECTS with random=False:")
    for name, item in _all_objects.items():
        if isinstance(item, CondDist) and not item.random:
            print(f"  {name}")
    print("List of all CondDist CLASSES:")
    for name, item in _all_objects.items():
        if isclass(item) and issubclass(item, CondDist):
            print(f"  {name}")


