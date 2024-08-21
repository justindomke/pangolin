import jax.tree_util
from pangolin import dag, util
from pangolin.ir.rv import RV
import numpy as np

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

        op = "~" if node.op.random else "="

        line = f"{shape_str} | {util.num2str(id)} {op} {str(node.op)}"
        if node.parents:
            line += "(" + par_id_str + ")"

        print(line)

        node_to_id[node] = id
        id += 1
