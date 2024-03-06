import pandas as pd

from . import dag, util

import numpy as np
import jax

from .ir import *

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class DB:
    conn: sqlite3.Connection
    var_to_id: dict
    id_to_var: dict
    dist_to_id: dict
    id_to_dist: dict


def get_db(vars):
    """
    Convert a set of RVs to an SQL database
    """

    db = sqlite3.connect(":memory:")

    all_vars = dag.upstream_nodes(vars)
    dist_id = -1
    var_id = 0
    dist_ids = dict()
    var_ids = dict()
    for var in all_vars:
        if var.cond_dist not in dist_ids:
            dist_ids[var.cond_dist] = dist_id
            dist_id -= 1

        var_ids[var] = var_id
        var_id += 1

    # create table for dists
    with db:
        dist_data = tuple((dist_id, dist.name) for dist, dist_id in dist_ids.items())
        cur = db.cursor()
        cur.execute("CREATE TABLE dist(dist_id,name)")
        cur.executemany("INSERT INTO dist VALUES(?,?)", dist_data)

    # create table for RVs
    with db:
        var_data = tuple(
            (var_id, dist_ids[var.cond_dist]) for var, var_id in var_ids.items()
        )
        cur = db.cursor()
        cur.execute("CREATE TABLE var(var_id,dist_id)")
        cur.executemany("INSERT INTO var VALUES(?,?)", var_data)

    # create var_dist merged table
    with db:
        cur = db.cursor()
        code = """
        CREATE TABLE vardist as
        SELECT
            var.var_id,
            dist.dist_id,
            dist.name
        FROM
            var inner join dist
                on var.dist_id = dist.dist_id
        """
        cur.execute(code)

    # create table for edges
    with db:
        edge_data = []
        for var in all_vars:
            for parnum, p in enumerate(var.parents):
                edge_data.append((var_ids[var], var_ids[p], parnum))

        cur = db.cursor()
        cur.execute("CREATE TABLE edge(var_id,par_id,parnum)")
        cur.executemany("INSERT INTO edge VALUES(?,?,?)", edge_data)

    id_to_vars = {value: key for key, value in var_ids.items()}
    id_to_dists = {value: key for key, value in dist_ids.items()}

    return DB(db, var_ids, id_to_vars, dist_ids, id_to_dists)


def query_df(vars, query, db=None):
    if db is None:
        db = get_db(vars)

    with db.conn:
        df = pd.read_sql_query(query, db.conn)
        return df


def query_nodes(vars, query, db=None):
    if db is None:
        db = get_db(vars)

    df = query_df(vars, query, db)

    if df.shape[1] == 1:
        print("HIYA")
        ids = df.iloc[:, 0]
        if ids[0] >= 0:
            return set(db.id_to_var[id] for id in ids)
        else:
            return set(db.id_to_dist[id] for id in ids)

    # convert columns to nodes
    stuff = []
    for column_name, ids in df.items():
        print(f"{column_name=} {ids=}")
        if ids[0] >= 0:
            my_stuff = tuple(db.id_to_var[id] for id in ids)
        else:
            my_stuff = tuple(db.id_to_dist[id] for id in ids)
        stuff.append(my_stuff)

    return set(zip(*stuff))


def upstream_nodes(vars):
    """
    For fun, implement upstream nodes using SQL
    """

    db = get_db(vars)
    starting_ids = tuple(str(db.var_to_id[var]) for var in vars)
    starting_ids_str = util.comma_separated(starting_ids)

    query = f"""
    with recursive up_var as (
        select
            var.var_id as var_id
        from var
        where var_id in {starting_ids_str}
        union all
        select
            edge.par_id as var_id
        from
            var inner join edge
                on var.var_id = edge.var_id
    )
    select
        *
    from up_var;
    """

    return query_nodes(vars, query, db)


def group_by_dist_and_parents(vars):
    """
    Group the nodes into groups that all have same parents and same cond_dists
    """

    query = """
            select
                var_id,
                hex(group_concat(par_id) || dist_id) 
            from (
                select
                    var.var_id,
                    dist_id,
                    par_id,
                    parnum
                from var left join edge 
                on var.var_id = edge.var_id
                order by
                edge.parnum
            )
            group by var_id
            """

    db = get_db(vars)
    df = query_df(vars, query, db)
    var_id = df.iloc[:, 0]
    sig = df.iloc[:, 1]

    valid_sigs = [s for s in np.unique(sig) if s != ""]

    return frozenset(
        frozenset(db.id_to_var[id] for id, s in zip(var_id, sig) if s == my_sig)
        for my_sig in valid_sigs
    )
