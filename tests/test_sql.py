from pangolin import sql
from pangolin.interface import *


def test_get_db():
    x = normal(0, 1)
    y = normal(x, 1)
    db = sql.get_db(
        [x, y]
    )  # db, var_to_id, dist_to_id, id_to_var, id_to_dist = sql.get_db([x, y])


def test_query_df():
    x = normal(0, 1)
    y = normal(x, 1)

    query = """
        SELECT * from var
        """
    rez = sql.query_df([x, y], query)
    print(rez)


def test_query_single_node():
    loc = makerv(0)
    query = """
    SELECT var.var_id from var
    """
    assert sql.query_nodes([loc], query) == {loc}


def test_query_single_node_with_dist():
    loc = makerv(0)
    query = """
        SELECT * from var
        """
    assert sql.query_nodes([loc], query) == {(loc, loc.cond_dist)}


def test_query_single_node_with_dist2():
    loc = makerv(0)
    query = """
        SELECT
            var.var_id,
            var.dist_id
         from var
        """
    assert sql.query_nodes([loc], query) == {(loc, loc.cond_dist)}


def test_query_single_node_with_dist3():
    loc = makerv(0)
    query = """
        SELECT
            var.dist_id,
            var.var_id
         from var
        """
    assert sql.query_nodes([loc], query) == {(loc.cond_dist, loc)}


def test_query_nodes():
    loc = makerv(0)
    scale = makerv(1)
    x = normal(loc, scale)
    y = normal(x, scale)

    query = """
    SELECT * from var
    """
    rez = sql.query_nodes([loc], query)
    expected = {(loc, loc.cond_dist)}
    # rez = sql.query_nodes([x, y], query)
    # expected = {loc,scale,x,y}
    print(f"{rez=}")
    print(f"{expected=}")
    assert rez == expected

    # assert rez == (x,y)
    # assert len(rez) == 5
    for node in rez:
        print(node)


def test_find_normals():
    x = normal(0, 1)
    y = normal(x, 1)

    query = """
    SELECT
        var.var_id
     from var inner join dist
        on var.dist_id = dist.dist_id
    where dist.name = 'normal_scale'
    """
    rez = sql.query_nodes([x, y], query)
    assert rez == {x, y}


def test_find_normals_with_dists():
    x = normal(0, 1)
    y = normal(x, 1)

    query = """
    SELECT
        var.var_id,
        dist.dist_id
     from var inner join dist
        on var.dist_id = dist.dist_id
    where dist.name = 'normal_scale'
    """
    rez = sql.query_nodes([x, y], query)
    assert rez == {(x, normal_scale), (y, normal_scale)}


def test_find_nodes_parent_pairs():
    query = """
    SELECT
        var.var_id,
        edge.par_id 
    from var inner join edge
        on var.var_id = edge.var_id
    """

    loc = makerv(0)
    scale = makerv(1)
    x = normal(loc, scale)
    rez = sql.query_nodes([x], query)
    assert rez == {(x, loc), (x, scale)}

    y = normal(x, scale)
    rez = sql.query_nodes([y], query)
    assert rez == {(x, loc), (x, scale), (y, x), (y, scale)}


def test_find_normal_normal_pairs():
    loc = makerv(0)
    scale = makerv(1)
    x = normal(loc, scale)
    y = normal(x, scale)
    z = normal(x, scale)

    query = """
    SELECT
        var.var_id,
        par.var_id
    from var
        inner join edge
            on var.var_id = edge.var_id
        left join dist
            on var.dist_id = dist.dist_id
        left join var as par
            on edge.par_id = par.var_id
        left join dist as par_dist
            on par.dist_id = par_dist.dist_id
    where
        dist.name = 'normal_scale' and
        par_dist.name = 'normal_scale'
    """

    rez = sql.query_nodes([x, y, z], query)
    assert rez == {(y, x), (z, x)}


def test_find_normal_normal_pairs2():
    loc = makerv(0)
    scale = makerv(1)
    x = normal(loc, scale)
    y = normal(x, scale)
    z = normal(x, scale)

    query = """
    SELECT
        var.var_id,
        par.var_id
    from edge
        inner join var
            on edge.var_id = var.var_id
        inner join dist
            on var.dist_id = dist.dist_id
        inner join var as par
            on edge.par_id = par.var_id
        inner join dist as par_dist
            on par.dist_id = par_dist.dist_id
    where
        dist.name = 'normal_scale' and
        par_dist.name = 'normal_scale'
    """

    rez = sql.query_nodes([x, y, z], query)
    assert rez == {(y, x), (z, x)}


def test_find_normal_normal_pairs3():
    loc = makerv(0)
    scale = makerv(1)
    x = normal(loc, scale)
    y = normal(x, scale)
    z = normal(x, scale)

    query = """
    SELECT
        vardist.var_id,
        pardist.var_id
    from edge
        inner join vardist
            on edge.var_id = vardist.var_id
        inner join vardist as pardist
            on edge.par_id = pardist.var_id
    where
        vardist.name = 'normal_scale' and
        pardist.name = 'normal_scale'
    """

    rez = sql.query_nodes([x, y, z], query)
    assert rez == {(y, x), (z, x)}


def test_upstream():
    loc = makerv(0)
    scale = makerv(1)
    x = normal(loc, scale)
    y = normal(x, scale)

    rez = sql.upstream_nodes([x])
    assert rez == {x, loc, scale}

    rez = sql.upstream_nodes([y])
    assert rez == {y, x, loc, scale}


def test_beta_bernoulli():
    a = exponential(1)
    b = exponential(1)
    x = beta(a, b)
    y = bernoulli(x)
    z = x + y

    query = """
    SELECT
        pardist.var_id,
        vardist.var_id
    from edge
        inner join vardist
            on edge.var_id = vardist.var_id
        inner join vardist as pardist
            on edge.par_id = pardist.var_id
    where
        vardist.name = 'bernoulli' and
        pardist.name = 'beta'
    """

    rez = sql.query_nodes([z], query)
    assert rez == {(x, y)}


def test_parent_hash():
    """
    take all the parents for each node, order them by parnum, concatenate them
    together, and then hash
    """
    # a = exponential(1)
    # b = exponential(1)
    # x = normal(a, b)
    # y = normal(a, b)
    # z = normal(x, b)

    a = makerv(1)
    b = makerv(2)
    vars = [a,b]
    vars += [normal(a,b) for i in range(2)]
    vars += [normal_prec(a, b) for i in range(2)]
    vars += [normal(b,a) for i in range(3)]
    vars += [exponential(a) for i in range(4)]
    vars += [exponential(b) for i in range(5)]

    # one row for each edge
    query = """
    select
        *
    from var left join edge
    on var.var_id = edge.var_id
    """
    df = sql.query_df(vars, query)
    print(df)

    # group by var
    query = """
    select
        *
    from var left join edge
    on var.var_id = edge.var_id
    group by var.var_id
    """
    df = sql.query_df(vars, query)
    print(df)

    # aggregate with min and max parent id
    query = """
    select
        min(edge.par_id),
        max(edge.par_id) 
    from var left join edge
    on var.var_id = edge.var_id
    group by var.var_id
    """
    df = sql.query_df(vars, query)
    print(df)


    # aggregate with all parent ids
    query = """
    select
        group_concat(edge.par_id) 
    from var left join edge
    on var.var_id = edge.var_id
    group by var.var_id
    """
    df = sql.query_df(vars, query)
    print(df)

    # aggregate with all parent ids using subquery
    query = """
    select
        group_concat(var_edge.par_id) 
    from (
        select
            var.var_id,
            edge.par_id,
            edge.parnum
        from var left join edge 
        on var.var_id = edge.var_id
    ) as var_edge
    group by var_edge.var_id
    """
    df = sql.query_df(vars, query)
    print(df)


    # aggregate with all parent ids, ordered by parnum
    query = """
    select
        group_concat(var_edge.par_id) 
    from (
        select
            var.var_id,
            edge.par_id,
            edge.parnum
        from var left join edge 
        on var.var_id = edge.var_id
        order by
        edge.parnum
    ) as var_edge
    group by var_edge.var_id
    """
    df = sql.query_df(vars, query)
    print(df)


    # aggregate with all parent ids, ordered by parnum
    query = """
    select
        group_concat(var_edge.par_id),
        var_edge.dist_id,
        group_concat(var_edge.par_id) || var_edge.dist_id,
        hex(group_concat(var_edge.par_id) || var_edge.dist_id) 
    from (
        select
            var.var_id,
            var.dist_id,
            edge.par_id,
            edge.parnum
        from var left join edge 
        on var.var_id = edge.var_id
        order by
        edge.parnum
    ) as var_edge
    group by var_edge.var_id
    """
    df = sql.query_df(vars, query)
    print(df)

    # get the signature only
    query = """
        select
            hex(group_concat(var_edge.par_id) || var_edge.dist_id) 
        from (
            select
                var.var_id,
                var.dist_id,
                edge.par_id,
                edge.parnum
            from var left join edge 
            on var.var_id = edge.var_id
            order by
            edge.parnum
        ) as var_edge
        group by var_edge.var_id
        """
    df = sql.query_df(vars, query)
    print(df)
    sig = df.iloc[:,0]
    assert len(np.unique(sig)) == 6

    # expect one signature for each row
    # vars = [a,b] #
    # vars += [normal(a,b) for i in range(2)]
    # vars += [normal_prec(a, b) for i in range(2)]
    # vars += [normal(b,a) for i in range(3)]
    # vars += [exponential(a) for i in range(4)]
    # vars += [exponential(b) for i in range(5)]

    # alternate version without so many names
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
    df = sql.query_df(vars, query)
    print(df)
    var_id = df.iloc[:,0]
    assert len(np.unique(var_id)) == 2+2+2+3+4+5
    sig = df.iloc[:,1]
    assert len(np.unique(sig)) == 6

def test_group_by_parents():
    a = makerv(0)
    b = makerv(1)
    c = normal(a,b)
    d = normal(a,b)
    e = normal(b,a)
    f = normal(b,a)
    g = normal(c,d)
    h = normal(e,f)

    rez = sql.group_by_dist_and_parents([c])
    assert rez == frozenset({frozenset({c})})

    rez = sql.group_by_dist_and_parents([c,d])
    assert rez == frozenset({frozenset({c,d})})

    rez = sql.group_by_dist_and_parents([c,d,e,f])
    assert rez == frozenset({
        frozenset({c,d}),
        frozenset({e,f})
    })

    rez = sql.group_by_dist_and_parents([c,d,e,f,g,h])
    assert rez == frozenset({
        frozenset({c,d}),
        frozenset({e,f}),
        frozenset({g}),
        frozenset({h})
    })

