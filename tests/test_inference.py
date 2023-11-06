from pangolin import inference, interface

normal = interface.normal

def test_upstream_with_descendent1():
    # TODO: variables to sample should become upstream_with_descendent
    z = normal(0, 1)
    x = normal(z, 1)
    nodes = inference.upstream_with_descendent([z], [x])
    assert nodes == [z, x]


def test_upstream_with_descendent2():
    z = normal(0, 1)
    x = normal(z, 1)
    nodes = inference.upstream_with_descendent([z], [])
    assert nodes == []


def test_upstream_with_descendent3():
    z = normal(0, 1)
    x = normal(0, 1)
    nodes = inference.upstream_with_descendent([z], [x])
    assert nodes == []


def test_upstream_with_descendent4():
    z = normal(0, 1)
    x = normal(0, 1)
    nodes = inference.upstream_with_descendent([x], [x])
    assert nodes == [x]


def test_upstream_with_descendent5():
    x = normal(0, 1)
    vars = inference.upstream_with_descendent([x], [])
    assert set(vars) == set()


def test_upstream_with_descendent6():
    x = normal(0, 1)
    y = normal(x, 1)
    vars = inference.upstream_with_descendent([y], [])
    assert set(vars) == set()


def test_upstream_with_descendent7():
    # x -> z <- y
    x = normal(0, 1)
    y = normal(0, 1)
    z = normal(x, y)
    vars = inference.upstream_with_descendent([x], [y])
    assert set(vars) == set()

    vars = inference.upstream_with_descendent([x], [z])
    assert set(vars) == {x, y, z}


def test_upstream_with_descendent8():
    x = normal(0, 1)
    y = x ** 2
    z = normal(y, 1)

    vars = inference.upstream_with_descendent([x], [z])
    assert set(vars) == {x, z}


def test_upstream_with_descendent9():
    x = normal(0, 1)
    y = x ** 2
    z = y ** 2

    vars = inference.upstream_with_descendent([x], [z])
    assert set(vars) == {x}


def test_upstream_with_descendent10():
    # x -> y -> z
    x = normal(0, 1)
    y = normal(x, 1)
    z = normal(y, 1)

    vars = inference.upstream_with_descendent([x], [])
    assert set(vars) == set()

    vars = inference.upstream_with_descendent([x], [y])
    assert set(vars) == {x, y}

    vars = inference.upstream_with_descendent([x], [z])
    assert set(vars) == {x, y, z}


def test_upstream_with_descendent11():
    a = normal(0, 1)
    b = normal(a, 1)
    c = normal(a, 1)
    d = normal(c, 1)

    vars = inference.upstream_with_descendent([a], [])
    assert set(vars) == set()

    vars = inference.upstream_with_descendent([a], [b])
    assert set(vars) == {a, b}

    vars = inference.upstream_with_descendent([a], [d])
    assert set(vars) == {a, c, d}


def test_upstream_with_descendent12():
    a = normal(0, 1)
    b = normal(a, 1)
    c = normal(b, 1)
    vars = inference.upstream_with_descendent([b], [c, a])
    assert set(vars) == {b, c}


def test_upstream_with_descendent13():
    a = normal(0, 1)
    b = normal(a, 1)
    c = normal(b, 1)
    d = normal(c, 1)
    e = normal(d, 1)
    vars = inference.upstream_with_descendent([c], [b, d])
    assert set(vars) == {c, d}
