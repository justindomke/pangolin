import pytest

from pangolin.ir import Add, Constant, RV, graph_upstream, print_upstream


@pytest.mark.parametrize("display", [print_upstream, graph_upstream])
def test_empty_display(display, capsys):
    display([])

    assert capsys.readouterr().out == "[empty vars, nothing to print]\n"


def test_print_upstream_accepts_pytrees_and_explicit_names(capsys):
    left = RV(Constant(1))
    right = RV(Constant(2))
    result = RV(Add(), left, right)

    print_upstream({"outputs": [result]}, {"left": left})

    assert capsys.readouterr().out == (
        "shape | statement\n"
        "----- | ---------\n"
        "()    | left = 1\n"
        "()    | a = 2\n"
        "()    | b = add(left, a)\n"
    )


def test_graph_upstream_displays_explicit_names(capsys):
    result = RV(Add(), RV(Constant(1)), RV(Constant(2)))

    graph_upstream(result, {"result": result})

    assert "add ← result" in capsys.readouterr().out


def test_print_upstream_names_from_locals(capsys):
    source = RV(Constant(1))
    result = RV(Add(), source, source)
    print_upstream(result, locals())
    assert capsys.readouterr().out == (
        "shape | statement\n----- | ---------\n"
        "()    | source = 1\n()    | result = add(source, source)\n"
    )


def test_print_upstream_nested_names_and_shared_nodes(capsys):
    source = RV(Constant(1))
    result = RV(Add(), source, source)
    print_upstream(result, {"params": {"source": source}, "outputs": [result, result]})
    assert capsys.readouterr().out == (
        "shape | statement\n----- | ---------\n"
        "()    | params['source'] = 1\n"
        "()    | outputs[0] = add(params['source'], params['source'])\n"
    )


def test_print_upstream_namespace_does_not_add_roots(capsys):
    source = RV(Constant(1))
    unrelated = RV(Constant(99))
    print_upstream(source, {"source": source, "unrelated": unrelated})
    assert capsys.readouterr().out == "shape | statement\n----- | ---------\n()    | source = 1\n"


def test_print_upstream_first_alias_wins(capsys):
    source = RV(Constant(1))
    print_upstream(source, {"nested": [source], "source": source})
    assert "nested[0] = 1" in capsys.readouterr().out


def test_print_upstream_reserves_later_names_and_unused_aliases(capsys):
    source = RV(Constant(1))
    result = RV(Add(), source, source)
    print_upstream(result, {"a": result, "b": result})
    assert capsys.readouterr().out == (
        "shape | statement\n----- | ---------\n"
        "()    | c = 1\n()    | a = add(c, c)\n"
    )


def test_print_upstream_empty_namespace_matches_omitted_namespace(capsys):
    source = RV(Constant(1))
    print_upstream(source)
    expected = capsys.readouterr().out
    print_upstream(source, {})
    assert capsys.readouterr().out == expected


def test_print_upstream_empty_roots_ignore_namespace_nodes(capsys):
    print_upstream([], {"source": RV(Constant(1))})
    assert capsys.readouterr().out == "[empty vars, nothing to print]\n"


def test_print_upstream_ambiguous_names_rejected_before_printing(capsys):
    x, y = RV(Constant(1)), RV(Constant(2))
    with pytest.raises(ValueError, match="refers to distinct objects"):
        print_upstream([x, y], {"a[0]": x, "a": [y]})
    assert capsys.readouterr().out == ""


def test_print_upstream_namespace_cycles_rejected(capsys):
    values: list[object] = []
    values.append(values)
    with pytest.raises(ValueError, match="acyclic"):
        print_upstream(RV(Constant(1)), {"values": values})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [1, "x", [RV(Constant(1)), 2]])
def test_print_upstream_non_rv_leaves_rejected(value, capsys):
    with pytest.raises(TypeError, match="PyTree of RVs"):
        print_upstream(value)
    assert capsys.readouterr().out == ""


def test_graph_upstream_honors_parent_numbering_mode(capsys):
    result = RV(Add(), RV(Constant(1)), RV(Constant(2)))

    graph_upstream(result, number_parents="off")
    without_numbers = capsys.readouterr().out
    graph_upstream(result, number_parents="on")
    with_numbers = capsys.readouterr().out

    assert without_numbers != with_numbers


def test_graph_upstream_names_from_locals(capsys):
    source = RV(Constant(1))
    result = RV(Add(), source, source)
    graph_upstream(result, locals())
    output = capsys.readouterr().out
    assert "1 ← source" in output
    assert "add ← result" in output


def test_graph_upstream_nested_names(capsys):
    source = RV(Constant(1))
    result = RV(Add(), source, source)
    graph_upstream({"roots": (result, result)}, {"values": [source, result], "alias": result})
    output = capsys.readouterr().out
    assert output.count("← values[0]") == 1
    assert output.count("← values[1]") == 1
    assert "alias" not in output


def test_graph_upstream_namespace_does_not_add_roots(capsys):
    source = RV(Constant(1))
    graph_upstream(source, {"source": source, "unrelated": RV(Constant(99))})
    output = capsys.readouterr().out
    assert "source" in output
    assert "99" not in output
    assert "unrelated" not in output


def test_graph_upstream_unnamed_nodes_have_no_annotation(capsys):
    source = RV(Constant(1))
    result = RV(Add(), source, source)
    graph_upstream(result, {"result": result})
    output = capsys.readouterr().out
    assert output.count("←") == 1
    graph_upstream(result)
    assert "←" not in capsys.readouterr().out


def test_graph_upstream_name_collision_rejected(capsys):
    x, y = RV(Constant(1)), RV(Constant(2))
    with pytest.raises(ValueError, match="refers to distinct objects"):
        graph_upstream([x, y], {"a[0]": x, "a": [y]})
    assert capsys.readouterr().out == ""


def test_graph_upstream_cycle_rejected(capsys):
    values: list[object] = []
    values.append(values)
    with pytest.raises(ValueError, match="acyclic"):
        graph_upstream(RV(Constant(1)), {"values": values})
    assert capsys.readouterr().out == ""


def test_graph_upstream_empty_roots_do_not_print_namespace(capsys):
    graph_upstream([], {"source": RV(Constant(1))})
    assert capsys.readouterr().out == "[empty vars, nothing to print]\n"


@pytest.mark.parametrize("value", [1, "x", [RV(Constant(1)), 2]])
def test_graph_upstream_invalid_leaf(value, capsys):
    with pytest.raises(TypeError, match="PyTree of RVs"):
        graph_upstream(value)
    assert capsys.readouterr().out == ""


def test_graph_upstream_marks_a_repeated_parent(capsys):
    value = RV(Constant(1))
    result = RV(Add(), value, value)

    graph_upstream(result, {"result": result})
    assert "│⁎" in capsys.readouterr().out

    graph_upstream(result, {"result": result}, number_parents="off")
    assert "⁎" not in capsys.readouterr().out


def test_graph_upstream_rejects_invalid_parent_numbering_mode():
    result = RV(Constant(1))

    with pytest.raises(ValueError, match="number_parents must be one of"):
        graph_upstream(result, number_parents="sometimes")  # type: ignore[arg-type]
