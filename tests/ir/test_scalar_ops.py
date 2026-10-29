from pangolin.ir import (
    Add,
    BernoulliLogit,
    Constant,
    Gamma,
    Lognormal,
    Normal,
    NormalPrec,
    RV,
    StudentT,
)


def test_generated_deterministic_documentation():
    assert Add.__doc__.startswith("Scalar `Add` operation.\n")
    assert (
        "expects the following scalar parents:\n\n" "* ``a``: parent 0\n" "* ``b``: parent 1"
    ) in Add.__doc__
    assert ">>> op.random\nFalse" in Add.__doc__
    assert "Represents a Add" not in Add.__doc__


def test_generated_distribution_documentation():
    assert Normal.__doc__.startswith("Scalar Normal distribution.\n")
    assert (
        "https://en.wikipedia.org/wiki/Normal_distribution" in Normal.__doc__.split("See Also")[1]
    )
    assert (
        "* ``mu``: location / mean\n" "* ``sigma``: scale / standard deviation"
    ) in Normal.__doc__
    assert ">>> op.random\nTrue" in Normal.__doc__


def test_wikipedia_overrides():
    assert "<https://en.wikipedia.org/wiki/Normal_distribution>" in NormalPrec.__doc__
    assert "<https://en.wikipedia.org/wiki/Log-normal_distribution>" in Lognormal.__doc__
    assert "wikipedia" not in BernoulliLogit.__doc__.lower()
    assert "See Also" not in BernoulliLogit.__doc__


def test_generated_notes_use_numpy_heading():
    assert "\nNotes\n-----\n" in Gamma.__doc__


def test_sameness():
    assert Normal() == Normal()
    assert Normal() != StudentT()


def test_hashing():
    assert hash(Normal()) == hash(Normal())
    assert hash(NormalPrec()) == hash(NormalPrec())


def test_wrong_number_of_args():
    for args in [(), (0,), (0, 1, 2), (0, 1, 2, 3)]:
        try:
            d = Normal()
            args = tuple(RV(Constant(a)) for a in args)  # convert to RVs
            x = RV(d, *args)
            assert False
        except TypeError as e:
            assert str(e) == f"Normal op got {len(args)} parent(s) but expected 2."


def test_non_scalar_parent_shapes():
    for parents_shapes in [((), (2,))]:
        try:
            d = Normal()
            d.get_shape(*parents_shapes)
            # args = tuple(RV(Constant(a)) for a in args) # convert to RVs
            # x = RV(d,*args)
            assert False
        except ValueError as e:
            assert str(e) == f"Normal op got parent shapes {parents_shapes} not all scalar."
