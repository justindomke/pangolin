from pangolin.ir import RV, Constant, Normal, StudentT, NormalPrec


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
            assert (
                str(e)
                == f"Normal op got parent shapes {parents_shapes} not all scalar."
            )
