import pytest

from pangolin import ir
from pangolin.ir import Add, Bernoulli, Composite, Normal, Op, Scan, VMap


def test_fixed_ops_report_randomness_and_discreteness():
    assert Add().random is False
    assert Normal().random is True
    assert Normal().discrete is False
    assert Bernoulli().random is True
    assert Bernoulli().discrete is True


def test_fixed_randomness_is_a_plain_boolean_attribute():
    assert Add.__dict__["random"] is False
    assert Normal.__dict__["random"] is True


def test_missing_randomness_is_rejected_when_instantiating():
    class MissingRandom(Op):
        def get_shape(self):
            return ()

    with pytest.raises(TypeError, match="abstract.*random"):
        MissingRandom()


def test_fixed_randomness_can_be_inherited():
    class CustomNormal(Normal):
        pass

    assert CustomNormal.random is True
    assert CustomNormal().random is True


def test_derived_randomness_can_be_inherited():
    class CustomVMap(VMap):
        pass

    assert CustomVMap(Normal(), (0, 0)).random is True
    assert CustomVMap(Add(), (0, 0)).random is False


@pytest.mark.parametrize("op_type", [VMap, Composite, Scan])
def test_derived_randomness_is_an_explicit_documented_property(op_type):
    assert isinstance(op_type.__dict__["random"], property)
    assert "random" in op_type.random.__doc__


@pytest.mark.parametrize("op", [Add(), Normal(), VMap(Normal(), (0, 0))])
def test_randomness_cannot_be_changed_on_an_op(op):
    with pytest.raises(TypeError, match="Ops are immutable"):
        op.random = not op.random


def test_fixed_discreteness_is_a_plain_boolean_attribute():
    assert Normal.__dict__["discrete"] is False
    assert Bernoulli.__dict__["discrete"] is True


@pytest.mark.parametrize("op_type", [VMap, Composite, Scan])
def test_derived_discreteness_is_documented_on_higher_order_ops(op_type):
    assert isinstance(op_type.__dict__["discrete"], property)
    assert "discrete" in op_type.discrete.__doc__


@pytest.mark.parametrize(
    "op",
    [
        Add(),
        VMap(Add(), (0, 0)),
        Composite(2, (Add(),), ((0, 1),)),
        Scan(Add(), 3, (0,)),
    ],
)
def test_discrete_is_none_for_deterministic_ops(op):
    assert op.random is False
    assert op.discrete is None


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (VMap(Normal(), (0, 0)), False),
        (VMap(Bernoulli(), (0,)), True),
        (Composite(1, (Bernoulli(),), ((0,),)), True),
        (Scan(Bernoulli(), 3, ()), True),
    ],
)
def test_higher_order_ops_derive_discreteness(op, expected):
    assert op.random is True
    assert op.discrete is expected


def test_random_op_must_declare_discreteness():
    class MissingDiscrete(Op):
        random: bool = True
        """True"""

        def get_shape(self):
            return ()

    with pytest.raises(TypeError, match="Random Ops must declare discrete as True or False"):
        MissingDiscrete()


@pytest.mark.parametrize("value", [None, 0, 1, "False"])
def test_random_op_rejects_nonboolean_discreteness(value):
    class InvalidDiscrete(Op):
        random = True
        discrete = value

        def get_shape(self):
            return ()

    with pytest.raises(TypeError, match="Random Ops must declare discrete as True or False"):
        InvalidDiscrete()


@pytest.mark.parametrize("value", [True, False])
def test_discreteness_validation_uses_initialized_instance_fields(value):
    class Configured(Op):
        random = True

        def __init__(self, value):
            self.value = value
            super().__init__()

        @property
        def discrete(self):
            return self.value

        def get_shape(self):
            return ()

    assert Configured(value).discrete is value


def test_scalar_primitives_obey_the_discreteness_contract():
    for name in ir.__all__:
        cls = getattr(ir, name)
        if isinstance(cls, type) and issubclass(cls, ir.ScalarOp) and cls is not ir.ScalarOp:
            op = cls()
            if op.random:
                assert isinstance(op.discrete, bool), name
            else:
                assert op.discrete is None, name


def test_fixed_and_derived_discreteness_can_be_inherited():
    class CustomNormal(Normal):
        pass

    class CustomVMap(VMap):
        pass

    assert CustomNormal().discrete is False
    assert CustomVMap(Normal(), (0, 0)).discrete is False
    assert CustomVMap(Bernoulli(), (0,)).discrete is True
    assert CustomVMap(Add(), (0, 0)).discrete is None


@pytest.mark.parametrize("op", [Add(), Normal(), VMap(Bernoulli(), (0,))])
def test_discreteness_cannot_be_changed_on_an_op(op):
    with pytest.raises(TypeError, match="Ops are immutable"):
        op.discrete = True
