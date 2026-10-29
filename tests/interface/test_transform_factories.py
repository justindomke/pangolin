"""Check configured factory coercion without changing generic Transform semantics."""

import pytest

from pangolin import functional, ir
from pangolin import interface as pi

FACTORIES = [("scale_by", (2,)), ("shift_by", (3,)), ("scaled_logit", (1, 5))]


def test_default_transform_library_belongs_to_default_interface():
    assert pi.tforms is pi.default.tforms
    assert pi.tforms.interface is pi.default
    other = pi.Interface()
    assert other.tforms.interface is other
    assert other.tforms is not pi.tforms


@pytest.mark.parametrize("name", functional.tforms.__all__)
def test_fixed_transforms_remain_shared_objects(name):
    if name in {factory for factory, _ in FACTORIES}:
        return
    other = pi.Interface()
    assert getattr(pi.tforms, name) is getattr(functional.tforms, name)
    assert getattr(other.tforms, name) is getattr(functional.tforms, name)


@pytest.mark.parametrize("name,args", FACTORIES)
@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
def test_factory_coerces_captured_parameters_but_does_not_force_output_interface(name, args, mode):
    owner = pi.Interface(broadcasting=mode)
    producer = pi.Interface()
    before = ir.RV._n
    transform = getattr(owner.tforms, name)(*args)
    assert type(transform) is functional.Transform
    assert ir.RV._n - before == len(args)

    distribution = producer.uniform if name == "scaled_logit" else producer.normal
    result = transform(distribution)(*args if name == "scaled_logit" else (0, 1))
    assert result.interface is producer
    assert isinstance(result.op, ir.Bijected)
    parameters = result.parents[: result.op.bijector.n_biject_params]
    assert len(parameters) == len(args)
    assert [parameter.op for parameter in parameters] == [ir.Constant(value) for value in args]
    assert all(parameter.interface is owner for parameter in parameters)


@pytest.mark.parametrize("name,args", FACTORIES)
def test_factory_preserves_existing_rvs_including_foreign_and_no_interface(name, args):
    owner = pi.Interface()
    parameters = tuple(ir.RV(ir.Constant(value)) for value in args)
    before = ir.RV._n
    transform = getattr(owner.tforms, name)(*parameters)
    assert ir.RV._n == before
    generic = getattr(functional.tforms, name)(*parameters)
    x = pi.constant(3)
    assert ir.rv_equal(transform.forward(x), generic.forward(x))
    result = transform(owner.normal)(0, 1)
    assert result.parents[: len(parameters)] == parameters
    assert all(parameter.interface is ir.NO_INTERFACE for parameter in parameters)


def test_scaled_logit_mixes_an_existing_bound_with_a_coerced_bound():
    owner = pi.Interface()
    lower = pi.constant(1)
    transform = owner.tforms.scaled_logit(lower=lower, upper=5)
    result = transform(pi.uniform)(1, 5)
    assert result.parents[0] is lower
    assert result.parents[1].op == ir.Constant(5)
    assert result.parents[1].interface is owner
    assert result.interface is pi.default


@pytest.mark.parametrize("name,args", FACTORIES)
def test_saved_factory_and_transform_work_across_overload(name, args):
    owner = pi.Interface(broadcasting="off")
    factory = getattr(owner.tforms, name)
    with owner.overload(broadcasting="numpy"):
        transform = factory(*args)
    result = transform(pi.normal)(0, 1)
    assert result.interface is pi.default
    assert result.parents[0].interface is owner
    assert owner._effective_config.broadcasting == "off"


def test_shift_by_accepts_array_values_without_broadcasting():
    transform = pi.tforms.shift_by([1, 2])
    x = pi.constant([3, 4])
    assert transform.forward(x).shape == (2,)
    with pytest.raises(ValueError, match="equal shapes"):
        transform.forward(pi.constant(3))


@pytest.mark.parametrize(
    "name,args", [("scale_by", ([1, 2],)), ("scaled_logit", ([1], 5)), ("scaled_logit", (1, [5]))]
)
def test_scalar_factories_reject_nonscalar_parameters(name, args):
    with pytest.raises(ValueError, match="scalar"):
        getattr(pi.tforms, name)(*args)


@pytest.mark.parametrize("name", ["exp", "log"])
def test_fixed_transform_preserves_the_argument_interface(name):
    owner = pi.Interface()
    x = ir.RV(ir.Constant(2))
    result = getattr(owner.tforms, name).forward(x)
    assert result.interface is ir.NO_INTERFACE
    assert result.parents == (x,)
