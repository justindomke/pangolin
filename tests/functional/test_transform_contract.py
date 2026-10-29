from pangolin import ir
from pangolin.ir import RV


import numpy as np
import pytest

from pangolin.interface import Transform
from pangolin.interface import bernoulli
from pangolin.interface import compose_transforms
from pangolin.interface import constant
from pangolin.interface import exp as interface_exp
from pangolin.interface import log as interface_log
from pangolin.interface import multi_normal
from pangolin.interface import normal
from pangolin.interface import tforms


def zero_log_det(x: RV, y: RV) -> RV:
    return constant(0.0)


def test_exp_constructs_a_transformed_rv_without_bijector_parameters():
    result = tforms.exp(normal)(0, 1)

    assert result.op == ir.Bijected(
        ir.Normal(),
        ir.Bijector(
            ir.Exp(),
            ir.Log(),
            ir.Composite(2, (ir.Log(),), ((1,),)),
            sign=ir.Constant(1),
        ),
    )
    assert [parent.op for parent in result.parents] == [ir.Constant(0), ir.Constant(1)]


def test_transform_calls_each_graph_function_once():
    calls = []

    def forward(x: RV) -> RV:
        calls.append("forward")
        return interface_exp(x)

    def inverse(y: RV) -> RV:
        calls.append("inverse")
        return interface_log(y)

    def log_det_jac(x: RV, y: RV) -> RV:
        calls.append("log_det_jac")
        return interface_log(y)

    transform = Transform(forward, inverse, log_det_jac)
    transform(normal)(0, 1)

    assert calls == ["forward", "inverse", "log_det_jac"]


def test_transform_names_returned_functions_honestly():
    transformed_normal = tforms.exp(normal)
    transformed_normal_and_base = tforms.exp.apply_and_invert(normal)

    assert transformed_normal.__name__ == "transformed_normal"
    assert transformed_normal.__qualname__ == "transformed_normal"
    assert not hasattr(transformed_normal, "__wrapped__")
    assert transformed_normal_and_base.__name__ == "transformed_normal_and_base"
    assert transformed_normal_and_base.__qualname__ == "transformed_normal_and_base"
    assert not hasattr(transformed_normal_and_base, "__wrapped__")


def test_forward_capture_order_is_canonical_for_all_bijector_operations():
    scale = constant(2)
    shift = constant(3)
    transform = Transform(
        lambda x: x * scale + shift,
        lambda y: (y - shift) / scale,
        lambda x, y: interface_log(abs(scale)),
    )

    result = transform(normal)(0, 1)

    assert result.parents[:2] == (scale, shift)
    assert result.op.bijector == ir.Bijector(
        ir.Composite(
            3,
            (ir.Mul(), ir.Add()),
            ((0, 1), (3, 2)),
        ),
        ir.Composite(
            3,
            (ir.Sub(), ir.Div()),
            ((0, 2), (3, 1)),
        ),
        ir.Composite(
            4,
            (ir.Abs(), ir.Log()),
            ((2,), (4,)),
        ),
        n_biject_params=2,
    )


def test_log_det_jac_may_ignore_captured_bijector_parameters():
    offset = constant(2)
    transform = Transform(
        lambda x: x + offset,
        lambda y: y - offset,
        zero_log_det,
    )

    result = transform(normal)(0, 1)

    assert result.parents[0] is offset
    assert result.op.bijector.log_det_jac == ir.Composite(
        3,
        (ir.Constant(0.0),),
        ((),),
    )


def test_internal_constants_are_embedded_in_each_operation():
    transform = Transform(
        lambda x: x + constant(1),
        lambda y: y - constant(1),
        zero_log_det,
    )

    result = transform(normal)(0, 1)

    assert result.op.bijector.n_biject_params == 0
    assert result.op.bijector.forward == ir.Composite(
        1,
        (ir.Constant(1), ir.Add()),
        ((), (0, 1)),
    )
    assert result.op.bijector.inverse == ir.Composite(
        1,
        (ir.Constant(1), ir.Sub()),
        ((), (0, 1)),
    )


def test_forward_and_inverse_must_capture_the_same_rv_objects():
    forward_offset = constant(1)
    inverse_offset = constant(1)
    transform = Transform(
        lambda x: x + forward_offset,
        lambda y: y - inverse_offset,
        zero_log_det,
    )

    with pytest.raises(ValueError, match="forward and inverse must capture exactly"):
        transform(normal)(0, 1)


def test_internal_and_captured_equal_constants_do_not_match():
    captured_one = constant(1)
    transform = Transform(
        lambda x: x + constant(1),
        lambda y: y - captured_one,
        zero_log_det,
    )

    with pytest.raises(ValueError, match="forward and inverse must capture exactly"):
        transform(normal)(0, 1)


def test_log_det_jac_cannot_introduce_a_captured_rv():
    parameter = constant(2)
    transform = Transform(
        lambda x: interface_exp(x),
        lambda y: interface_log(y),
        lambda x, y: parameter + constant(0),
    )

    with pytest.raises(ValueError, match="log_det_jac cannot capture"):
        transform(normal)(0, 1)


def test_identity_forward_and_inverse_are_supported():
    identity = Transform(lambda x: x, lambda y: y, zero_log_det)

    result = identity(normal)(0, 1)

    assert result.op.bijector == ir.Bijector(
        ir.Identity(),
        ir.Identity(),
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
    )


def test_identity_forward_keeps_x_and_y_as_distinct_log_det_inputs():
    identity = Transform(
        lambda x: x,
        lambda y: y,
        lambda x, y: (x - y) * 0,
    )

    result = identity(normal)(0, 1)

    assert result.op.bijector.log_det_jac == ir.Composite(
        2,
        (ir.Sub(), ir.Constant(0), ir.Mul()),
        ((0, 1), (), (2, 3)),
    )


def test_log_det_jac_may_return_a_formal_input_directly():
    transform = Transform(
        interface_exp,
        interface_log,
        lambda x, y: y,
    )

    result = transform(normal)(0, 1)

    assert result.op.bijector.log_det_jac == ir.Composite(
        2,
        (ir.Identity(),),
        ((1,),),
    )


def test_forward_must_depend_on_its_input():
    transform = Transform(
        lambda x: constant(1),
        lambda y: y,
        zero_log_det,
    )

    with pytest.raises(ValueError, match="forward must depend on its input"):
        transform(normal)(0, 1)


def test_inverse_must_depend_on_its_input():
    transform = Transform(
        interface_exp,
        lambda y: constant(1),
        zero_log_det,
    )

    with pytest.raises(ValueError, match="inverse must depend on its input"):
        transform(normal)(0, 1)


def test_transform_requires_an_rv_from_the_wrapped_function():
    def bad_function() -> RV:
        return 1  # type: ignore[return-value]

    with pytest.raises(TypeError, match="transformed function must return exactly one RV"):
        tforms.exp(bad_function)()


@pytest.mark.parametrize("which", ["forward", "inverse", "log_det_jac"])
def test_transform_requires_an_rv_from_each_graph_function(which):
    def bad_unary(x: RV) -> RV:
        return 1  # type: ignore[return-value]

    def bad_binary(x: RV, y: RV) -> RV:
        return 1  # type: ignore[return-value]

    transform = Transform(
        bad_unary if which == "forward" else lambda x: interface_exp(x),
        bad_unary if which == "inverse" else lambda y: interface_log(y),
        bad_binary if which == "log_det_jac" else lambda x, y: interface_log(y),
    )

    with pytest.raises(TypeError, match="leaves are RV"):
        transform(normal)(0, 1)


@pytest.mark.parametrize("which", ["forward", "inverse", "log_det_jac"])
def test_transform_requires_one_rv_rather_than_an_rv_pytree(which):
    def bad_unary(x: RV) -> RV:
        return [interface_exp(x)]  # type: ignore[return-value]

    def bad_binary(x: RV, y: RV) -> RV:
        return [constant(0)]  # type: ignore[return-value]

    transform = Transform(
        bad_unary if which == "forward" else lambda x: interface_exp(x),
        bad_unary if which == "inverse" else lambda y: interface_log(y),
        bad_binary if which == "log_det_jac" else lambda x, y: interface_log(y),
    )

    with pytest.raises(TypeError, match=rf"{which} must return exactly one RV"):
        transform(normal)(0, 1)


def test_transform_rejects_an_unrelated_existing_output():
    existing = constant(3)
    transform = Transform(lambda x: existing, lambda y: y, zero_log_det)

    with pytest.raises(ValueError, match="forward must depend on its input RV"):
        transform(normal)(0, 1)


def test_transform_rejects_a_random_forward_operation():
    transform = Transform(
        lambda x: normal(x, 1),
        lambda y: y,
        zero_log_det,
    )

    with pytest.raises(ValueError, match="forward op cannot be random"):
        transform(normal)(0, 1)


def test_transform_rejects_a_random_intermediate_operation():
    def forward(x: RV) -> RV:
        sample = normal(x, 1)
        return sample + 1

    transform = Transform(forward, lambda y: y, zero_log_det)

    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        transform(normal)(0, 1)


def test_transform_rejects_a_deterministic_base_operation():
    with pytest.raises(ValueError, match="base_op must be random"):
        tforms.exp(interface_exp)(constant(1))


def test_transform_rejects_a_discrete_base_operation():
    with pytest.raises(ValueError, match="base_op must be continuous"):
        tforms.exp(bernoulli)(0.5)


def test_transform_rejects_an_incompatible_inverse_shape():
    transform = Transform(
        lambda x: x + constant([1, 2]),
        lambda y: y,
        zero_log_det,
    )

    with pytest.raises(ValueError, match="forward and inverse ops have incompatible shapes"):
        transform(normal)(0, 1)


def test_transform_rejects_a_non_scalar_log_det_jacobian():
    identity = Transform(lambda x: x, lambda y: y, lambda x, y: y)

    with pytest.raises(ValueError, match="log_det_jac must return a scalar"):
        identity(multi_normal)(np.zeros(2), np.eye(2))


def test_apply_and_invert_uses_the_compiled_inverse_operation():
    offset = constant(2)
    transform = tforms.shift_by(offset)

    transformed, reconstructed = transform.apply_and_invert(normal)(0, 1)

    assert reconstructed.op is transformed.op.bijector.inverse
    assert reconstructed.parents == (transformed, offset)


def test_reverse_exchanges_forward_and_inverse():
    result = tforms.log(normal)(0, 1)

    assert result.op.bijector.forward == ir.Log()
    assert result.op.bijector.inverse == ir.Exp()


def test_composition_applies_transforms_in_mathematical_order():
    scale = constant(2)
    shift = constant(3)
    transform = compose_transforms([tforms.scale_by(scale), tforms.shift_by(shift)])

    result = transform(normal)(0, 1)

    assert result.parents[:2] == (scale, shift)
    assert result.op.bijector.forward == ir.Composite(
        3,
        (ir.Mul(), ir.Add()),
        ((0, 1), (3, 2)),
    )
    assert result.op.bijector.inverse == ir.Composite(
        3,
        (ir.Sub(), ir.Div()),
        ((0, 2), (3, 1)),
    )


def test_composition_deduplicates_a_capture_used_by_multiple_transforms():
    parameter = constant(2)
    transform = compose_transforms([tforms.scale_by(parameter), tforms.shift_by(parameter)])

    result = transform(normal)(0, 1)

    assert result.parents[0] is parameter
    assert result.op.bijector.n_biject_params == 1


def test_forward_log_det_composition_uses_forward_order():
    calls = []

    def make_transform(name):
        def forward(x):
            calls.append(f"forward {name}")
            return x + 1

        def inverse(y):
            calls.append(f"inverse {name}")
            return y - 1

        def log_det_jac(x, y):
            calls.append(f"log_det {name}")
            return constant(0)

        return Transform(forward, inverse, log_det_jac)

    transform = compose_transforms([make_transform("a"), make_transform("b")])
    transform.log_det_jac(constant(0), constant(2))

    assert calls == ["forward a", "log_det a", "forward b", "log_det b"]


def test_inverse_log_det_composition_uses_inverse_order():
    calls = []

    def make_transform(name):
        def forward(x):
            calls.append(f"forward {name}")
            return x + 1

        def inverse(y):
            calls.append(f"inverse {name}")
            return y - 1

        def log_det_jac(x, y):
            calls.append(f"log_det {name}")
            return constant(0)

        return Transform(forward, inverse, log_det_jac)

    transform = compose_transforms(
        [make_transform("a"), make_transform("b")],
        log_det_direction="inverse",
    )
    transform.log_det_jac(constant(0), constant(2))

    assert calls == ["inverse b", "log_det b", "inverse a", "log_det a"]


def test_empty_composition_is_the_identity_transform():
    result = compose_transforms([])(normal)(0, 1)

    assert result.op.bijector == ir.Bijector(
        ir.Identity(),
        ir.Identity(),
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
        sign=ir.Constant(1),
    )


def test_composition_copies_its_input_sequence():
    transforms = [tforms.exp]
    composed = compose_transforms(transforms)
    transforms.append(tforms.log)

    result = composed(normal)(0, 1)

    assert result.op.bijector.forward == ir.Exp()


def test_composition_rejects_a_non_transform_item():
    with pytest.raises(TypeError, match="every item"):
        compose_transforms([tforms.exp, "log"])  # type: ignore[list-item]


def test_composition_rejects_an_invalid_log_det_direction():
    with pytest.raises(ValueError, match="log_det_direction"):
        compose_transforms([], "sideways")  # type: ignore[arg-type]


def test_transform_repr_identifies_its_three_functions():
    representation = repr(tforms.exp)

    assert representation.startswith("Transform(")
    assert "exp" in representation
    assert "log" in representation
