import pytest

from pangolin import functional, ir
from pangolin.functional import transforming
from pangolin.functional.transforming import Transform, compose_transforms


def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale)


def exponential_transform() -> Transform:
    return Transform(
        lambda x: ir.RV(ir.Exp(), x),
        lambda y: ir.RV(ir.Log(), y),
        lambda x, y: ir.RV(ir.Log(), y),
    )


def zero_log_det(x: ir.RV, y: ir.RV) -> ir.RV:
    return ir.RV(ir.Constant(0.0))


def test_public_api_contains_transform_tools():
    assert transforming.__all__ == ["Transform", "compose_transforms"]
    assert functional.Transform is Transform
    assert functional.compose_transforms is compose_transforms


def test_transform_constructs_the_expected_ir():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    result = exponential_transform()(normal)(mean, scale)

    assert result.op == ir.Bijected(
        ir.Normal(),
        ir.Bijector(
            ir.Exp(),
            ir.Log(),
            ir.Composite(2, (ir.Log(),), ((1,),)),
        ),
    )
    assert result.parents == (mean, scale)


def test_transform_calls_each_function_once_and_preserves_distribution_arguments():
    calls: list[str] = []
    configuration = object()

    def distribution(mean: ir.RV, config: object, *, scale: ir.RV) -> ir.RV:
        calls.append("distribution")
        assert config is configuration
        return ir.RV(ir.Normal(), mean, scale)

    def forward(x: ir.RV) -> ir.RV:
        calls.append("forward")
        return ir.RV(ir.Exp(), x)

    def inverse(y: ir.RV) -> ir.RV:
        calls.append("inverse")
        return ir.RV(ir.Log(), y)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        calls.append("log_det_jac")
        return ir.RV(ir.Log(), y)

    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    result = Transform(forward, inverse, log_det_jac)(distribution)(
        mean,
        configuration,
        scale=scale,
    )

    assert calls == ["distribution", "forward", "inverse", "log_det_jac"]
    assert result.parents == (mean, scale)


def test_transform_uses_forward_capture_order_for_every_bijector_operation():
    factor = ir.RV(ir.Constant(2))
    offset = ir.RV(ir.Constant(3))

    def forward(x: ir.RV) -> ir.RV:
        scaled = ir.RV(ir.Mul(), x, factor)
        return ir.RV(ir.Add(), scaled, offset)

    def inverse(y: ir.RV) -> ir.RV:
        shifted = ir.RV(ir.Sub(), y, offset)
        return ir.RV(ir.Div(), shifted, factor)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        magnitude = ir.RV(ir.Abs(), factor)
        return ir.RV(ir.Log(), magnitude)

    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    result = Transform(forward, inverse, log_det_jac)(normal)(mean, scale)

    assert result.parents == (factor, offset, mean, scale)
    assert isinstance(result.op, ir.Bijected)
    assert result.op.bijector == ir.Bijector(
        ir.Composite(3, (ir.Mul(), ir.Add()), ((0, 1), (3, 2))),
        ir.Composite(3, (ir.Sub(), ir.Div()), ((0, 2), (3, 1))),
        ir.Composite(4, (ir.Abs(), ir.Log()), ((2,), (4,))),
        n_biject_params=2,
    )


def test_transform_embeds_constants_created_inside_callbacks():
    transform = Transform(
        lambda x: ir.RV(ir.Add(), x, ir.RV(ir.Constant(1))),
        lambda y: ir.RV(ir.Sub(), y, ir.RV(ir.Constant(1))),
        zero_log_det,
    )
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    result = transform(normal)(mean, scale)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.bijector.n_biject_params == 0
    assert result.op.bijector.forward == ir.Composite(
        1,
        (ir.Constant(1), ir.Add()),
        ((), (0, 1)),
    )


def test_transform_supports_identity_callbacks():
    identity = Transform(lambda x: x, lambda y: y, zero_log_det)
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    result = identity(normal)(mean, scale)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.bijector == ir.Bijector(
        ir.Identity(),
        ir.Identity(),
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
    )


def test_transform_enforces_capture_contracts():
    first = ir.RV(ir.Constant(1))
    second = ir.RV(ir.Constant(1))
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    mismatched = Transform(
        lambda x: ir.RV(ir.Add(), x, first),
        lambda y: ir.RV(ir.Sub(), y, second),
        zero_log_det,
    )
    with pytest.raises(ValueError, match="forward and inverse must capture exactly"):
        mismatched(normal)(mean, scale)

    extra_log_det_capture = Transform(
        lambda x: ir.RV(ir.Exp(), x),
        lambda y: ir.RV(ir.Log(), y),
        lambda x, y: ir.RV(ir.Add(), first, ir.RV(ir.Constant(0))),
    )
    with pytest.raises(ValueError, match="log_det_jac cannot capture"):
        extra_log_det_capture(normal)(mean, scale)


def test_transform_requires_forward_and_inverse_to_depend_on_their_inputs():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    bad_forward = Transform(
        lambda x: ir.RV(ir.Constant(1)),
        lambda y: y,
        zero_log_det,
    )
    with pytest.raises(ValueError, match="forward must depend on its input"):
        bad_forward(normal)(mean, scale)

    bad_inverse = Transform(
        lambda x: ir.RV(ir.Exp(), x),
        lambda y: ir.RV(ir.Constant(1)),
        zero_log_det,
    )
    with pytest.raises(ValueError, match="inverse must depend on its input"):
        bad_inverse(normal)(mean, scale)


def test_transform_rejects_invalid_functions_and_operations():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    def invalid_distribution() -> ir.RV:
        return 1  # type: ignore[return-value]

    with pytest.raises(TypeError, match="transformed function must return exactly one RV"):
        exponential_transform()(invalid_distribution)()

    random_forward = Transform(
        lambda x: ir.RV(ir.Normal(), x, ir.RV(ir.Constant(1))),
        lambda y: y,
        zero_log_det,
    )
    with pytest.raises(ValueError, match="forward op cannot be random"):
        random_forward(normal)(mean, scale)

    with pytest.raises(ValueError, match="base_op must be random"):
        exponential_transform()(lambda x: ir.RV(ir.Exp(), x))(mean)

    with pytest.raises(ValueError, match="base_op must be continuous"):
        exponential_transform()(lambda p: ir.RV(ir.Bernoulli(), p))(scale)


def test_transform_validates_inverse_shape_and_scalar_log_det():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    incompatible_inverse = Transform(
        lambda x: ir.RV(ir.VMap(ir.Add(), (None, 0), 2), x, ir.RV(ir.Constant([1, 2]))),
        lambda y: y,
        zero_log_det,
    )
    with pytest.raises(ValueError, match="forward and inverse ops have incompatible shapes"):
        incompatible_inverse(normal)(mean, scale)

    vector = ir.RV(ir.Constant([0.0, 0.0]))
    covariance = ir.RV(ir.Constant([[1.0, 0.0], [0.0, 1.0]]))
    nonscalar_log_det = Transform(lambda x: x, lambda y: y, lambda x, y: y)
    with pytest.raises(ValueError, match="log_det_jac must return a scalar"):
        nonscalar_log_det(lambda: ir.RV(ir.MultiNormal(), vector, covariance))()


def test_forward_and_inverse_outputs_determine_result_interfaces():
    distribution_interface = ir.RVInterface()
    forward_interface = ir.RVInterface()
    inverse_interface = ir.RVInterface()
    mean = ir.RV(ir.Constant(0), interface=distribution_interface)
    scale = ir.RV(ir.Constant(1), interface=distribution_interface)

    transform = Transform(
        lambda x: ir.RV(ir.Exp(), x, interface=forward_interface),
        lambda y: ir.RV(ir.Log(), y, interface=inverse_interface),
        lambda x, y: ir.RV(ir.Log(), y),
    )
    transformed, reconstructed = transform.apply_and_invert(
        lambda: ir.RV(ir.Normal(), mean, scale, interface=distribution_interface)
    )()

    assert type(transformed) is ir.RV
    assert type(reconstructed) is ir.RV
    assert transformed.interface is forward_interface
    assert reconstructed.interface is inverse_interface
    assert transformed.parents == (mean, scale)
    assert reconstructed.parents == (transformed,)
    assert isinstance(transformed.op, ir.Bijected)
    assert reconstructed.op is transformed.op.bijector.inverse


def test_reverse_and_composition():
    exp = exponential_transform()
    identity = compose_transforms([exp, exp.reverse])
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    result = identity(normal)(mean, scale)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.bijector.forward == ir.Composite(
        1,
        (ir.Exp(), ir.Log()),
        ((0,), (1,)),
    )
    assert result.op.bijector.inverse == ir.Composite(
        1,
        (ir.Exp(), ir.Log()),
        ((0,), (1,)),
    )


def test_composition_validates_its_arguments():
    with pytest.raises(TypeError, match="every item"):
        compose_transforms([exponential_transform(), "log"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="log_det_direction"):
        compose_transforms([], "sideways")  # type: ignore[arg-type]


def test_transform_names_returned_functions():
    transformed = exponential_transform()(normal)
    transformed_and_base = exponential_transform().apply_and_invert(normal)

    assert transformed.__name__ == "transformed_normal"
    assert transformed.__qualname__ == "transformed_normal"
    assert transformed_and_base.__name__ == "transformed_normal_and_base"
    assert transformed_and_base.__qualname__ == "transformed_normal_and_base"
