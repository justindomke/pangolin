import pytest

from pangolin import ir
from pangolin.ir import Scan


@pytest.mark.parametrize(
    "base, axes",
    [(ir.Exp(), ()), (ir.Add(), (0,)), (ir.Add(), (None,)), (ir.StudentT(), (0, None))],
)
def test_prints_axes_as_a_list(base, axes):
    op = Scan(base, 3, axes)
    assert repr(op) == f"Scan({base!r}, 3, {list(axes)})"
    assert str(op) == f"scan({base}, 3, {list(axes)})"
    assert eval(repr(op), dict(vars(ir))) == op
    assert op.in_axes == axes


def scalar_three_input_op() -> ir.Composite:
    return ir.Composite(3, (ir.Add(), ir.Add()), [[0, 1], [3, 2]])


def test_exponential():
    op = Scan(ir.Exponential(), 5, ())
    assert op.get_shape(()) == (5,)


def test_multi_normal_fixed_cov():
    ndims = 3
    length = 5
    op = Scan(ir.MultiNormal(), length, (None,))
    assert op.get_shape((ndims,), (ndims, ndims)) == (length, ndims)


def test_multi_normal_varying_cov():
    ndims = 3
    length = 5
    op = Scan(ir.MultiNormal(), length, (0,))
    assert op.get_shape((ndims,), (length, ndims, ndims)) == (length, ndims)


def test_multi_normal_varying_cov_dim1():
    ndims = 3
    length = 5
    op = Scan(ir.MultiNormal(), length, (1,))
    assert op.get_shape((ndims,), (ndims, length, ndims)) == (length, ndims)


def test_multi_normal_varying_cov_dim2():
    ndims = 3
    length = 5
    op = Scan(ir.MultiNormal(), length, (2,))
    assert op.get_shape((ndims,), (ndims, ndims, length)) == (length, ndims)


def test_matrix_mul():
    ndims = 3
    length = 5
    op = Scan(ir.Matmul(), length, (None,))
    assert op.get_shape((ndims,), (ndims, ndims)) == (length, ndims)


def test_vmap_autoregressive():
    dims = 3
    length = 5
    base_op = ir.VMap(ir.Mul(), (0, 0))
    op = Scan(base_op, length, (None,))
    assert op.get_shape((dims,), (dims,)) == (length, dims)


def test_vmap_autoregressive_mapped0():
    dims = 3
    length = 5
    base_op = ir.VMap(ir.Mul(), (0, 0))
    op = Scan(base_op, length, (0,))
    assert op.get_shape((dims,), (length, dims)) == (length, dims)


def test_vmap_autoregressive_mapped1():
    dims = 3
    length = 5
    base_op = ir.VMap(ir.Mul(), (0, 0))
    op = Scan(base_op, length, (1,))
    assert op.get_shape((dims,), (dims, length)) == (length, dims)


def test_scan_accepts_negative_mapped_axis():
    ndims = 3
    length = 5
    op = Scan(ir.MultiNormal(), length, (-1,))

    assert op.get_shape((ndims,), (ndims, ndims, length)) == (length, ndims)


def test_scan_validates_all_mapped_inputs():
    length = 5
    op = Scan(scalar_three_input_op(), length, (0, 0))

    assert op.get_shape((), (length,), (length,)) == (length,)

    with pytest.raises(ValueError, match="scanned input 1 has mapped axis length 4, expected 5"):
        op.get_shape((), (length,), (length - 1,))


def test_scan_allows_unmapped_inputs_with_unrelated_dimensions():
    ndims = 3
    length = 5
    op = Scan(ir.MultiNormal(), length, (None,))

    assert op.get_shape((ndims,), (ndims, ndims)) == (length, ndims)


def test_scan_requires_base_output_shape_to_match_carry():
    op = Scan(ir.Sum(axis=0), 5, ())

    with pytest.raises(ValueError, match=r"output shape \(3,\) must match carry shape \(2, 3\)"):
        op.get_shape((2, 3))


@pytest.mark.parametrize(
    ("axis", "mapped_shape"),
    [
        (0, (6, 3, 3)),
        (1, (3, 6, 3)),
        (2, (3, 3, 6)),
        (-1, (3, 3, 6)),
    ],
)
def test_scan_rejects_mapped_axis_length_different_from_scan_length(axis, mapped_shape):
    op = Scan(ir.MultiNormal(), 5, (axis,))

    with pytest.raises(ValueError, match="mapped axis length 6, expected 5"):
        op.get_shape((3,), mapped_shape)


@pytest.mark.parametrize("axis", [3, -4])
def test_scan_rejects_out_of_bounds_mapped_axis(axis):
    op = Scan(ir.MultiNormal(), 5, (axis,))

    with pytest.raises(ValueError, match=r"in_axes\[0\].* is out of bounds"):
        op.get_shape((3,), (3, 3, 5))


def test_scan_rejects_mapping_over_scalar_input():
    op = Scan(ir.Add(), 5, (0,))

    with pytest.raises(ValueError, match="input with 0 dimensions"):
        op.get_shape((), ())


@pytest.mark.parametrize("other_shapes", [(), ((), ())])
def test_scan_requires_one_axis_entry_per_scanned_input(other_shapes):
    op = Scan(ir.Add(), 5, (None,))

    with pytest.raises(ValueError, match="1 in_axes entries"):
        op.get_shape((), *other_shapes)


def test_scan_requires_op():
    with pytest.raises(TypeError, match="base_op must be an Op"):
        Scan("exponential", 5, ())  # type: ignore[arg-type]


@pytest.mark.parametrize("length", [True, None, 1.5, "5"])
def test_scan_requires_integer_length(length):
    with pytest.raises(TypeError, match="length must be an integer"):
        Scan(ir.Exponential(), length, ())


@pytest.mark.parametrize("length", [0, -1])
def test_scan_requires_positive_length(length):
    with pytest.raises(ValueError, match="length must be positive"):
        Scan(ir.Exponential(), length, ())


def test_scan_requires_in_axes_sequence():
    with pytest.raises(TypeError, match="in_axes must be a sequence"):
        Scan(ir.Exponential(), 5, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", [True, 0.5, "0"])
def test_scan_requires_integer_or_none_axes(axis):
    with pytest.raises(TypeError, match="each in_axes entry must be an integer or None"):
        Scan(ir.Add(), 5, (axis,))
