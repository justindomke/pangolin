from re import M
import jax.tree_util

from pangolin import ir
from pangolin import interface as pi
from pangolin.interface import vmap
from pangolin.interface.base import InfixRV, AbstractOp, constant
from pangolin.interface.vfor import extract_from_rvs
from pangolin.interface.vfor import vfor, extract_from_rvs_single_axis, Axis
import numpy as np
import pytest


def test_step1_extract_mixed_dims():
    a = constant([1.0, 2.0, 3.0])
    b = constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    i = InfixRV(AbstractOp(()))
    j = InfixRV(AbstractOp(()))

    out = a[i] * b[i, j]

    arrays, dims, replay = extract_from_rvs([i, j], out)

    assert len(arrays) == 2
    assert arrays[0] is a
    assert arrays[1] is b
    assert dims[0] == [0, None]
    assert dims[1] == [0, 1]


def test_minimal_dims_positions():
    """
    a[i] * b[i, j]: dims must record, per array, for each dummy index,
    0 if that index is used (once — indexing consumes it) else None.
    This is the contract documented in extract_from_rvs's own docstring.
    """
    a = constant([1.0, 2.0, 3.0])
    b = constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    i = InfixRV(AbstractOp(()))
    j = InfixRV(AbstractOp(()))

    out = a[i] * b[i, j]
    arrays, dims, replay = extract_from_rvs([i, j], out)

    assert arrays == [a, b]
    assert dims == [[0, None], [0, 1]]


def test_step2_single_axis_extraction_trouble3():
    """
    trouble3's c, decomposed per-axis as vfor does internally.

    c(i, j) = normal(a[i], b[i, j])  where a has shape (I,), b has shape (I, J).

    For the j-axis (size J): extract_from_rvs_single_axis should report that
    'a' (as a[i]) is NOT indexed by j, while 'b' IS. The indexed_dims must
    line up so that the vmap over j maps b but not a.
    """
    I, J = 10, 20

    a = vfor(lambda i: pi.normal(0, 10), i=I)  # shape (I,)
    b = vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)  # shape (I, J)
    assert a.shape == (I,)
    assert b.shape == (I, J)

    # Build the body with real Axis RVs, as extract_from_rvs_single_axis expects
    ax_j = InfixRV(Axis(J))
    # We can't easily build a[i] with a dummy i here since extract_from_rvs_single_axis
    # takes ONE axis. Instead, test what happens when we ask about the j-axis on
    # an expression that involves both a[i] and b[i,j] — use constant arrays directly.

    a_const = constant([1.0] * I)  # shape (I,)
    b_const = constant([[1.0] * J] * I)  # shape (I, J)

    ax_i = InfixRV(Axis(I))
    ax_j2 = InfixRV(Axis(J))

    out = pi.normal(a_const[ax_i], b_const[ax_i, ax_j2])

    indexed_rvs, indexed_dims, replay = extract_from_rvs_single_axis(ax_j2, out)

    # a_const[ax_i] does not involve ax_j2 → should not appear, or appear with dim None
    # b_const[ax_i, ax_j2] involves ax_j2 at position 1 → dim should reflect that
    print("indexed_rvs:", indexed_rvs)
    print("indexed_dims:", indexed_dims)

    # b must be in there with a j-mapping; a must either be absent or None-mapped
    assert len(indexed_rvs) >= 1
    for rv, dim in zip(indexed_rvs, indexed_dims):
        if rv is b_const:
            assert dim is not None, "b should be mapped over j"


def test_step3_manual_nested_vmap_trouble3_corrected():
    """
    c(i,j) = a[i] + b[i,j], a:(I,), b:(I,J).

    Correct manual nesting: inner vmap must use in_axes=0 for b, because the
    outer vmap slices b to (J,) before the inner one sees it. Axis 1 of the
    ORIGINAL b becomes axis 0 of the SLICED b[i].
    """
    I, J = 10, 20
    a = constant([1.0] * I)
    b = constant([[1.0] * J] * I)

    def body(a_i, b_ij):
        return a_i + b_ij

    f_j = vmap(body, (None, 0), J)  # <- 0, not 1
    f_i = vmap(f_j, (0, 0), I)
    c = f_i(a, b)
    print("c.shape =", c.shape)
    assert c.shape == (I, J)


def test_step3_control_trouble2():
    """
    trouble2 pattern: c(i,j) = f(a[i], b[j]), a:(I,), b:(J,)
    dims=[[0,None],[None,0]] (a uses i, b uses j)
      idx=1 (j): in_axes=(None, 0), size=J
      idx=0 (i): in_axes=(0, None), size=I
    """
    I, J = 10, 20
    a = constant([1.0] * I)
    b = constant([1.0] * J)

    def body(a_i, b_j):
        return a_i + b_j

    f_j = vmap(body, (None, 0), J)  # b over axis 0 (j), a not mapped
    f_i = vmap(f_j, (0, None), I)  # a over axis 0 (i), b-result not mapped
    c = f_i(a, b)
    print("c.shape =", c.shape)
    assert c.shape == (I, J)


def test_step3_vmap_none_axis1():
    """Simplest possible test: vmap with in_axes=(None, 1)."""
    b = constant([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    a = constant([10.0, 20.0])  # (2,)

    def body(x, y):
        return x + y

    f = vmap(body, (None, 1), 2)  # a not mapped, b mapped over axis 1
    result = f(a, b)
    print("result.shape =", result.shape)
    assert result.shape == (2, 2)


def test_vmap_mixed_mapped_unmapped_no_broadcast():
    """
    Simplest possible: two scalars in body, one from mapped, one from unmapped.
    No broadcasting confusion.
    """
    a = constant(5.0)  # scalar, NOT mapped
    b = constant([[1.0, 2.0], [3.0, 4.0]])  # (2,2), mapped over axis 1

    def body(x, y):
        return x + y

    f = vmap(body, (None, 1), 2)
    result = f(a, b)
    # result[j] = a + b[:, j], each is scalar + (2,) = (2,)
    # so result should be (2, 2)
    print("result.shape =", result.shape)
    assert result.shape == (2, 2)


def test_step4_nested_vmap_over_vfor_outputs():
    """
    Identical to test_step3_manual_nested_vmap_trouble3_corrected (which passes),
    except a and b come from vfor instead of constant().
    """
    I, J = 10, 20
    a = vfor(lambda i: pi.normal(0, 10), i=I)  # (I,)
    b = vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)  # (I, J)
    assert a.shape == (I,)
    assert b.shape == (I, J)

    def body(a_i, b_ij):
        return a_i + b_ij

    f_j = vmap(body, (None, 0), J)
    f_i = vmap(f_j, (0, 0), I)
    c = f_i(a, b)
    print("c.shape =", c.shape)
    assert c.shape == (I, J), f"expected ({I}, {J}), got {c.shape}"


def test_step5_vfor_pipeline_manual():
    """
    Replicate vfor's internal steps for trouble3's c, checking the in_axes
    and output shape at each stage.
    """

    I, J = 10, 20
    a = vfor(lambda i: pi.normal(0, 10), i=I)
    b = vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)

    dummy_i = InfixRV(AbstractOp(()))
    dummy_j = InfixRV(AbstractOp(()))
    output = pi.normal(a[dummy_i], b[dummy_i, dummy_j])

    dummy_indices = [dummy_i, dummy_j]
    arrays, dims, replay = extract_from_rvs(dummy_indices, output)
    print("dims =", dims)

    def base_fun(*args):
        return replay(*args)

    # vfor's loop, but printing in_axes each iteration
    new_fun = base_fun
    for idx in range(1, -1, -1):  # idx = 1 (j), then 0 (i)
        in_axes = tuple(dim[idx] for dim in dims)
        axis_size = (I, J)[idx]
        print(f"idx={idx}  in_axes={in_axes}  axis_size={axis_size}")
        new_fun = vmap(new_fun, in_axes, axis_size)

    c = new_fun(*arrays)
    print("c.shape =", c.shape)
    assert c.shape == (I, J), f"expected ({I}, {J}), got {c.shape}"


def test_step6_nested_axis_correctA():
    """Inner vmap maps axis 0 of the sliced array — correct usage, passes."""
    I, J = 10, 20
    b = constant(np.zeros((I, J)))

    f_j = vmap(lambda b_ij: b_ij + 1.0, 0, J)  # axis 0 of the (J,) slice
    f_i = vmap(f_j, 0, I)

    c = f_i(b)
    assert c.shape == (I, J)


def test_step6_nested_axis_correctB():
    """Inner vmap maps axis 0 of the sliced array — correct usage, passes."""
    I, J = 10, 20
    b = constant(np.zeros((I, J)))

    c = vmap(vmap(lambda b_ij: b_ij + 1.0, axis_size=J), axis_size=I)(b)

    assert c.shape == (I, J)


def test_step6_nested_axis_correctC():
    """Inner vmap maps axis 0 of the sliced array — correct usage, passes."""
    I, J = 10, 20
    b = constant(np.zeros((I, J)))

    c = vmap(vmap(lambda b_ij: b_ij + 1.0, 0, axis_size=J), 0, axis_size=I)(b)

    assert c.shape == (I, J)


def test_step7_mixed_none_nested():
    """
    Inner map: b mapped (axis 0 of the slice), a NOT mapped (None).
    This is the exact structure trouble3's c needs for its j-axis.
    """
    I, J = 10, 20
    a = constant(np.zeros(I))  # (I,)
    b = constant(np.zeros((I, J)))  # (I, J)

    def body(a_i, b_ij):
        return a_i + b_ij  # scalars → scalar

    f_j = vmap(body, (None, 0), J)  # a unmapped, b mapped over axis 0
    f_i = vmap(f_j, (0, 0), I)  # both mapped over axis 0
    c = f_i(a, b)
    assert c.shape == (I, J)


def test_reduce_vfor_two_dummies_one_array():
    """
    vfor over ONE array indexed by TWO dummies → dims=[[0,1]].
    Theory: inner (j) map uses slot 1, but the (I,J) array is already
    sliced by the i-map, so axis 1 doesn't exist → IndexError.
    """
    I, J = 10, 20
    x = constant(np.zeros((I, J)))

    c = vfor(lambda i, j: x[i, j] + 1.0, i=I, j=J)
    assert c.shape == (I, J)


def test_reduce_manual_raw_slot_CRASHES():
    """
    Manually replicate vfor's exact in_axes for dims=[[0,1]].
    idx=1 (j): in_axes=(1,)  → raw slot, uncorrected.
    idx=0 (i): in_axes=(0,)
    Theory predicts IndexError: tuple index out of range.
    """
    I, J = 10, 20
    x = constant(np.zeros((I, J)))

    f_j = vmap(lambda s: s + 1.0, 1, J)  # slot 1 — what vfor currently emits
    f_i = vmap(f_j, 0, I)
    with pytest.raises(IndexError):
        f_i(x)


def test_reduce_manual_shifted_axis_PASSES():
    """
    Same computation, but inner map uses the axis AFTER the i-map's shift.
    idx=1 (j): dims slot 1 − (one consumed axis before it) = 0.
    Theory predicts this passes.
    """
    I, J = 10, 20
    x = constant(np.zeros((I, J)))

    f_j = vmap(lambda s: s + 1.0, 0, J)  # shifted axis 0
    f_i = vmap(f_j, 0, I)
    c = f_i(x)
    assert c.shape == (I, J)


def test_decompose_extract_single_array():
    """What does extract_from_rvs return for x[i,j]? Confirm dims and arrays."""
    I, J = 10, 20
    x = constant(np.zeros((I, J)))
    i = InfixRV(AbstractOp(()))
    j = InfixRV(AbstractOp(()))
    out = x[i, j] + 1.0

    arrays, dims, replay = extract_from_rvs([i, j], out)
    print("arrays:", arrays)
    print("dims:", dims)
    assert arrays == [x]
    assert dims == [[0, 1]]


def test_decompose_replay_body_shape():
    """
    The body is replay(x[i,j]-replacement). vmap's _set_shape calls the base
    body with FULL x (shape (I,J)), not a scalar. What shape does replay return
    when given the full array?
    """
    I, J = 10, 20
    x = constant(np.zeros((I, J)))
    i = InfixRV(AbstractOp(()))
    j = InfixRV(AbstractOp(()))
    out = x[i, j] + 1.0

    arrays, dims, replay = extract_from_rvs([i, j], out)

    # vmap's _set_shape calls base_fun with the full (un-sliced) argument:
    base_out = replay(x.op.value)  # x is (I, J), NOT a scalar
    print("replay(x).shape =", base_out.shape)


def test_decompose_replay_1d():
    """
    The body is replay(x[i,j]-replacement). vmap's _set_shape calls the base
    body with FULL x (shape (I,J)), not a scalar. What shape does replay return
    when given the full array?
    """
    I = 10
    x = constant(np.zeros(I))
    i = InfixRV(AbstractOp(()))
    out = x[i] + 1.0

    arrays, dims, replay = extract_from_rvs([i], out)

    assert arrays == [x]
    assert dims == [[0]]

    base_out = replay(x.op.value[0])
    assert base_out.shape == ()


def test_decompose_replay_2d():
    """
    The body is replay(x[i,j]-replacement). vmap's _set_shape calls the base
    body with FULL x (shape (I,J)), not a scalar. What shape does replay return
    when given the full array?
    """
    I = 10
    J = 20
    x = constant(np.zeros((I, J)))
    i = InfixRV(AbstractOp(()))
    j = InfixRV(AbstractOp(()))
    out = x[i, j] + 1.0

    arrays, dims, replay = extract_from_rvs([i, j], out)

    assert arrays == [x]
    assert dims == [[0, 1]]

    base_out = replay(x.op.value[0, 0])
    assert base_out.shape == ()
