from pangolin import inference_jags
from pangolin import *
import numpy as np


def test_reference1():
    r0 = inference_jags.Reference("sup", (5, 3))
    r1 = r0.index(0, "i")
    assert str(r0) == "sup"
    assert str(r1) == "sup[i,]"


def test_reference2():
    r0 = inference_jags.Reference("sup", (5, 3))
    r1 = r0.index(1, "i")
    assert str(r0) == "sup"
    assert str(r1) == "sup[,i]"


def test_reference3():
    r0 = inference_jags.Reference("sup", (5, 3))
    r1 = r0.index(0, "i")
    r2 = r1.index(0, "j")
    assert str(r0) == "sup"
    assert str(r1) == "sup[i,]"
    assert str(r2) == "sup[i,j]"


def test_reference4():
    r0 = inference_jags.Reference("sup", (9, 9, 9))
    r1 = r0.index(1, "i")
    r2 = r1.index(1, "j")
    r3 = r2.index(0, "k")
    assert str(r0) == "sup"
    assert str(r1) == "sup[,i,]"
    assert str(r2) == "sup[,i,j]"
    assert str(r3) == "sup[k,i,j]"


# def assert_sample_close(var, value):
#     """
#     assert that a single sample of var is close to value
#     """
#     # inf = inference_jags.JAGSInference(niter=1)
#     [out] = inference_jags.sample_flat([var], [], [], niter=1)
#     assert np.allclose(out[0], value)
#
#
# def test_indexing1():
#     x = np.random.randn(5)
#     y = makerv(x)[:]
#     expected = x[:]
#     assert_sample_close(y, expected)
#
#
# def test_indexing2():
#     x = np.random.randn(10)
#     y = makerv(x)[2:8]
#     expected = x[2:8]
#     assert_sample_close(y, expected)
#
#
# def test_indexing3():
#     x = np.random.randn(5, 7)
#     y = makerv(x)[:]
#     expected = x[:]
#     assert_sample_close(y, expected)
#
#
# def test_indexing4():
#     x = np.random.randn(5, 7)
#     y = makerv(x)[2:4]
#     expected = x[2:4]
#     assert_sample_close(y, expected)
#
#
# def test_indexing5():
#     x = np.random.randn(5, 7)
#     y = makerv(x)[2:4, 1:2]
#     expected = x[2:4, 1:2]
#     assert_sample_close(y, expected)
#
#
# def test_indexing6():
#     x = np.random.randn(5, 7)
#     y = makerv(x)[[2, 3]]
#     expected = x[[2, 3]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing7():
#     x = np.random.randn(3, 3)
#     y = makerv(x)[2]
#     expected = x[2]
#     assert_sample_close(y, expected)
#
#
# def test_indexing8():
#     x = np.random.randn(3, 3)
#     y = makerv(x)[2, :]
#     expected = x[2, :]
#     assert_sample_close(y, expected)
#
#
# def test_indexing9():
#     x = np.random.randn(3, 3)
#     y = makerv(x)[2, 1]
#     expected = x[2, 1]
#     assert_sample_close(y, expected)
#
#
# def test_indexing10():
#     x = np.random.randn(3, 3, 5)
#     y = makerv(x)[2, 1]
#     expected = x[2, 1]
#     assert_sample_close(y, expected)
#
#
# def test_indexing11():
#     x = np.random.randn(3, 4, 5)
#     y = makerv(x)[2, :, 4]
#     expected = x[2, :, 4]
#     assert_sample_close(y, expected)
#
#
# def test_indexing12():
#     x = np.random.randn(5, 5)
#     y = makerv(x)[:, [1, 4, 2]]
#     expected = x[:, [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing13():
#     x = np.random.randn(5, 5, 5)
#     y = makerv(x)[:, [1, 4, 2]]
#     expected = x[:, [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing14():
#     x = np.random.randn(5, 5)
#     y = makerv(x)[[2, 1, 2], [1, 4, 2]]
#     expected = x[[2, 1, 2], [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing15():
#     x = np.random.randn(5, 5, 7)
#     y = makerv(x)[[2, 1, 2], [1, 4, 2]]
#     expected = x[[2, 1, 2], [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing16():
#     x = np.random.randn(5, 5, 7)
#     y = makerv(x)[:, [2, 1, 2], [1, 4, 2]]
#     expected = x[:, [2, 1, 2], [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing17():
#     x = np.random.randn(6, 5, 7)
#     y = makerv(x)[[2, 1, 2], 1:4, [1, 4, 2]]
#     expected = x[[2, 1, 2], 1:4, [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing18():
#     x = np.random.randn(5, 5, 7)
#     y = makerv(x)[[2, 1, 2], 1:4]
#     expected = x[[2, 1, 2], 1:4]
#     assert_sample_close(y, expected)
#
#
# def test_indexing19():
#     x = np.random.randn(3, 5)
#     y = plate(makerv(x))(lambda x_i: x_i[1])
#
#     expected = x[:, 1]
#     assert_sample_close(y, expected)
#
#
# def test_indexing20():
#     x = np.random.randn(3, 5)
#     y = plate(makerv(x))(lambda x_i: x_i[[3, 1, 0, 2]])
#
#     expected = x[:, [3, 1, 0, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing21():
#     x = np.random.randn(7, 5)
#     y = plate(makerv(x), in_axes=1)(lambda x_i: x_i[[3, 1, 0, 2]])
#
#     expected = x[[3, 1, 0, 2], :].T
#     assert_sample_close(y, expected)
#
#
# def test_indexing22():
#     x = np.random.randn(7, 5)
#     y = plate(makerv(x), in_axes=1)(lambda x_i: x_i[:])
#
#     expected = x.T
#     assert_sample_close(y, expected)
#
#
# def test_indexing23():
#     x = np.random.randn(7, 5)
#     expected = jax.vmap(lambda x_i: x_i[:], in_axes=1)(x)
#     y = vmap(lambda x_i: x_i[:], in_axes=1)(makerv(x))
#     assert_sample_close(y, expected)
#
#
# def test_indexing24():
#     x = np.random.randn(7, 5, 11)
#     expected = jax.vmap(lambda x_i: x_i[4, 2:4], in_axes=1)(x)
#     y = vmap(lambda x_i: x_i[4, 2:4], in_axes=1)(makerv(x))
#     assert_sample_close(y, expected)
#
#
# def test_indexing29():
#     x = np.random.randn(7, 5, 11, 13, 17)
#     expected = jax.vmap(lambda x_i: x_i[:, [1, 2, 1], 2:4, :], in_axes=2)(x)
#     y = vmap(lambda x_i: x_i[:, [1, 2, 1], 2:4, :], in_axes=2)(makerv(x))
#     assert_sample_close(y, expected)
#
#
# def test_indexing30():
#     """
#     simplest possible test to trigger advanced indexing
#     """
#     x = np.random.randn(4, 5, 6, 7)
#     y = makerv(x)[:, [2, 1, 2], :, [1, 4, 2]]
#     expected = x[:, [2, 1, 2], :, [1, 4, 2]]
#     assert_sample_close(y, expected)
#
#
# def test_indexing31():
#     """
#     comically complicated indexing and vmap
#     this tries to test *everything*:
#     * the starting array has 7 dimensions
#     * we vmap over all pairs of 2 axes
#     * then we do a super complex index using a mixture of slicing and advanced
#     indexing and also dimensions that are just left implicit
#     """
#
#     x = np.random.randn(4, 5, 6, 4, 5, 3, 3)
#
#     # try all possible vmap pairs!
#     for in_axis1 in range(7):
#         for in_axis2 in range(6):
#             y = vmap(
#                 lambda x_i: vmap(
#                     lambda x_ij: x_ij[:, [1, 2, 1], 2:4, [0, 0, 0]], in_axes=in_axis2
#                 )(x_i),
#                 in_axes=in_axis1,
#             )(makerv(x))
#             expected = jax.vmap(
#                 lambda x_i: jax.vmap(
#                     lambda x_ij: x_ij[:, [1, 2, 1], 2:4, [0, 0, 0]], in_axes=in_axis2
#                 )(x_i),
#                 in_axes=in_axis1,
#             )(x)
#             assert_sample_close(y, expected)
#
#
# def test_indexing32():
#     """
#     test 2-d index arrays!
#     """
#     x = np.random.randn(5)
#     idx = np.array([[0, 2], [3, 4]])
#     y = makerv(x)[idx]
#     expected = x[idx]
#     assert_sample_close(y, expected)
#
#
# def test_indexing33():
#     x = np.random.randn(5, 6)
#     idx = np.array([[0, 2], [3, 4]])
#     y = makerv(x)[:, idx]
#     expected = x[:, idx]
#     assert_sample_close(y, expected)
#
#
# def test_indexing34():
#     x = np.random.randn(5, 6)
#     idx = np.array([[0, 2], [3, 4]])
#     y = makerv(x)[idx, :]
#     expected = x[idx, :]
#     assert_sample_close(y, expected)
#
#
# def test_indexing35():
#     """
#     indices separated to trigger advanced indexing
#     """
#     x = np.random.randn(5, 6, 5)
#     idx = np.array([[0, 2], [3, 4]])
#     y = makerv(x)[idx, :, idx]
#     expected = x[idx, :, idx]
#     assert_sample_close(y, expected)
#
#
# def test_indexing36():
#     """
#     indices together to trigger non-advanced indexing
#     """
#     x = np.random.randn(5, 6, 5)
#     idx = [[0, 2], [3, 4]]
#     y = makerv(x)[:, idx, idx]
#     expected = x[:, idx, idx]
#     assert_sample_close(y, expected)
#
#
# def test_indexing37():
#     """
#     indices together to trigger non-advanced indexing
#     """
#     x = np.random.randn(5, 6, 5, 4)
#     idx = [[0, 2], [3, 4]]
#     y = makerv(x)[:, idx, idx, :]
#     expected = x[:, idx, idx, :]
#     assert_sample_close(y, expected)
#
#
# def test_indexing38():
#     """
#     indices together inside vmap
#     """
#     x = np.random.randn(5, 6, 5, 7)
#     idx = [[0, 2], [3, 4]]
#     y = vmap(lambda x_i: x_i[:, idx, idx], in_axes=0)(makerv(x))
#     expected = jax.vmap(lambda x_i: x_i[:, idx, idx], in_axes=0)(x)
#     assert_sample_close(y, expected)
#
#
# def test_indexing39():
#     """
#     indices separated inside vmap
#     """
#     x = np.random.randn(5, 6, 5, 7)
#     idx = [[0, 2], [3, 4]]
#     y = vmap(lambda x_i: x_i[idx, :, idx], in_axes=0)(makerv(x))
#     expected = jax.vmap(lambda x_i: x_i[idx, :, idx], in_axes=0)(x)
#     assert_sample_close(y, expected)
#
#
# def test_indexing40():
#     """
#     even more comically complicated indexing and vmap
#     this tries to test *everything*:
#     * the starting array has 7 dimensions
#     * we vmap over all pairs of 2 axes
#     * then have a mixture of full slices, short slice, and 2-d indices
#     * then we do a super complex index using a mixture of slicing and advanced
#     indexing and also dimensions that are just left implicit
#     """
#
#     x = np.random.randn(4, 5, 4, 4, 5, 4, 4)
#
#     # try all possible vmap pairs!
#     for in_axis1 in range(7):
#         for in_axis2 in range(6):
#             y = vmap(
#                 lambda x_i: vmap(
#                     lambda x_ij: x_ij[
#                         :, [[1, 2, 1], [0, 0, 0]], 2:4, [[0, 0, 0], [1, 0, 1]]
#                     ],
#                     in_axes=in_axis2,
#                 )(x_i),
#                 in_axes=in_axis1,
#             )(makerv(x))
#             expected = jax.vmap(
#                 lambda x_i: jax.vmap(
#                     lambda x_ij: x_ij[
#                         :, [[1, 2, 1], [0, 0, 0]], 2:4, [[0, 0, 0], [1, 0, 1]]
#                     ],
#                     in_axes=in_axis2,
#                 )(x_i),
#                 in_axes=in_axis1,
#             )(x)
#             assert_sample_close(y, expected)
#
#
# def test_indexing41():
#     """
#     like previous but indices put next to each other to avoid advanced indexing
#     """
#
#     x = np.random.randn(4, 5, 4, 4, 5, 4, 4)
#
#     # try all possible vmap pairs!
#     for in_axis1 in range(1):
#         for in_axis2 in range(1):
#             y = vmap(
#                 lambda x_i: vmap(
#                     lambda x_ij: x_ij[
#                         :, [[1, 2, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 1]], 2:4
#                     ],
#                     in_axes=in_axis2,
#                 )(x_i),
#                 in_axes=in_axis1,
#             )(makerv(x))
#             expected = jax.vmap(
#                 lambda x_i: jax.vmap(
#                     lambda x_ij: x_ij[
#                         :, [[1, 2, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 1]], 2:4
#                     ],
#                     in_axes=in_axis2,
#                 )(x_i),
#                 in_axes=in_axis1,
#             )(x)
#             assert_sample_close(y, expected)
