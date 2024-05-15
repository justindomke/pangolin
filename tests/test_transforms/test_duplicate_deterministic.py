from pangolin.interface import normal, normal_scale, makerv, exp
from pangolin import dag
from pangolin.transforms.transforms_util import replace
import numpy as np

from pangolin import transforms
from pangolin.transforms.duplicate_deterministic import duplicate_deterministic
from pangolin.transforms.transforms import apply_transforms

# For now this whole file seems to be obsolete because duplicate deterministic RVs
# are the same by definition

# def test_duplicate1():
#     x = normal(0, 1)
#     y = exp(x)
#     z = exp(x)
#     [new_y, new_z], _, _ = duplicate_deterministic([y, z], [], [])
#     assert y != z
#     assert new_y == new_z
#
#
# def test_duplicate1_apply():
#     x = normal(0, 1)
#     y = exp(x)
#     z = exp(x)
#     tforms = [duplicate_deterministic]
#     [new_y, new_z], tmp, _ = apply_transforms(tforms, [y, z], None, None)
#     assert y != z
#     assert new_y == new_z
#
#     rv_dict = {"y": y, "z": z}
#
#     new_rv_dict, _, _ = apply_transforms(tforms, rv_dict, None, None)
#     new_y = new_rv_dict["y"]
#     new_z = new_rv_dict["z"]
#     assert y != z
#     assert new_y == new_z
#
#
# def test_duplicate2():
#     x = makerv(1.3)
#     y = makerv(1.3)
#     [new_x, new_y], _, _ = duplicate_deterministic([x, y], [], [])
#     assert x != y
#     assert new_x == new_y
#
#
# def test_duplicate3():
#     x = normal(0, 1)
#     y = exp(x)
#     z = exp(x)
#     a = normal(y, z)
#     [new_y, new_z, new_a], _, _ = duplicate_deterministic([y, z, a], [], [])
#     assert y != z
#     assert new_y == new_z
#     assert new_y.parents == (x,)
#     assert new_z.parents == (x,)
#     assert new_a != a
#     assert new_a.parents == (new_y, new_z)
#
#
# def test_duplicate4():
#     """
#     same as previous except a considered given
#     """
#     x = normal(0, 1)
#     y = exp(x)
#     z = exp(x)
#     a = normal(y, z)
#     [new_y, new_z], [new_a], [new_val] = duplicate_deterministic([y, z], [a], [1.0])
#     print(f"{new_val=}")
#     assert y != z
#     assert new_y == new_z
#     assert new_y.parents == (x,)
#     assert new_z.parents == (x,)
#     assert new_a != a
#     assert new_a.parents == (new_y, new_z)
#     assert new_a.parents[0] == new_a.parents[1]
#
#
# def test_duplicate5():
#     """
#     same as previous except y and z not explicitly passed
#     """
#     x = normal(0, 1)
#     y = exp(x)
#     z = exp(x)
#     a = normal(y, z)
#     [new_a], _, _ = duplicate_deterministic([a], [], [])
#     (new_y, new_z) = new_a.parents
#     assert y != z
#     assert new_y == new_z
#     assert new_y.parents == (x,)
#     assert new_z.parents == (x,)
#     assert new_a != a
#     assert new_a.parents == (new_y, new_z)
#
#
# def test_duplicate6():
#     loc = normal(0, 1)
#     log_scale = normal(2, 3)
#     x = [normal(loc, exp(log_scale)) for i in range(5)]
#     new_x, _, _ = duplicate_deterministic(x, [], [])
#     for xi in new_x:
#         assert xi.cond_dist == normal_scale
#         assert xi.parents == new_x[0].parents
#
#
# def test_duplicate7():
#     loc = normal(0, 1)
#     log_scale = normal(2, 3)
#     x = [normal(loc, exp(log_scale)) for i in range(5)]
#     tforms = [duplicate_deterministic]
#     new_x, _, _ = apply_transforms(tforms, x, [], [])
#     for xi in new_x:
#         assert xi.cond_dist == normal_scale
#         assert xi.parents == new_x[0].parents  # thus, x-node shared
