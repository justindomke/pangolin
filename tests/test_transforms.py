from pangolin import dag
from pangolin.interface import *

# from pangolin import new_infer as infer
from pangolin import transforms


def test_non_centered1():
    loc = makerv(1.1)
    scale = makerv(2.2)
    x = normal(loc, scale)
    rule = transforms.NonCenteredNormalTransformationRule()
    tform = rule.get_transform(x.cond_dist, pars_included=None, obs_below=None)
    [new_x] = tform.regenerate((loc, scale))
    assert new_x.cond_dist == add
    assert new_x.parents[0] == loc
    assert new_x.parents[1].cond_dist == mul
    assert new_x.parents[1].parents[0] == scale
    assert new_x.parents[1].parents[1].cond_dist == normal_scale


def test_non_centered2():
    # model
    loc = makerv(1.1)
    scale = makerv(2.2)
    x = normal(loc, scale)
    # transform
    rule = transforms.NonCenteredNormalTransformationRule()
    replacements = rule.apply(x)
    new_x = replacements[x]
    # check
    assert new_x.cond_dist == add
    assert new_x.parents[0] == loc
    assert new_x.parents[1].cond_dist == mul
    assert new_x.parents[1].parents[0] == scale
    assert new_x.parents[1].parents[1].cond_dist == normal_scale


def test_normal_normal1():
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = normal_scale(a, b)
    x = normal_scale(z, c)
    rule = transforms.NormalNormalTransformationRule()
    tform = rule.get_transform(x.cond_dist, z.cond_dist, obs_below=[True, False])
    [new_x, new_z] = tform.regenerate((z, c), (a, b))

    assert new_x.cond_dist == normal_scale
    assert new_z.cond_dist == normal_scale


def test_normal_normal2():
    # model
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = normal_scale(a, b)
    x = normal_scale(z, c)
    # transform
    rule = transforms.NormalNormalTransformationRule()
    replacements = rule.apply(x, observed_vars=[x])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert new_x.cond_dist == normal_scale
    assert new_z.cond_dist == normal_scale


def test_vmapped_transformation_apply1():
    # model
    a = makerv([1.1, 2.2])
    b = makerv([3.3, 4.4])
    c = makerv([5.5, 6.6])
    z = vmap(normal_scale, 0)(a, b)
    x = vmap(normal_scale, 0)(z, c)
    # transform
    base_rule = transforms.NormalNormalTransformationRule()
    rule = transforms.VMappedTransformationRule(base_rule)
    replacements = rule.apply(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert isinstance(new_x.cond_dist, VMapDist)
    assert isinstance(new_z.cond_dist, VMapDist)
    assert new_x.cond_dist.base_cond_dist == normal_scale
    assert new_z.cond_dist.base_cond_dist == normal_scale
    assert new_x.shape == (2,)
    assert new_z.shape == (2,)


def test_vmapped_transformation_apply2():
    # model
    a = makerv(1.1)
    b = makerv([3.3, 4.4])
    c = makerv([5.5, 6.6])
    z = vmap(normal_scale, (None, 0))(a, b)
    x = vmap(normal_scale, 0)(z, c)
    # transform
    base_rule = transforms.NormalNormalTransformationRule()
    rule = transforms.VMappedTransformationRule(base_rule)
    replacements = rule.apply(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert isinstance(new_x.cond_dist, VMapDist)
    assert isinstance(new_z.cond_dist, VMapDist)
    assert new_x.cond_dist.base_cond_dist == normal_scale
    assert new_z.cond_dist.base_cond_dist == normal_scale
    assert new_x.shape == (2,)
    assert new_z.shape == (2,)


def test_vmapped_transformation_apply3():
    # model
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv([5.5, 6.6])
    z = vmap(normal_scale, (None, None), axis_size=2)(a, b)
    x = vmap(normal_scale, (0, 0))(z, c)
    # transform
    base_rule = transforms.NormalNormalTransformationRule()
    rule = transforms.VMappedTransformationRule(base_rule)
    replacements = rule.apply(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert isinstance(new_x.cond_dist, VMapDist)
    assert isinstance(new_z.cond_dist, VMapDist)
    assert new_x.cond_dist.base_cond_dist == normal_scale
    assert new_z.cond_dist.base_cond_dist == normal_scale
    assert new_x.shape == (2,)
    assert new_z.shape == (2,)


def test_beta_binomial1():
    # model
    a = 2
    b = 3
    n = 10
    z = beta(a, b)
    x = binomial(n, z)
    # transform
    rule = transforms.BetaBinomialTransformationRule()
    replacements = rule.apply(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert new_x.cond_dist == beta_binomial
    assert new_z.cond_dist == beta


def test_vmap_beta_binomial1():
    # model
    a = makerv([1.1, 2.2])
    b = 3
    n = 10
    z = vmap(beta, (0, None))(a, b)
    x = vmap(binomial, (None, 0))(n, z)
    # transform
    base_rule = transforms.BetaBinomialTransformationRule()
    rule = transforms.VMappedTransformationRule(base_rule)
    replacements = rule.apply(x, [x])
    print(f"{replacements=}")
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert isinstance(new_x.cond_dist, VMapDist)
    assert new_x.cond_dist.base_cond_dist == beta_binomial
    assert isinstance(new_z.cond_dist, VMapDist)
    assert new_z.cond_dist.base_cond_dist == beta


def test_double_vmap_normal_normal():
    double_normal = vmap(vmap(normal_scale, 0), 0)
    locs = makerv([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    scales = makerv([[9.8, 8.7, 7.6], [6.5, 5.4, 4.3]])
    z = double_normal(locs, scales)
    x = double_normal(z, scales)

    base_rule1 = transforms.NormalNormalTransformationRule()
    base_rule2 = transforms.VMappedTransformationRule(base_rule1)
    rule = transforms.VMappedTransformationRule(base_rule2)
    replacements = rule.apply(x, [x])

    new_x = replacements[x]
    new_z = replacements[z]

    assert new_x.cond_dist.base_cond_dist.base_cond_dist == normal_scale
    assert new_z.cond_dist.base_cond_dist.base_cond_dist == normal_scale

    # display(viz_upstream(new_z))


def test_observed_descendents1():
    x = normal(0, 1)
    y = normal(x, 1)
    z = normal(y, 1)

    out = transforms.check_observed_descendents([x, y], [z])

    assert out == (False, True)


def test_observed_descendents2():
    z = normal(0, 1)

    out = transforms.check_observed_descendents([z], [z])

    assert out == (True,)


def test_observed_descendents3():
    x = normal(0, 1)
    y = normal(x, 1)
    z = normal(y, 1)

    out = transforms.check_observed_descendents([x, y, z], [z])

    assert out == (False, False, True)


def test_observed_descendents4():
    x = normal(0, 1)
    y = normal(x, 1)
    z = normal(x + y, 1)

    out = transforms.check_observed_descendents([x, y], [z])

    assert out == (True, True)


# def test_

# def test_non_centered1():
#     loc = makerv(1.1)
#     scale = makerv(2.2)
#     x = normal(loc, scale)
#     t = infer.NonCenteredNormalTransformationRule
#     t.check(x, observed_vars=[])
#     extracted = t.extract(x)
#     new_x, *new_pars = t.apply(x, *extracted)
#     assert new_x.cond_dist == add
#     assert new_x.parents[0] == loc
#     assert new_x.parents[1].cond_dist == mul
#
#
# def test_non_centered2():
#     loc = makerv(1.1)
#     scale = makerv(2.2)
#     x = normal(loc, scale)
#     t = infer.NonCenteredNormalTransformationRule
#     [new_x] = infer.apply_transformation_rules([x], [t], [])
#     assert new_x.cond_dist == add
#     assert new_x.parents[0].cond_dist == loc.cond_dist  # only cond_dist equal!
#     assert new_x.parents[1].cond_dist == mul
#
#
# def test_normal_normal1():
#     a = makerv(1.1)
#     b = makerv(2.2)
#     c = makerv(3.3)
#     z = normal_scale(a, b)
#     x = normal_scale(z, c)
#     t = infer.NormalNormalTransformationRule
#     t.check(x, observed_vars=[x])
#     extracted = t.extract(x)
#     new_x, *new_pars = t.apply(x, *extracted)
#
#
# def test_normal_normal2():
#     a = makerv(1.1)
#     b = makerv(2.2)
#     c = makerv(3.3)
#     z = normal_scale(a, b)
#     x = normal_scale(z, c)
#     assert z in dag.upstream_nodes(x)
#     assert not x in dag.upstream_nodes(z)
#     t = infer.NormalNormalTransformationRule
#     [new_x, new_z] = infer.apply_transformation_rules([x, z], [t], [x])
#     assert not new_z in dag.upstream_nodes(new_x)
#     assert new_x in dag.upstream_nodes(new_z)
#
#
# def test_normal_normal_shrink2():
#     a = makerv(1.1)
#     b = makerv(2.2)
#     c = makerv(3.3)
#     z = normal_scale(a, b)
#     x = normal_scale(z, c)
#     assert z in dag.upstream_nodes(x)
#     assert not x in dag.upstream_nodes(z)
#     t1 = infer.NormalNormalTransformationRule
#     t2 = infer.ConstantOpTransformationRule
#     [new_x1, new_z1] = infer.apply_transformation_rules([x, z], [t1, t2], [x])
#     assert not new_z1 in dag.upstream_nodes(new_x1)
#     assert new_x1 in dag.upstream_nodes(new_z1)
#
#
# def test_constant_op1():
#     a = makerv(1.1)
#     b = makerv(2.2)
#     x = a + b
#     t = infer.ConstantOpTransformationRule
#     t.check(x, observed_vars=[])
#
# # def test_vmap_normal_normal():
# #     locs = makerv([1.1, 2.2])
# #     scales = makerv([3.3, 4.4])
# #     vec_normal = vmap(normal_scale, (0, 0))
# #     z = vec_normal(locs, scales)
# #     x = vec_normal(z, scales)
# #     t0 = infer.NormalNormalTransformationRule
# #     t = infer.VMapTransformationRule(t0)
# #     t.check(x, z, scales, observed_vars=[x])
# #     extracted = t.extract(x, z, scales)
