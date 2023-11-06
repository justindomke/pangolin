from pangolin.interface import *

# import infer
# import new_infer
from pangolin import new_infer
from jax import numpy as jnp

from numpyro import distributions as dists

# TODO: infer.py will go away—should migrate all these tests elsewhere
# TODO: for testing actual inference results, should gradually increase
# number of samples

# def test_log_prob_dist1():
#     d = normal_scale
#     logp = infer.log_prob_dist(d, 0.0, 0.0, 1.0)
#     expected = np.log(1 / np.sqrt(2 * np.pi))
#     print(f"{logp=}")
#     print(f"{expected=}")
#     assert np.allclose(logp, expected)


def test_sample_dist1():
    d = normal_scale
    key = jax.random.PRNGKey(0)
    x = new_infer.sample_dist(d, key, 0.0, 1.0)
    assert x.shape == ()


def test_sample_dist2():
    d = VMapDist(normal_scale, in_axes=[0, 0], axis_size=3)
    locs = np.array([1, 2, 3])
    scales = np.array([4, 5, 6])
    key = jax.random.PRNGKey(0)
    x = new_infer.sample_dist(d, key, locs, scales)
    assert x.shape == (3,)


def test_sample_dist3():
    d = VMapDist(normal_scale, in_axes=[0, None], axis_size=3)
    locs = np.array([1, 2, 3])
    scales = np.array(4)
    key = jax.random.PRNGKey(0)
    x = new_infer.sample_dist(d, key, locs, scales)
    assert x.shape == (3,)


# def test_sample_and_log_prob1():
#     d = VMapDist(normal_scale, in_axes=[0, None], axis_size=3)
#     locs = np.array([1, 2, 3])
#     scales = np.array(4)
#     key = jax.random.PRNGKey(0)
#     x = infer.sample_dist(d, key, locs, scales)
#     assert x.shape == (3,)
#     logp = new_infer.log_prob_dist(d, x, locs, scales)
#     assert logp.shape == ()


def test_log_prob_flat1():
    x = normal(0, 1)
    y = normal(x, 1)
    l1 = new_infer.ancestor_log_prob_flat([x], [2.0], [], [])
    l2 = new_infer.ancestor_log_prob_flat([y], [-1.0], [x], [1])
    assert np.allclose(l1, l2)


# ancestor log prob doesn't exist
# def test_log_prob1():
#     x = normal(0, 1)
#     y = normal(x, 1)
#     l1 = new_infer.ancestor_log_prob(x)
#     l2 = new_infer.ancestor_log_prob(y, x)
#
#     assert np.allclose(l1(0.), l2(0., 0.))
#     assert np.allclose(l1(2.), l2(1., -1.))
#     assert np.allclose(l1(2.), l2(0.5, 2.5))


def test_deterministic_sample1():
    x = normal(0, 1)
    y = x ** 2
    (xs, ys) = new_infer.sample((x, y), niter=100)
    assert xs.shape == (100,)
    assert ys.shape == (100,)
    assert np.allclose(ys, xs ** 2)


def test_sample_flat1():
    z = normal(0.0, 1.0)
    [zs] = new_infer.sample_flat([z], niter=10)
    assert zs.shape == (10,)


def test_sample_flat2():
    z = normal(0.0, 1.0)
    x = normal(z, 1.0)
    [zs] = new_infer.sample_flat([z], [x], [1.0], niter=10)
    assert zs.shape == (10,)


def test_sample1():
    z = normal(0.0, 1.0)
    zs = new_infer.sample(z, niter=10)
    assert zs.shape == (10,)


def test_sample2():
    z = normal(0.0, 1.0)
    x = normal(z, 1)
    y = normal(z, 1)
    ys = new_infer.sample(y, x, 1.0, niter=10)
    assert ys.shape == (10,)


def test_sample3():
    z = normal(0.0, 1.0)
    x = normal(z, 0.00001)
    y = normal(z, 0.00001)
    x_obs = 0.5
    zs = new_infer.sample(z, x, x_obs, niter=10)
    print(f"{zs=}")


def test_sample4():
    z = normal(0.0, 0.999)
    x = normal(z, 0.00001)
    y = normal(z, 0.00002)
    x_obs = 0.5
    ys = new_infer.sample(y, x, x_obs, niter=10)
    print(f"{ys=}")

    # ys = infer.sample(y, x, x_obs, niter=10)
    # assert ys.shape == (10,)
    # print(f"{ys=}")
    # assert np.all(np.abs(ys - (x_obs+2)) < .01)


def test_new_cond_log_prob_flat1():
    x = normal(0.1, 0.2)
    y = normal(x, 0.3)
    l = new_infer.ancestor_log_prob_flat([x], [0.4], [], [])
    expected = new_infer.log_prob_dist(normal_scale, 0.4, 0.1, 0.2)
    assert l == expected

    l = new_infer.ancestor_log_prob_flat([y], [0.4], [x], [0.5])
    expected = new_infer.log_prob_dist(normal_scale, 0.4, 0.5, 0.3)
    assert l == expected

    l = new_infer.ancestor_log_prob_flat([x, y], [0.4, 0.5], [], [])
    expected = new_infer.log_prob_dist(
        normal_scale, 0.4, 0.1, 0.2
    ) + new_infer.log_prob_dist(normal_scale, 0.5, 0.4, 0.3)
    assert l == expected


def test_new_ancestor_sample1():
    x = normal(0.1, 0.2)
    y = normal(x, 0.3)

    # get expected samples
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    expected_x_samp = new_infer.sample_dist(normal_scale, subkey, 0.1, 0.2)
    key, subkey = jax.random.split(key)
    expected_y_samp = new_infer.sample_dist(normal_scale, subkey, expected_x_samp, 0.3)

    # reset key
    key = jax.random.PRNGKey(0)

    # sample just x
    [x_samp] = new_infer.ancestor_sample_flat(key, [x], [], [])
    assert x_samp == expected_x_samp

    # sample just y (x done in background)
    [y_samp] = new_infer.ancestor_sample_flat(key, [y], [], [])
    assert y_samp == expected_y_samp

    # sample both together with x listed first
    [x_samp, y_samp] = new_infer.ancestor_sample_flat(key, [x, y], [], [])
    assert x_samp == expected_x_samp
    assert y_samp == expected_y_samp

    # sample both together with y listed first
    [y_samp, x_samp] = new_infer.ancestor_sample_flat(key, [y, x], [], [])
    assert x_samp == expected_x_samp
    assert y_samp == expected_y_samp


def test_new_ancestor_sample2():
    x = normal(0.1, 0.2)
    y = normal(x, 0.3)
    z = x + 2 * y

    # get expected samples
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    expected_x_samp = new_infer.sample_dist(normal_scale, subkey, 0.1, 0.2)
    key, subkey = jax.random.split(key)
    expected_y_samp = new_infer.sample_dist(normal_scale, subkey, expected_x_samp, 0.3)
    expected_z_samp = expected_x_samp + 2 * expected_y_samp

    # reset key
    key = jax.random.PRNGKey(0)

    # sample just x
    [x_samp] = new_infer.ancestor_sample_flat(key, [x], [], [])
    assert x_samp == expected_x_samp

    # sample just y (x done in background)
    [y_samp] = new_infer.ancestor_sample_flat(key, [y], [], [])
    assert y_samp == expected_y_samp

    # sample just z (x and y done in background)
    [z_samp] = new_infer.ancestor_sample_flat(key, [z], [], [])
    assert z_samp == expected_z_samp

    # sample all pairs in all orders
    [x_samp, y_samp] = new_infer.ancestor_sample_flat(key, [x, y], [], [])
    assert (x_samp, y_samp) == (expected_x_samp, expected_y_samp)
    [y_samp, x_samp] = new_infer.ancestor_sample_flat(key, [y, x], [], [])
    assert (y_samp, x_samp) == (expected_y_samp, expected_x_samp)

    [x_samp, z_samp] = new_infer.ancestor_sample_flat(key, [x, z], [], [])
    assert (x_samp, z_samp) == (expected_x_samp, expected_z_samp)
    [z_samp, x_samp] = new_infer.ancestor_sample_flat(key, [z, x], [], [])
    assert (z_samp, x_samp) == (expected_z_samp, expected_x_samp)

    [y_samp, z_samp] = new_infer.ancestor_sample_flat(key, [y, z], [], [])
    assert (y_samp, z_samp) == (expected_y_samp, expected_z_samp)
    [z_samp, y_samp] = new_infer.ancestor_sample_flat(key, [z, y], [], [])
    assert (z_samp, y_samp) == (expected_z_samp, expected_y_samp)

    # sample all three in all orders
    [x_samp, y_samp, z_samp] = new_infer.ancestor_sample_flat(key, [x, y, z], [], [])
    assert (x_samp, y_samp, z_samp) == (
        expected_x_samp,
        expected_y_samp,
        expected_z_samp,
    )
    [x_samp, z_samp, y_samp] = new_infer.ancestor_sample_flat(key, [x, z, y], [], [])
    assert (x_samp, z_samp, y_samp) == (
        expected_x_samp,
        expected_z_samp,
        expected_y_samp,
    )
    [y_samp, x_samp, z_samp] = new_infer.ancestor_sample_flat(key, [y, x, z], [], [])
    assert (y_samp, x_samp, z_samp) == (
        expected_y_samp,
        expected_x_samp,
        expected_z_samp,
    )
    [y_samp, z_samp, x_samp] = new_infer.ancestor_sample_flat(key, [y, z, x], [], [])
    assert (y_samp, z_samp, x_samp) == (
        expected_y_samp,
        expected_z_samp,
        expected_x_samp,
    )
    [z_samp, x_samp, y_samp] = new_infer.ancestor_sample_flat(key, [z, x, y], [], [])
    assert (z_samp, x_samp, y_samp) == (
        expected_z_samp,
        expected_x_samp,
        expected_y_samp,
    )
    [z_samp, y_samp, x_samp] = new_infer.ancestor_sample_flat(key, [z, y, x], [], [])
    assert (z_samp, y_samp, x_samp) == (
        expected_z_samp,
        expected_y_samp,
        expected_x_samp,
    )

    # sample x twice and z three times
    [x_samp1, z_samp1, x_samp2, z_samp2, z_samp3] = new_infer.ancestor_sample_flat(
        key, [x, z, x, z, z], [], []
    )
    assert x_samp1 == x_samp2
    assert z_samp1 == z_samp2
    assert z_samp1 == z_samp3
    assert x_samp1 == expected_x_samp
    assert z_samp1 == expected_z_samp


def test_new_sample_flat1():
    x = normal(0.5, 2.7)
    niter = 10_000
    # won't actually do MCMC because it can use ancestor sampling
    [xs] = new_infer.sample_flat([x], niter=niter)
    assert xs.shape == (niter,)
    assert np.abs(np.mean(xs) - 0.5) < 0.1
    assert np.abs(np.std(xs) - 2.7) < 0.1


def test_new_sample_flat2():
    x = normal(0.5, 2.7)
    y = 3 * x + x ** 2
    niter = 10_000
    # won't actually do MCMC because it can use ancestor sampling
    [xs, ys] = new_infer.sample_flat([x, y], niter=niter)
    assert xs.shape == (niter,)
    assert ys.shape == (niter,)
    assert np.abs(np.mean(xs) - 0.5) < 0.1
    assert np.abs(np.std(xs) - 2.7) < 0.1
    assert np.allclose(ys, 3 * xs + xs ** 2)


def test_new_sample_flat3():
    x = normal(0, 1)
    y = normal(x, 1)
    niter = 10_000

    [xs] = new_infer.sample_flat([x], [y], [1], niter=niter)
    assert xs.shape == (niter,)
    print(f"{np.mean(xs)=}")
    print(f"{np.var(xs)=}")
    assert np.abs(np.mean(xs) - 0.5) < 0.03
    assert np.abs(np.var(xs) - 0.5) < 0.03


def test_new_sample_flat4():
    x = normal(0, 1)
    y = normal(x, 1)
    z = normal(y, 1)
    niter = 10_000

    vars = new_infer.variables_to_sample([y], [x, z])
    assert set(vars) == {y, z}

    [ys] = new_infer.sample_flat([y], [x, z], [-2, 7], niter=niter)
    assert ys.shape == (niter,)
    print(f"{np.mean(ys)=}")
    print(f"{np.var(ys)=}")
    assert np.abs(np.mean(ys) - (-2 + 7) / 2) < 0.03
    assert np.abs(np.var(ys) - 1 / 2) < 0.03


# def test_vmap1():
#     x = plate(N=2)(lambda:
#                    normal(0, 1))
#     assert x.shape == (2,)
#     xs = infer.sample(x, niter=10)
#
#     assert xs.shape == (10, 2)
#
#
# def test_vmap2():
#     x = normal(0, 1)
#     y = plate(N=2)(lambda:
#                    normal(x, 1))
#     assert x.shape == ()
#     assert y.shape == (2,)
#     [xs, ys] = infer.sample([x, y], niter=10)
#
#     assert xs.shape == (10,)
#     assert ys.shape == (10, 2)
#
#
# def test_vmap3():
#     x = normal(0, 1)
#     y = plate(N=2)(lambda:
#                    normal(1, x * x + .01)
#                    )
#     [xs, ys] = infer.sample([x, y], niter=10)
#
#     assert xs.shape == (10,)
#     assert ys.shape == (10, 2)
#
#
# # def test_transforms1():
# #     x = normal(0, 1)
# #     y = softplus(x)
# #     ys = infer.sample(y, niter=1000)
#
#
#
# def test_transforms1():
#     x = softplus(normal_scale)(0, 1)
#     xs = infer.sample(x, niter=10)
#     assert xs.shape == (10,)


def test_mixture1():
    locs = [-0.11, 0.22]
    scales = [0.33, 4.4]
    weights = [0.333, 0.667]
    x_val = 0.279

    normal_mixture = Mixture(normal_scale, (0, 0))  # new CondDist
    x = normal_mixture(weights, locs, scales)  # new RV
    l = new_infer.ancestor_log_prob_flat([x], [x_val], [], [])

    from numpyro import distributions as dists

    l0 = dists.Normal(locs[0], scales[0]).log_prob(x_val)
    l1 = dists.Normal(locs[1], scales[1]).log_prob(x_val)
    expected = jnp.log(weights[0] * jnp.exp(l0) + weights[1] * jnp.exp(l1))

    print(f"{l=}")
    print(f"{expected=}")

    assert np.allclose(l, expected)


def test_mixture2():
    locs = [-0.11, 0.22]
    scale = 0.33
    weights = [0.333, 0.667]
    x_val = 0.279

    normal_mixture = Mixture(normal_scale, (0, None))  # new CondDist
    x = normal_mixture(weights, locs, scale)  # new RV
    l = new_infer.ancestor_log_prob_flat([x], [x_val], [], [])

    from numpyro import distributions as dists

    l0 = dists.Normal(locs[0], scale).log_prob(x_val)
    l1 = dists.Normal(locs[1], scale).log_prob(x_val)
    expected = jnp.log(weights[0] * jnp.exp(l0) + weights[1] * jnp.exp(l1))

    print(f"{l=}")
    print(f"{expected=}")

    assert np.allclose(l, expected)


def test_mixture3():
    loc = -0.11
    scales = [0.33, 4.4]
    weights = [0.333, 0.667]
    x_val = 0.279

    normal_mixture = Mixture(normal_scale, (None, 0))  # new CondDist
    x = normal_mixture(weights, loc, scales)  # new RV
    l = new_infer.ancestor_log_prob_flat([x], [x_val], [], [])

    l0 = dists.Normal(loc, scales[0]).log_prob(x_val)
    l1 = dists.Normal(loc, scales[1]).log_prob(x_val)
    expected = jnp.log(weights[0] * jnp.exp(l0) + weights[1] * jnp.exp(l1))

    print(f"{l=}")
    print(f"{expected=}")

    assert np.allclose(l, expected)


def test_mixture4():
    # silly to take a mixture of distributions with identical parameters but whatever
    loc = -0.11
    scale = 0.33
    weights = [0.333, 0.667]
    x_val = 0.279

    normal_mixture = Mixture(normal_scale, (None, None))  # new CondDist
    x = normal_mixture(weights, loc, scale)  # new RV
    l = new_infer.ancestor_log_prob_flat([x], [x_val], [], [])

    from numpyro import distributions as dists

    l0 = dists.Normal(loc, scale).log_prob(x_val)
    l1 = dists.Normal(loc, scale).log_prob(x_val)
    expected = jnp.log(weights[0] * jnp.exp(l0) + weights[1] * jnp.exp(l1))

    print(f"{l=}")
    print(f"{expected=}")

    assert np.allclose(l, expected)


# def test_make_mixture1():
#     loc = -0.11
#     scale = 0.33
#     weights = [0.333, 0.667]
#
#     z = bernoulli(weights[1])
#     x = make_mixture(z, lambda z: normal_scale(loc, scale))
#     x_val = 0.279
#
#     l = new_infer.ancestor_log_prob_flat([x], [x_val], [], [])
#
#     l0 = dists.Normal(loc, scale).log_prob(x_val)
#     l1 = dists.Normal(loc, scale).log_prob(x_val)
#     expected = jnp.log(weights[0] * jnp.exp(l0) + weights[1] * jnp.exp(l1))
#
#     print(f"{l=}")
#     print(f"{expected=}")
#
#     assert np.allclose(l, expected)


def test_cond_prob1():
    cond_dist = CondProb(bernoulli)
    out = new_infer.eval_dist(cond_dist, 0, 0.25)
    expected = 0.75
    assert np.allclose(out, expected)


def test_cond_prob2():
    cond_dist = CondProb(normal_scale)
    x_val = 0.123
    loc = .456
    scale = .789
    out = new_infer.eval_dist(cond_dist, x_val, loc, scale)
    expected = jnp.exp(dists.Normal(loc, scale).log_prob(x_val))
    assert np.allclose(out, expected)


def test_indexing1():
    x_numpy = np.random.randn(3, 7)
    y_numpy = x_numpy[:, [0, 1, 2]]
    x = makerv(x_numpy)
    y = x[:, [0, 1, 2]]
    y_samp = new_infer.sample(y, niter=1)[0]
    assert np.allclose(y_numpy, y_samp)


def test_indexing2():
    x_numpy = np.random.randn(3, 5, 7)
    idx1 = np.random.randint(low=0, high=3, size=[11, 13])
    idx2 = np.random.randint(low=0, high=3, size=[11, 13])
    y_numpy = x_numpy[:, idx1, idx2]
    x = makerv(x_numpy)
    y = x[:, idx1, idx2]
    y_samp = new_infer.sample(y, niter=1)[0]
    assert np.allclose(y_numpy, y_samp)


def test_indexing3():
    x_numpy = np.random.randn(3, 5, 7)
    idx0 = np.random.randint(low=0, high=3, size=[11, 13])
    idx2 = np.random.randint(low=0, high=3, size=[11, 13])
    y_numpy = x_numpy[idx0, 1::2, idx2]
    x = makerv(x_numpy)
    y = x[idx0, 1::2, idx2]
    y_samp = new_infer.sample(y, niter=1)[0]
    assert np.allclose(y_numpy, y_samp)
