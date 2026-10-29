"""Joint densities and gradients for nested scans with random initialization and vector events."""

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist

from pangolin import dag, interface as pi, jax_backend
from pangolin.functional import loop


def check_density(nodes, values, reference, *, gradients=True):
    actual = jax_backend.ancestor_log_prob(tuple(nodes))
    expected = reference(values)
    np.testing.assert_allclose(actual(values), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(jax.jit(actual)(values), expected, rtol=2e-6, atol=2e-6)
    if gradients:
        a, b = jax.jit(jax.grad(actual))(values), jax.grad(reference)(values)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-5)


def test_three_random_initialization_arrays_and_state_dependent_scale():
    z = pi.normal(0.0, 1.0)

    def model(prev, t, s, u):
        if t is None:
            return pi.normal(z + s + u, 0.5)
        if s is None:
            return pi.normal(z + t - u, 0.6)
        if u is None:
            return pi.normal(z + t + 2 * s, 0.7)
        a, b, c = prev[t - 1, s, u], prev[t, s - 1, u], prev[t, s, u - 1]
        return pi.normal(0.2 * a + 0.3 * b + 0.4 * c + z, 0.7 + 0.1 * a * a)

    x = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
        u=[2],
    )
    nodes = [rv for rv in dag.upstream_nodes(x) if rv.op.random]
    assert nodes[0] is z and nodes[-1] is x
    shapes = [(), (3, 2), (2, 2), (2, 3), (2, 3, 2)]
    assert len(nodes) == len(shapes)
    by_shape = {rv.shape: rv for rv in nodes}
    assert set(by_shape) == set(shapes)
    nodes = [by_shape[shape] for shape in shapes]
    values = (
        jnp.array(0.2),
        *(jnp.arange(np.prod(rv.shape), dtype=float).reshape(rv.shape) / 5 for rv in nodes[1:]),
    )

    def reference(v):
        z, a, b, c, x = v
        old_t = jnp.concatenate((a[None, :, :], x[:-1]), axis=0)
        old_s = jnp.concatenate((b[:, None, :], x[:, :-1]), axis=1)
        old_u = jnp.concatenate((c[:, :, None], x[:, :, :-1]), axis=2)
        return (
            dist.Normal(0, 1).log_prob(z)
            + dist.Normal(z + jnp.arange(3)[:, None] + jnp.arange(2), 0.5).log_prob(a).sum()
            + dist.Normal(z + jnp.arange(2)[:, None] - jnp.arange(2), 0.6).log_prob(b).sum()
            + dist.Normal(z + jnp.arange(2)[:, None] + 2 * jnp.arange(3), 0.7).log_prob(c).sum()
            + dist.Normal(0.2 * old_t + 0.3 * old_s + 0.4 * old_u + z, 0.7 + 0.1 * old_t**2)
            .log_prob(x)
            .sum()
        )

    check_density(nodes, values, reference)


def test_multivariate_random_carries_with_outer_and_intermediate_vectorization():
    A_value = jnp.array([[0.2, -0.1], [0.4, 0.3]])
    B_value = jnp.array([[0.3, 0.2], [-0.2, 0.1]])
    Q_value = jnp.array([[1.0, 0.35], [0.35, 0.8]])
    A, B, Q = pi.constant(A_value), pi.constant(B_value), pi.constant(Q_value)
    shared = pi.multi_normal([0.0, 0.0], np.eye(2))

    def init_t(i, j, s):
        return pi.multi_normal(shared + i + 2 * j + 3 * s, Q)

    def init_s(i, t, j):
        return pi.multi_normal(shared + 2 * i + t - j, Q)

    def model(prev, i, t, j, s):
        if t is None:
            return init_t(i, j, s)
        if s is None:
            return init_s(i, t, j)
        mean = A @ prev[i, t - 1, j, s, :] + B @ prev[i, t, j, s - 1, :] + shared
        return pi.multi_normal(mean, Q)

    x = loop(model, interface=pi.default, i=2, t=[2], j=2, s=[3])
    nodes = [rv for rv in dag.upstream_nodes(x) if rv.op.random]
    assert nodes[0] is shared and nodes[-1] is x
    assert [rv.shape for rv in nodes] == [(2,), (2, 2, 3, 2), (2, 2, 2, 2), (2, 2, 2, 3, 2)]
    values = tuple(
        jnp.arange(np.prod(rv.shape), dtype=float).reshape(rv.shape) / 17 - 0.2 for rv in nodes
    )

    def reference(v):
        z, initial_t, initial_s, x = v
        old_t = jnp.concatenate((initial_t[:, None, :, :, :], x[:, :-1]), axis=1)
        old_s = jnp.concatenate((initial_s[:, :, :, None, :], x[:, :, :, :-1]), axis=3)
        mean_t = (
            jnp.arange(2)[:, None, None]
            + 2 * jnp.arange(2)[None, :, None]
            + 3 * jnp.arange(3)[None, None, :]
        )[..., None] + z
        mean_s = (
            2 * jnp.arange(2)[:, None, None]
            + jnp.arange(2)[None, :, None]
            - jnp.arange(2)[None, None, :]
        )[..., None] + z
        return (
            dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)).log_prob(z)
            + dist.MultivariateNormal(mean_t, Q_value).log_prob(initial_t).sum()
            + dist.MultivariateNormal(mean_s, Q_value).log_prob(initial_s).sum()
            + dist.MultivariateNormal(old_t @ A_value.T + old_s @ B_value.T + z, Q_value)
            .log_prob(x)
            .sum()
        )

    check_density(nodes, values, reference)


def test_discrete_initializers_and_binary_grid_joint_density():
    def model(prev, t, s):
        if t is None:
            return pi.bernoulli(0.3)
        if s is None:
            return pi.bernoulli(0.4)
        return pi.bernoulli_logit(0.8 * prev[t - 1, s] - 0.5 * prev[t, s - 1])

    x = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    nodes = [rv for rv in dag.upstream_nodes(x) if rv.op.random]
    assert [rv.shape for rv in nodes] == [(3,), (2,), (2, 3)]
    assert all(rv.op.discrete for rv in nodes)
    values = (jnp.array([0, 1, 0]), jnp.array([1, 0]), jnp.array([[0, 1, 1], [1, 0, 1]]))

    def reference(v):
        a, b, x = v
        old_t = jnp.concatenate((a[None, :], x[:-1]), axis=0)
        old_s = jnp.concatenate((b[:, None], x[:, :-1]), axis=1)
        return (
            dist.Bernoulli(0.3).log_prob(a).sum()
            + dist.Bernoulli(0.4).log_prob(b).sum()
            + dist.Bernoulli(logits=0.8 * old_t - 0.5 * old_s).log_prob(x).sum()
        )

    check_density(nodes, values, reference, gradients=False)
    sampler = jax_backend.ancestor_sampler(tuple(nodes))
    samples = jax.jit(jax.vmap(sampler))(jax.random.split(jax.random.key(15), 64))
    for sample in samples:
        assert np.all((sample == 0) | (sample == 1))
    actual = jax.jit(jax.vmap(jax_backend.ancestor_log_prob(tuple(nodes))))(samples)
    np.testing.assert_allclose(actual, jax.vmap(reference)(samples), rtol=1e-6)


def test_full_gaussian_mean_and_covariance_include_shared_initialization():
    shared = pi.normal(0.0, 0.4)

    def model(prev, i, t, s):
        if t is None:
            return pi.normal(shared + i + s, 0.5)
        if s is None:
            return pi.normal(shared - i + t, 0.6)
        return pi.normal(0.2 * prev[i, t - 1, s] + 0.3 * prev[i, t, s - 1] + shared, 0.7)

    output = loop(
        model,
        interface=pi.default,
        i=2,
        t=[2],
        s=[2],
    )

    def reference(noise):
        z = 0.4 * noise[0]
        initial_t = z + np.arange(2)[:, None] + np.arange(2) + 0.5 * noise[1:5].reshape(2, 2)
        initial_s = z - np.arange(2)[:, None] + np.arange(2) + 0.6 * noise[5:9].reshape(2, 2)
        innovations = 0.7 * noise[9:].reshape(2, 2, 2)
        x = np.zeros((2, 2, 2))
        for i, t, s in np.ndindex(x.shape):
            a = initial_t[i, s] if t == 0 else x[i, t - 1, s]
            b = initial_s[i, t] if s == 0 else x[i, t, s - 1]
            x[i, t, s] = 0.2 * a + 0.3 * b + z + innovations[i, t, s]
        return x.ravel()

    mean = reference(np.zeros(17))
    factor = np.stack([reference(column) - mean for column in np.eye(17)], axis=1)
    covariance = factor @ factor.T
    sampler = jax_backend.ancestor_sampler(output)
    samples = np.asarray(
        jax.jit(jax.vmap(sampler))(jax.random.split(jax.random.key(77), 6000))
    ).reshape(6000, 8)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.045)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance, atol=0.065)
    assert covariance[0, 4] > 0.1  # Separate individuals share the same latent variable.
