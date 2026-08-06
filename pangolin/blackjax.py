"""
This module defines a convenient interface to call
`Blackjax <https://blackjax-devs.github.io/blackjax/>`_
to do inference. You could of course just call
`pangolin.jax_backend.ancestor_log_prob` to get a plain jax function and
then call Blackjax yourself. But this module abstracts away all the
details.

The module has three layers:

- **Inference drivers** (`run_nuts`, `run_hmc`, `run_pathfinder`,
  `run_meanfield_vi`, `run_fullrank_vi`, `run_smc`, `run_rwm`):
  functions with the uniform signature
  ``run_inf(log_prob, key, initial_state, num_samples, **options)``,
  taking an unconstrained log-density and returning flat samples with
  leading axis ``num_samples``. These are public: write your own driver
  with this signature and wrap it with `blackjax_calculate`.
- **Engines** (`nuts`, `hmc`, `pathfinder`, `meanfield_vi`,
  `fullrank_vi`, `smc`, `rwm`):
  :class:`pangolin.calculate.Calculate` objects pre-bound to a driver
  via `blackjax_calculate`. This is the recommended user interface.
- **Defaults** (`sample`, `E`, `var`, `std`, `sample_arviz`):
  methods of the default engine (NUTS, 1000 samples).
"""

from __future__ import annotations
from jax import numpy as jnp
import jax
import jax.tree_util
import numpy as np
from typing import Callable, Optional, Iterable
from pangolin import ir
from pangolin.ir import RV
import blackjax
import optax
import textwrap
import inspect
import functools

from pangolin import dag
from pangolin import jax_backend
from pangolin.calculate import Calculate
from pangolin.jax_backend.bijectors import default_bijector_dict

__all__ = [
    # inference drivers
    "run_nuts",
    "run_hmc",
    "run_pathfinder",
    "run_meanfield_vi",
    "run_fullrank_vi",
    "run_smc",
    "run_rwm",
    # generic pipeline
    "sample_flat",
    # factory
    "blackjax_calculate",
    # engines
    "nuts",
    "hmc",
    "pathfinder",
    "meanfield_vi",
    "fullrank_vi",
    "smc",
    "rwm",
    # defaults
    "sample",
    "E",
    "var",
    "std",
    "sample_arviz",
]


################################################################################
# Inference drivers
#
# All drivers share the signature
#     run_inf(log_prob, key, initial_state, num_samples, **options)
# and return samples in the *unconstrained* (bijector-flattened) space
# with leading axis `num_samples`. Transformation back to the
# constrained space happens in `sample_flat`.
################################################################################


def mcmc_inference_loop(rng_key, kernel, initial_state, num_samples):
    """
    Run a Blackjax transition kernel for a fixed number of steps.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        Key used to generate one fresh subkey per transition.
    kernel : Callable
        A Blackjax step function with signature
        ``kernel(rng_key, state) -> (state, info)``, e.g.
        ``blackjax.nuts(log_prob, **parameters).step``.
    initial_state
        Kernel state to start from (typically the last state of an
        adaptation run).
    num_samples : int
        Number of transitions to perform.

    Returns
    -------
    last_state
        The final kernel state — a single state, not a stacked trace.
        Suitable for continuing the chain or seeding a subsequent
        sampling phase.
    states
        Stacked kernel states, one per step.
    infos
        Stacked kernel info objects (acceptance rates, divergences,
        etc.), one per step.
    """

    @jax.jit
    def one_step(states, rng_key):
        states, infos = kernel(rng_key, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    last_state, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return last_state, (states, infos)


def run_nuts(log_prob, key, initial_state, num_samples):
    """
    Sample from a density using the No-U-Turn Sampler (NUTS).

    Warmup uses Blackjax's window adaptation to tune the step size and
    mass matrix. Note that `num_samples` is used both as the length of
    the adaptation run and as the number of posterior draws.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into warmup and sampling
        keys.
    initial_state
        Initial position in the unconstrained latent space.
    num_samples : int
        Number of adaptation steps *and* number of draws returned.

    Returns
    -------
    samples
        Positions of the returned states, with leading axis
        `num_samples`.

    Examples
    --------
    Draw from a 3-dimensional standard normal (no pangolin needed — any
    jax log-density works):

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_nuts(log_prob, key, jnp.zeros(3), num_samples=10)
    >>> samps.shape
    (10, 3)
    """

    sample_key, warmup_key = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.nuts, log_prob)

    (last_state, parameters), _ = adapt.run(warmup_key, initial_state, num_samples)  # type: ignore

    kernel = blackjax.nuts(log_prob, **parameters).step
    _, (states, _) = mcmc_inference_loop(sample_key, kernel, last_state, num_samples)
    return states.position


def run_hmc(log_prob, key, initial_state, num_samples, num_integration_steps=60):
    """
    Sample from a density using Hamiltonian Monte Carlo (HMC).

    Warmup uses Blackjax's window adaptation to tune the step size and
    mass matrix; the trajectory length is fixed via
    `num_integration_steps` and is *not* adapted. As with `run_nuts`,
    `num_samples` is used both as the adaptation length and as the
    number of posterior draws.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into warmup and sampling
        keys.
    initial_state
        Initial position in the unconstrained latent space.
    num_samples : int
        Number of adaptation steps *and* number of draws returned.
    num_integration_steps : int, optional
        Fixed number of leapfrog steps per trajectory (default 60).
        Larger values explore further per iteration at higher cost;
        if you find yourself needing many steps, prefer NUTS, which
        adapts trajectory length automatically.

    Returns
    -------
    samples
        Positions of the returned states, with leading axis
        `num_samples`.

    Examples
    --------
    Draw from a 3-dimensional standard normal:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_hmc(log_prob, key, jnp.zeros(3), num_samples=10)
    >>> samps.shape
    (10, 3)
    """

    sample_key, warmup_key = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.hmc, log_prob, num_integration_steps=num_integration_steps)

    (last_state, parameters), _ = adapt.run(warmup_key, initial_state, num_samples)  # type: ignore

    kernel = blackjax.hmc(log_prob, **parameters).step
    _, (states, _) = mcmc_inference_loop(sample_key, kernel, last_state, num_samples)
    return states.position


def run_pathfinder(log_prob, key, initial_state, num_samples, elbo_samples=200, **lbfgs_kwargs):
    """
    Draw approximate posterior samples using Blackjax pathfinder.

    Pathfinder runs L-BFGS optimization from the initial position and
    fits a Gaussian approximation along the optimization path, then
    draws from it. The draws are i.i.d. samples from the
    *approximation*, not a Markov chain: if the true posterior is
    badly non-Gaussian, results are biased in a way that more draws
    will not fix. Use `run_nuts` when exactness matters.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into approximation and
        sampling keys.
    initial_state
        Initial position in the unconstrained latent space, used as
        the L-BFGS starting point. Must not be exactly at the mode
        (zero gradient prevents curvature estimation).
    num_samples : int
        Number of draws returned from the fitted approximation.
    elbo_samples : int, optional
        Number of draws used internally to estimate the ELBO along the
        optimization path (default 200). Controls approximation
        quality, not the size of the output.
    **lbfgs_kwargs
        Additional keyword arguments forwarded to
        `blackjax.vi.pathfinder.approximate`, e.g. `maxiter`,
        `maxcor`, `ftol`, `gtol`.

    Returns
    -------
    samples : jnp.ndarray
        Draws from the Gaussian approximation, with leading axis
        `num_samples`.

    Examples
    --------
    Draw approximate samples from a 3-dimensional standard normal
    (starting away from the mode so L-BFGS can estimate curvature):

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_pathfinder(log_prob, key, jnp.ones(3), num_samples=10)
    >>> samps.shape
    (10, 3)
    """

    approx_key, sample_key = jax.random.split(key)
    state, _ = blackjax.vi.pathfinder.approximate(approx_key, log_prob, initial_state, elbo_samples, **lbfgs_kwargs)
    samples, _logq = blackjax.vi.pathfinder.sample(sample_key, state, num_samples)
    return samples


def run_meanfield_vi(log_prob, key, initial_state, num_samples, n_iter=500, learning_rate=0.05):
    """
    Approximate a density using mean-field variational inference.

    Fits a factorized (diagonal-covariance) Gaussian to the target by
    stochastic optimization of the ELBO, then draws from it. The
    approximation cannot capture posterior correlations between
    latents; for a full-covariance Gaussian use `run_fullrank_vi`, and
    for exact sampling use `run_nuts`.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into optimization and
        sampling keys.
    initial_state
        Initial position in the unconstrained latent space, used to
        initialize the variational mean.
    num_samples : int
        Number of draws returned from the fitted approximation.
    n_iter : int, optional
        Number of ELBO optimization steps (default 500).
    learning_rate : float, optional
        Learning rate for the Adam optimizer (default 0.05).

    Returns
    -------
    samples
        Draws from the fitted mean-field Gaussian, with leading axis
        `num_samples`.

    Examples
    --------
    Fit a 2-dimensional standard normal and draw 10 samples:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_meanfield_vi(log_prob, key, jnp.zeros(2), num_samples=10, n_iter=50)
    >>> samps.shape
    (10, 2)
    """

    opt_key, sample_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    vi = blackjax.vi.meanfield_vi.as_top_level_api(log_prob, optimizer)

    state = vi.init(initial_state)

    def step_fn(state, k):
        state, _ = vi.step(k, state)
        return state, None

    keys = jax.random.split(opt_key, n_iter)
    state, _ = jax.lax.scan(step_fn, state, keys)

    return vi.sample(sample_key, state, num_samples)


def run_fullrank_vi(log_prob, key, initial_state, num_samples, n_iter=500, learning_rate=0.05):
    """
    Approximate a density using full-rank variational inference.

    Fits a Gaussian with a full (Cholesky-parameterized) covariance to
    the target by stochastic optimization of the ELBO, then draws from
    it. Unlike `run_meanfield_vi`, the approximation captures posterior
    correlations between latents; it is still a Gaussian approximation,
    so badly non-Gaussian posteriors are biased in a way that more
    optimization will not fix.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into optimization and
        sampling keys.
    initial_state
        Initial position in the unconstrained latent space, used to
        initialize the variational mean.
    num_samples : int
        Number of draws returned from the fitted approximation.
    n_iter : int, optional
        Number of ELBO optimization steps (default 500).
    learning_rate : float, optional
        Learning rate for the Adam optimizer (default 0.05).

    Returns
    -------
    samples
        Draws from the fitted full-rank Gaussian, with leading axis
        `num_samples`.

    Examples
    --------
    Fit a 2-dimensional standard normal and draw 10 samples:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_fullrank_vi(log_prob, key, jnp.zeros(2), num_samples=10, n_iter=50)
    >>> samps.shape
    (10, 2)
    """

    opt_key, sample_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    vi = blackjax.vi.fullrank_vi.as_top_level_api(log_prob, optimizer)

    state = vi.init(initial_state)

    def step_fn(state, k):
        state, _ = vi.step(k, state)
        return state, None

    keys = jax.random.split(opt_key, n_iter)
    state, _ = jax.lax.scan(step_fn, state, keys)

    return vi.sample(sample_key, state, num_samples)


def run_smc(
    log_prob,
    key,
    initial_state,
    num_samples,
    step_size=1.0,
    inverse_mass_matrix=None,
    num_integration_steps=10,
    num_mcmc_steps=10,
    target_ess=0.5,
    max_iters=100,
):
    """
    Sample from a density using adaptive tempered Sequential Monte
    Carlo.

    SMC maintains a population of `num_samples` particles that is moved
    from a prior toward the posterior along a temperature schedule
    chosen adaptively to control the effective sample size. Unlike
    gradient-based MCMC (NUTS/HMC), SMC can move between separated
    posterior modes, and unlike VI it is asymptotically exact. It is
    the most expensive option per sample.

    Because tempered SMC is written as prior^tempering x likelihood,
    a Gaussian pseudo-prior centered at the initial position is used
    internally, with the target log-density playing the role of the
    likelihood. `initial_state` therefore seeds the initial particle
    cloud.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into initialization and
        iteration keys.
    initial_state
        Position in the unconstrained latent space around which the
        initial particle cloud is centered.
    num_samples : int
        Number of particles, which is also the number of draws
        returned.
    step_size : float, optional
        Step size of the inner HMC kernel (default 1.0).
    inverse_mass_matrix : optional
        Inverse mass matrix of the inner HMC kernel. Defaults to the
        identity.
    num_integration_steps : int, optional
        Number of leapfrog steps of the inner HMC kernel (default 10).
    num_mcmc_steps : int, optional
        Number of MCMC kernel applications per particle per
        temperature step (default 10).
    target_ess : float, optional
        Target effective sample size, as a fraction of `num_samples`,
        used to choose the next temperature (default 0.5).
    max_iters : int, optional
        Safety cap on the number of temperature steps (default 100);
        raises if reached.

    Returns
    -------
    particles
        Final particle population, with leading axis `num_samples`.

    Examples
    --------
    Sample a 2-dimensional standard normal with 20 particles:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_smc(log_prob, key, jnp.zeros(2), num_samples=20)
    >>> samps.shape
    (20, 2)
    """

    init_key, iter_key = jax.random.split(key)

    flat_init = jax.tree_util.tree_leaves(initial_state)
    dim = sum(int(np.prod(x.shape)) for x in flat_init) if flat_init else 1
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.ones(dim)

    logprior_fn = lambda x: -0.5 * jnp.sum(x**2)
    loglikelihood_fn = lambda x: log_prob(x) - logprior_fn(x)

    hmc_kernel = blackjax.hmc.build_kernel()
    hmc_init = blackjax.hmc.init
    hmc_parameters = dict(
        step_size=jnp.array([step_size]),
        inverse_mass_matrix=jnp.atleast_1d(inverse_mass_matrix)[jnp.newaxis, ...],
        num_integration_steps=jnp.array([num_integration_steps]),
    )

    adaptive_tempered = blackjax.adaptive_tempered_smc(
        logprior_fn,
        loglikelihood_fn,
        hmc_kernel,
        hmc_init,
        hmc_parameters,
        blackjax.smc.resampling.systematic,  # type: ignore[attr-defined]
        target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    particles = jax.vmap(lambda k: initial_state + jax.random.normal(k, initial_state.shape))(
        jax.random.split(init_key, num_samples)
    )
    state = adaptive_tempered.init(particles)  # type: ignore[attr-defined]

    n_iter = 0
    while state.tempering_param < 1:  # type: ignore[attr-defined]
        iter_key, subkey = jax.random.split(iter_key)
        state, _ = adaptive_tempered.step(subkey, state)
        n_iter += 1
        if n_iter >= max_iters:
            raise RuntimeError(f"run_smc did not reach tempering_param=1 within {max_iters} iterations")

    return state.particles  # type: ignore[attr-defined]


def run_rwm(log_prob, key, initial_state, num_samples, step_size=1.0):
    """
    Sample from a density using random-walk Metropolis (RWM).

    Each step proposes a Gaussian perturbation of the current position
    and accepts or rejects it via the Metropolis ratio. No gradients
    are used, so this works on any (even non-differentiable)
    log-density; the cost is slow, diffusive exploration that scales
    poorly with dimension. Prefer NUTS whenever gradients are
    available. There is no adaptation: `step_size` must be chosen by
    hand, and the first `num_samples` steps are discarded as warmup.

    Parameters
    ----------
    log_prob : Callable
        Log-density of the target distribution, as a jitted jax
        function over the unconstrained latent space.
    key : jax.random.PRNGKey
        Randomness source; split internally into warmup and sampling
        keys.
    initial_state
        Initial position in the unconstrained latent space.
    num_samples : int
        Number of draws returned. An equal number of warmup steps is
        discarded first.
    step_size : float, optional
        Standard deviation of the Gaussian proposal (default 1.0).
        Tune so that the acceptance rate is roughly 0.2-0.4.

    Returns
    -------
    samples
        Positions of the returned states, with leading axis
        `num_samples`.

    Examples
    --------
    Draw from a 3-dimensional standard normal:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    >>> key = jax.random.PRNGKey(0)
    >>> samps = run_rwm(log_prob, key, jnp.zeros(3), num_samples=10)
    >>> samps.shape
    (10, 3)
    """

    step_size = jnp.array(step_size)
    rw = blackjax.additive_step_random_walk(log_prob, blackjax.mcmc.random_walk.normal(step_size))
    sample_key, warmup_key, init_key = jax.random.split(key, 3)
    initial_rwm_state = rw.init(initial_state, init_key)

    last_state, _ = mcmc_inference_loop(warmup_key, rw.step, initial_rwm_state, num_samples)
    _, (states, _) = mcmc_inference_loop(sample_key, rw.step, last_state, num_samples)
    return states.position


################################################################################
# Generic sampling method
################################################################################


def sample_flat(
    vars: list[RV],
    given_vars: list[RV],
    given_vals: list,
    *,
    run_inf: Callable,
    bijector_dict: Optional[dict] = default_bijector_dict,
    **inf_args,
) -> list[jnp.ndarray]:
    """
    Given a "flat" specification of an inference problem, do inference
    using a jax-based inference driver. The basic algorithm is:

    1. Find the latent (random, unconditioned) RVs upstream of `vars`
       and `given_vars`.
    2. Compile a jitted log-density over the unconstrained latent
       space via `jax_backend.ancestor_log_prob_flat`.
    3. Draw an initial latent position by ancestor sampling.
    4. Call `run_inf` to produce unconstrained samples.
    5. Map samples back to the constrained space via the bijectors,
       and extract the entries corresponding to `vars`.

    Parameters
    ----------
    vars
        The RVs you want to sample from.
    given_vars
        The RVs you want to condition on. Must all be random.
    given_vals
        The observed values for `given_vars`.
    run_inf
        Inference driver, with signature
        ``run_inf(log_prob, key, initial_state, num_samples, **options)``,
        where ``log_prob`` is a jitted jax function over the
        unconstrained latent space and ``key`` is a jax PRNGKey,
        returning unconstrained samples with leading axis
        ``num_samples``. See `run_nuts`, `run_hmc`, `run_pathfinder`,
        `run_meanfield_vi`, `run_fullrank_vi`, `run_smc`, `run_rwm`.
        The driver need not be gradient-based — any method that can
        sample from a jax log-density works.
    bijector_dict
        Mapping from ops to bijectors used to unconstrain latent RVs,
        or None to work directly in the constrained space.
    **inf_args
        Forwarded verbatim to `run_inf`. Must include `num_samples`,
        which is also used here to size the final `vmap`.

    Returns
    -------
    samples: list[jnp.ndarray]
        Samples for each variable in `vars`, each with leading axis
        ``inf_args["num_samples"]``.

    Examples
    --------
    >>> x = ir.RV(ir.Constant(0.5))
    >>> y = ir.RV(ir.Normal(), x, x)
    >>> z = ir.RV(ir.Normal(), y, x)
    >>> [samps_x, samps_y] = sample_flat([x, y], [z], [3.0], run_inf=run_nuts, num_samples=30)
    >>> samps_x.shape
    (30,)
    >>> samps_y.shape
    (30,)
    >>> np.allclose(samps_x, 0.5)
    True
    >>> np.allclose(samps_y, 0.5)
    False
    """

    if len(given_vars) != len(given_vals):
        raise ValueError("length of given_vars not equal to length of given_vals")

    if any(not v.op.random for v in given_vars):
        nonrandom_ops = [v.op for v in given_vars if not v.op.random]
        raise ValueError(f"Cannot condition on RV with non-random op(s) {nonrandom_ops}")

    given_vals = [jnp.array(val) for val in given_vals]
    all_vars = dag.upstream_nodes(tuple(vars) + tuple(given_vars))
    latent_vars = [var for var in all_vars if var.op.random and var not in given_vars]

    for v in latent_vars:
        if v.op.discrete:
            raise ValueError(
                f"sample_flat does not support discrete latent/unobserved RV "
                f"(latents are compiled to a continuous unconstrained space; "
                f"saw op {v.op})"
            )

    @jax.jit
    def log_prob(latent_vals):
        return jax_backend.ancestor_log_prob_flat(latent_vars + given_vars, latent_vals + given_vals, bijector_dict)

    seed = np.random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    init_key, inf_key = jax.random.split(key)

    latent_vals = jax_backend.ancestor_sample_flat(latent_vars, init_key, bijector_dict=bijector_dict)
    latent_samps = run_inf(log_prob, inf_key, latent_vals, **inf_args)

    if bijector_dict is not None and len(latent_samps) > 0:

        def constrain(latent_vals):
            all_unconstrained = jax_backend.ancestor_constrain(
                latent_vars + given_vars, latent_vals + given_vals, bijector_dict
            )
            latent_unconstrained = all_unconstrained[: len(latent_vars)]
            return latent_unconstrained

        latent_samps = jax.vmap(constrain)(latent_samps)

    def fill(latent_vals):
        return jax_backend.fill_in(latent_vars + given_vars, latent_vals + given_vals, vars)

    # TODO: extracting num_samples like this is not elegant
    return jax.vmap(fill, axis_size=inf_args["num_samples"])(latent_samps)


################################################################################
# Factory
################################################################################


def blackjax_calculate(run_inf, frozen: Iterable[str] = (), **options) -> Calculate:
    """
    Wrap an inference driver into a convenient `Calculate` object.

    The driver is always frozen: an engine bound to `run_nuts` cannot
    later be called with a different `run_inf`. All other options are
    overridable at call time.

    Parameters
    ----------
    run_inf
        Inference driver, with signature
        ``run_inf(log_prob, key, initial_state, num_samples, **options) -> samples``,
        where ``log_prob`` is a jitted jax function over the
        unconstrained latent space, ``key`` is a jax PRNGKey, and
        ``initial_state`` is a position in that space. See `run_nuts`,
        `run_hmc`, `run_pathfinder` for examples. Nothing requires the
        driver to use Blackjax internally.
    frozen
        Additional option names that cannot be overridden at call
        time. ``"run_inf"`` is always included.
    **options
        Default options forwarded to `run_inf` (and typically
        including ``num_samples``).

    Returns
    -------
    Calculate
        Calculator bound to `sample_flat` with the given driver. Its
        runtime docstring forwards the driver's documentation, so
        ``help(engine)`` shows the driver's parameters.

    Examples
    --------
    >>> my_nuts = blackjax_calculate(run_nuts, num_samples=500)
    >>> my_engine = blackjax_calculate(run_smc, num_particles=500)
    """

    calc = Calculate(sample_flat, run_inf=run_inf, frozen=frozenset(frozen) | {"run_inf"}, **options)
    calc.__doc__ = (
        f"Inference engine (a pangolin.calculate.Calculate) using {run_inf.__name__}.\n"
        f"See pangolin.calculate.Calculate for available methods "
        f"(sample, E, var, std, sample_arviz).\n\n"
        f"Options are forwarded to {run_inf.__name__}:\n\n"
        + textwrap.indent(inspect.getdoc(run_inf) or "(undocumented)", "    ")
    )
    return calc


################################################################################
# Engines (pre-bound drivers)
################################################################################

nuts = blackjax_calculate(run_nuts)
"""
NUTS inference engine; options are forwarded to `run_nuts`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, etc.).
"""

hmc = blackjax_calculate(run_hmc)
"""
HMC inference engine; options are forwarded to `run_hmc`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, etc.).
"""

pathfinder = blackjax_calculate(run_pathfinder)
"""
Pathfinder inference engine; options are forwarded to `run_pathfinder`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, etc.).
"""

meanfield_vi = blackjax_calculate(run_meanfield_vi)
"""
Mean-field VI inference engine; options are forwarded to `run_meanfield_vi`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, etc.).
"""

fullrank_vi = blackjax_calculate(run_fullrank_vi)
"""
Full-rank VI inference engine; options are forwarded to `run_fullrank_vi`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, `std`, `sample_arviz`).
"""

smc = blackjax_calculate(run_smc)
"""
Tempered SMC inference engine; options are forwarded to `run_smc`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, `std`, `sample_arviz`).
"""

rwm = blackjax_calculate(run_rwm)
"""
Random-walk Metropolis inference engine; options are forwarded to `run_rwm`.

This is a :class:`pangolin.calculate.Calculate` — see that class for the
available methods (`sample`, `E`, `var`, `std`, `sample_arviz`).
"""


################################################################################
# Default calculator and convenience methods (NUTS, 1000 samples)
################################################################################

calc = blackjax_calculate(run_nuts, num_samples=1000)

sample = calc.sample
"""
Default version of `Calculate.sample` that draws 1000 samples via NUTS.
"""
E = calc.E
"""
Default version of `Calculate.E` that uses 1000 samples via NUTS.
"""
var = calc.var
"""
Default version of `Calculate.var` that uses 1000 samples via NUTS.
"""
std = calc.std
"""
Default version of `Calculate.std` that uses 1000 samples via NUTS.
"""
sample_arviz = calc.sample_arviz
"""
Default version of `Calculate.sample_arviz` that uses 1000 samples via NUTS.
"""
