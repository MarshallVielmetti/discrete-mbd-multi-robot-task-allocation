"""Record denoising distributions without changing the benchmark MBD update.

The upstream checkout returns only selected decisions. This instrumented scan
matches its RNG splits and update at MBD_UPSTREAM_REVISION; the equivalence test
guards against drift when that checkout changes. The scan's current decision is
not an elitist incumbent. Best-so-far selection belongs to the visualization.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

# Importing the adapter puts the external solver checkout on sys.path.
from baseline.discrete_mbd_adapter import discrete_mbd  # noqa: F401
from solver.discrete_mbd import construct_Qs, sample_categorical


class DiffusionHistory(NamedTuple):
    decisions: jnp.ndarray
    probabilities: jnp.ndarray
    objectives: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1, 2))
def discrete_mbd_history(args, cost_fn, sampler=None, seed=42):
    """Return selected x, updated p, and cost at each reverse step M..1.

    probabilities[k] is the distribution used to draw the second candidate
    batch at step k, before the selected decision is carried into the next step.
    The default sampler matches upstream; an optional constrained sampler may
    return (samples, log_importance_correction) for the binary (A, P) adapter.
    """
    first_count = int(args.Nsample * args.sample_ratio)
    second_count = args.Nsample - first_count
    if args.Ndiffuse < 1 or first_count < 1 or second_count < 1:
        raise ValueError("Require positive steps and at least one sample in each batch")
    if args.temp_sample <= 0:
        raise ValueError("temp_sample must be positive")
    if sampler is None:
        def sampler(rng, probabilities, count):
            samples = sample_categorical(rng, probabilities, count)
            return samples, jnp.zeros((), dtype=probabilities.dtype)
    transitions, cumulative = construct_Qs(args.Ndiffuse, args.K)
    previous = jnp.concatenate([jnp.eye(args.K)[None], cumulative[:-1]], axis=0)
    rng, initial_rng = jax.random.split(jax.random.PRNGKey(seed))
    uniform = jnp.ones((args.N, args.K)) / args.K
    initial = sampler(initial_rng, uniform, 1)[0][0]

    def reverse_step(carry, step):
        rng, current = carry
        rng, proposal_rng, final_rng = jax.random.split(rng, 3)
        signal = cumulative[step]
        proposal = current @ signal.T
        proposal = proposal / jnp.sum(proposal, axis=-1, keepdims=True)
        candidates, correction = sampler(proposal_rng, proposal, first_count)
        posterior = jnp.einsum(
            "rt,kt,jt->rjk", current, transitions[step],
            1.0 / jnp.clip(signal, 1e-30),
        )
        posterior = posterior * previous[step, None]
        costs = jax.vmap(cost_fn)(candidates)
        weights = jax.nn.softmax(-costs / args.temp_sample + correction)
        weighted = jnp.einsum("s,srj->rj", weights, candidates)
        probabilities = jnp.einsum("rj,rjk->rk", weighted, posterior)
        final_candidates, _ = sampler(final_rng, probabilities, second_count)
        final_costs = jax.vmap(cost_fn)(final_candidates)
        all_candidates = jnp.concatenate([candidates, final_candidates], axis=0)
        all_costs = jnp.concatenate([costs, final_costs], axis=0)
        selected_index = jnp.argmin(all_costs)
        selected = all_candidates[selected_index]
        return (rng, selected), (selected, probabilities, all_costs[selected_index])

    _, history = jax.lax.scan(
        reverse_step, (rng, initial), jnp.arange(args.Ndiffuse - 1, -1, -1)
    )
    return DiffusionHistory(*history)
