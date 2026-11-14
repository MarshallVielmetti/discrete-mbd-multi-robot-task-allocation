"""Check MRTA integration against the cleaning branch's MBD API."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from baseline.discrete_mbd_adapter import (
    ASSIGNMENT_PERMUTATION_METHOD, SolverArgs, UNCONSTRAINED_METHOD,
    _select_unconstrained_result, discrete_mbd, objective, problem_from_env,
    solve, unconstrained_objective,
)
from env.task_env import TaskEnv
from scripts import run_seed_benchmark as benchmark


class MBDBackendTest(unittest.TestCase):
    def setUp(self):
        self.env = TaskEnv(
            per_species_range=(2, 2), species_range=(2, 2), tasks_range=(3, 3),
            traits_dim=2, max_task_size=2, agent_skill_mode="single_skill", seed=3,
        )
        self.args = SolverArgs(
            Ndiffuse=3, Nsample=8, temp_sample=1.0, N=12, K=4,
            sample_ratio=0.5, elite_frac=0.5, T0=1.0, alpha=0.95,
        )

    def test_unconstrained_adapter_matches_upstream(self):
        problem = problem_from_env(self.env)
        cost_fn = lambda decision: unconstrained_objective(
            problem, decision, self.args.infeasibility_penalty)
        history = discrete_mbd(self.args, cost_fn, seed=7)
        expected = _select_unconstrained_result(self.args, problem, history)
        result = solve(self.env, UNCONSTRAINED_METHOD, self.args, seed=7, warmup=False)
        np.testing.assert_array_equal(result.decision, expected[0])
        self.assertAlmostEqual(result.predicted_makespan, float(expected[1]))

    def test_benchmark_mbd_matches_upstream_with_a_dynamic_seed(self):
        problem = problem_from_env(self.env)
        cost_fn = lambda decision: objective(problem, decision)
        history = discrete_mbd(self.args, cost_fn, seed=7)
        expected = history[jnp.argmin(jax.vmap(cost_fn)(history))]
        benchmark.load_search_backend(["MBD"])
        result = benchmark.run_solver("MBD", "exact", self.args, problem, jnp.int32(7))
        np.testing.assert_array_equal(result[0], expected)

    def test_instrumented_history_matches_upstream(self):
        from baseline.discrete_mbd_history import discrete_mbd_history
        cost_fn = lambda decision: jnp.sum(jnp.argmax(decision, axis=-1))
        expected = discrete_mbd(self.args, cost_fn, seed=7)
        actual = discrete_mbd_history(self.args, cost_fn, seed=jnp.int32(7))
        np.testing.assert_array_equal(actual.decisions, expected)
        np.testing.assert_array_equal(actual.objectives, jax.vmap(cost_fn)(expected))
        self.assertTrue(np.isfinite(actual.probabilities).all())

    def test_assignment_permutation_solver_preserves_permutations(self):
        from dataclasses import replace
        args = replace(self.args, N=12 + 3**2, K=2)
        result = solve(self.env, ASSIGNMENT_PERMUTATION_METHOD, args, seed=7, warmup=False)
        np.testing.assert_array_equal(result.permutation.sum(axis=0), np.ones(3))
        np.testing.assert_array_equal(result.permutation.sum(axis=1), np.ones(3))
        self.assertTrue(np.isfinite(result.penalized_objective))


if __name__ == "__main__":
    unittest.main()
