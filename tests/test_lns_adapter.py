"""Verify that LNS uses the upstream search and the shared MRTA semantics."""

import contextlib
from dataclasses import replace
import io
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from baseline.discrete_mbd_adapter import (
    SolverArgs,
    UNCONSTRAINED_LNS_METHOD,
    _select_unconstrained_result,
    lns,
    objective,
    problem_from_env,
    solve,
    unconstrained_objective,
)
from env.task_env import TaskEnv


class LNSAdapterTest(unittest.TestCase):
    def setUp(self):
        self.env = TaskEnv(
            per_species_range=(2, 2),
            species_range=(2, 2),
            tasks_range=(3, 3),
            traits_dim=2,
            max_task_size=2,
            agent_skill_mode="single_skill",
            seed=3,
        )
        self.args = SolverArgs(
            Ndiffuse=3, Nsample=4, temp_sample=1.0,
            N=self.env.agents_num * self.env.tasks_num,
            K=self.env.tasks_num + 1, sample_ratio=0.5,
            elite_frac=0.5, T0=1.0, alpha=0.95,
            destroy_k_ratio=0.25,
        )

    def test_exact_adapter_matches_upstream_and_replays(self):
        problem = problem_from_env(self.env)
        cost_fn = lambda decision: objective(problem, decision)
        history = lns(self.args, cost_fn, seed=7)
        expected = history[jnp.argmin(jax.vmap(cost_fn)(history))]
        result = solve(self.env, "lns", self.args, seed=7)

        np.testing.assert_array_equal(result.decision, expected)
        self.assertTrue(result.feasible)
        self.env.init_state()
        for agent_id, route in enumerate(result.routes):
            self.env.pre_set_route(list(route[1:]), agent_id)
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.execute_by_route(plot_figure=False)
        _, finished = self.env.get_episode_reward(100)
        self.assertTrue(np.all(finished))
        self.assertAlmostEqual(
            result.predicted_makespan, self.env.current_time, places=4
        )

    def test_unconstrained_adapter_matches_upstream_with_multi_skill_teams(self):
        abilities = np.asarray([[1, 1], [1, 0]], dtype=int)
        self.env.species_dict["abilities"] = abilities
        for agent in self.env.agent_dic.values():
            agent["abilities"] = abilities[agent["species"]].copy()
        problem = problem_from_env(self.env, require_one_hot=False)
        cost_fn = lambda decision: unconstrained_objective(
            problem, decision, self.args.infeasibility_penalty
        )
        history = lns(self.args, cost_fn, seed=7)
        expected = _select_unconstrained_result(self.args, problem, history)
        result = solve(
            self.env, UNCONSTRAINED_LNS_METHOD, self.args,
            seed=7, warmup=False,
        )

        np.testing.assert_array_equal(result.decision, expected[0])
        self.assertAlmostEqual(result.predicted_makespan, float(expected[1]))
        self.assertAlmostEqual(result.penalized_objective, float(expected[2]))
        self.assertEqual(result.feasibility_violation, float(expected[3]))
        self.assertEqual(result.infeasible_tasks, int(expected[4]))

    def test_invalid_destroy_ratios_fail_before_search(self):
        for method in ("lns", UNCONSTRAINED_LNS_METHOD):
            for ratio in (0.0, -0.1, 1.1, float("nan")):
                with self.subTest(method=method, ratio=ratio):
                    with self.assertRaisesRegex(ValueError, "destroy_k_ratio"):
                        solve(
                            self.env, method,
                            replace(self.args, destroy_k_ratio=ratio),
                            warmup=False,
                        )

    def test_small_positive_destroy_ratio_matches_upstream(self):
        args = replace(self.args, destroy_k_ratio=0.001)
        problem = problem_from_env(self.env)
        cost_fn = lambda decision: objective(problem, decision)
        history = lns(args, cost_fn, seed=7)
        result = solve(self.env, "lns", args, seed=7, warmup=False)
        np.testing.assert_array_equal(result.decision, history[-1])

    def test_single_iteration_can_change_all_variables(self):
        result = solve(
            self.env, "lns",
            replace(self.args, Ndiffuse=1, destroy_k_ratio=1.0),
            warmup=False,
        )
        self.assertTrue(result.feasible)
        self.assertTrue(np.isfinite(result.objective))


if __name__ == "__main__":
    unittest.main()
