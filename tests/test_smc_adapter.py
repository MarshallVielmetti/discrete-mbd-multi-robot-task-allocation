"""Verify upstream SMC integration and its temperature/history constraints."""

import contextlib
from dataclasses import replace
import io
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from baseline.discrete_mbd_adapter import (
    SolverArgs,
    UNCONSTRAINED_SMC_METHOD,
    feasibility_metrics,
    objective,
    problem_from_env,
    schedule_unconstrained_decision,
    smc,
    solve,
    unconstrained_objective,
)
from env.task_env import TaskEnv


class SMCAdapterTest(unittest.TestCase):
    def setUp(self):
        self.env = TaskEnv(
            per_species_range=(2, 2), species_range=(2, 2),
            tasks_range=(3, 3), traits_dim=2, max_task_size=2,
            agent_skill_mode="single_skill", seed=3,
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
        history = smc(self.args, cost_fn, seed=7)
        self.assertEqual(history.shape[0], self.args.Ndiffuse - 1)
        expected = history[jnp.argmin(jax.vmap(cost_fn)(history))]
        result = solve(self.env, "smc", self.args, seed=7)

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
        history = smc(self.args, cost_fn, seed=7)
        costs = np.asarray(jax.vmap(cost_fn)(history))
        base_costs = np.asarray(jax.vmap(
            lambda decision: schedule_unconstrained_decision(problem, decision)[0]
        )(history))
        violations, infeasible_tasks = jax.vmap(
            lambda decision: feasibility_metrics(problem, decision)
        )(history)
        feasible = np.asarray(violations) == 0
        best = (
            np.argmin(np.where(feasible, base_costs, np.inf))
            if feasible.any() else np.argmin(costs)
        )
        result = solve(
            self.env, UNCONSTRAINED_SMC_METHOD, self.args,
            seed=7, warmup=False,
        )

        np.testing.assert_array_equal(result.decision, history[best])
        self.assertAlmostEqual(result.predicted_makespan, float(base_costs[best]))
        self.assertAlmostEqual(result.penalized_objective, float(costs[best]))
        self.assertEqual(result.feasible, bool(feasible[best]))
        self.assertEqual(result.feasibility_violation, float(violations[best]))
        self.assertEqual(result.infeasible_tasks, int(infeasible_tasks[best]))

    def test_invalid_configurations_fail_before_search(self):
        invalid = [
            {"Ndiffuse": 1},
            {"T0": 0.0}, {"T0": -1.0}, {"T0": float("inf")},
            {"T0": float("nan")},
            {"alpha": 0.0}, {"alpha": -0.1}, {"alpha": 1.1},
            {"alpha": float("nan")},
            {"destroy_k_ratio": 0.0}, {"destroy_k_ratio": -0.1},
            {"destroy_k_ratio": 1.1}, {"destroy_k_ratio": float("nan")},
        ]
        for method in ("smc", UNCONSTRAINED_SMC_METHOD):
            for config in invalid:
                with self.subTest(method=method, config=config):
                    with self.assertRaises(ValueError):
                        solve(self.env, method, replace(self.args, **config), warmup=False)

    def test_minimum_history_and_small_mutation_block_preserve_upstream_behavior(self):
        # Upstream clamps a sub-one-variable SMC mutation block to one variable.
        args = replace(self.args, Ndiffuse=2, destroy_k_ratio=0.001, alpha=1.0)
        problem = problem_from_env(self.env)
        history = smc(args, lambda decision: objective(problem, decision), seed=2)
        result = solve(self.env, "smc", args, seed=2, warmup=False)
        np.testing.assert_array_equal(result.decision, history[0])
        self.assertTrue(result.feasible)
        self.assertTrue(np.isfinite(result.objective))


if __name__ == "__main__":
    unittest.main()
