import contextlib
import io
import math
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from baseline.discrete_mbd_adapter import (
    UNCONSTRAINED_CEM_METHOD,
    UNCONSTRAINED_METHOD,
    UNCONSTRAINED_SA_METHOD,
    assignment_permutation_feasibility_metrics,
    assignment_permutation_objective,
    BenchmarkProblem,
    decode_decision,
    decode_assignment_permutation_decision,
    decode_unconstrained_decision,
    feasibility_metrics,
    make_assignment_permutation_sampler,
    make_iid_sampler,
    problem_from_env,
    routes_from_assignment_permutation_decision,
    routes_from_decision,
    routes_from_unconstrained_decision,
    schedule_assignment_permutation_decision,
    schedule_decision,
    schedule_unconstrained_decision,
    solve,
    SolverArgs,
    unconstrained_objective,
)
from env.task_env import TaskEnv
from scripts.generate_paper_scenarios import PAPER_CONFIGURATIONS, make_paper_environment


class DiscreteMBDAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_paper_environment("sa_bt", PAPER_CONFIGURATIONS[1], seed=0)

    def make_ordered_decision(self):
        levels = np.broadcast_to(
            np.arange(self.env.tasks_num),
            (self.env.agents_num, self.env.tasks_num),
        )
        return jax.nn.one_hot(
            jnp.asarray(levels.reshape(-1)), self.env.tasks_num + 1
        )

    def make_assignment_permutation_decision(self, assignment, order):
        permutation = np.zeros(
            (self.env.tasks_num, self.env.tasks_num), dtype=np.int32
        )
        permutation[np.asarray(order), np.arange(self.env.tasks_num)] = 1
        bits = np.concatenate(
            [
                np.asarray(assignment, dtype=np.int32).reshape(-1),
                permutation.reshape(-1),
            ]
        )
        return jax.nn.one_hot(jnp.asarray(bits), 2)

    @staticmethod
    def make_problem(capabilities, requirements):
        capabilities = np.asarray(capabilities, dtype=np.float32)
        requirements = np.asarray(requirements, dtype=np.float32)
        num_agents = capabilities.shape[0]
        num_tasks = requirements.shape[0]
        return BenchmarkProblem(
            agent_species=jnp.arange(num_agents, dtype=jnp.int32),
            capabilities=jnp.asarray(capabilities),
            requirements=jnp.asarray(requirements),
            durations=jnp.ones(num_tasks, dtype=jnp.float32),
            task_distances=jnp.zeros(
                (num_tasks, num_tasks), dtype=jnp.float32
            ),
            depot_distances=jnp.zeros(
                (num_agents, num_tasks), dtype=jnp.float32
            ),
            velocities=jnp.ones(num_agents, dtype=jnp.float32),
        )

    @staticmethod
    def make_unconstrained_decision(levels, categories=4):
        return jax.nn.one_hot(
            jnp.asarray(levels, dtype=jnp.int32).reshape(-1), categories
        )

    def test_decoder_builds_exact_task_teams(self):
        problem = problem_from_env(self.env)
        decision = self.make_ordered_decision()
        _, membership, _, _ = decode_decision(problem, decision)
        abilities = np.asarray(self.env.species_dict["abilities"])[
            np.asarray(problem.agent_species)
        ]
        coverage = np.asarray(membership, dtype=np.int32).T @ abilities
        np.testing.assert_array_equal(coverage, np.asarray(problem.requirements))

    def test_predicted_makespan_matches_task_env_replay(self):
        problem = problem_from_env(self.env)
        decision = self.make_ordered_decision()
        predicted = float(schedule_decision(problem, decision)[0])
        routes = routes_from_decision(problem, decision)

        self.env.init_state()
        for agent_id, route in enumerate(routes):
            self.env.pre_set_route(list(route[1:]), agent_id)
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.execute_by_route(plot_figure=False)
        _, finished = self.env.get_episode_reward(100)

        self.assertTrue(np.all(finished))
        self.assertAlmostEqual(predicted, self.env.current_time, places=4)

    def test_unconstrained_decoder_does_not_force_feasibility(self):
        problem = problem_from_env(self.env)
        decision = jax.nn.one_hot(
            jnp.zeros(self.env.agents_num * self.env.tasks_num, dtype=jnp.int32),
            self.env.tasks_num + 1,
        )

        _, membership, _, _ = decode_unconstrained_decision(problem, decision)
        violation, infeasible_tasks = feasibility_metrics(problem, decision)

        self.assertFalse(bool(jnp.any(membership)))
        self.assertEqual(float(violation), float(jnp.sum(problem.requirements)))
        self.assertEqual(int(infeasible_tasks), self.env.tasks_num)

    def test_unconstrained_objective_penalizes_capability_shortfall(self):
        problem = problem_from_env(self.env)
        decision = jax.nn.one_hot(
            jnp.zeros(self.env.agents_num * self.env.tasks_num, dtype=jnp.int32),
            self.env.tasks_num + 1,
        )
        penalty = 123.0
        makespan = float(schedule_unconstrained_decision(problem, decision)[0])
        violation = float(feasibility_metrics(problem, decision)[0])
        penalized = float(unconstrained_objective(problem, decision, penalty))

        self.assertAlmostEqual(penalized, makespan + penalty * violation, places=4)

    def test_unconstrained_objective_decodes_decision_once(self):
        problem = problem_from_env(self.env)
        decision = self.make_unconstrained_decision(
            np.arange(self.env.agents_num * self.env.tasks_num).reshape(
                self.env.agents_num, self.env.tasks_num
            )
            % (self.env.tasks_num + 1),
            categories=self.env.tasks_num + 1,
        )
        penalty = 123.0
        expected = schedule_unconstrained_decision(problem, decision)[0] + (
            penalty * feasibility_metrics(problem, decision)[0]
        )

        with mock.patch(
            "baseline.discrete_mbd_adapter.decode_unconstrained_decision",
            wraps=decode_unconstrained_decision,
        ) as decode:
            actual = unconstrained_objective(problem, decision, penalty)

        self.assertEqual(decode.call_count, 1)
        self.assertEqual(float(actual), float(expected))

    def test_unconstrained_decoder_trims_multi_skill_coalition(self):
        problem = self.make_problem(
            capabilities=[[1, 1], [1, 0], [0, 1]],
            requirements=[[1, 1]],
        )
        decision = self.make_unconstrained_decision([[1], [2], [2]])

        _, membership, _, _ = decode_unconstrained_decision(problem, decision)
        violation, infeasible_tasks = feasibility_metrics(problem, decision)

        np.testing.assert_array_equal(
            np.asarray(membership).reshape(-1), [True, False, False]
        )
        self.assertEqual(float(violation), 0.0)
        self.assertEqual(int(infeasible_tasks), 0)

    def test_unconstrained_decoder_preserves_repeated_skill_requirements(self):
        problem = self.make_problem(
            capabilities=[[1, 1], [1, 1], [1, 0], [0, 1]],
            requirements=[[2, 1]],
        )
        decision = self.make_unconstrained_decision([[1], [2], [3], [3]])

        _, membership, _, _ = decode_unconstrained_decision(problem, decision)
        coverage = (
            np.asarray(membership, dtype=np.float32).T
            @ np.asarray(problem.capabilities)
        )

        self.assertTrue(np.all(coverage >= np.asarray(problem.requirements)))
        self.assertEqual(int(np.asarray(membership).sum()), 2)

    def test_unconstrained_decoder_trims_without_hiding_existing_shortfall(self):
        problem = self.make_problem(
            capabilities=[[1, 0], [1, 0], [0, 1]],
            requirements=[[1, 1]],
        )
        decision = self.make_unconstrained_decision([[1], [2], [0]])

        _, membership, _, _ = decode_unconstrained_decision(problem, decision)
        violation, infeasible_tasks = feasibility_metrics(problem, decision)

        np.testing.assert_array_equal(
            np.asarray(membership).reshape(-1), [True, False, False]
        )
        self.assertEqual(float(violation), 1.0)
        self.assertEqual(int(infeasible_tasks), 1)

    def test_multi_skill_trimming_preserves_one_hot_decoder_behavior(self):
        problem = problem_from_env(self.env)
        rng = np.random.default_rng(22)
        levels = rng.integers(
            0,
            self.env.tasks_num + 1,
            size=(self.env.agents_num, self.env.tasks_num),
        )
        decision = self.make_unconstrained_decision(
            levels, categories=self.env.tasks_num + 1
        )

        _, membership, _, _ = decode_unconstrained_decision(problem, decision)
        expected = np.zeros_like(levels, dtype=bool)
        requirements = np.asarray(problem.requirements, dtype=np.int32)
        agent_species = np.asarray(problem.agent_species)
        for task_id in range(self.env.tasks_num):
            for species_id in range(self.env.species_num):
                eligible = [
                    agent_id
                    for agent_id in range(self.env.agents_num)
                    if agent_species[agent_id] == species_id
                    and levels[agent_id, task_id] > 0
                ]
                eligible.sort(
                    key=lambda agent_id: (
                        levels[agent_id, task_id],
                        agent_id,
                    )
                )
                count = requirements[task_id, species_id]
                expected[eligible[:count], task_id] = True

        np.testing.assert_array_equal(np.asarray(membership), expected)

    def test_unconstrained_solver_accepts_multi_skill_environment(self):
        env = TaskEnv(
            per_species_range=(1, 1),
            species_range=(2, 2),
            tasks_range=(2, 2),
            traits_dim=2,
            max_task_size=2,
            seed=3,
        )
        abilities = np.asarray([[1, 1], [1, 0]], dtype=int)
        env.species_dict["abilities"] = abilities
        for agent in env.agent_dic.values():
            agent["abilities"] = abilities[agent["species"]].copy()
        for task in env.task_dic.values():
            task["requirements"] = np.asarray([1, 1], dtype=int)
            task["status"] = task["requirements"].copy()

        args = SolverArgs(
            Ndiffuse=1,
            Nsample=2,
            temp_sample=1.0,
            N=env.agents_num * env.tasks_num,
            K=env.tasks_num + 1,
            sample_ratio=0.5,
            elite_frac=0.5,
            T0=1.0,
            alpha=0.95,
            infeasibility_penalty=100.0,
        )
        for method in (
            UNCONSTRAINED_METHOD,
            UNCONSTRAINED_SA_METHOD,
            UNCONSTRAINED_CEM_METHOD,
        ):
            with self.subTest(method=method):
                result = solve(
                    env,
                    method,
                    args,
                    seed=0,
                    warmup=False,
                )
                self.assertEqual(
                    result.assignment.shape,
                    (env.agents_num, env.tasks_num),
                )

    def test_feasible_unconstrained_schedule_matches_task_env_replay(self):
        problem = problem_from_env(self.env)
        task_levels = np.arange(1, self.env.tasks_num + 1, dtype=np.int32)
        levels = np.broadcast_to(
            task_levels, (self.env.agents_num, self.env.tasks_num)
        )
        decision = jax.nn.one_hot(
            jnp.asarray(levels.reshape(-1)), self.env.tasks_num + 1
        )
        violation, _ = feasibility_metrics(problem, decision)
        predicted = float(schedule_unconstrained_decision(problem, decision)[0])
        routes = routes_from_unconstrained_decision(problem, decision)

        self.assertEqual(float(violation), 0.0)
        self.env.init_state()
        for agent_id, route in enumerate(routes):
            self.env.pre_set_route(list(route[1:]), agent_id)
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.execute_by_route(plot_figure=False)
        _, finished = self.env.get_episode_reward(100)

        self.assertTrue(np.all(finished))
        self.assertAlmostEqual(predicted, self.env.current_time, places=4)

    def test_assignment_permutation_sampler_always_returns_permutations(self):
        num_agents = 3
        num_tasks = 4
        num_variables = num_agents * num_tasks + num_tasks**2
        probabilities = jax.random.uniform(
            jax.random.PRNGKey(10),
            (num_variables, 2),
            minval=0.05,
            maxval=1.0,
        )
        sampler = make_assignment_permutation_sampler(num_agents, num_tasks)

        samples, log_correction = sampler(
            jax.random.PRNGKey(11), probabilities, 128
        )
        permutation_bits = jnp.argmax(samples, axis=-1)[
            :, num_agents * num_tasks :
        ].reshape(128, num_tasks, num_tasks)

        np.testing.assert_array_equal(
            np.asarray(permutation_bits.sum(axis=1)),
            np.ones((128, num_tasks), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(permutation_bits.sum(axis=2)),
            np.ones((128, num_tasks), dtype=np.int32),
        )
        self.assertEqual(samples.shape, (128, num_variables, 2))
        self.assertEqual(log_correction.shape, (128,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(log_correction))))

        _, uniform_correction = sampler(
            jax.random.PRNGKey(12), jnp.ones((num_variables, 2)), 8
        )
        expected = math.lgamma(num_tasks + 1) - num_tasks**2 * math.log(2.0)
        np.testing.assert_allclose(
            np.asarray(uniform_correction), expected, rtol=1e-6
        )

    def test_iid_sampler_matches_categorical_probabilities(self):
        probabilities = jnp.asarray(
            [
                [0.1, 0.2, 0.7],
                [3.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        sampler = make_iid_sampler(num_variables=2, num_categories=3)

        samples, log_correction = sampler(
            jax.random.PRNGKey(13), probabilities, 100_000
        )
        frequencies = jnp.mean(samples, axis=0)
        expected = probabilities / jnp.sum(
            probabilities, axis=-1, keepdims=True
        )

        self.assertEqual(samples.shape, (100_000, 2, 3))
        self.assertEqual(log_correction.shape, ())
        np.testing.assert_allclose(
            np.asarray(frequencies), np.asarray(expected), atol=5e-3
        )

    def test_assignment_permutation_decoder_induces_per_robot_routes(self):
        problem = problem_from_env(self.env)
        assignment = np.zeros(
            (self.env.agents_num, self.env.tasks_num), dtype=np.int32
        )
        assignment[0, [0, 2]] = 1
        assignment[1, [1, 2]] = 1
        order = np.arange(self.env.tasks_num)
        decision = self.make_assignment_permutation_decision(assignment, order)

        decoded_assignment, permutation, priorities, decoded_order = (
            decode_assignment_permutation_decision(problem, decision)
        )
        routes = routes_from_assignment_permutation_decision(problem, decision)

        np.testing.assert_array_equal(np.asarray(decoded_assignment), assignment)
        np.testing.assert_array_equal(np.asarray(permutation), np.eye(self.env.tasks_num))
        np.testing.assert_array_equal(np.asarray(priorities), order + 1)
        np.testing.assert_array_equal(np.asarray(decoded_order), order)
        self.assertEqual(routes[0], [0, 1, 3, 0])
        self.assertEqual(routes[1], [0, 2, 3, 0])

    def test_assignment_permutation_penalizes_capability_shortfall(self):
        problem = problem_from_env(self.env)
        assignment = np.zeros(
            (self.env.agents_num, self.env.tasks_num), dtype=np.int32
        )
        decision = self.make_assignment_permutation_decision(
            assignment, np.arange(self.env.tasks_num)
        )
        penalty = 123.0

        makespan = float(
            schedule_assignment_permutation_decision(problem, decision)[0]
        )
        violation, infeasible_tasks = (
            assignment_permutation_feasibility_metrics(problem, decision)
        )
        penalized = float(
            assignment_permutation_objective(problem, decision, penalty)
        )

        self.assertEqual(float(violation), float(jnp.sum(problem.requirements)))
        self.assertEqual(int(infeasible_tasks), self.env.tasks_num)
        self.assertAlmostEqual(
            penalized, makespan + penalty * float(violation), places=4
        )

    def test_global_permutation_does_not_serialize_disjoint_robots(self):
        problem = BenchmarkProblem(
            agent_species=jnp.asarray([0, 0], dtype=jnp.int32),
            capabilities=jnp.ones((2, 1), dtype=jnp.float32),
            requirements=jnp.asarray([[1], [1]], dtype=jnp.int32),
            durations=jnp.asarray([2.0, 3.0], dtype=jnp.float32),
            task_distances=jnp.zeros((2, 2), dtype=jnp.float32),
            depot_distances=jnp.zeros((2, 2), dtype=jnp.float32),
            velocities=jnp.ones(2, dtype=jnp.float32),
        )
        assignment = np.eye(2, dtype=np.int32)
        permutation = np.eye(2, dtype=np.int32)
        bits = np.concatenate([assignment.reshape(-1), permutation.reshape(-1)])
        decision = jax.nn.one_hot(jnp.asarray(bits), 2)

        makespan, starts, finishes, _ = (
            schedule_assignment_permutation_decision(problem, decision)
        )

        self.assertEqual(float(makespan), 3.0)
        np.testing.assert_array_equal(np.asarray(starts), np.zeros(2))
        np.testing.assert_array_equal(np.asarray(finishes), [2.0, 3.0])

    def test_feasible_assignment_permutation_schedule_matches_task_env_replay(self):
        problem = problem_from_env(self.env)
        random_key_decision = self.make_ordered_decision()
        _, membership, _, order = decode_decision(problem, random_key_decision)
        decision = self.make_assignment_permutation_decision(membership, order)
        violation, _ = assignment_permutation_feasibility_metrics(problem, decision)
        predicted = float(
            schedule_assignment_permutation_decision(problem, decision)[0]
        )
        routes = routes_from_assignment_permutation_decision(problem, decision)

        self.assertEqual(float(violation), 0.0)
        self.env.init_state()
        for agent_id, route in enumerate(routes):
            self.env.pre_set_route(list(route[1:]), agent_id)
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.execute_by_route(plot_figure=False)
        _, finished = self.env.get_episode_reward(100)

        self.assertTrue(np.all(finished))
        self.assertAlmostEqual(predicted, self.env.current_time, places=4)


if __name__ == "__main__":
    unittest.main()
