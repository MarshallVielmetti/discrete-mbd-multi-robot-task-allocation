import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from baseline import ctas_d
from env.task_env import TaskEnv


class CTASBackendTest(unittest.TestCase):
    def fixture(self, root, tasks=1):
        env = TaskEnv(per_species_range=(1, 1), species_range=(1, 1),
                      tasks_range=(tasks, tasks), traits_dim=1, max_task_size=2,
                      agent_skill_mode="single_skill", seed=0)
        path = root / "env_0.pkl"
        path.write_bytes(pickle.dumps(env))
        result_dir = root / "solution"
        result_dir.mkdir()
        ctas_d.write_ctas_inputs(env, result_dir)
        return env, path, result_dir

    def test_replay_completes_task_and_returns_to_depot(self):
        with tempfile.TemporaryDirectory() as temporary:
            env, path, result_dir = self.fixture(Path(temporary))
            (result_dir / "results.yaml").write_text(yaml.safe_dump({
                "result": {"plannerTime": 30.05},
                "vehicle": {"v1": {"type": 1, "node": [2, 1, 0]}},
            }))
            result = ctas_d.evaluate_instance(path, result_dir)
            distance = np.linalg.norm(env.task_dic[0]["location"] - env.agent_dic[0]["depot"])
            expected = 2 * distance / env.agent_dic[0]["velocity"] + env.task_dic[0]["time"]
            self.assertEqual(result["solved"], 1)
            self.assertAlmostEqual(result["makespan"], expected)
            self.assertEqual(result["stored_solver_time"], 30.05)

    def test_missing_solution_and_incomplete_plan_are_unsolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, path, result_dir = self.fixture(Path(temporary), tasks=2)
            missing = ctas_d.evaluate_instance(path, result_dir)
            self.assertEqual(missing["solved"], 0)
            self.assertTrue(math.isnan(missing["makespan"]))
            (result_dir / "results.yaml").write_text(yaml.safe_dump({
                "result": {"plannerTime": 30},
                "vehicle": {"v1": {"type": 1, "node": [3, 1, 0]}},
            }))
            incomplete = ctas_d.evaluate_instance(path, result_dir)
            self.assertEqual(incomplete["solved"], 0)
            self.assertTrue(math.isnan(incomplete["makespan"]))

    def test_export_and_portable_parameters_preserve_capabilities_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, _, folder = self.fixture(root)
            vehicles = yaml.safe_load((folder / "vehicle_param.yaml").read_text())
            self.assertEqual(vehicles["vehicle0"]["capVector"], env.species_dict["abilities"][0].tolist())
            portable = root / "portable.yaml"
            ctas_d.prepare_parameters(folder / "planner_param.yaml", portable, 30)
            parameters = yaml.safe_load(portable.read_text())
            self.assertEqual(parameters["solverMaxTime"], 30)
            self.assertTrue(Path(parameters["graphFile"]).is_absolute())

    def test_license_failure_propagates_before_solver_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            solver = root / "solver"
            solver.touch()
            with mock.patch.object(ctas_d, "check_gurobi_license", side_effect=RuntimeError("unlicensed")):
                with self.assertRaisesRegex(RuntimeError, "unlicensed"):
                    ctas_d.preflight(ctas_d.CTASConfig(solver=solver), root)

    def test_legacy_main_pickles_load_without_the_original_exporter(self):
        with tempfile.TemporaryDirectory() as temporary:
            env, path, _ = self.fixture(Path(temporary))
            with mock.patch.object(TaskEnv, "__module__", "__main__"), \
                    mock.patch.object(sys.modules["__main__"], "TaskEnv", TaskEnv, create=True):
                path.write_bytes(pickle.dumps(env))
            loaded = ctas_d.load_instance(path)
            self.assertIsInstance(loaded, TaskEnv)
            self.assertEqual(loaded.tasks_num, 1)


if __name__ == "__main__":
    unittest.main()
