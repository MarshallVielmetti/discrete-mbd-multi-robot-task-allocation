import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from env.task_env import TaskEnv
from scripts.generate_paper_scenarios import (
    PAPER_CONFIGURATIONS,
    generate_suites,
    make_paper_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def environment_arrays(env):
    return {
        "abilities": np.asarray(env.species_dict["abilities"]),
        "requirements": np.asarray(
            env.get_matrix(env.task_dic, "requirements")
        ),
        "task_locations": np.asarray(env.get_matrix(env.task_dic, "location")),
        "task_durations": np.asarray(env.get_matrix(env.task_dic, "time")),
        "depot_locations": np.asarray(env.get_matrix(env.depot_dic, "location")),
    }


class PaperScenarioGeneratorTest(unittest.TestCase):
    def test_table_configuration_matrix(self):
        observed = [
            (item.agents, item.species, item.tasks, item.skills)
            for item in PAPER_CONFIGURATIONS
        ]
        self.assertEqual(
            observed,
            [
                (9, 3, 20, 3),
                (15, 5, 20, 5),
                (25, 5, 20, 5),
                (25, 5, 50, 5),
                (50, 5, 50, 5),
            ],
        )

    @unittest.skipUnless((REPO_ROOT / "RALTestSet/env_0.pkl").is_file(),
                         "Optional upstream RALTestSet is not included in this repository")
    def test_reconstructs_released_sa_bt_instance(self):
        sys.modules["__main__"].TaskEnv = TaskEnv
        with (REPO_ROOT / "RALTestSet" / "env_0.pkl").open("rb") as stream:
            released = pickle.load(stream)

        reconstructed = make_paper_environment(
            "sa_bt", PAPER_CONFIGURATIONS[1], seed=0
        )

        for name, expected in environment_arrays(released).items():
            np.testing.assert_array_equal(
                environment_arrays(reconstructed)[name], expected, err_msg=name
            )

    def test_scenario_semantics_and_fleet_feasibility(self):
        configuration = PAPER_CONFIGURATIONS[0]
        for scenario in ("sa_bt", "sa_at", "ma_at"):
            with self.subTest(scenario=scenario):
                env = make_paper_environment(scenario, configuration, seed=4)
                abilities = np.asarray(env.species_dict["abilities"])
                requirements = np.asarray(
                    env.get_matrix(env.task_dic, "requirements")
                )

                self.assertEqual(env.agents_num, configuration.agents)
                self.assertEqual(env.species_num, configuration.species)
                self.assertEqual(env.tasks_num, configuration.tasks)
                self.assertEqual(env.traits_dim, configuration.skills)
                self.assertTrue(np.all(requirements.sum(axis=1) > 0))
                self.assertTrue(
                    np.all(
                        np.asarray(env.species_dict["number"]) @ abilities
                        >= requirements
                    )
                )

                if scenario.startswith("sa_"):
                    np.testing.assert_array_equal(
                        abilities, np.eye(configuration.skills)
                    )
                else:
                    self.assertTrue(np.all(np.isin(abilities, (0, 1))))
                    self.assertEqual(
                        np.unique(abilities, axis=0).shape[0],
                        configuration.species,
                    )
                    self.assertTrue(np.any(abilities.sum(axis=1) > 1))

                expected_values = (0, 1) if scenario == "sa_bt" else (0, 1, 2)
                self.assertTrue(np.all(np.isin(requirements, expected_values)))
                if scenario.endswith("_at"):
                    self.assertIn(2, requirements)

    def test_generation_is_repeatable_and_writes_manifest(self):
        configuration = PAPER_CONFIGURATIONS[0]
        first = make_paper_environment("ma_at", configuration, seed=17)
        second = make_paper_environment("ma_at", configuration, seed=17)
        for name, expected in environment_arrays(first).items():
            np.testing.assert_array_equal(
                environment_arrays(second)[name], expected, err_msg=name
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = generate_suites(
                output_dir=Path(temporary_directory),
                scenarios=("sa_at",),
                configurations=(configuration,),
                instances=2,
                seed_base=10,
                released_test_set=None,
                force=False,
            )
            suite = Path(manifest["suites"][0]["path"])
            self.assertTrue((suite / "env_0.pkl").is_file())
            self.assertTrue((suite / "env_1.pkl").is_file())
            self.assertTrue((suite / "manifest.json").is_file())
            self.assertTrue((Path(temporary_directory) / "manifest.json").is_file())
            self.assertEqual(manifest["suites"][0]["seeds"], [10, 11])


if __name__ == "__main__":
    unittest.main()
