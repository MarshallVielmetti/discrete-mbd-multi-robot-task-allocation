import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from env.task_env import TaskEnv
from baseline.rl.traits import pad_policy_traits


class PaperRLTraitsTest(unittest.TestCase):
    def test_three_skill_suite_uses_checkpoint_feature_dimensions(self):
        env = TaskEnv(
            per_species_range=(1, 1), species_range=(3, 3),
            tasks_range=(2, 2), traits_dim=3, max_task_size=2,
            agent_skill_mode="single_skill", seed=0,
        )
        original_requirements = np.asarray(env.get_matrix(env.task_dic, "requirements"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "env_0.pkl"
            with path.open("wb") as stream:
                pickle.dump(env, stream)
            with path.open("rb") as stream:
                loaded = pickle.load(stream)
            pad_policy_traits(loaded)
            loaded.init_state()
            self.assertEqual(loaded.traits_dim, 5)
            np.testing.assert_array_equal(
                np.asarray(loaded.get_matrix(loaded.task_dic, "requirements"))[:, :3],
                original_requirements,
            )
            self.assertFalse(np.any(
                np.asarray(loaded.get_matrix(loaded.task_dic, "requirements"))[:, 3:]
            ))
            self.assertEqual(loaded.get_current_agent_status(loaded.agent_dic[0]).shape[1], 11)
            self.assertEqual(loaded.get_current_task_status(loaded.agent_dic[0]).shape[1], 15)
            np.testing.assert_array_equal(
                np.asarray(loaded.get_matrix(loaded.agent_dic, "abilities"))[:, :3],
                np.asarray(env.get_matrix(env.agent_dic, "abilities")),
            )
            np.testing.assert_array_equal(
                np.asarray(loaded.get_matrix(loaded.agent_dic, "abilities"))
                @ np.asarray(loaded.get_matrix(loaded.task_dic, "requirements")).T,
                np.asarray(env.get_matrix(env.agent_dic, "abilities"))
                @ original_requirements.T,
            )
            with path.open("rb") as stream:
                self.assertEqual(pickle.load(stream).traits_dim, 3)

    def test_padding_is_idempotent(self):
        env = TaskEnv(
            per_species_range=(1, 1), species_range=(3, 3),
            tasks_range=(2, 2), traits_dim=3, max_task_size=2,
            agent_skill_mode="single_skill", seed=0,
        )
        pad_policy_traits(env)
        abilities = env.species_dict["abilities"].copy()
        pad_policy_traits(env)
        np.testing.assert_array_equal(env.species_dict["abilities"], abilities)


if __name__ == "__main__":
    unittest.main()
