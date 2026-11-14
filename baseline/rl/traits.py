"""Adapt smaller capability vectors to the released five-trait RL policy."""

import numpy as np


def pad_policy_traits(env, policy_traits=5):
    """Zero-extend capabilities and requirements in a loaded environment.

    Extra traits have no capability or demand, so the allocation problem is
    unchanged while TaskEnv produces the checkpoint's input feature widths.
    This operates only on the in-memory environment, before its first reset.
    """

    if env.traits_dim > policy_traits:
        raise ValueError(
            f"The policy supports {policy_traits} traits, got {env.traits_dim}"
        )
    padding = policy_traits - env.traits_dim
    if not padding:
        return env
    env.species_dict["abilities"] = np.pad(
        np.asarray(env.species_dict["abilities"]), ((0, 0), (0, padding))
    )
    for agent in env.agent_dic.values():
        agent["abilities"] = np.pad(np.asarray(agent["abilities"]), (0, padding))
    for task in env.task_dic.values():
        for key in ("requirements", "status"):
            task[key] = np.pad(np.asarray(task[key]), (0, padding))
        if task.get("optimized_ability") is not None:
            task["optimized_ability"] = np.pad(
                np.asarray(task["optimized_ability"]),
                [(0, 0)] * (np.ndim(task["optimized_ability"]) - 1) + [(0, padding)],
            )
    env.traits_dim = policy_traits
    return env
