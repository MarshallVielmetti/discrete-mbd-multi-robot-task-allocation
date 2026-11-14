"""Adapt discrete_mbd's categorical solvers to the released MRTA instances.

The original adapter optimizes categorical random keys.  For every task and
species, the required number of agents with the lowest keys form the team.
The unconstrained adapter instead samples team eligibility and task priority
directly, discards redundant eligible agents, and penalizes any capability
shortfall in the objective.  The assignment-permutation adapter directly
optimizes a binary robot-task matrix A and a binary task-priority permutation
matrix P, enforcing P in its sampler and penalizing capability shortfall.
All representations induce one shared task order, which makes every agent's
route acyclic.  The objective evaluates the real benchmark geometry, task
durations, synchronized team starts, and return-to-depot time.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import os
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


DISCRETE_MBD_ROOT = Path(os.environ.get(
    "DISCRETE_MBD_ROOT", Path(__file__).resolve().parents[1] / "external/discrete_mbd"
)).expanduser().resolve()

if str(DISCRETE_MBD_ROOT) not in sys.path:
    sys.path.insert(0, str(DISCRETE_MBD_ROOT))

from solver.cem import cem  # noqa: E402
from solver.discrete_mbd import discrete_mbd  # noqa: E402
from solver.lns import lns  # noqa: E402
from solver.sa import sa  # noqa: E402
from solver.smc import smc  # noqa: E402


MBD_UPSTREAM_REVISION = "8e5929824ac803f28029cac0ceb58f3230ee6c2c"
UNCONSTRAINED_METHOD = "unconstrained_mbd"
UNCONSTRAINED_SA_METHOD = "unconstrained_sa"
UNCONSTRAINED_CEM_METHOD = "unconstrained_cem"
UNCONSTRAINED_LNS_METHOD = "unconstrained_lns"
UNCONSTRAINED_SMC_METHOD = "unconstrained_smc"
UNCONSTRAINED_METHODS = frozenset(
    (
        UNCONSTRAINED_METHOD,
        UNCONSTRAINED_SA_METHOD,
        UNCONSTRAINED_CEM_METHOD,
        UNCONSTRAINED_LNS_METHOD,
        UNCONSTRAINED_SMC_METHOD,
    )
)
ASSIGNMENT_PERMUTATION_METHOD = "assignment_permutation_mbd"
SOLVERS = {
    "discrete_mbd": discrete_mbd,
    UNCONSTRAINED_METHOD: discrete_mbd,
    UNCONSTRAINED_SA_METHOD: sa,
    UNCONSTRAINED_CEM_METHOD: cem,
    ASSIGNMENT_PERMUTATION_METHOD: discrete_mbd,
    "sa": sa,
    "cem": cem,
    "lns": lns,
    UNCONSTRAINED_LNS_METHOD: lns,
    "smc": smc,
    UNCONSTRAINED_SMC_METHOD: smc,
}


class BenchmarkProblem(NamedTuple):
    """Static arrays needed to score one released benchmark instance."""

    agent_species: jnp.ndarray
    capabilities: jnp.ndarray
    requirements: jnp.ndarray
    durations: jnp.ndarray
    task_distances: jnp.ndarray
    depot_distances: jnp.ndarray
    velocities: jnp.ndarray


@dataclass(unsafe_hash=True)
class SolverArgs:
    """Common configuration expected by the external categorical solvers."""

    Ndiffuse: int
    Nsample: int
    temp_sample: float
    N: int
    K: int
    sample_ratio: float
    elite_frac: float
    T0: float
    alpha: float
    infeasibility_penalty: float = 10_000.0
    destroy_k_ratio: float = 0.25
    # Only used by ``decode_decision(..., fill="cover")``: drop redundant
    # robots after the greedy fill. Ignored by the species fill.
    prune: bool = True


@dataclass
class SolveResult:
    decision: np.ndarray
    assignment: np.ndarray
    permutation: np.ndarray
    priorities: np.ndarray
    routes: list
    objective: float
    predicted_makespan: float
    predicted_waiting_time: float
    predicted_time_cost: float
    predicted_travel_dist: float
    predicted_efficiency: float
    penalized_objective: float
    feasible: bool
    feasibility_violation: float
    infeasible_tasks: int
    jit_warmup_time: float
    solver_time: float


def _pairwise_distances(points):
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def problem_from_env(env, require_one_hot=True):
    """Convert a ``TaskEnv`` instance to immutable JAX arrays.

    The exact-feasible random-key decoder needs one-hot species capabilities
    and integer requirements. The unconstrained and direct ``(A, P)``
    representations use the actual capability matrix, so their callers disable
    that restriction.
    """

    abilities = np.asarray(env.species_dict["abilities"], dtype=np.float32)
    expected = np.eye(env.species_num, env.traits_dim, dtype=np.float32)
    if require_one_hot and (
        abilities.shape != expected.shape or not np.array_equal(abilities, expected)
    ):
        raise ValueError(
            "The exact-feasible random-key decoder requires one-hot species "
            "capabilities; "
            f"got {abilities.tolist()}"
        )

    requirements = np.asarray(
        env.get_matrix(env.task_dic, "requirements"), dtype=np.float32
    )
    agent_species = np.asarray(
        env.get_matrix(env.agent_dic, "species"), dtype=np.int32
    )
    capabilities = abilities[agent_species]
    if require_one_hot:
        converted_requirements = np.rint(requirements).astype(np.int32)
        if not np.array_equal(requirements, converted_requirements):
            raise ValueError("Task capability requirements must be integer counts")

        available = np.asarray(env.species_dict["number"], dtype=np.int32)
        if np.any(converted_requirements < 0) or np.any(
            converted_requirements > available[None, :]
        ):
            raise ValueError(
                "A task requests more agents of a species than are available"
            )
    else:
        converted_requirements = requirements
        fleet_capability = np.sum(capabilities, axis=0)
        if np.any(converted_requirements < 0) or np.any(
            converted_requirements > fleet_capability[None, :]
        ):
            raise ValueError(
                "A task requests more capability than the fleet can provide"
            )

    task_locations = np.asarray(
        env.get_matrix(env.task_dic, "location"), dtype=np.float32
    )
    agent_depots = np.asarray(
        env.get_matrix(env.agent_dic, "depot"), dtype=np.float32
    )
    depot_distances = np.linalg.norm(
        agent_depots[:, None, :] - task_locations[None, :, :], axis=-1
    )

    return BenchmarkProblem(
        agent_species=jnp.asarray(agent_species, dtype=jnp.int32),
        capabilities=jnp.asarray(capabilities, dtype=jnp.float32),
        requirements=jnp.asarray(converted_requirements),
        durations=jnp.asarray(
            env.get_matrix(env.task_dic, "time"), dtype=jnp.float32
        ),
        task_distances=jnp.asarray(
            _pairwise_distances(task_locations), dtype=jnp.float32
        ),
        depot_distances=jnp.asarray(depot_distances, dtype=jnp.float32),
        velocities=jnp.asarray(
            env.get_matrix(env.agent_dic, "velocity"), dtype=jnp.float32
        ),
    )


FILL_MODES = ("species", "cover")


def _cover_membership(problem, scores, prune):
    """Greedy-cover teams for multi-skill robots (``fill="cover"``).

    For each task, robots are visited in ascending score order and a robot is
    added iff it lowers the task's capability shortfall
    ``sum(max(requirement - coverage, 0))``. A robot skipped once can never
    help later (the unmet skills only shrink), so the fill ends with zero
    shortfall whenever the whole fleet covers the task. With ``prune`` the
    team is then visited in descending score order and a robot is removed iff
    the shortfall does not increase without it, which leaves an
    inclusion-minimal team.
    """

    num_agents = scores.shape[0]
    capabilities = problem.capabilities.astype(jnp.float32)
    requirements = problem.requirements.astype(jnp.float32)

    def shortfall(requirement, coverage):
        return jnp.sum(jnp.maximum(requirement - coverage, 0.0))

    def cover_task(task_scores, requirement):
        fill_order = jnp.argsort(task_scores)

        def try_add(state, agent_id):
            members, coverage = state
            candidate = coverage + capabilities[agent_id]
            add = (
                shortfall(requirement, candidate)
                < shortfall(requirement, coverage) - 1e-6
            )
            members = members.at[agent_id].set(add)
            coverage = jnp.where(add, candidate, coverage)
            return (members, coverage), None

        initial = (
            jnp.zeros(num_agents, dtype=bool),
            jnp.zeros_like(requirement),
        )
        (members, coverage), _ = jax.lax.scan(try_add, initial, fill_order)
        if not prune:
            return members

        def try_remove(state, agent_id):
            members, coverage = state
            is_member = members[agent_id]
            candidate = coverage - capabilities[agent_id]
            remove = is_member & (
                shortfall(requirement, candidate)
                <= shortfall(requirement, coverage) + 1e-6
            )
            members = members.at[agent_id].set(is_member & ~remove)
            coverage = jnp.where(remove, candidate, coverage)
            return (members, coverage), None

        (members, _), _ = jax.lax.scan(
            try_remove, (members, coverage), fill_order[::-1]
        )
        return members

    return jax.vmap(cover_task, in_axes=(1, 0), out_axes=1)(
        scores, requirements
    )


def decode_decision(problem, decision, fill="species", prune=True):
    """Decode categorical random keys into teams and a shared task order.

    Every level is a rank key (lower = picked first; agent id breaks ties).

    ``fill="species"`` (SA-BT / SA-AT, the original decoder, unchanged): for
    every task and species, the ``requirements[t, s]`` lowest-key robots of
    species ``s`` form the team. Needs one-hot species and integer counts.

    ``fill="cover"`` (MA-AT): greedy cover on full capability vectors, see
    :func:`_cover_membership`; ``prune`` removes redundant robots afterwards.
    On one-hot instances it returns the same teams as ``"species"``.

    Both fills are feasible whenever the fleet can cover every task, and both
    use the same priority (mean team level) and shared task order.
    ``fill`` and ``prune`` must be Python values (they select the traced code).
    """

    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}; got {fill!r}")

    num_agents = problem.agent_species.shape[0]
    num_tasks = problem.requirements.shape[0]
    levels = jnp.argmax(decision, axis=-1).reshape(num_agents, num_tasks)

    # Agent-id tie-breaking makes the within-species ranking deterministic.
    tie_break = jnp.arange(num_agents, dtype=jnp.float32)[:, None] * 1e-3
    scores = levels.astype(jnp.float32) + tie_break
    if fill == "species":
        same_species = (
            problem.agent_species[:, None] == problem.agent_species[None, :]
        )
        lower_score = scores[None, :, :] < scores[:, None, :]
        rank_within_species = jnp.sum(
            lower_score & same_species[:, :, None], axis=1
        )
        required_by_agent = problem.requirements[:, problem.agent_species].T
        membership = rank_within_species < required_by_agent
    else:
        membership = _cover_membership(problem, scores, prune)

    team_size = jnp.sum(membership, axis=0)
    priority = jnp.sum(
        jnp.where(membership, levels, 0), axis=0
    ) / jnp.maximum(team_size, 1)
    # The smallest possible difference between two averages is much larger
    # than this tie-break for the released 15-agent instances.
    priority = priority + jnp.arange(num_tasks, dtype=jnp.float32) * 1e-4
    order = jnp.argsort(priority)
    return levels, membership, priority, order


def decode_unconstrained_decision(problem, decision):
    """Decode independently sampled eligibility and per-task priorities.

    Category zero excludes an agent from a task.  Categories 1..K-1 include
    the agent in the eligible pool and specify its requested task priority.
    Redundant eligible agents are discarded in descending priority order when
    removing them does not increase the task's capability shortfall. This
    reverse-delete rule uses each agent's complete capability vector, so it
    supports both one-hot and overlapping multi-skill species. If the eligible
    coalition cannot meet the requirement, the decoder retains every agent
    whose removal would make the existing shortfall worse. Unlike
    :func:`decode_decision`, it can therefore return an infeasible task for the
    objective to penalize.
    """

    num_agents = problem.agent_species.shape[0]
    num_tasks = problem.requirements.shape[0]
    levels = jnp.argmax(decision, axis=-1).reshape(num_agents, num_tasks)
    eligible = levels > 0

    # Lower sampled levels win when the eligible coalition contains redundant
    # robots. Agent-id tie-breaking makes equal-level pruning deterministic.
    tie_break = jnp.arange(num_agents, dtype=jnp.float32)[:, None] * 1e-3
    scores = jnp.where(eligible, levels.astype(jnp.float32) + tie_break, jnp.inf)

    def trim_task(task_eligible, task_scores, requirement):
        removal_order = jnp.argsort(task_scores)[::-1]
        initial_coverage = (
            task_eligible.astype(jnp.float32) @ problem.capabilities
        )

        def try_remove(state, agent_id):
            members, coverage = state
            is_member = members[agent_id]
            candidate_coverage = coverage - jnp.where(
                is_member,
                problem.capabilities[agent_id],
                jnp.zeros_like(problem.capabilities[agent_id]),
            )
            current_shortfall = jnp.sum(jnp.maximum(requirement - coverage, 0.0))
            candidate_shortfall = jnp.sum(
                jnp.maximum(requirement - candidate_coverage, 0.0)
            )
            remove = is_member & (
                candidate_shortfall <= current_shortfall + 1e-6
            )
            members = members.at[agent_id].set(is_member & ~remove)
            coverage = jnp.where(remove, candidate_coverage, coverage)
            return (members, coverage), None

        (members, _), _ = jax.lax.scan(
            try_remove,
            (task_eligible, initial_coverage),
            removal_order,
        )
        return members

    membership = jax.vmap(trim_task, in_axes=(1, 1, 0), out_axes=1)(
        eligible, scores, problem.requirements
    )

    team_size = jnp.sum(membership, axis=0)
    priority = jnp.sum(jnp.where(membership, levels, 0), axis=0) / jnp.maximum(
        team_size, 1
    )
    priority = priority + jnp.arange(num_tasks, dtype=jnp.float32) * 1e-4
    order = jnp.argsort(priority)
    return levels, membership, priority, order


def make_iid_sampler(num_variables, num_categories):
    """Build an independent categorical sampler using one uniform per row.
    """

    def sampler(rng, probabilities, num_samples):
        weights = probabilities + 1e-12
        cdf = jnp.cumsum(weights, axis=-1)
        cdf = cdf / cdf[:, -1:]
        uniform = jax.random.uniform(rng, (num_samples, num_variables))
        indices = jnp.sum(
            uniform[:, :, None] >= cdf[None, :, :-1], axis=-1
        )
        samples = jax.nn.one_hot(indices, num_categories)
        return samples, jnp.zeros((), dtype=probabilities.dtype)

    return sampler


def make_assignment_permutation_sampler(num_agents, num_tasks):
    """Build a DMBD sampler for binary ``A`` and permutation ``P`` blocks.

    The decision passed to DMBD has one two-category row for every binary
    entry of ``A`` followed by every binary entry of ``P``. Assignment bits
    are sampled independently. The permutation rows are visited in a random
    order and each row selects an unused column, using the same masked
    sequential construction as distributed MBD.

    ``log_correction`` is ``log(phi / psi)`` for the permutation block, where
    ``phi`` is the independent binary proposal and ``psi`` is the constrained
    sequential proposal. DMBD adds this value to its importance logits.
    """

    assignment_variables = num_agents * num_tasks
    permutation_variables = num_tasks * num_tasks
    num_variables = assignment_variables + permutation_variables

    def sampler(rng, probabilities, num_samples):
        probabilities = probabilities / (
            jnp.sum(probabilities, axis=-1, keepdims=True) + 1e-12
        )
        assignment_probabilities = probabilities[:assignment_variables]
        permutation_probabilities = probabilities[assignment_variables:]
        permutation_probabilities = permutation_probabilities.reshape(
            num_tasks, num_tasks, 2
        )

        assignment_rng, permutation_rng = jax.random.split(rng)
        assignment_logits = jnp.log(assignment_probabilities + 1e-12)
        assignment_logits = jnp.broadcast_to(
            assignment_logits[None, :, :],
            (num_samples, assignment_variables, 2),
        )
        assignment_bits = jax.random.categorical(
            assignment_rng, assignment_logits, axis=-1
        )

        # Probability under the independent binary proposal of each possible
        # one-hot row. Choosing column l means P[j,l]=1 and every other bit in
        # row j is zero.
        log_p_off = jnp.log(permutation_probabilities[:, :, 0] + 1e-12)
        log_p_on = jnp.log(permutation_probabilities[:, :, 1] + 1e-12)
        row_pattern_logits = (
            jnp.sum(log_p_off, axis=1, keepdims=True) + log_p_on - log_p_off
        )

        def sample_permutation(sample_rng):
            order_rng, columns_rng = jax.random.split(sample_rng)
            row_order = jax.random.permutation(order_rng, num_tasks)
            column_keys = jax.random.split(columns_rng, num_tasks)

            def select_column(state, step):
                available, permutation, log_correction = state
                row = row_order[step]
                logits = jnp.where(
                    available, row_pattern_logits[row], -jnp.inf
                )
                column = jax.random.categorical(column_keys[step], logits)
                # For this row, log(phi / psi) is the log normalizer over the
                # still-available columns.
                log_correction = log_correction + jsp.special.logsumexp(logits)
                return (
                    available.at[column].set(False),
                    permutation.at[row, column].set(1),
                    log_correction,
                ), None

            initial_state = (
                jnp.ones(num_tasks, dtype=bool),
                jnp.zeros((num_tasks, num_tasks), dtype=jnp.int32),
                jnp.asarray(0.0, dtype=probabilities.dtype),
            )
            (_, permutation, log_correction), _ = jax.lax.scan(
                select_column, initial_state, jnp.arange(num_tasks)
            )
            return permutation, log_correction

        permutations, log_correction = jax.vmap(sample_permutation)(
            jax.random.split(permutation_rng, num_samples)
        )
        bits = jnp.concatenate(
            [assignment_bits, permutations.reshape(num_samples, -1)], axis=1
        )
        samples = jax.nn.one_hot(bits, 2)
        return samples.reshape(num_samples, num_variables, 2), log_correction

    return sampler


def decode_assignment_permutation_decision(problem, decision):
    """Decode binary DMBD rows into ``A``, ``P``, ranks, and task order."""

    num_agents = problem.agent_species.shape[0]
    num_tasks = problem.requirements.shape[0]
    assignment_variables = num_agents * num_tasks
    bits = jnp.argmax(decision, axis=-1)
    assignment = bits[:assignment_variables].reshape(num_agents, num_tasks)
    permutation = bits[assignment_variables:].reshape(num_tasks, num_tasks)
    priority_columns = jnp.argmax(permutation, axis=1)
    priorities = priority_columns + 1
    order = jnp.argsort(priority_columns)
    return assignment, permutation, priorities, order


def _schedule_membership(problem, membership, order):
    """Evaluate a fixed membership matrix in one shared task order."""

    num_agents = problem.agent_species.shape[0]
    num_tasks = problem.requirements.shape[0]
    agent_ids = jnp.arange(num_agents)

    def schedule_task(state, task_id):
        free_time, last_task, starts, finishes, total_waiting = state
        previous = jnp.clip(last_task, 0, num_tasks - 1)
        from_previous = problem.task_distances[previous, task_id]
        from_depot = problem.depot_distances[:, task_id]
        travel = jnp.where(last_task >= 0, from_previous, from_depot)
        arrival = free_time + travel / problem.velocities
        team = membership[:, task_id]
        start = jnp.max(jnp.where(team, arrival, 0.0))
        finish = start + problem.durations[task_id]
        waiting = jnp.sum(jnp.where(team, start - arrival, 0.0))
        free_time = jnp.where(team, finish, free_time)
        last_task = jnp.where(team, task_id, last_task)
        starts = starts.at[task_id].set(start)
        finishes = finishes.at[task_id].set(finish)
        return (
            free_time,
            last_task,
            starts,
            finishes,
            total_waiting + waiting,
        ), task_id

    initial_state = (
        jnp.zeros(num_agents, dtype=jnp.float32),
        -jnp.ones(num_agents, dtype=jnp.int32),
        jnp.zeros(num_tasks, dtype=jnp.float32),
        jnp.zeros(num_tasks, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    final_state, _ = jax.lax.scan(schedule_task, initial_state, order)
    free_time, last_task, starts, finishes, total_waiting = final_state
    final_task = jnp.clip(last_task, 0, num_tasks - 1)
    return_travel = problem.depot_distances[agent_ids, final_task]
    return_time = jnp.where(
        last_task >= 0,
        free_time + return_travel / problem.velocities,
        0.0,
    )
    makespan = jnp.max(return_time)
    return makespan, starts, finishes, total_waiting


def schedule_decision(problem, decision, fill="species", prune=True):
    """Evaluate an exact-feasible random-key decision."""

    _, membership, _, order = decode_decision(
        problem, decision, fill=fill, prune=prune
    )
    return _schedule_membership(problem, membership, order)


def schedule_unconstrained_decision(problem, decision):
    """Evaluate a directly sampled decision before applying feasibility cost."""

    _, membership, _, order = decode_unconstrained_decision(problem, decision)
    return _schedule_membership(problem, membership, order)


def schedule_assignment_permutation_decision(problem, decision):
    """Compute ``T(A, P)`` including travel, synchronization, and return."""

    assignment, _, _, order = decode_assignment_permutation_decision(
        problem, decision
    )
    return _schedule_membership(problem, assignment.astype(bool), order)


def objective(problem, decision, fill="species", prune=True):
    """Return the benchmark's return-to-depot makespan objective."""

    return schedule_decision(problem, decision, fill=fill, prune=prune)[0]


def _membership_feasibility_metrics(problem, membership):
    """Return feasibility metrics for an already decoded membership matrix."""

    coverage = membership.astype(jnp.float32).T @ problem.capabilities
    shortfall = jnp.maximum(problem.requirements - coverage, 0.0)
    capability_infeasible = jnp.any(shortfall > 0.0, axis=1)
    empty = jnp.sum(membership, axis=0) == 0
    uncovered_empty = empty & ~capability_infeasible
    violation = jnp.sum(shortfall) + jnp.sum(uncovered_empty)
    infeasible_tasks = jnp.sum(capability_infeasible | empty)
    return violation, infeasible_tasks


def decision_feasibility_metrics(problem, decision, fill="species", prune=True):
    """Shortfall and infeasible-task count for a random-key decision.

    Always zero when the fleet covers every task; kept as a sanity check.
    """

    _, membership, _, _ = decode_decision(
        problem, decision, fill=fill, prune=prune
    )
    return _membership_feasibility_metrics(problem, membership)


def feasibility_metrics(problem, decision):
    """Return total capability shortfall and the number of infeasible tasks."""

    _, membership, _, _ = decode_unconstrained_decision(problem, decision)
    return _membership_feasibility_metrics(problem, membership)


def assignment_permutation_feasibility_metrics(problem, decision):
    """Return assignment/capability violations for directly represented ``A``."""

    assignment, _, _, _ = decode_assignment_permutation_decision(
        problem, decision
    )
    coverage = assignment.astype(jnp.float32).T @ problem.capabilities
    shortfall = jnp.maximum(problem.requirements - coverage, 0.0)
    capability_infeasible = jnp.any(shortfall > 0.0, axis=1)
    empty = jnp.sum(assignment, axis=0) == 0
    # Positive requirements already penalize an empty task. Add one unit only
    # when an all-zero requirement vector would otherwise leave it unpenalized.
    uncovered_empty = empty & ~capability_infeasible
    violation = jnp.sum(shortfall) + jnp.sum(uncovered_empty)
    infeasible_tasks = jnp.sum(capability_infeasible | empty)
    return violation, infeasible_tasks


def unconstrained_objective(problem, decision, infeasibility_penalty):
    """Return makespan plus a finite penalty for missing capability units."""

    _, membership, _, order = decode_unconstrained_decision(problem, decision)
    makespan = _schedule_membership(problem, membership, order)[0]
    violation, _ = _membership_feasibility_metrics(problem, membership)
    return makespan + infeasibility_penalty * violation


def assignment_permutation_objective(
    problem, decision, infeasibility_penalty
):
    """Return ``T(A, P)`` plus a penalty per missing capability unit."""

    makespan = schedule_assignment_permutation_decision(problem, decision)[0]
    violation, _ = assignment_permutation_feasibility_metrics(problem, decision)
    return makespan + infeasibility_penalty * violation


def _select_unconstrained_result(args, problem, history):
    """Select the best feasible history item, or least-penalized fallback."""

    cost_fn = lambda decision: unconstrained_objective(
        problem, decision, args.infeasibility_penalty
    )
    costs = jax.vmap(cost_fn)(history)
    base_costs = jax.vmap(
        lambda decision: schedule_unconstrained_decision(problem, decision)[0]
    )(history)
    violations, infeasible_task_counts = jax.vmap(
        lambda decision: feasibility_metrics(problem, decision)
    )(history)
    feasible = violations <= 0.0
    best_feasible_index = jnp.argmin(jnp.where(feasible, base_costs, jnp.inf))
    best_penalized_index = jnp.argmin(costs)
    best_index = jnp.where(
        jnp.any(feasible), best_feasible_index, best_penalized_index
    )
    return (
        history[best_index],
        base_costs[best_index],
        costs[best_index],
        violations[best_index],
        infeasible_task_counts[best_index],
    )


@partial(jax.jit, static_argnums=(0,))
def _run_unconstrained(args, problem, seed):
    """Run and score unconstrained MBD with ``problem`` kept dynamic.

    Upstream makes ``cost_fn`` a static JIT argument. Constructing a new closure
    for each benchmark instance would therefore compile the same-shaped solver
    50 times. Tracing the current upstream implementation inside this wrapper
    lets one compiled sweep cell accept every RAL instance as array data.
    """

    cost_fn = lambda decision: unconstrained_objective(
        problem, decision, args.infeasibility_penalty
    )
    history = discrete_mbd.__wrapped__(args, cost_fn, seed=seed)
    return _select_unconstrained_result(args, problem, history)


@partial(jax.jit, static_argnums=(0,))
def _run_unconstrained_sa(args, problem, seed):
    """Run SA on the unconstrained encoding with dynamic problem arrays."""

    cost_fn = lambda decision: unconstrained_objective(
        problem, decision, args.infeasibility_penalty
    )
    history = sa.__wrapped__(args, cost_fn, seed=seed)
    return _select_unconstrained_result(args, problem, history)


@partial(jax.jit, static_argnums=(0,))
def _run_unconstrained_cem(args, problem, seed):
    """Run CEM on the unconstrained encoding with dynamic problem arrays."""

    cost_fn = lambda decision: unconstrained_objective(
        problem, decision, args.infeasibility_penalty
    )
    history = cem.__wrapped__(args, cost_fn, seed=seed)
    return _select_unconstrained_result(args, problem, history)


@partial(jax.jit, static_argnums=(0,))
def _run_unconstrained_lns(args, problem, seed):
    """Run upstream LNS on eligibility/priorities with dynamic problem arrays."""

    cost_fn = lambda decision: unconstrained_objective(
        problem, decision, args.infeasibility_penalty
    )
    history = lns.__wrapped__(args, cost_fn, seed=seed)
    return _select_unconstrained_result(args, problem, history)


@partial(jax.jit, static_argnums=(0,))
def _run_unconstrained_smc(args, problem, seed):
    """Run upstream SMC on eligibility/priorities with dynamic problem arrays."""

    cost_fn = lambda decision: unconstrained_objective(
        problem, decision, args.infeasibility_penalty
    )
    history = smc.__wrapped__(args, cost_fn, seed=seed)
    return _select_unconstrained_result(args, problem, history)


@partial(jax.jit, static_argnums=(0,))
def _run_assignment_permutation(args, problem, seed):
    """Run binary ``(A, P)`` DMBD with a dynamically supplied problem."""

    # Upstream no longer accepts constrained proposals. The local history scan
    # retains that extension while sharing upstream's transition schedule.
    from baseline.discrete_mbd_history import discrete_mbd_history

    num_agents = problem.agent_species.shape[0]
    num_tasks = problem.requirements.shape[0]
    sampler = make_assignment_permutation_sampler(num_agents, num_tasks)
    cost_fn = lambda decision: assignment_permutation_objective(
        problem, decision, args.infeasibility_penalty
    )
    history = discrete_mbd_history.__wrapped__(
        args, cost_fn, sampler=sampler, seed=seed
    ).decisions
    costs = jax.vmap(cost_fn)(history)
    base_costs = jax.vmap(
        lambda decision: schedule_assignment_permutation_decision(
            problem, decision
        )[0]
    )(history)
    violations, infeasible_task_counts = jax.vmap(
        lambda decision: assignment_permutation_feasibility_metrics(
            problem, decision
        )
    )(history)
    feasible = violations <= 0.0
    best_feasible_index = jnp.argmin(jnp.where(feasible, base_costs, jnp.inf))
    best_penalized_index = jnp.argmin(costs)
    best_index = jnp.where(
        jnp.any(feasible), best_feasible_index, best_penalized_index
    )
    return (
        history[best_index],
        base_costs[best_index],
        costs[best_index],
        violations[best_index],
        infeasible_task_counts[best_index],
    )


def _routes_from_membership(membership, order):
    """Convert membership and order into TACO-compatible one-based routes."""

    membership = np.asarray(membership)
    order = np.asarray(order)
    routes = []
    for agent_id in range(membership.shape[0]):
        tasks = [int(task_id) + 1 for task_id in order if membership[agent_id, task_id]]
        routes.append([0, *tasks, 0])
    return routes


def _solution_metrics(problem, membership, order):
    """Compute full-coalition metrics using the same semantics as ``T``."""

    makespan, starts, _, total_waiting = _schedule_membership(
        problem, membership, order
    )
    membership = np.asarray(membership, dtype=bool)
    order = np.asarray(order)
    task_distances = np.asarray(problem.task_distances)
    depot_distances = np.asarray(problem.depot_distances)
    travel_distance = 0.0
    for agent_id in range(membership.shape[0]):
        tasks = [
            int(task_id)
            for task_id in order
            if membership[agent_id, task_id]
        ]
        if not tasks:
            continue
        travel_distance += float(depot_distances[agent_id, tasks[0]])
        travel_distance += sum(
            float(task_distances[first, second])
            for first, second in zip(tasks, tasks[1:])
        )
        travel_distance += float(depot_distances[agent_id, tasks[-1]])

    coverage = membership.astype(np.float32).T @ np.asarray(problem.capabilities)
    requirements = np.asarray(problem.requirements)
    requirement_total = requirements.sum(axis=1)
    task_efficiency = np.divide(
        coverage.sum(axis=1),
        requirement_total,
        out=np.ones_like(requirement_total, dtype=np.float32),
        where=requirement_total > 0,
    )
    return {
        "makespan": float(makespan),
        "waiting_time": float(total_waiting) / membership.shape[0],
        "time_cost": float(jnp.mean(starts)),
        "travel_dist": travel_distance,
        "efficiency": float(np.mean(task_efficiency)),
    }


def routes_from_decision(problem, decision, fill="species", prune=True):
    """Convert an exact-feasible decision into one-based agent routes."""

    _, membership, _, order = decode_decision(
        problem, jnp.asarray(decision), fill=fill, prune=prune
    )
    return _routes_from_membership(membership, order)


def routes_from_unconstrained_decision(problem, decision):
    """Convert a directly sampled decision into one-based agent routes."""

    _, membership, _, order = decode_unconstrained_decision(
        problem, jnp.asarray(decision)
    )
    return _routes_from_membership(membership, order)


def routes_from_assignment_permutation_decision(problem, decision):
    """Convert binary ``(A, P)`` into one-based routes for ``TaskEnv``."""

    assignment, _, _, order = decode_assignment_permutation_decision(
        problem, jnp.asarray(decision)
    )
    return _routes_from_membership(assignment, order)


def solve(env, method, args, seed=0, warmup=True):
    """Run one external solver and return its best decoded route solution."""

    if method not in SOLVERS:
        raise ValueError(f"Unknown solver {method!r}; choose from {sorted(SOLVERS)}")
    if args.Nsample < 2:
        raise ValueError("Nsample must be at least 2")
    if args.Ndiffuse < 1:
        raise ValueError("Ndiffuse must be at least 1")
    if method in ("lns", UNCONSTRAINED_LNS_METHOD, "smc", UNCONSTRAINED_SMC_METHOD):
        if not 0.0 < args.destroy_k_ratio <= 1.0:
            raise ValueError("destroy_k_ratio must be in (0, 1]")
    if method in ("smc", UNCONSTRAINED_SMC_METHOD):
        if args.Ndiffuse < 2:
            raise ValueError("SMC requires Ndiffuse >= 2 to produce a search history")
        if not np.isfinite(args.T0) or args.T0 <= 0.0:
            raise ValueError("SMC requires a finite positive T0")
        if not 0.0 < args.alpha <= 1.0:
            raise ValueError("SMC requires alpha in (0, 1]")

    unconstrained = method in UNCONSTRAINED_METHODS
    assignment_permutation = method == ASSIGNMENT_PERMUTATION_METHOD
    problem = problem_from_env(
        env,
        require_one_hot=not (assignment_permutation or unconstrained),
    )
    if assignment_permutation:
        expected_variables = (
            env.agents_num * env.tasks_num + env.tasks_num * env.tasks_num
        )
        if args.N != expected_variables or args.K != 2:
            raise ValueError(
                f"{ASSIGNMENT_PERMUTATION_METHOD} requires "
                f"N={expected_variables} binary variables and K=2; "
                f"got N={args.N}, K={args.K}"
            )
    else:
        expected_variables = env.agents_num * env.tasks_num
        if args.N != expected_variables or args.K < 2:
            raise ValueError(
                f"SolverArgs must use N={expected_variables} and K>=2; "
                f"got N={args.N}, K={args.K}"
            )

    if assignment_permutation:
        if args.infeasibility_penalty <= 0:
            raise ValueError("infeasibility_penalty must be positive")
        base_cost_fn = lambda decision: schedule_assignment_permutation_decision(
            problem, decision
        )[0]
        cost_fn = lambda decision: assignment_permutation_objective(
            problem, decision, args.infeasibility_penalty
        )
        metrics_fn = assignment_permutation_feasibility_metrics
        route_fn = routes_from_assignment_permutation_decision
    elif unconstrained:
        if args.infeasibility_penalty <= 0:
            raise ValueError("infeasibility_penalty must be positive")
        base_cost_fn = lambda decision: schedule_unconstrained_decision(
            problem, decision
        )[0]
        cost_fn = lambda decision: unconstrained_objective(
            problem, decision, args.infeasibility_penalty
        )
        metrics_fn = feasibility_metrics
        route_fn = routes_from_unconstrained_decision
    else:
        base_cost_fn = lambda decision: objective(problem, decision)
        cost_fn = base_cost_fn
        metrics_fn = lambda _problem, _decision: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        route_fn = routes_from_decision

    solver = SOLVERS[method]

    if assignment_permutation:
        dynamic_seed = jnp.asarray(seed, dtype=jnp.int32)

        def run_once():
            result = _run_assignment_permutation(args, problem, dynamic_seed)
            result[2].block_until_ready()
            return result

    elif unconstrained:
        dynamic_seed = jnp.asarray(seed, dtype=jnp.int32)
        unconstrained_runner = {
            UNCONSTRAINED_METHOD: _run_unconstrained,
            UNCONSTRAINED_SA_METHOD: _run_unconstrained_sa,
            UNCONSTRAINED_CEM_METHOD: _run_unconstrained_cem,
            UNCONSTRAINED_LNS_METHOD: _run_unconstrained_lns,
            UNCONSTRAINED_SMC_METHOD: _run_unconstrained_smc,
        }[method]

        def run_once():
            result = unconstrained_runner(args, problem, dynamic_seed)
            result[2].block_until_ready()
            return result

    else:
        score_history = jax.jit(jax.vmap(cost_fn))
        base_score_history = jax.jit(jax.vmap(base_cost_fn))
        feasibility_history = jax.jit(
            jax.vmap(lambda decision: metrics_fn(problem, decision))
        )

        def run_once():
            history = solver(args, cost_fn, seed=seed)
            costs = score_history(history)
            base_costs = base_score_history(history)
            violations, infeasible_task_counts = feasibility_history(history)
            feasible = violations <= 0.0
            feasible_costs = jnp.where(feasible, base_costs, jnp.inf)
            best_feasible_index = jnp.argmin(feasible_costs)
            best_penalized_index = jnp.argmin(costs)
            best_index = jnp.where(
                jnp.any(feasible), best_feasible_index, best_penalized_index
            )
            best_decision = history[best_index]
            result = (
                best_decision,
                base_costs[best_index],
                costs[best_index],
                violations[best_index],
                infeasible_task_counts[best_index],
            )
            result[2].block_until_ready()
            return result

    warmup_time = 0.0
    if warmup:
        start = time.perf_counter()
        run_once()
        warmup_time = time.perf_counter() - start

    start = time.perf_counter()
    decision, best_cost, penalized_cost, violation, infeasible_tasks = run_once()
    solver_time = time.perf_counter() - start
    decision = np.asarray(decision)
    predicted_makespan = float(best_cost)
    violation = float(violation)
    if assignment_permutation:
        membership, permutation, priorities, order = (
            decode_assignment_permutation_decision(problem, decision)
        )
    elif unconstrained:
        _, membership, _, order = decode_unconstrained_decision(problem, decision)
        priority_columns = jnp.zeros(env.tasks_num, dtype=jnp.int32).at[order].set(
            jnp.arange(env.tasks_num)
        )
        priorities = priority_columns + 1
        permutation = jax.nn.one_hot(priority_columns, env.tasks_num)
    else:
        _, membership, _, order = decode_decision(problem, decision)
        priority_columns = jnp.zeros(env.tasks_num, dtype=jnp.int32).at[order].set(
            jnp.arange(env.tasks_num)
        )
        priorities = priority_columns + 1
        permutation = jax.nn.one_hot(priority_columns, env.tasks_num)
    solution_metrics = _solution_metrics(problem, membership, order)
    return SolveResult(
        decision=decision,
        assignment=np.asarray(membership, dtype=np.int32),
        permutation=np.asarray(permutation, dtype=np.int32),
        priorities=np.asarray(priorities, dtype=np.int32),
        routes=route_fn(problem, decision),
        objective=float(penalized_cost),
        predicted_makespan=predicted_makespan,
        predicted_waiting_time=solution_metrics["waiting_time"],
        predicted_time_cost=solution_metrics["time_cost"],
        predicted_travel_dist=solution_metrics["travel_dist"],
        predicted_efficiency=solution_metrics["efficiency"],
        penalized_objective=float(penalized_cost),
        feasible=violation <= 0.0,
        feasibility_violation=violation,
        infeasible_tasks=int(infeasible_tasks),
        jit_warmup_time=warmup_time,
        solver_time=solver_time,
    )
