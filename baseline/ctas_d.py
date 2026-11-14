"""CTAS-D input export, Gurobi execution and deadlock-aware route evaluation.

The paper runner owns suite selection, seeds and the results schema. This module
owns the CTAS-D backend and has no standalone experiment CLI.
"""

import contextlib
import csv
import io
import math
import os
import pickle
import subprocess
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from env.task_env import TaskEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUROBI_HOME = REPO_ROOT / "external/gurobi1003/linux64"


@dataclass(frozen=True)
class CTASConfig:
    solver: Path = REPO_ROOT / "external/resilient_team_planner/build/main"
    gurobi_home: Path = DEFAULT_GUROBI_HOME
    time_limit: float = 30.0
    threads: Optional[int] = None


def environment_with_libraries(gurobi_home):
    env = os.environ.copy()
    library_paths = [
        str(gurobi_home / "lib"),
        str(REPO_ROOT / "external" / "install" / "yaml-cpp" / "lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return env

def check_gurobi_license(gurobi_home, output_dir, env):
    executable = gurobi_home / "bin" / "gurobi_cl"
    model = gurobi_home / "examples" / "data" / "coins.lp"
    if not executable.is_file() or not model.is_file():
        raise FileNotFoundError(
            f"Cannot run the Gurobi license check under {gurobi_home}"
        )
    check_dir = output_dir / "license-check"
    check_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        f"LogFile={check_dir / 'gurobi.log'}",
        f"ResultFile={check_dir / 'coins.sol'}",
        str(model),
    ]
    process = subprocess.run(
        command,
        cwd=check_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (check_dir / "console.log").write_text(process.stdout)
    if process.returncode != 0:
        last_lines = "\n".join(process.stdout.rstrip().splitlines()[-8:])
        raise RuntimeError(
            "Gurobi is installed but its license check failed. "
            f"Details: {check_dir / 'console.log'}\n{last_lines}"
        )

def prepare_parameters(source, destination, time_limit):
    with source.open() as stream:
        parameters = yaml.safe_load(stream)
    source_dir = source.parent
    for key, filename in (
        ("vehicleParamFile", "vehicle_param.yaml"),
        ("taskParamFile", "task_param.yaml"),
        ("graphFile", "graph.yaml"),
    ):
        input_path = (source_dir / filename).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        parameters[key] = str(input_path)
    if parameters.get("flagSolver") != "TEAMPLANNER_CONDET":
        raise ValueError(f"{source} is not configured for CTAS-D")
    if time_limit is not None:
        parameters["solverMaxTime"] = float(time_limit)
    with destination.open("w") as stream:
        yaml.safe_dump(parameters, stream, sort_keys=False)


def pairwise_distances(points):
    return {
        source: {
            target: 0.0
            if source == target
            else math.hypot(
                points[source][0] - points[target][0],
                points[source][1] - points[target][1],
            )
            for target in range(len(points))
        }
        for source in range(len(points))
    }

def write_ctas_inputs(env, destination):
    destination.mkdir(parents=True, exist_ok=True)
    task_locations = np.asarray(env.get_matrix(env.task_dic, "location"))
    depot_locations = np.asarray(env.get_matrix(env.depot_dic, "location"))
    task_distances = pairwise_distances(task_locations)
    task_pairs = list(permutations(range(env.tasks_num), 2))

    vehicles = {}
    graph = {}
    for species in range(env.species_num):
        vehicles[f"vehicle{species}"] = {
            "engCap": 1e6,
            "engCost": 0.0,
            "capVector": env.species_dict["abilities"][species].tolist(),
            "capVar": [0.0] * env.traits_dim,
        }
        species_graph = {
            f"edge{index}": [
                source,
                target,
                0,
                task_distances[source][target],
                0,
                task_distances[source][target] / 0.2,
            ]
            for index, (source, target) in enumerate(task_pairs)
        }
        edge_offset = len(task_pairs)
        for task_id, task_location in enumerate(task_locations):
            distance = float(np.linalg.norm(depot_locations[species] - task_location))
            species_graph[f"edge{edge_offset + 2 * task_id}"] = [
                env.tasks_num + species,
                task_id,
                0,
                distance,
                0,
                distance / 0.2,
            ]
            species_graph[f"edge{edge_offset + 2 * task_id + 1}"] = [
                task_id,
                env.tasks_num + env.species_num + species,
                0,
                distance,
                0,
                distance / 0.2,
            ]
        for task_id, task in env.task_dic.items():
            species_graph[f"node{task_id}"] = float(task["time"])
        graph[f"vehicle{species}"] = species_graph

    tasks = {}
    for task in env.task_dic.values():
        constraints = {}
        constraint_index = 0
        for capability, requirement in enumerate(task["requirements"]):
            if requirement:
                constraints[f"and{constraint_index}"] = {
                    "or0": {
                        "geq": True,
                        "capId": int(capability),
                        "capReq": float(requirement),
                        "capVar": 0.0,
                    }
                }
                constraint_index += 1
        tasks[f"task{task['ID']}"] = constraints

    planner = {
        "flagOptimizeCost": True,
        "flagTaskComplete": True,
        "flagSprAddCutToSameType": True,
        "taskCompleteReward": 10000,
        "timePenalty": 100,
        "recoursePenalty": 1.0,
        "taskRiskPenalty": 0.0,
        "LARGETIME": 10000.0,
        "MAXTIME": 1000.0,
        "MAXENG": 1e8,
        "flagSolver": "TEAMPLANNER_CONDET",
        "CcpBeta": 0.95,
        "taskBeta": 0.95,
        "solverMaxTime": 600.0,
        "solverIterMaxTime": 50.0,
        "flagNotUseUnralavant": True,
        "MAXALPHA": 20.0,
        "taskNum": int(env.tasks_num),
        "vehNum": int(env.species_num),
        "capNum": int(env.traits_dim),
        "vehTypeNum": int(env.species_num),
        "vehNumPerType": [int(value) for value in env.species_dict["number"]],
        "sampleNum": 500,
        "randomType": 0,
        "capType": [0] * env.traits_dim,
        "vehicleParamFile": "vehicle_param.yaml",
        "taskParamFile": "task_param.yaml",
        "graphFile": "graph.yaml",
    }

    for name, payload in (
        ("vehicle_param.yaml", vehicles),
        ("task_param.yaml", tasks),
        ("graph.yaml", graph),
        ("planner_param.yaml", planner),
    ):
        with (destination / name).open("w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)


def write_gurobi_env(run_dir, seed, threads):
    lines = [f"Seed {seed}"]
    if threads is not None:
        lines.append(f"Threads {threads}")
    (run_dir / "gurobi.env").write_text("\n".join(lines) + "\n")

def seed_was_applied(log_text, seed):
    """Gurobi echoes parameters read from gurobi.env, e.g. 'Set parameter Seed to value 3'."""
    return f"Seed to value {seed}" in log_text


class _InstanceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("__main__", "yamlGenerator") and name == "TaskEnv":
            return TaskEnv
        return super().find_class(module, name)


def load_instance(path):
    """Read both public legacy pickles and generated TaskEnv instances."""
    with Path(path).open("rb") as stream:
        return _InstanceUnpickler(stream).load()


def load_routes(env, folder):
    """Assign CTAS-D's species routes to individual robots, as in the release."""
    result_path = folder / "results.yaml"
    if not result_path.is_file():
        return False
    with result_path.open() as stream:
        result = yaml.safe_load(stream) or {}
    if "vehicle" not in result:
        return False
    with (folder / "planner_param.yaml").open() as stream:
        parameters = yaml.safe_load(stream)
    if parameters["flagSolver"] != "TEAMPLANNER_CONDET":
        raise ValueError("CTAS-D replay requires TEAMPLANNER_CONDET species routes")
    groups = {species: [] for species in range(len(parameters["vehNumPerType"]))}
    for vehicle in result["vehicle"].values():
        groups[vehicle["type"] - 1].append(vehicle["node"])
    for species, routes in groups.items():
        for route in routes:
            agent = env.species_dict[species].pop(0)
            env.pre_set_route(route[1:], agent)
    return True


def evaluate_instance(path, result_dir):
    """Measure a plan by replay; solver success alone does not establish feasibility."""
    env = load_instance(path)
    env.init_state()
    result_path = result_dir / "results.yaml"
    planner_time = float("nan")
    if result_path.is_file():
        with result_path.open() as stream:
            result = yaml.safe_load(stream) or {}
        planner_time = float(result.get("result", {}).get("plannerTime", float("nan")))
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = load_routes(env, result_dir)
        if loaded:
            replay_routes(env)
    solved = False
    if loaded:
        _, finished_tasks = env.get_episode_reward(100)
        solved = np.mean(finished_tasks) == 1.0
    return {
        "solved": float(solved),
        "makespan": float(env.current_time) if solved else float("nan"),
        "stored_solver_time": planner_time,
    }


def replay_routes(env):
    """Use TaskEnv's route semantics, stopping if unfinished tasks cannot progress."""
    env.plot_figure = False
    env.max_waiting_time = 200
    while not env.finished and env.current_time < 200:
        released, env.current_time = env.next_decision()
        agents = released[0] + released[1]
        if not agents:
            break
        for agent in agents:
            route = env.agent_dic[agent]["pre_set_route"]
            if not route:
                env.agent_step(agent, 0, 0)
                env.agent_dic[agent]["next_decision"] = np.nan
            else:
                env.agent_step(agent, route.pop(0), 0)
        env.finished = env.check_finished()


def preflight(config, output_dir):
    solver = config.solver.resolve()
    if not solver.is_file():
        raise FileNotFoundError(f"CTAS-D executable not found: {solver}; run scripts/setup_external_solvers.sh")
    environment = environment_with_libraries(config.gurobi_home.resolve())
    check_gurobi_license(config.gurobi_home.resolve(), output_dir.resolve(), environment)
    return environment


def run_suite(suite, output_dir, seeds, config, fields, *, limit=None, dry_run=False, environment=None):
    """Run missing seed/instance pairs and append to the paper runner's shared CSV."""
    files = sorted(suite.glob("env_*.pkl"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No env_*.pkl in {suite}")
    runs_csv = output_dir / "runs.csv"
    done = set()
    if runs_csv.is_file() and runs_csv.stat().st_size:
        with runs_csv.open() as stream:
            for row in csv.DictReader(stream):
                if row["method"] == "CTAS-D" and row["decoder"] == "milp":
                    done.add((int(row["seed"]), int(row["instance"])))
    todo = [(seed, path) for seed in seeds for path in files
            if (seed, int(path.stem.rsplit("_", 1)[1])) not in done]
    scenario, configuration = suite.parent.name, suite.name
    print(f"{scenario}/{configuration} CTAS-D: {len(todo)} runs to do "
          f"({len(seeds)} seeds x {len(files)} instances, {len(done)} done)", flush=True)
    if not todo or dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "ctas_inputs"
    for path in files:
        destination = inputs_dir / path.stem
        if not (destination / "planner_param.yaml").is_file():
            write_ctas_inputs(load_instance(path), destination)
    new_file = not runs_csv.is_file() or runs_csv.stat().st_size == 0
    with runs_csv.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new_file:
            writer.writeheader()
        seed_warned = False
        for position, (seed, path) in enumerate(todo, 1):
            instance = int(path.stem.rsplit("_", 1)[1])
            solver_seed = 1000 * seed + instance
            run_dir = output_dir / f"seed_{seed}" / path.stem
            run_dir.mkdir(parents=True, exist_ok=True)
            prepare_parameters(inputs_dir / path.stem / "planner_param.yaml",
                               run_dir / "planner_param.yaml", config.time_limit)
            write_gurobi_env(run_dir, solver_seed, config.threads)
            for name in ("results.yaml", "sampleLog.yaml"):
                (run_dir / name).unlink(missing_ok=True)
            process = subprocess.run(
                [str(config.solver.resolve()), str(run_dir / "planner_param.yaml"),
                 str(run_dir / "results.yaml"), str(run_dir / "sampleLog.yaml")],
                cwd=run_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
            (run_dir / "run.log").write_text(process.stdout)
            if process.returncode:
                raise RuntimeError(f"CTAS-D failed on {run_dir} (exit {process.returncode}); see run.log")
            if solver_seed != 0 and not seed_warned:
                logs = process.stdout + "".join(
                    log.read_text(errors="ignore") for log in run_dir.glob("*.log") if log.name != "run.log")
                if not seed_was_applied(logs, solver_seed):
                    seed_warned = True
                    print(f"  warning: no 'Seed to value {solver_seed}' in {run_dir} logs "
                          "(shown once per suite); verify the Gurobi seed was applied", flush=True)
            result = evaluate_instance(path, run_dir)
            solved = result["solved"] == 1.0
            makespan = result["makespan"] if solved else float("nan")
            writer.writerow(dict(
                scenario=scenario, configuration=configuration, decoder="milp", method="CTAS-D",
                seed=seed, instance=instance, solver_seed=solver_seed, levels="",
                makespan=makespan, penalized_cost=makespan, feasible=int(solved),
                violation=0.0 if solved else float("nan"), infeasible_tasks=0 if solved else "",
                replay_makespan=makespan, solver_time=result["stored_solver_time"],
            ))
            stream.flush()
            if position % 10 == 0 or position == len(todo):
                print(f"  CTAS-D {position}/{len(todo)}", flush=True)
