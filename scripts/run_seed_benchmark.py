#!/usr/bin/env python3
"""Reproduce the full MRTA paper suite, including CTAS-D and RL(s.10).

With no arguments, prepare 15 suites of 50 instances, run all eight methods over
five seeds, and write runs.csv, summary.csv and the combined paper table.tex.
Search uses 350 iterations and 2048 samples; CTAS-D has a 30-second budget and
RL returns the best of ten sampled rollouts. Requires the external search solver
checkout, the released policy checkpoint, and a built, licensed CTAS-D solver.

The six search methods use the same categorical encoding, decoder and cost, and get
the same budget (``--steps`` iterations x ``--samples`` cost evaluations per iteration):

* SA-BT / SA-AT: exact-feasible random-key decoder (``decode_decision``), K = #tasks levels
  (0 .. K-1), cost = return-to-depot makespan. Every sample is feasible.
* MA-AT (default, ``--ma-decoder cover``): the same random-key decoder with the greedy-cover
  fill (``decode_decision(..., fill="cover")``), K = #tasks levels, cost = makespan. Every
  sample is feasible. Redundant robots are pruned unless ``--no-prune`` is given; the decoder
  column then reads ``cover_noprune`` instead of ``cover``.
* MA-AT (``--ma-decoder eligibility``): eligibility decoder (``decode_unconstrained_decision``),
  K = #tasks + 1 levels (level 0 = excluded), cost = makespan + penalty x missing capability
  units.

For each suite, method, seed s and instance j the solver seed is ``1000 * s + j``, identical
across methods. Each run returns the best solution in its search history (best feasible
makespan, or least-penalized if none is feasible). Results are appended to
``<output-dir>/<scenario>/<configuration>/runs.csv`` (resumable per decoder, method, seed and
instance), and ``summary.csv`` reports, per decoder and method, the mean over instances for each seed and then mean +/- std (ddof=1) across seeds.
Infeasible runs (MA-AT eligibility only) are excluded from that seed's mean and counted in ``infeasible``.

Examples (from the repository root):
    .venv/bin/python scripts/run_seed_benchmark.py --scenarios sa_bt --configurations kn9_ks3_km20_kb3
    .venv/bin/python scripts/run_seed_benchmark.py                      # all 15 suites
    .venv/bin/python scripts/run_seed_benchmark.py --summary-only       # rebuild the table
"""

import argparse
import contextlib
import csv
import hashlib
import io
import importlib
import json
import math
import os
import pickle
import sys
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

SCENARIOS = ("sa_bt", "sa_at", "ma_at")
CONFIGURATIONS = (
    "kn9_ks3_km20_kb3",
    "kn15_ks5_km20_kb5",
    "kn25_ks5_km20_kb5",
    "kn25_ks5_km50_kb5",
    "kn50_ks5_km50_kb5",
)
SEARCH_METHODS = ("MBD", "CEM", "LNS", "SA", "SMC", "ROUNDED_MBD")
METHODS = ("CTAS-D", "LNS", "SA", "RL(s.10)", "CEM", "SMC", "ROUNDED_MBD", "MBD")
METHOD_LABELS = {"CTAS-D": "CTAS-D", "ROUNDED_MBD": "Rounded MBD", "MBD": "Discrete MBD (ours)"}
SCENARIO_LABELS = {"sa_bt": "SS-BT", "sa_at": "SS-AT", "ma_at": "MS-AT"}


def load_search_backend(methods=SEARCH_METHODS):
    """Load/compile search only when requested, so inspection needs no JAX/GPU."""
    global jax, jnp, run_solver, SOLVERS
    try:
        if not SOLVERS:
            os.environ.setdefault("JAX_PLATFORMS", "cuda")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            import jax
            import jax.numpy as jnp
            adapter = importlib.import_module("baseline.discrete_mbd_adapter")
            names = (
                "SolverArgs", "decision_feasibility_metrics", "feasibility_metrics",
                "make_iid_sampler", "objective", "problem_from_env", "routes_from_decision",
                "routes_from_unconstrained_decision", "schedule_unconstrained_decision",
                "unconstrained_objective",
            )
            globals().update({name: getattr(adapter, name) for name in names})
            SOLVERS.update(MBD=adapter.discrete_mbd, CEM=cem_iid, LNS=adapter.lns,
                           SA=adapter.sa, SMC=adapter.smc)
            run_solver = jax.jit(run_solver, static_argnums=(0, 1, 2))
        if "ROUNDED_MBD" in methods and "ROUNDED_MBD" not in SOLVERS:
            SOLVERS["ROUNDED_MBD"] = importlib.import_module("solver.rounded_mbd").rounded_mbd
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Search solvers are unavailable. Run uv sync --frozen and "
            "git submodule update --init --recursive (or set DISCRETE_MBD_ROOT)."
        ) from error


def cem_iid(args, cost_fn, seed=42):
    """Upstream solver/cem.py, step for step, except that each iteration draws its samples with
    the local per-row inverse-CDF categorical sampler (``make_iid_sampler``).

    The sampling distribution is identical. Upstream ``jax.random.categorical`` on logits
    broadcast to (Nsample, N, K) builds several Nsample x N x K float buffers (Gumbel noise,
    logits, one-hot), which exceeds memory on the 50-robot/50-task suites (>6 GB at 2048 samples,
    N = 2500, K = 50); this version needs about a third of that.
    """
    rng = jax.random.PRNGKey(seed=seed)
    probs = jnp.ones((args.N, args.K)) / args.K
    n_elite = max(1, int(args.Nsample * args.elite_frac))
    draw = make_iid_sampler(args.N, args.K)

    def cem_iteration(carry, _):
        rng, probs = carry
        rng, sample_rng = jax.random.split(rng)
        samples, _ = draw(sample_rng, probs, args.Nsample)          # (Nsample, N, K) one-hot
        costs = jax.vmap(cost_fn)(samples)
        elite_idx = jnp.argsort(costs)[:n_elite]
        probs = jnp.mean(samples[elite_idx], axis=0)
        return (rng, probs), samples[jnp.argmin(costs)]

    _, history = jax.lax.scan(cem_iteration, (rng, probs), None, length=args.Ndiffuse)
    return history


cem_iid.__wrapped__ = cem_iid
# ROUNDED_MBD: Gaussian MBD on the {-1,1} relaxation with DIFUSCO rounding before every cost
# evaluation. The decoders handle feasibility; its history
# is the rounded mean round(Y_{i-1}) of every reverse step, as returned by solver/rounded_mbd.py.
SOLVERS = {}
FIELDS = ["scenario", "configuration", "decoder", "method", "seed", "instance", "solver_seed",
          "levels", "makespan", "penalized_cost", "feasible", "violation", "infeasible_tasks",
          "replay_makespan", "solver_time"]


COVER_DECODERS = ("cover", "cover_noprune")


def decoder_for(scenario, ma_decoder="cover", prune=True):
    """Decoder label for a scenario; it is also part of the resume key in runs.csv."""
    if scenario.startswith("sa_"):
        return "exact"
    if ma_decoder == "cover":
        return "cover" if prune else "cover_noprune"
    return "eligibility"


def run_solver(method, decoder, args, problem, seed):
    """One solver run with ``problem`` and ``seed`` as traced data (compiled once per suite)."""
    if decoder == "exact":
        cost_fn = lambda x: objective(problem, x)
    elif decoder in COVER_DECODERS:
        cost_fn = lambda x: objective(problem, x, fill="cover", prune=args.prune)
    else:
        cost_fn = lambda x: unconstrained_objective(problem, x, args.infeasibility_penalty)
    solver = SOLVERS[method].__wrapped__
    history = solver(args, cost_fn, seed=seed)

    costs = jax.vmap(cost_fn)(history)
    if decoder == "exact":
        makespans = costs
        violations = jnp.zeros_like(costs)
        infeasible = jnp.zeros(costs.shape, dtype=jnp.int32)
    elif decoder in COVER_DECODERS:
        # Feasible by construction; the metrics are a sanity check.
        makespans = costs
        violations, infeasible = jax.vmap(lambda x: decision_feasibility_metrics(
            problem, x, fill="cover", prune=args.prune))(history)
    else:
        makespans = jax.vmap(lambda x: schedule_unconstrained_decision(problem, x)[0])(history)
        violations, infeasible = jax.vmap(lambda x: feasibility_metrics(problem, x))(history)
    feasible = violations <= 0.0
    best = jnp.where(jnp.any(feasible),
                     jnp.argmin(jnp.where(feasible, makespans, jnp.inf)),
                     jnp.argmin(costs))
    return history[best], makespans[best], costs[best], violations[best], infeasible[best]


def solver_args(cli, method, env, decoder):
    tasks = env.tasks_num
    is_smc = method == "SMC"
    return SolverArgs(
        Ndiffuse=cli.steps, Nsample=cli.samples, temp_sample=cli.temperature,
        N=env.agents_num * tasks, K=tasks if decoder != "eligibility" else tasks + 1,
        sample_ratio=cli.sample_ratio, elite_frac=cli.elite_frac,
        T0=cli.smc_t0 if is_smc else cli.sa_t0,
        alpha=cli.smc_alpha if is_smc else cli.sa_alpha,
        infeasibility_penalty=cli.penalty, destroy_k_ratio=cli.destroy_k_ratio,
        prune=decoder != "cover_noprune",
    )


def replay_makespan(env, routes):
    """Execute the routes in TaskEnv (as run_discrete_mbd.py does) and return its makespan."""
    env.init_state()
    for agent_id, route in enumerate(routes):
        env.pre_set_route(list(route[1:]), agent_id)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route("/tmp/", "replay", False)
    _, finished = env.get_episode_reward(100)
    return float(env.current_time) if np.mean(finished) == 1.0 else np.nan


def instance_files(suite, limit):
    files = sorted(suite.glob("env_*.pkl"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return files[:limit] if limit else files


def run_suite(cli, scenario, configuration):
    load_search_backend(cli.methods)
    from env.task_env import TaskEnv
    # Legacy instances refer to __main__.TaskEnv rather than env.task_env.TaskEnv.
    sys.modules["__main__"].TaskEnv = TaskEnv
    suite = cli.scenario_dir / scenario / configuration
    files = instance_files(suite, cli.limit)
    if not files:
        raise FileNotFoundError(f"No env_*.pkl in {suite}; run scripts/generate_paper_scenarios.py")
    decoder = decoder_for(scenario, cli.ma_decoder, not cli.no_prune)
    out_dir = cli.output_dir / scenario / configuration
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    done = set()
    if runs_csv.exists():
        with runs_csv.open() as stream:
            for row in csv.DictReader(stream):
                done.add((row["decoder"], row["method"], int(row["seed"]), int(row["instance"])))

    envs = []
    for path in files:
        with path.open("rb") as stream:
            env = pickle.load(stream)
        envs.append((int(path.stem.rsplit("_", 1)[1]), env))
    problems = {j: problem_from_env(env, require_one_hot=decoder == "exact") for j, env in envs}

    new_file = not runs_csv.exists() or runs_csv.stat().st_size == 0
    with runs_csv.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for method in (method for method in cli.methods if method in SEARCH_METHODS):
            args = solver_args(cli, method, envs[0][1], decoder)
            todo = [(s, j, env) for s in cli.seeds for j, env in envs if (decoder, method, s, j) not in done]
            if not todo:
                print(f"{scenario}/{configuration} {method}: done", flush=True)
                continue
            # Compile once per suite and method; excluded from the reported solver times.
            start = time.perf_counter()
            jax.block_until_ready(run_solver(method, decoder, args, problems[todo[0][1]],
                                             jnp.int32(0)))
            print(f"{scenario}/{configuration} {method}: compiled in "
                  f"{time.perf_counter() - start:.1f}s, {len(todo)} runs", flush=True)
            for k, (s, j, env) in enumerate(todo, 1):
                solver_seed = 1000 * s + j
                start = time.perf_counter()
                result = jax.block_until_ready(
                    run_solver(method, decoder, args, problems[j], jnp.int32(solver_seed)))
                elapsed = time.perf_counter() - start
                decision, makespan, cost, violation, infeasible = map(np.asarray, result)
                feasible = bool(violation <= 0.0)
                replay = np.nan
                if cli.replay and feasible:
                    if decoder == "exact":
                        routes = routes_from_decision(problems[j], decision)
                    elif decoder in COVER_DECODERS:
                        routes = routes_from_decision(problems[j], decision, fill="cover",
                                                      prune=args.prune)
                    else:
                        routes = routes_from_unconstrained_decision(problems[j], decision)
                    replay = replay_makespan(env, routes)
                writer.writerow(dict(
                    scenario=scenario, configuration=configuration, decoder=decoder,
                    method=method, seed=s, instance=j, solver_seed=solver_seed, levels=args.K,
                    makespan=float(makespan), penalized_cost=float(cost), feasible=int(feasible),
                    violation=float(violation), infeasible_tasks=int(infeasible),
                    replay_makespan=replay, solver_time=elapsed))
                stream.flush()
                if k % 25 == 0 or k == len(todo):
                    print(f"  {method} {k}/{len(todo)}", flush=True)


def ctas_config(cli):
    from baseline.ctas_d import CTASConfig
    return CTASConfig(solver=cli.ctas_solver, gurobi_home=cli.gurobi_home,
                      time_limit=cli.ctas_time_limit, threads=cli.ctas_threads)


def preflight(cli):
    """Fail on missing runtimes before starting the multi-hour paper sweep."""
    if "RL(s.10)" in cli.methods:
        if not cli.checkpoint.is_file():
            raise FileNotFoundError(f"RL checkpoint not found: {cli.checkpoint}")
        code = "import torch"
        if cli.rl_device == "cuda":
            code += "; assert torch.cuda.is_available(), 'RL requires CUDA; use --rl-device cpu for a CPU run'"
        subprocess.run([sys.executable, "-c", code], check=True, env=rl_environment(cli))
    if "CTAS-D" in cli.methods:
        from baseline import ctas_d
        config = ctas_config(cli)
        cli._ctas_context = (config, ctas_d.preflight(config, cli.output_dir))
    if set(cli.methods) & set(SEARCH_METHODS):
        load_search_backend(cli.methods)
        jax.devices()  # Resolve the requested platform before preparing or running suites.


def search_solver_fingerprint():
    """Identify solver sources, including local edits and alternate checkouts."""
    root = Path(os.environ.get("DISCRETE_MBD_ROOT", REPO_ROOT / "external/discrete_mbd"))
    sources = sorted((root.expanduser() / "solver").glob("*.py"))
    if not sources:
        raise FileNotFoundError(f"No search solver sources under {root}")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode() + b"\0")
        digest.update(source.read_bytes() + b"\0")
    return digest.hexdigest()


def record_run_settings(cli):
    """Keep changed budgets, solvers or checkpoints from reusing completed rows."""
    path = cli.output_dir / "run_settings.json"
    settings = json.loads(path.read_text()) if path.is_file() else {}
    common = {"scenario_dir": str(cli.scenario_dir), "instances": cli.instances}
    if settings and settings.get("suite") != common:
        raise ValueError(f"Suite settings differ from {path}; use a separate --output-dir")
    settings["suite"] = common
    methods = settings.setdefault("methods", {})
    solver_fingerprint = (search_solver_fingerprint()
                          if set(cli.methods) & set(SEARCH_METHODS) else None)
    for method in cli.methods:
        if method in SEARCH_METHODS:
            names = ["steps", "samples", "ma_decoder", "no_prune"]
            names += {"MBD": ["temperature", "sample_ratio"], "CEM": ["elite_frac"],
                      "SA": ["sa_t0", "sa_alpha"], "LNS": ["destroy_k_ratio"],
                      "SMC": ["destroy_k_ratio", "smc_t0", "smc_alpha"],
                      "ROUNDED_MBD": ["temperature"]}[method]
            if cli.ma_decoder == "eligibility":
                names.append("penalty")
        elif method == "CTAS-D":
            names = ["ctas_time_limit", "ctas_threads", "ctas_solver", "gurobi_home"]
        else:
            names = ["checkpoint", "rl_device", "rl_nccl_library"]
        values = {name: str(getattr(cli, name)) if isinstance(getattr(cli, name), Path)
                  else getattr(cli, name) for name in names}
        if method in SEARCH_METHODS:
            values["solver_sha256"] = solver_fingerprint
        if method == "RL(s.10)":
            values["rollouts"] = 10
            values["checkpoint_sha256"] = hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest()
        if method in methods and methods[method] != values:
            raise ValueError(f"{method} settings differ from {path}; use a separate --output-dir")
        methods[method] = values
    if not path.is_file() and any(cli.output_dir.glob("*/*/runs.csv")):
        print("Existing runs.csv files have no recorded settings; verify their budgets before reuse.", file=sys.stderr)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_suites(cli):
    """Reuse existing instances, and deterministically prepare missing suites."""
    from scripts.generate_paper_scenarios import PAPER_CONFIGURATIONS, generate_suites
    from env.task_env import TaskEnv
    sys.modules["__main__"].TaskEnv = TaskEnv
    configurations = {configuration.name: configuration for configuration in PAPER_CONFIGURATIONS}
    index_path = cli.scenario_dir / "manifest.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    records = {(record["scenario"], record["configuration"]): record
               for record in index.get("suites", [])}
    for scenario in cli.scenarios:
        for name in cli.configurations:
            suite = cli.scenario_dir / scenario / name
            files = instance_files(suite, None)
            count = min(cli.instances, cli.limit) if cli.limit else cli.instances
            expected = set(range(count))
            found = {int(path.stem.rsplit("_", 1)[1]) for path in files}
            if expected <= found:
                if cli.limit is None and found != expected:
                    raise ValueError(f"{suite} must contain exactly {count} instances")
                continue
            index = generate_suites(
                cli.scenario_dir, scenarios=(scenario,), configurations=(configurations[name],),
                instances=count,
            )
            records.update({(record["scenario"], record["configuration"]): record
                            for record in index["suites"]})
            index["suites"] = list(records.values())
            temporary = index_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(index, indent=2) + "\n")
            temporary.replace(index_path)


def run_ctas_suite(cli, scenario, configuration):
    from baseline import ctas_d
    config, environment = cli._ctas_context
    ctas_d.run_suite(cli.scenario_dir / scenario / configuration,
                     cli.output_dir / scenario / configuration, cli.seeds, config, FIELDS,
                     limit=cli.limit, environment=environment)


def rl_environment(cli):
    environment = os.environ.copy()
    if cli.rl_nccl_library is not None:
        library = cli.rl_nccl_library.resolve()
        if not library.is_file():
            raise FileNotFoundError(library)
        existing = environment.get("LD_PRELOAD", "")
        environment["LD_PRELOAD"] = str(library) + (" " + existing if existing else "")
    return environment


def run_rl_suite(cli, scenario, configuration):
    """Run the existing ten-rollout policy evaluator and normalize resumable rows."""
    suite = cli.scenario_dir / scenario / configuration
    files = instance_files(suite, cli.limit)
    expected = {int(path.stem.rsplit("_", 1)[1]) for path in files}
    if not expected:
        raise FileNotFoundError(f"No env_*.pkl in {suite}")
    folder = cli.output_dir / scenario / configuration
    folder.mkdir(parents=True, exist_ok=True)
    runs_csv = folder / "runs.csv"
    done = set()
    if runs_csv.is_file() and runs_csv.stat().st_size:
        with runs_csv.open() as stream:
            for row in csv.DictReader(stream):
                if row["method"] == "RL(s.10)" and row["decoder"] == "policy":
                    done.add((int(row["seed"]), int(row["instance"])))
    for seed in cli.seeds:
        missing = {instance for instance in expected if (seed, instance) not in done}
        if not missing:
            continue
        # Raw output remains inspectable; interrupted ingestion can reuse a complete CSV.
        raw = folder / f"rl_sample_10_seed{seed}.csv"
        frame = pd.read_csv(raw) if raw.is_file() and raw.stat().st_size else None
        required = {"instance", "solved", "makespan", "computation_time"}
        if (frame is None or not required <= set(frame.columns)
                or len(frame) != len(expected) or set(frame.instance) != expected):
            command = [
                sys.executable, "-m", "baseline.rl.evaluator",
                "--test-set", str(suite), "--checkpoint", str(cli.checkpoint),
                "--device", cli.rl_device, "--samples", "10",
                "--seed", str(1000 * seed), "--output", str(raw),
            ]
            if cli.limit is not None:
                command.extend(["--limit", str(cli.limit)])
            print(f"{scenario}/{configuration} RL(s.10) seed {seed}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, env=rl_environment(cli), check=True)
            frame = pd.read_csv(raw)
        if (not required <= set(frame.columns) or len(frame) != len(expected)
                or set(frame.instance) != expected or not frame.solved.isin([0, 1]).all()):
            raise ValueError(f"Incomplete or invalid RL output: {raw}")
        new_file = not runs_csv.exists() or runs_csv.stat().st_size == 0
        with runs_csv.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            for row in frame.to_dict("records"):
                instance = int(row["instance"])
                if instance not in missing:
                    continue
                solved = int(row["solved"])
                makespan = row["makespan"] if solved else float("nan")
                writer.writerow(dict(
                    scenario=scenario, configuration=configuration, decoder="policy",
                    method="RL(s.10)", seed=seed, instance=instance,
                    solver_seed=1000 * seed + instance, levels="", makespan=makespan,
                    penalized_cost=makespan, feasible=solved,
                    violation=0 if solved else float("nan"), infeasible_tasks="",
                    replay_makespan=makespan, solver_time=row["computation_time"],
                ))
                stream.flush()


def render_paper_table(summary, cli):
    """Render the paper's scenario blocks, paired configuration columns and ranks."""
    configurations = cli.configurations
    columns = 1 + 2 * len(configurations)
    lines = [
        r"% Requires \usepackage{booktabs,graphicx} and \usepackage[table]{xcolor}",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.90}",
        (r"\caption{Mean makespan and computation time on Heterogeneous MRTA over "
         f"{min(cli.instances, cli.limit) if cli.limit else cli.instances} problem instances across "
         f"{len(cli.seeds)} solver seeds. "
         r"CTAS-D superscripts indicate success rates. \textbf{Bold}/\underline{underline} "
         r"indicate best/second-best makespan, respectively.}"),
        r"\label{tab:paper-suite-compact}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{" + str(len(configurations)) + r"}{c@{\hspace{2pt}}c}}",
        r"\toprule",
    ]
    parameters = [tuple(int(part[2:]) for part in name.split("_")) for name in configurations]
    for first in (True, False):
        cells = [""]
        for agents, species, tasks, skills in parameters:
            content = f"R{{=}}{agents},\\ S{{=}}{species}" if first else f"T{{=}}{tasks},\\ B{{=}}{skills}"
            cells.append(r"\multicolumn{2}{c}{$" + content + "$}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append("".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(configurations))))
    lines.append(" & ".join(["Method"] + [
        r"\footnotesize Makespan $\downarrow$ & \footnotesize Time (s) $\downarrow$"
    ] * len(configurations)) + r" \\")
    lines.append(r"\midrule")
    lookup = {(row.scenario, row.configuration, row.method): row
              for row in summary.itertuples(index=False)}
    for scenario_index, scenario in enumerate(cli.scenarios):
        lines.extend([rf"\multicolumn{{{columns}}}{{l}}{{\textit{{{SCENARIO_LABELS[scenario]}}}}} \\",
                      r"\specialrule{0.4pt}{0.15em}{0.15em}"])
        for method in cli.methods:
            if method == "CEM":
                lines.append(r"\addlinespace[5pt]")
            if method == "MBD":
                lines.append(r"\rowcolor[gray]{0.93}")
            label = METHOD_LABELS.get(method, method)
            if method == "CTAS-D":
                label += f" ({cli.ctas_time_limit:g} s)"
            cells = [label]
            for configuration in configurations:
                row = lookup.get((scenario, configuration, method))
                if row is None:
                    cells.extend(["--", "--"])
                    continue
                value = "--" if not math.isfinite(row.mean) else f"{row.mean:.2f}"
                if value != "--":
                    # Rank displayed precision, so rounded ties share a rank.
                    ranked = sorted({round(other.mean, 2) for key, other in lookup.items()
                                     if key[:2] == (scenario, configuration)
                                     and math.isfinite(other.mean)})
                    rank = ranked.index(round(row.mean, 2))
                    if rank < 2:
                        command = "textbf" if rank == 0 else "underline"
                        value = f"\\{command}{{{value}}}"
                    if method == "CTAS-D":
                        success = r"$<1\%$" if 0 < row.success < 1 else f"{row.success:.0f}\\%"
                        value += r"\textsuperscript{\ " + success + "}"
                runtime = cli.ctas_time_limit if method == "CTAS-D" else row.solver_time_mean
                cells.extend([value, f"{runtime:g}" if method == "CTAS-D" else f"{runtime:.2f}"])
            lines.append(" & ".join(cells) + r" \\")
        if scenario_index < len(cli.scenarios) - 1:
            lines.extend([r"\midrule", r"\addlinespace[5pt]"])
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""])
    return "\n".join(lines)


def summarize(cli):
    frames = [pd.read_csv(p) for p in sorted(cli.output_dir.glob("*/*/runs.csv")) if p.stat().st_size]
    if not frames:
        print("No results yet.")
        return
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs.scenario.isin(cli.scenarios) & runs.configuration.isin(cli.configurations)
                & runs.method.isin(cli.methods) & runs.seed.isin(cli.seeds)]
    search = runs.method.isin(SEARCH_METHODS)
    expected_decoders = runs.scenario.map(lambda scenario: decoder_for(scenario, cli.ma_decoder, not cli.no_prune))
    runs = runs[(search & (runs.decoder == expected_decoders))
                | ((runs.method == "CTAS-D") & (runs.decoder == "milp"))
                | ((runs.method == "RL(s.10)") & (runs.decoder == "policy"))]
    expected_instances = min(cli.instances, cli.limit) if cli.limit else cli.instances
    runs = runs[runs.instance.isin(range(expected_instances))]
    if runs.empty:
        print("No results for the selected methods, seeds and decoder.")
        return
    if runs.duplicated(["scenario", "configuration", "decoder", "method", "seed", "instance"]).any():
        raise ValueError("Duplicate suite/method/seed/instance rows in runs.csv")
    rows = []
    for (scenario, configuration, decoder, method), group in runs.groupby(
            ["scenario", "configuration", "decoder", "method"], sort=False):
        per_seed = group[group.feasible == 1].groupby("seed").makespan.mean()
        n_inst = group.groupby("seed").instance.nunique()
        rows.append(dict(
            scenario=scenario, configuration=configuration, method=method,
            decoder=decoder, seeds=int(group.seed.nunique()),
            instances_per_seed=int(n_inst.min()),
            success=100.0 * group.feasible.mean(),
            mean=per_seed.mean(), std=per_seed.std(ddof=1) if per_seed.size > 1 else np.nan,
            infeasible=int((group.feasible == 0).sum()),
            solver_time_mean=group.solver_time.mean(),
            per_seed_means=" ".join(f"{v:.3f}" for v in per_seed.values),
        ))
    summary = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    summary["_s"] = summary.scenario.map({s: i for i, s in enumerate(SCENARIOS)})
    summary["_c"] = summary.configuration.map({c: i for i, c in enumerate(CONFIGURATIONS)})
    summary["_m"] = summary.method.map(order)
    summary = summary.sort_values(["_s", "_c", "decoder", "_m"], kind="stable").drop(
        columns=["_s", "_c", "_m"])
    summary.to_csv(cli.output_dir / "summary.csv", index=False)
    table_path = cli.output_dir / "table.tex"
    table_path.write_text(render_paper_table(summary, cli))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        table = summary.assign(result=[f"{m:.2f} ± {s:.2f}" for m, s in zip(summary["mean"],
                                                                              summary["std"])])
        print(table[["scenario", "configuration", "decoder", "method", "result", "seeds",
                     "instances_per_seed", "infeasible", "solver_time_mean"]].to_string(index=False))
    expected_groups = len(cli.scenarios) * len(cli.configurations) * len(cli.methods)
    if (len(summary) != expected_groups or (summary.seeds != len(cli.seeds)).any()
            or (summary.instances_per_seed != expected_instances).any()):
        print("Warning: incomplete results; table.tex contains partial means or missing cells.")
    print(f"\nWrote {cli.output_dir / 'summary.csv'} and {table_path}")


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def method_name(value):
    name = value.strip().upper()
    aliases = {"CTAS_D": "CTAS-D", "CTAS": "CTAS-D", "RL": "RL(s.10)",
               "RL_S10": "RL(s.10)", "RL(S.10)": "RL(s.10)"}
    return aliases.get(name, name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", type=comma_list, default=list(SCENARIOS))
    p.add_argument("--configurations", type=comma_list, default=list(CONFIGURATIONS))
    p.add_argument("--methods", type=lambda v: [method_name(m) for m in comma_list(v)], default=list(METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--limit", type=int, help="Use only the first N instances of each suite.")
    p.add_argument("--instances", type=int, default=50, help="Instances per suite (generated if missing).")
    p.add_argument("--scenario-dir", type=Path, default=REPO_ROOT / "PaperScenarioSets")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "seed_benchmark")
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1e-5, help="MBD weighting temperature lambda.")
    p.add_argument("--sample-ratio", type=float, default=0.75, help="MBD share of samples in S_i^1.")
    p.add_argument("--elite-frac", type=float, default=0.01, help="CEM elite fraction.")
    p.add_argument("--destroy-k-ratio", type=float, default=0.1, help="LNS/SMC fraction of variables resampled.")
    p.add_argument("--sa-t0", type=float, default=1.0)
    p.add_argument("--sa-alpha", type=float, default=0.95)
    p.add_argument("--smc-t0", type=float, default=1.0)
    p.add_argument("--smc-alpha", type=float, default=0.95)
    p.add_argument("--penalty", type=float, default=10_000.0,
                   help="MA-AT eligibility cost per missing capability unit.")
    p.add_argument("--ma-decoder", choices=("cover", "eligibility"), default="cover",
                   help="MA-AT decoder: greedy-cover random keys (default) or eligibility + penalty.")
    p.add_argument("--no-prune", action="store_true",
                   help="MA-AT cover: skip removing redundant robots (faster; logged as cover_noprune).")
    p.add_argument("--replay", action="store_true", help="Also replay each solution in TaskEnv.")
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "model/save/checkpoint.pth")
    p.add_argument("--rl-device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--rl-nccl-library", type=Path, help="Optional NCCL library preloaded only for RL subprocesses.")
    p.add_argument("--ctas-time-limit", type=float, default=30.0)
    p.add_argument("--ctas-threads", type=int, help="Gurobi thread count (default: Gurobi's choice).")
    p.add_argument("--ctas-solver", type=Path,
                   default=REPO_ROOT / "external/resilient_team_planner/build/main")
    p.add_argument("--gurobi-home", type=Path,
                   default=Path(os.environ.get("GUROBI_HOME", REPO_ROOT / "external/gurobi1003/linux64")))
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    cli = p.parse_args(argv)
    for name, values, allowed in (("scenarios", cli.scenarios, SCENARIOS),
                                  ("configurations", cli.configurations, CONFIGURATIONS),
                                  ("methods", cli.methods, METHODS)):
        unknown = sorted(set(values) - set(allowed))
        if not values or len(values) != len(set(values)):
            p.error(f"--{name} must be a nonempty selection without duplicates")
        if unknown:
            p.error(f"unknown {name}: {', '.join(unknown)} (choose from {', '.join(allowed)})")
    if any(seed < 0 or 1000 * seed + cli.instances - 1 > 2_000_000_000 for seed in cli.seeds):
        p.error("--seeds must yield nonnegative Gurobi seeds no larger than 2000000000")
    if len(cli.seeds) != len(set(cli.seeds)):
        p.error("--seeds must not contain duplicates")
    if cli.instances < 1 or cli.samples < 2 or cli.steps < 2:
        p.error("--instances must be positive; --samples and --steps must be at least 2")
    if cli.limit is not None and not 1 <= cli.limit <= cli.instances:
        p.error("--limit must be between 1 and --instances")
    for name in ("temperature", "penalty", "sa_t0", "smc_t0", "ctas_time_limit"):
        if not math.isfinite(getattr(cli, name)) or getattr(cli, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("elite_frac", "destroy_k_ratio", "sa_alpha", "smc_alpha"):
        if not 0 < getattr(cli, name) <= 1:
            p.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if not 0 < cli.sample_ratio < 1:
        p.error("--sample-ratio must be in (0, 1)")
    if int(cli.samples * cli.sample_ratio) < 1 or int(cli.samples * cli.sample_ratio) >= cli.samples:
        p.error("--sample-ratio must allocate at least one sample to each batch")
    if cli.ctas_threads is not None and cli.ctas_threads < 1:
        p.error("--ctas-threads must be positive")
    for name in ("scenario_dir", "output_dir", "checkpoint", "ctas_solver", "gurobi_home"):
        setattr(cli, name, getattr(cli, name).resolve())
    return cli


def main(argv=None):
    cli = parse_args(argv)
    if not cli.summary_only:
        if not cli.dry_run:
            preflight(cli)
            record_run_settings(cli)
            prepare_suites(cli)
        for scenario in cli.scenarios:
            for configuration in cli.configurations:
                if cli.dry_run:
                    print(f"{scenario}/{configuration}: decoder={decoder_for(scenario, cli.ma_decoder, not cli.no_prune)}, "
                          f"methods={','.join(cli.methods)}, seeds={cli.seeds}")
                    continue
                if "CTAS-D" in cli.methods:
                    run_ctas_suite(cli, scenario, configuration)
                if "RL(s.10)" in cli.methods:
                    run_rl_suite(cli, scenario, configuration)
                if set(cli.methods) & set(SEARCH_METHODS):
                    run_suite(cli, scenario, configuration)
    if not cli.dry_run:
        summarize(cli)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
