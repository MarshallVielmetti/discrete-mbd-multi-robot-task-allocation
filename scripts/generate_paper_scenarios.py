#!/usr/bin/env python3
"""Generate deterministic reconstructions of the paper's benchmark suites."""

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv


SCENARIOS = ("sa_bt", "sa_at", "ma_at")
RELEASED_CONFIGURATION = "kn15_ks5_km20_kb5"


@dataclass(frozen=True)
class PaperConfiguration:
    """One parameter block from the paper's comparison tables."""

    agents: int
    species: int
    tasks: int
    skills: int

    @property
    def agents_per_species(self):
        if self.agents % self.species:
            raise ValueError(
                f"{self.agents} agents cannot be divided evenly among "
                f"{self.species} species"
            )
        return self.agents // self.species

    @property
    def name(self):
        return (
            f"kn{self.agents}_ks{self.species}_"
            f"km{self.tasks}_kb{self.skills}"
        )


PAPER_CONFIGURATIONS = (
    PaperConfiguration(agents=9, species=3, tasks=20, skills=3),
    PaperConfiguration(agents=15, species=5, tasks=20, skills=5),
    PaperConfiguration(agents=25, species=5, tasks=20, skills=5),
    PaperConfiguration(agents=25, species=5, tasks=50, skills=5),
    PaperConfiguration(agents=50, species=5, tasks=50, skills=5),
)


def make_paper_environment(scenario, configuration, seed):
    """Construct one SA-BT, SA-AT, or MA-AT environment.

    ``TaskEnv.random_int`` uses an exclusive upper bound. Binary tasks
    therefore use ``max_task_size=2``, while additive requirements spanning
    the paper's inclusive range 0..2 use ``max_task_size=3``.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if scenario.startswith("sa_") and configuration.species != configuration.skills:
        raise ValueError("single-skill suites require one species per skill")

    return TaskEnv(
        per_species_range=(
            configuration.agents_per_species,
            configuration.agents_per_species,
        ),
        species_range=(configuration.species, configuration.species),
        tasks_range=(configuration.tasks, configuration.tasks),
        traits_dim=configuration.skills,
        decision_dim=10,
        max_task_size=2 if scenario == "sa_bt" else 3,
        duration_scale=5,
        seed=seed,
        plot_figure=False,
        agent_skill_mode=(
            "single_skill" if scenario.startswith("sa_") else "random_binary"
        ),
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(path)


def _write_pickle(path, environment):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(environment, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def _released_source(released_test_set, scenario, configuration, index, seed):
    if (
        released_test_set is None
        or scenario != "sa_bt"
        or configuration.name != RELEASED_CONFIGURATION
        or seed != index
    ):
        return None
    candidate = released_test_set / f"env_{index}.pkl"
    return candidate if candidate.is_file() else None


def generate_suites(
    output_dir,
    scenarios=SCENARIOS,
    configurations=PAPER_CONFIGURATIONS,
    instances=50,
    seed_base=0,
    released_test_set=REPO_ROOT / "RALTestSet",
    force=False,
):
    """Write selected suites and return the top-level manifest."""

    output_dir = Path(output_dir)
    released_test_set = (
        None if released_test_set is None else Path(released_test_set)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_records = []

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        for configuration in configurations:
            suite = output_dir / scenario / configuration.name
            suite.mkdir(parents=True, exist_ok=True)
            previous_records = {}
            previous_manifest_path = suite / "manifest.json"
            if previous_manifest_path.is_file():
                previous_manifest = json.loads(previous_manifest_path.read_text())
                previous_records = {
                    record["file"]: record
                    for record in previous_manifest.get("records", ())
                }
            instance_records = []
            for index in range(instances):
                seed = seed_base + index
                destination = suite / f"env_{index}.pkl"
                released_source = _released_source(
                    released_test_set, scenario, configuration, index, seed
                )

                if destination.is_file() and not force:
                    previous = previous_records.get(destination.name)
                    if (
                        previous is None
                        or previous.get("seed") != seed
                        or previous.get("sha256") != _sha256(destination)
                    ):
                        raise FileExistsError(
                            f"{destination} does not match the requested seed; "
                            "use --force or choose a new output directory"
                        )
                    source = previous.get("source", "existing")
                else:
                    if released_source is not None:
                        shutil.copy2(released_source, destination)
                        source = "released_RALTestSet"
                    else:
                        environment = make_paper_environment(
                            scenario, configuration, seed
                        )
                        _write_pickle(destination, environment)
                        source = "deterministic_reconstruction"

                instance_records.append(
                    {
                        "index": index,
                        "seed": seed,
                        "file": destination.name,
                        "sha256": _sha256(destination),
                        "source": source,
                    }
                )

            suite_manifest = {
                "scenario": scenario,
                "configuration": {
                    **asdict(configuration),
                    "agents_per_species": configuration.agents_per_species,
                },
                "instances": instances,
                "seeds": [seed_base + index for index in range(instances)],
                "task_requirement_values": (
                    [0, 1] if scenario == "sa_bt" else [0, 1, 2]
                ),
                "species_capabilities": (
                    "one_hot"
                    if scenario.startswith("sa_")
                    else "unique_nonzero_binary_vectors"
                ),
                "records": instance_records,
            }
            _write_json(suite / "manifest.json", suite_manifest)
            suite_records.append(
                {
                    "scenario": scenario,
                    "configuration": configuration.name,
                    "path": str(suite.resolve()),
                    "instances": instances,
                    "seeds": suite_manifest["seeds"],
                    "manifest": str((suite / "manifest.json").resolve()),
                    "uses_released_instances": any(
                        record["source"] == "released_RALTestSet"
                        for record in instance_records
                    ),
                }
            )

    manifest = {
        "format_version": 1,
        "paper": {
            "title": (
                "Heterogeneous Multi-robot Task Allocation and Scheduling "
                "via Reinforcement Learning"
            ),
            "url": "https://marmotlab.org/publications/73-RAL2025-HetMRTA.pdf",
        },
        "status": (
            "The released SA-BT kn15/ks5/km20/kb5 files are reused when "
            "available. Every other suite is a deterministic reconstruction "
            "because the original test files were not released."
        ),
        "generation_rules": {
            "locations": "uniform in [0,1] x [0,1]",
            "task_duration": "uniform in [0,5)",
            "agent_speed": 0.2,
            "nonempty_task_requirements": True,
            "fleet_feasibility_checked": True,
            "sa_capabilities": "one-hot with one species per skill",
            "ma_capabilities": (
                "independent binary vectors, rejecting zero and duplicate "
                "species vectors, as in the public TaskEnv generator"
            ),
            "bt_requirements": "integer samples from {0,1}",
            "at_requirements": "integer samples from {0,1,2}",
        },
        "seed_base": seed_base,
        "instances_per_suite": instances,
        "suites": suite_records,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _comma_separated(value):
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=_comma_separated,
        default=SCENARIOS,
        help="Comma-separated subset of sa_bt,sa_at,ma_at.",
    )
    parser.add_argument(
        "--configurations",
        type=_comma_separated,
        default=tuple(item.name for item in PAPER_CONFIGURATIONS),
        help=(
            "Comma-separated configuration names; the default is all five "
            "paper configurations."
        ),
    )
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "PaperScenarioSets",
    )
    parser.add_argument(
        "--released-test-set",
        type=Path,
        default=REPO_ROOT / "RALTestSet",
        help=(
            "Reuse the released SA-BT kn15/ks5/km20/kb5 pickles from this "
            "directory when seed-base is zero."
        ),
    )
    parser.add_argument(
        "--no-reuse-released",
        action="store_true",
        help="Regenerate the released configuration instead of copying it.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected suite paths without writing any files.",
    )
    args = parser.parse_args()

    unknown_scenarios = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown_scenarios:
        parser.error(f"unknown scenarios: {', '.join(unknown_scenarios)}")
    configurations_by_name = {
        item.name: item for item in PAPER_CONFIGURATIONS
    }
    unknown_configurations = sorted(
        set(args.configurations) - set(configurations_by_name)
    )
    if unknown_configurations:
        parser.error(
            "unknown configurations: " + ", ".join(unknown_configurations)
        )
    if args.instances < 1:
        parser.error("--instances must be at least 1")
    args.configurations = tuple(
        configurations_by_name[name] for name in args.configurations
    )
    if args.no_reuse_released:
        args.released_test_set = None
    return args


def main():
    args = parse_args()
    if args.dry_run:
        for scenario in args.scenarios:
            for configuration in args.configurations:
                print(args.output_dir / scenario / configuration.name)
        return

    manifest = generate_suites(
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        configurations=args.configurations,
        instances=args.instances,
        seed_base=args.seed_base,
        released_test_set=args.released_test_set,
        force=args.force,
    )
    print(
        f"Wrote {len(manifest['suites'])} suites and "
        f"{args.output_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
