import contextlib
import csv
import io
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts import run_seed_benchmark as benchmark


class PaperSuiteTest(unittest.TestCase):
    def test_search_backend_loads_from_the_checked_out_submodule(self):
        process = subprocess.run([
            sys.executable, "-c",
            "from scripts.run_seed_benchmark import load_search_backend, SOLVERS; "
            "load_search_backend(); assert len(SOLVERS) == 6",
        ], cwd=benchmark.REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_defaults_cover_the_full_paper(self):
        cli = benchmark.parse_args([])
        self.assertEqual(cli.scenarios, ["sa_bt", "sa_at", "ma_at"])
        self.assertEqual(len(cli.configurations), 5)
        self.assertEqual(set(cli.methods), {
            "MBD", "CEM", "LNS", "SA", "SMC", "ROUNDED_MBD", "CTAS-D", "RL(s.10)",
        })
        self.assertEqual(cli.seeds, [0, 1, 2, 3, 4])
        self.assertEqual((cli.steps, cli.samples, cli.instances), (350, 2048, 50))
        self.assertEqual((cli.temperature, cli.sample_ratio), (1e-5, 0.75))
        self.assertEqual((cli.destroy_k_ratio, cli.sa_t0, cli.sa_alpha), (0.1, 1.0, 0.95))
        self.assertEqual(cli.ctas_time_limit, 30)

    def test_aliases_and_invalid_empty_selections(self):
        self.assertEqual(benchmark.parse_args(["--methods", "ctas_d,rl"]).methods,
                         ["CTAS-D", "RL(s.10)"])
        for arguments in (["--methods", ""], ["--scenarios", ""],
                          ["--seeds", "0", "0"], ["--samples", "1"],
                          ["--ctas-time-limit", "nan"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    benchmark.parse_args(arguments)

    def test_dry_run_needs_no_solver_packages_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            process = subprocess.run([
                sys.executable, str(benchmark.REPO_ROOT / "scripts/run_seed_benchmark.py"),
                "--dry-run", "--output-dir", str(output),
                "--scenario-dir", str(Path(temporary) / "missing"),
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("CTAS-D", process.stdout)
            self.assertIn("RL(s.10)", process.stdout)
            self.assertEqual(len(process.stdout.strip().splitlines()), 15)
            self.assertFalse(output.exists())

    def test_default_main_dispatches_every_method_for_all_suites(self):
        with mock.patch.object(benchmark, "preflight"), \
                mock.patch.object(benchmark, "record_run_settings"), \
                mock.patch.object(benchmark, "prepare_suites") as prepare, \
                mock.patch.object(benchmark, "run_suite") as search, \
                mock.patch.object(benchmark, "run_rl_suite") as rl, \
                mock.patch.object(benchmark, "run_ctas_suite") as ctas, \
                mock.patch.object(benchmark, "summarize"):
            benchmark.main([])
        prepare.assert_called_once()
        self.assertEqual((search.call_count, rl.call_count, ctas.call_count), (15, 15, 15))

    def test_rl_ingests_seeds_and_resumes_without_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "scenarios/sa_bt/kn9_ks3_km20_kb3"
            suite.mkdir(parents=True)
            for instance in range(2):
                (suite / f"env_{instance}.pkl").touch()
            cli = benchmark.parse_args([
                "--methods", "rl", "--seeds", "0", "1", "--limit", "2",
                "--scenario-dir", str(root / "scenarios"),
                "--output-dir", str(root / "results"),
            ])

            def evaluate(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([
                    dict(instance=0, solved=1, makespan=25, computation_time=1.5),
                    dict(instance=1, solved=0, makespan=float("nan"), computation_time=2),
                ]).to_csv(output, index=False)

            with mock.patch.object(benchmark.subprocess, "run", side_effect=evaluate) as run:
                benchmark.run_rl_suite(cli, "sa_bt", "kn9_ks3_km20_kb3")
                benchmark.run_rl_suite(cli, "sa_bt", "kn9_ks3_km20_kb3")
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.args[0][1:3] == ["-m", "baseline.rl.evaluator"]
                                for call in run.call_args_list))
            self.assertEqual([call.args[0][call.args[0].index("--seed") + 1]
                              for call in run.call_args_list], ["0", "1000"])
            self.assertTrue(all(call.args[0][call.args[0].index("--samples") + 1] == "10"
                                for call in run.call_args_list))
            rows = pd.read_csv(root / "results/sa_bt/kn9_ks3_km20_kb3/runs.csv")
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(rows.method), {"RL(s.10)"})
            self.assertEqual(set(rows.solver_seed), {0, 1, 1000, 1001})
            self.assertEqual(list(rows.feasible), [1, 0, 1, 0])

    def test_ctas_resume_does_not_mistake_search_rows_for_ctas(self):
        from baseline import ctas_d as ctas
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "scenarios/sa_bt/kn9_ks3_km20_kb3"
            suite.mkdir(parents=True)
            (suite / "env_0.pkl").touch()
            output = root / "results/sa_bt/kn9_ks3_km20_kb3"
            output.mkdir(parents=True)
            with (output / "runs.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=benchmark.FIELDS)
                writer.writeheader()
                writer.writerow(dict(method="MBD", decoder="exact", seed=0, instance=0))
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                ctas.run_suite(suite, output, [0], ctas.CTASConfig(), benchmark.FIELDS, dry_run=True)
            self.assertIn("1 runs to do", printed.getvalue())

    def test_prepare_creates_all_fifteen_suites_and_reuses_existing_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            cli = benchmark.parse_args(["--scenario-dir", temporary, "--limit", "1"])
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.prepare_suites(cli)
            instances = list(Path(temporary).glob("*/*/env_0.pkl"))
            self.assertEqual(len(instances), 15)
            before = {path: path.read_bytes() for path in instances}
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.prepare_suites(cli)
            self.assertEqual(before, {path: path.read_bytes() for path in instances})
            self.assertEqual(len(list(Path(temporary).glob("*/*/manifest.json"))), 15)
            self.assertEqual(len(json.loads((Path(temporary) / "manifest.json").read_text())["suites"]), 15)

    def test_ctas_adapter_exports_inputs_and_shares_the_resumable_csv(self):
        from baseline import ctas_d as ctas
        from scripts.generate_paper_scenarios import PAPER_CONFIGURATIONS, make_paper_environment
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = benchmark.parse_args([
                "--methods", "ctas", "--seeds", "1", "--limit", "1",
                "--scenario-dir", str(root / "scenarios"),
                "--output-dir", str(root / "results"), "--ctas-threads", "2",
            ])
            suite = cli.scenario_dir / "sa_bt/kn9_ks3_km20_kb3"
            suite.mkdir(parents=True)
            with (suite / "env_0.pkl").open("wb") as stream:
                pickle.dump(make_paper_environment("sa_bt", PAPER_CONFIGURATIONS[0], 0), stream)
            options = benchmark.ctas_config(cli)
            cli._ctas_context = (options, ctas.environment_with_libraries(cli.gurobi_home))
            completed = subprocess.CompletedProcess([], 0, stdout="Set parameter Seed to value 1000\n")
            with mock.patch.object(ctas.subprocess, "run", return_value=completed) as run, \
                    mock.patch.object(ctas, "evaluate_instance", return_value={
                        "solved": 1, "makespan": 31.39, "stored_solver_time": 30.05,
                    }), contextlib.redirect_stdout(io.StringIO()):
                benchmark.run_ctas_suite(cli, "sa_bt", "kn9_ks3_km20_kb3")
                benchmark.run_ctas_suite(cli, "sa_bt", "kn9_ks3_km20_kb3")
            self.assertEqual(run.call_count, 1)
            folder = cli.output_dir / "sa_bt/kn9_ks3_km20_kb3"
            self.assertIn("Seed 1000", (folder / "seed_1/env_0/gurobi.env").read_text())
            self.assertIn("Threads 2", (folder / "seed_1/env_0/gurobi.env").read_text())
            self.assertIn("solverMaxTime: 30.0", (folder / "seed_1/env_0/planner_param.yaml").read_text())
            rows = pd.read_csv(folder / "runs.csv")
            self.assertEqual(list(rows.method), ["CTAS-D"])
            self.assertEqual(rows.makespan.item(), 31.39)

    def test_missing_prerequisites_fail_before_scenarios_are_generated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError), mock.patch.object(benchmark, "prepare_suites") as prepare:
                benchmark.main(["--methods", "rl", "--checkpoint", str(root / "missing.pth"),
                                "--scenario-dir", str(root / "scenarios"),
                                "--output-dir", str(root / "results")])
            prepare.assert_not_called()
            self.assertFalse((root / "results").exists())

    def test_resume_rejects_changed_budgets_but_allows_adding_methods_and_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = ["--output-dir", temporary, "--methods", "MBD", "--seeds", "0"]
            benchmark.record_run_settings(benchmark.parse_args(arguments))
            benchmark.record_run_settings(benchmark.parse_args(arguments + ["--seeds", "0", "1"]))
            benchmark.record_run_settings(benchmark.parse_args(["--output-dir", temporary, "--methods", "ctas"]))
            with self.assertRaisesRegex(ValueError, "MBD settings differ"):
                benchmark.record_run_settings(benchmark.parse_args(arguments + ["--samples", "4096"]))

    def test_resume_rejects_changed_solver_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            cli = benchmark.parse_args(["--output-dir", temporary, "--methods", "MBD"])
            with mock.patch.object(benchmark, "search_solver_fingerprint", create=True,
                                   return_value="original"):
                benchmark.record_run_settings(cli)
            with mock.patch.object(benchmark, "search_solver_fingerprint", create=True,
                                   return_value="changed"):
                with self.assertRaisesRegex(ValueError, "MBD settings differ"):
                    benchmark.record_run_settings(cli)

    def test_solver_fingerprint_tracks_source_edits_in_an_alternate_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            solver = Path(temporary) / "solver"
            solver.mkdir()
            source = solver / "discrete_mbd.py"
            source.write_text("original implementation\n")
            with mock.patch.dict("os.environ", DISCRETE_MBD_ROOT=temporary):
                original = benchmark.search_solver_fingerprint()
                self.assertEqual(original, benchmark.search_solver_fingerprint())
                source.write_text("changed implementation\n")
                self.assertNotEqual(original, benchmark.search_solver_fingerprint())

    def test_summary_and_table_combine_methods_without_mixing_decoders(self):
        with tempfile.TemporaryDirectory() as temporary:
            cli = benchmark.parse_args(["--output-dir", temporary])
            folder = Path(temporary) / "ma_at/kn9_ks3_km20_kb3"
            folder.mkdir(parents=True)
            rows = []
            for method, decoder, makespan, feasible in (
                ("MBD", "cover", 35.45, 1), ("SMC", "cover", 35.49, 1),
                ("MBD", "eligibility", 999, 1), ("RL(s.10)", "policy", 41.44, 1),
                ("CTAS-D", "milp", 62.29, 1), ("CTAS-D", "milp", float("nan"), 0),
            ):
                rows.append(dict(scenario="ma_at", configuration=folder.name,
                                 method=method, decoder=decoder, seed=0,
                                 instance=len(rows), makespan=makespan, feasible=feasible,
                                 solver_time=31 if method == "CTAS-D" else 0.26))
            pd.DataFrame(rows).to_csv(folder / "runs.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.summarize(cli)
            summary = pd.read_csv(Path(temporary) / "summary.csv")
            self.assertEqual(summary[summary.method == "MBD"]["mean"].item(), 35.45)
            self.assertEqual(summary[summary.method == "CTAS-D"].success.item(), 50)
            table = (Path(temporary) / "table.tex").read_text()
            self.assertIn(r"\textbf{35.45}", table)
            self.assertIn(r"\underline{35.49}", table)
            self.assertIn(r"62.29\textsuperscript{\ 50\%} & 30", table)
            self.assertIn("Discrete MBD (ours)", table)
            self.assertIn(r"\rowcolor[gray]{0.93}", table)
            self.assertNotIn("999", table)


if __name__ == "__main__":
    unittest.main()
