#!/usr/bin/env python3
"""Build the LaTeX results table from scripts/run_seed_benchmark.py output.

Reads every ``<results-dir>/<scenario>/<configuration>/runs.csv`` and reports, for each
configuration, scenario and method:

* Makespan: mean +/- std across solver seeds of the per-seed mean makespan over instances
  (infeasible runs are excluded from that seed's mean),
* Time (s): mean solver time per run (compilation excluded),
* Feas. (MA-AT only): percentage of runs, over all seeds and instances, that ended feasible.

The lowest mean makespan in each (configuration, scenario) cell group is bold. Missing results
print as "--", and incomplete groups (fewer seeds or instances than expected) are listed on
stdout so a half-finished run is never mistaken for a final number.

Example:
    .venv/bin/python scripts/make_seed_table.py --output results/seed_benchmark/table.tex
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("sa_bt", "sa_at", "ma_at")
SCENARIO_NAMES = {"sa_bt": "SA-BT", "sa_at": "SA-AT", "ma_at": "MA-AT"}
CONFIGURATIONS = (
    "kn9_ks3_km20_kb3",
    "kn15_ks5_km20_kb5",
    "kn25_ks5_km20_kb5",
    "kn25_ks5_km50_kb5",
    "kn50_ks5_km50_kb5",
)
METHODS = ("SA", "CEM", "LNS", "SMC", "ROUNDED_MBD", "MBD")
METHOD_LABELS = {"MBD": "MBD (ours)", "ROUNDED_MBD": "Rounded MBD"}


def load(results_dir):
    frames = [pd.read_csv(p) for p in sorted(results_dir.glob("*/*/runs.csv")) if p.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"No runs.csv under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def aggregate(runs):
    stats = {}
    for (scenario, configuration, method), g in runs.groupby(["scenario", "configuration", "method"]):
        per_seed = g[g.feasible == 1].groupby("seed").makespan.mean()
        stats[configuration, scenario, method] = dict(
            mean=per_seed.mean() if per_seed.size else np.nan,
            std=per_seed.std(ddof=1) if per_seed.size > 1 else np.nan,
            seeds=g.seed.nunique(),
            instances=int(g.groupby("seed").instance.nunique().min()),
            feasible=100.0 * g.feasible.mean(),
            time=g.solver_time.mean(),
        )
    return stats


def config_cell(name, rows):
    kn, ks, km, kb = re.match(r"kn(\d+)_ks(\d+)_km(\d+)_kb(\d+)", name).groups()
    return (rf"\multirow{{{rows}}}{{*}}{{\shortstack{{$k_n={kn}$\\$k_s={ks}$\\"
            rf"$k_m={km}$\\$k_b={kb}$}}}}")


def fmt_makespan(s, bold):
    if s is None or np.isnan(s["mean"]):
        return "--"
    text = f"{s['mean']:.2f}" + (rf" $\pm$ {s['std']:.2f}" if not np.isnan(s["std"]) else "")
    return rf"\textbf{{{text}}}" if bold else text


def build_table(stats, methods, caption, label, scenarios=SCENARIOS, configurations=CONFIGURATIONS):
    cols = {s: (3 if s == "ma_at" else 2) for s in scenarios}
    group, rule, sub, start = [], [], [], 3
    for sc in scenarios:
        group.append(rf"\multicolumn{{{cols[sc]}}}{{c}}{{{SCENARIO_NAMES[sc]}}}")
        rule.append(rf"\cmidrule(lr){{{start}-{start + cols[sc] - 1}}}")
        sub += ([r"Feas. (\%)"] if sc == "ma_at" else []) + ["Makespan", "Time (s)"]
        start += cols[sc]
    lines = [
        r"% Requires \usepackage{booktabs,multirow,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * sum(cols.values()) + "}",
        r"\toprule",
        " & & " + " & ".join(group) + r" \\",
        "".join(rule),
        "Configuration & Method & " + " & ".join(sub) + r" \\",
        r"\midrule",
    ]
    for ci, configuration in enumerate(configurations):
        best = {}
        for scenario in scenarios:
            means = {m: stats.get((configuration, scenario, m), {}).get("mean", np.nan) for m in methods}
            finite = {m: v for m, v in means.items() if not np.isnan(v)}
            best[scenario] = min(finite, key=finite.get) if finite else None
        for mi, method in enumerate(methods):
            cells = [config_cell(configuration, len(methods)) if mi == 0 else "",
                     METHOD_LABELS.get(method, method)]
            for scenario in scenarios:
                s = stats.get((configuration, scenario, method))
                if scenario == "ma_at":
                    cells.append("--" if s is None else f"{s['feasible']:.0f}")
                cells.append(fmt_makespan(s, best[scenario] == method))
                cells.append("--" if s is None else f"{s['time']:.2f}")
            lines.append(" & ".join(cells) + r" \\")
        if ci != len(configurations) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    return "\n".join(lines)


def preview(stats, methods, scenarios=SCENARIOS, configurations=CONFIGURATIONS):
    rows = []
    for configuration in configurations:
        for method in methods:
            row = {"configuration": configuration, "method": method}
            for scenario in scenarios:
                s = stats.get((configuration, scenario, method))
                name = SCENARIO_NAMES[scenario]
                row[name] = "--" if s is None or np.isnan(s["mean"]) else (
                    f"{s['mean']:.2f} ± {s['std']:.2f}" if not np.isnan(s["std"]) else f"{s['mean']:.2f}")
                if scenario == "ma_at":
                    row["MA-AT feas."] = "--" if s is None else f"{s['feasible']:.0f}%"
            rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "seed_benchmark")
    p.add_argument("--output", type=Path, default=None, help="Default: <results-dir>/table.tex")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--scenarios", default=",".join(SCENARIOS), help="Columns to include, e.g. sa_bt.")
    p.add_argument("--configurations", default=",".join(CONFIGURATIONS), help="Rows to include.")
    p.add_argument("--seeds", type=int, default=5, help="Expected seeds per cell (for the completeness check).")
    p.add_argument("--instances", type=int, default=50, help="Expected instances per seed.")
    p.add_argument("--label", default="tab:seed-benchmark")
    p.add_argument("--caption", default=(
        r"Makespan (mean $\pm$ std over 5 solver seeds of the mean over 50 instances) and mean "
        r"solver time per instance. All methods use the same encoding, decoder, cost and budget "
        r"(350 iterations $\times$ 2048 samples). Best mean in bold."))
    args = p.parse_args()
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    configurations = [x.strip() for x in args.configurations.split(",") if x.strip()]

    stats = aggregate(load(args.results_dir))
    output = args.output or args.results_dir / "table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_table(stats, methods, args.caption, args.label, scenarios, configurations))

    print(preview(stats, methods, scenarios, configurations))
    incomplete = [f"{c}/{s}/{m}: {v['seeds']} seeds, {v['instances']} instances/seed"
                  for (c, s, m), v in sorted(stats.items())
                  if s in scenarios and c in configurations and (v["seeds"] < args.seeds or v["instances"] < args.instances)]
    missing = [f"{c}/{s}/{m}" for c in configurations for s in scenarios for m in methods
               if (c, s, m) not in stats]
    if incomplete:
        print("\nIncomplete (not final numbers):\n  " + "\n  ".join(incomplete))
    if missing:
        print(f"\nMissing: {len(missing)} cells (shown as --)")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
