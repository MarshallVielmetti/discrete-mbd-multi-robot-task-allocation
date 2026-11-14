#!/usr/bin/env python3
"""Visualize Discrete MBD denoising on one paper scenario.

The script runs the instrumented Discrete MBD (``baseline/discrete_mbd_history.py``)
with the benchmark settings on one instance of ``PaperScenarioSets`` and draws the
iterates x^(i), from the random start i = M to the final plan i = 0:

* tasks as discs on the ground, labeled ``T#`` with one chip per required
  capability; the number in a chip is the amount required,
* robots drawn by species (drone, mobile manipulator, quadruped, humanoid, drone
  with manipulator) in species colors. By default each depot shows one robot
  labeled with the robots it stands for (e.g. ``R1-R3``) and their capability chips,
* the candidate plans MBD evaluates around x^(i) at reverse step i as faint
  curves: nearly uniform at i = M, collapsing onto one plan as i -> 0,
* the plan x^(i) in full species color, and its makespan.

Outputs (written to ``--out-dir``):

* ``<stem>.pdf`` (or ``.svg``; ``--output all`` adds ``.png``): a row of panels for
  the paper (double-column width by default),
* ``<stem>_final.pdf`` (same formats): one large, fully labeled frame of the final plan,
* ``<stem>.gif`` / ``.mp4``: an animation over the iterates, with the schedule of
  the current plan (travel, wait, and task bars per robot) next to the scene.

The MBD run is cached in ``<out-dir>/cache``, so restyling re-renders in seconds.
Defaults are in ``DEFAULTS``; every field is also a command-line flag.

Examples (from the repository root; the solver is found through DISCRETE_MBD_ROOT):
    DISCRETE_MBD_ROOT=~/Desktop/discrete_mbd python scripts/visualize_denoising.py
    python scripts/visualize_denoising.py --scenario sa_bt --instance 3 --output gif
    python scripts/visualize_denoising.py --azimuth 70 --elevation 55 --panels 350,200,60,0
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, font_manager, rcParams  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import viz3d as V  # noqa: E402

# ----------------------------------------------------------------------------- settings
DEFAULTS = dict(
    scenario="ma_at",
    configuration="kn9_ks3_km20_kb3",
    instance=1,
    seed=0,
    steps=350,
    samples=2048,
    temperature=1e-5,
    sample_ratio=0.75,
    cloud="proposal",
    cloud_samples=48,
    output="all",
    panels="auto",
    panel_labels="first",
    frame_stride=5,
    fps=12,
    gantt=True,
    gif_dpi=100,
    gif_colors=128,
    depot_style="single",
    azimuth=50.0,
    elevation=35.0,
    width=7.0,
    out_dir=REPO_ROOT / "results" / "figures",
)

HELP = dict(
    scenario="Scenario set: sa_bt, sa_at, or ma_at.",
    configuration="Configuration folder, e.g. kn9_ks3_km20_kb3 (9 robots, 3 species, 20 tasks, 3 capabilities).",
    instance="Instance number: PaperScenarioSets/<scenario>/<configuration>/env_<instance>.pkl.",
    seed="Benchmark seed s; the solver seed is 1000 * s + instance, as in run_seed_benchmark.py.",
    steps="Number of reverse (denoising) steps M (benchmark: 350).",
    samples="Samples per step (benchmark: 2048).",
    temperature="MBD temperature of the cost weights (benchmark: 1e-5).",
    sample_ratio="Fraction of samples drawn around the current iterate (benchmark: 0.75).",
    cloud="Faint plans: 'proposal' = candidates MBD evaluates at each step; 'posterior' = draws from p.",
    cloud_samples="Number of faint plans drawn per step.",
    output="pdf | svg (panels and final frame), gif | mp4 (animation), all (pdf, png, and gif).",
    panels="Iterates i shown in the paper panels: 'auto' (M, 0.6 M, 0.25 M, 0) or a list like 350,200,60,0.",
    panel_labels="Which paper panels get T#/R# labels: first, all, or none.",
    frame_stride="Animation: show every k-th iterate.",
    fps="Animation frames per second.",
    gantt="Animation: draw the schedule of the current plan next to the scene.",
    gif_dpi="Animation resolution (100: about 1200 px wide; 140: about 1700 px).",
    gif_colors="GIF palette size; fewer colors give a smaller file but can shift colors.",
    depot_style="single: one robot per depot labeled with its robot range; ring: every robot around its depot.",
    azimuth="Camera rotation about the vertical axis, in degrees.",
    elevation="Camera tilt in degrees: 90 is top-down, lower is more tilted.",
    width="Width of the paper figures in inches (7.0 = double column).",
    out_dir="Output folder.",
)

CHOICES = dict(
    scenario=["sa_bt", "sa_at", "ma_at"],
    cloud=["proposal", "posterior"],
    output=["pdf", "svg", "gif", "mp4", "all"],
    panel_labels=["first", "all", "none"],
    depot_style=["single", "ring"],
)

# Species are drawn in this order of designs and colors (red, orange, blue, purple, green).
SPECIES_DESIGNS = ["drone", "mobile_manipulator", "quadruped", "humanoid", "drone_manipulator"]
SPECIES_COLORS = ["#E03131", "#F08C00", "#1C7ED6", "#6E56CF", "#2F9E44"]
CAPABILITY_COLORS = ["#8BD3C7", "#F6D06F", "#F4A7B9", "#B9D7F7", "#D6C6A8"]
INK = "#1f2329"
BOARD = "#eceff2"
FINAL_FILL, FINAL_EDGE = "#D3F9D8", "#2B8A3E"  # light green title box for the final plan

ROBOT_SCALE = 1.9  # robot size in world units (the ground is 10 x 10)
RING_LIMIT = 10  # depot_style="ring": depots with more robots fall back to one robot

# Animation layout (inches). The figure height is SCENE_HEIGHT times the scene's aspect ratio.
SCENE_WIDTH, SCHEDULE_WIDTH, SCENE_HEIGHT = 6.776, 5.3625, 6.16
SUBTITLE_Y = 0.835  # figure fraction of the two panel sub-titles


def _first_font(families):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in families if f in installed), "DejaVu Sans")


FONT = _first_font(["Helvetica", "Arial", "Inter", "DejaVu Sans"])
rcParams.update({
    "font.family": FONT,
    # Math text (the italic i) in the same font as everything else.
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


# ----------------------------------------------------------------------------- data
def load_env(scenario, configuration, instance):
    """Load one paper scenario (a pickled ``TaskEnv``)."""
    from env.task_env import TaskEnv
    sys.modules["__main__"].TaskEnv = TaskEnv  # the pickles refer to __main__.TaskEnv
    path = REPO_ROOT / "PaperScenarioSets" / scenario / configuration / f"env_{instance}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def run_denoising(cfg):
    """Run the instrumented MBD with the benchmark settings and decode, for every step,
    the selected plan and the faint candidate plans. Returns a dict of numpy arrays."""
    import jax
    import jax.numpy as jnp
    from baseline.discrete_mbd_adapter import SolverArgs, decode_decision, objective, problem_from_env
    from baseline.discrete_mbd_history import discrete_mbd_history
    from solver.discrete_mbd import construct_Qs, sample_categorical

    env = load_env(cfg.scenario, cfg.configuration, cfg.instance)
    exact = cfg.scenario.startswith("sa_")
    fill = "species" if exact else "cover"  # same decoders as the benchmark
    problem = problem_from_env(env, require_one_hot=exact)
    args = SolverArgs(Ndiffuse=cfg.steps, Nsample=cfg.samples, temp_sample=cfg.temperature,
                      N=env.agents_num * env.tasks_num, K=env.tasks_num,
                      sample_ratio=cfg.sample_ratio, elite_frac=0.01, T0=1.0, alpha=0.95, prune=True)
    cost_fn = lambda x: objective(problem, x, fill=fill, prune=True)  # noqa: E731
    solver_seed = 1000 * cfg.seed + cfg.instance
    history = discrete_mbd_history(args, cost_fn, seed=solver_seed)

    def decode(x):
        _, membership, _, order = decode_decision(problem, x, fill=fill, prune=True)
        return membership, order

    decode_many = jax.jit(jax.vmap(jax.vmap(decode)))
    selected_m, selected_o = decode_many(history.decisions[:, None])

    # The random start x^(M), drawn exactly as in discrete_mbd_history.
    _, initial_rng = jax.random.split(jax.random.PRNGKey(solver_seed))
    initial = sample_categorical(initial_rng, jnp.ones((args.N, args.K)) / args.K, 1)[0]
    initial_m, initial_o = decode_many(initial[None, None])

    # Faint plans at step k. "proposal": MBD's first-stage candidates Cat(x Qbar^T) around the
    # iterate carried into step k (x^(M) for k = 0, else the previous selection).
    current = jnp.concatenate([initial[None], history.decisions[:-1]], axis=0)
    _, cumulative = construct_Qs(args.Ndiffuse, args.K)
    steps_t = jnp.arange(args.Ndiffuse - 1, -1, -1)  # scan order of the history

    def proposal(x, t):
        probs = x @ cumulative[t].T
        return probs / probs.sum(-1, keepdims=True)

    probs = jax.vmap(proposal)(current, steps_t) if cfg.cloud == "proposal" else history.probabilities
    keys = jax.random.split(jax.random.PRNGKey(10_000 + solver_seed), args.Ndiffuse)
    draws = jax.vmap(lambda k, p: sample_categorical(k, p, cfg.cloud_samples))(keys, probs)
    cloud_m, cloud_o = decode_many(draws)

    return dict(
        task_xy=np.asarray(env.get_matrix(env.task_dic, "location"), dtype=float),
        requirements=np.rint(np.asarray(env.get_matrix(env.task_dic, "requirements"), dtype=float)).astype(int),
        agent_species=np.asarray(env.get_matrix(env.agent_dic, "species"), dtype=int),
        depot_xy=np.stack([np.asarray(d["location"], dtype=float) for d in env.depot_dic.values()]),
        abilities=np.asarray(env.species_dict["abilities"], dtype=int),
        makespan=np.asarray(history.objectives, dtype=float),
        initial_membership=np.asarray(initial_m[0, 0], dtype=bool),
        initial_order=np.asarray(initial_o[0, 0], dtype=int),
        initial_makespan=np.asarray(cost_fn(initial), dtype=float),
        selected_membership=np.asarray(selected_m[:, 0], dtype=bool),
        selected_order=np.asarray(selected_o[:, 0], dtype=int),
        cloud_membership=np.asarray(cloud_m, dtype=bool),
        cloud_order=np.asarray(cloud_o, dtype=int),
    )


def load_or_run(cfg):
    """Cached ``run_denoising`` plus the task durations, robot speeds and depots."""
    # "_v2": the cache also stores the random start x^(M); older caches are recomputed.
    name = (f"denoise_{cfg.scenario}_{cfg.configuration}_env{cfg.instance}_seed{cfg.seed}"
            f"_M{cfg.steps}_S{cfg.samples}_{cfg.cloud}{cfg.cloud_samples}_v2.npz")
    cache = Path(cfg.out_dir) / "cache" / name
    if cache.exists() and not cfg.rerun:
        data = dict(np.load(cache))
    else:
        data = run_denoising(cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
    env = load_env(cfg.scenario, cfg.configuration, cfg.instance)
    data["durations"] = np.asarray(env.get_matrix(env.task_dic, "time"), dtype=float)
    data["velocities"] = np.asarray(env.get_matrix(env.agent_dic, "velocity"), dtype=float)
    data["agent_depot_xy"] = np.asarray(env.get_matrix(env.agent_dic, "depot"), dtype=float)
    return data


def schedule(data, membership, order):
    """Per-robot timeline of one plan, following ``_schedule_membership`` in the adapter:
    robots travel at their own speed, a team starts a task when its last member arrives,
    and every robot returns to its depot.

    Returns ``(segments, makespan)`` with segments ``(robot, kind, t0, t1, task)`` and
    kind in {"travel", "wait", "work", "return"}.
    """
    xy, depots, vel = data["task_xy"], data["agent_depot_xy"], data["velocities"]
    n = membership.shape[0]
    free = np.zeros(n)
    here = depots.copy()
    visited = np.zeros(n, dtype=bool)
    segments = []
    for t in order:
        team = np.flatnonzero(membership[:, t])
        if len(team) == 0:
            continue
        arrival = free[team] + np.linalg.norm(xy[t] - here[team], axis=1) / vel[team]
        start = arrival.max()
        finish = start + data["durations"][t]
        for r, a in zip(team, arrival):
            segments.append((r, "travel", free[r], a, t))
            if start - a > 1e-9:
                segments.append((r, "wait", a, start, t))
            segments.append((r, "work", start, finish, t))
            free[r], here[r], visited[r] = finish, xy[t], True
    end = free.copy()
    for r in np.flatnonzero(visited):
        end[r] = free[r] + np.linalg.norm(depots[r] - here[r]) / vel[r]
        segments.append((r, "return", free[r], end[r], -1))
    return segments, float(end.max()) if visited.any() else 0.0


def routes(membership, order):
    """Per-robot ordered task lists from a membership matrix and a shared task order."""
    return [[int(t) for t in order if membership[r, t]] for r in range(membership.shape[0])]


# ----------------------------------------------------------------------------- scene
class Scene:
    """Data and static layout shared by every frame: camera, robot positions, axis limits.

    Iterates are indexed by i = M (random start), ..., 0 (final plan); see ``plan``.
    """

    def __init__(self, data, view, depot_style="single"):
        self.data, self.view = data, view
        self.M = len(data["makespan"])
        self.tasks = V.ground(data["task_xy"])
        self.depots = V.ground(data["depot_xy"])
        species = data["agent_species"]
        n = len(species)
        self.robot_pos = np.zeros((n, 3))
        self.robot_shown = np.zeros(n, dtype=bool)
        self.depot_lead = {}  # depot -> the one robot drawn for it (single style)
        self.ring_radius = {}
        for d in range(len(self.depots)):
            members = np.flatnonzero(species == d)
            if depot_style == "single" or len(members) > RING_LIMIT:
                self.robot_pos[members] = self.depots[d]
                self.robot_shown[members[0]] = True
                self.depot_lead[d] = members[0]
                self.ring_radius[d] = ROBOT_SCALE * 0.66
                continue
            radius = ROBOT_SCALE * (0.42 + 0.10 * len(members))
            self.ring_radius[d] = radius
            start = math.radians(-view.azimuth - 90)  # first robot at the front of the ring
            for j, r in enumerate(members):
                a = start + 2 * math.pi * j / len(members)
                self.robot_pos[r] = self.depots[d] + radius * np.array([math.cos(a), math.sin(a), 0.0])
                self.robot_shown[r] = True

        # Robots face their first task in the final plan; a depot's single robot faces the
        # first task any of its robots visits.
        final = routes(*self.plan(0))
        self.robot_yaw = np.full(n, math.radians(-view.azimuth - 90))
        for r, tasks in enumerate(final):
            if tasks:
                v = self.tasks[tasks[0]] - self.robot_pos[r]
                self.robot_yaw[r] = math.atan2(v[1], v[0])
        for d, lead in self.depot_lead.items():
            first = next((tasks[0] for r, tasks in enumerate(final) if species[r] == d and tasks), None)
            if first is not None:
                v = self.tasks[first] - self.depots[d]
                self.robot_yaw[lead] = math.atan2(v[1], v[0])

        # Axis limits: the ground plus room for robots; ``ylim`` also leaves room for a title.
        corners = V.ground(np.array([[-0.06, -0.06], [1.06, -0.06], [1.06, 1.06], [-0.06, 1.06]]))
        pad = 0.25
        screen = view.project(np.vstack([corners, corners + [0, 0, 1.6 * ROBOT_SCALE]]))
        self.xlim = (screen[:, 0].min() - pad, screen[:, 0].max() + pad)
        bottom = view.project(corners - [0, 0, 0.4])[:, 1].min() - pad
        top = max(view.project(np.vstack([corners, self.robot_pos + [0, 0, 1.1 * ROBOT_SCALE]]))[:, 1].max(),
                  view.project(corners)[:, 1].max() + 0.6)
        self.ylim_tight = (bottom, top + 0.2)
        self.ylim = (bottom, top + 2.4)

    def plan(self, i):
        """Membership and task order of x^(i). History row M - 1 - i holds x^(i) for i < M
        (reverse step i + 1 produced it); x^(M) is the random start."""
        d = self.data
        if i == self.M:
            return d["initial_membership"], d["initial_order"]
        return d["selected_membership"][self.M - 1 - i], d["selected_order"][self.M - 1 - i]

    def cost(self, i):
        """Makespan of x^(i)."""
        d = self.data
        return float(d["initial_makespan"]) if i == self.M else float(d["makespan"][self.M - 1 - i])

    def cloud(self, i):
        """Faint plans around x^(i) (evaluated at reverse step i); None for the final x^(0)."""
        if i == 0:
            return None
        k = self.M - i
        return self.data["cloud_membership"][k], self.data["cloud_order"][k]

    def robot_label(self, r):
        """Label of a drawn robot: its id, or the ids a depot's single robot stands for."""
        s = self.data["agent_species"][r]
        if s not in self.depot_lead:
            return f"R{r + 1}"
        ids = np.flatnonzero(self.data["agent_species"] == s) + 1
        if len(ids) == 1:
            return f"R{ids[0]}"
        if np.all(np.diff(ids) == 1):
            return f"R{ids[0]}–R{ids[-1]}"
        return ",".join(f"R{r}" for r in ids)


def robot_label_anchor(scene, r):
    """Screen point halfway between where the robot stands and the front (lowest) edge of
    its depot circle, so the robot's label sits right below it, inside the circle."""
    view = scene.view
    d = scene.data["agent_species"][r]
    ring = V.ground_ellipse(view, scene.depots[d] + [0, 0, 0.01], scene.ring_radius[d])
    base = view.project(scene.robot_pos[r])
    return np.array([base[0], 0.5 * (base[1] + ring[:, 1].min())])


def species_color(scene, r):
    return SPECIES_COLORS[scene.data["agent_species"][r] % len(SPECIES_COLORS)]


def draw_board(ax, scene):
    """Ground slab with a light grid."""
    view = scene.view
    slab = V.box((5.0, 5.0, -0.18), (10 + 1.2, 10 + 1.2, 0.36), BOARD)
    V.draw_mesh(ax, view, slab, zorder=0, edge_width=0.5)
    lines = []
    for g in np.linspace(-0.6, 10.6, 15):
        lines.append(np.array([[g, -0.6, 0.002], [g, 10.6, 0.002]]))
        lines.append(np.array([[-0.6, g, 0.002], [10.6, g, 0.002]]))
    V.draw_curves(ax, view, lines, colors=[(1, 1, 1, 0.9)] * len(lines), widths=0.5, zorder=0.5)


def draw_tasks(ax, scene, scale):
    """Task locations as white discs with a center dot."""
    view = scene.view
    for p in scene.tasks:
        xy = V.ground_ellipse(view, p + [0, 0, 0.01], 0.22)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="white", edgecolor=INK, lw=0.8 * scale, zorder=3)
        ax.plot(*view.project(p), "o", ms=1.3 * scale, color=INK, zorder=3.1)


def draw_rings(ax, scene, scale):
    """Depot circles."""
    for d, p in enumerate(scene.depots):
        xy = V.ground_ellipse(scene.view, p + [0, 0, 0.01], scene.ring_radius[d])
        ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.3 * scale, zorder=2.6, solid_capstyle="round")


def plan_curves(scene, plan_routes, rng, bend_scale, base_bend):
    """Curves of one plan, depot -> tasks in order. Each robot bends its legs by its own
    amount, so robots that share a leg stay distinguishable; ``bend_scale`` adds random
    bends that fan out the faint plans."""
    curves, owners = [], []
    n = len(plan_routes)
    for r, tasks in enumerate(plan_routes):
        if not tasks:
            continue
        points = [scene.robot_pos[r]] + [scene.tasks[t] for t in tasks]
        own_bend = base_bend * (2.0 * r / max(n - 1, 1) - 1.0)
        for a, b in zip(points[:-1], points[1:]):
            bend = own_bend + bend_scale * rng.uniform(-1, 1)
            curves.append(V.bezier(a + [0, 0, 0.03], b + [0, 0, 0.03], bend))
            owners.append(r)
    return curves, owners


def draw_routes(ax, scene, i, scale, rng):
    """Faint candidate plans around x^(i), then x^(i) itself, which gets stronger as i -> 0."""
    progress = 1.0 - i / scene.M  # 0 at x^(M), 1 at x^(0)
    curves, colors = [], []
    alpha = 0.075 + 0.2 * progress ** 1.5
    cloud = scene.cloud(i)
    for s in range(0 if cloud is None else cloud[0].shape[0]):
        plan = routes(cloud[0][s], cloud[1][s])
        c, owners = plan_curves(scene, plan, rng, bend_scale=0.35 * (1 - progress) + 0.05, base_bend=0.12)
        curves += c
        colors += [(*to_rgb(species_color(scene, r)), alpha) for r in owners]
    V.draw_curves(ax, scene.view, curves, colors, widths=0.6 * scale, zorder=1.5)

    c, owners = plan_curves(scene, routes(*scene.plan(i)), rng, bend_scale=0.0, base_bend=0.18)
    strength = 0.25 + 0.75 * progress ** 2
    plan_colors = [(*to_rgb(species_color(scene, r)), strength) for r in owners]
    halo = [(1, 1, 1, 0.8 * strength)] * len(c)
    V.draw_curves(ax, scene.view, c, halo, widths=(0.9 + 1.6 * progress) * scale, zorder=2.7)
    V.draw_curves(ax, scene.view, c, plan_colors, widths=(0.5 + 1.4 * progress) * scale, zorder=2.8)


def draw_robots(ax, scene, scale):
    """Robots (back to front), with a ground shadow under aerial ones."""
    view, data = scene.view, scene.data
    order = sorted(np.flatnonzero(scene.robot_shown), key=lambda r: -view.depth(scene.robot_pos[r]))
    for rank, r in enumerate(order):
        design = SPECIES_DESIGNS[data["agent_species"][r] % len(SPECIES_DESIGNS)]
        pos = scene.robot_pos[r]
        if design in V.AERIAL:
            V.draw_shadow(ax, view, pos, 0.24 * ROBOT_SCALE, zorder=2.9)
        mesh = V.place(V.ROBOT_BUILDERS[design](species_color(scene, r)), pos, yaw=scene.robot_yaw[r],
                       scale=ROBOT_SCALE)
        V.draw_mesh(ax, view, mesh, zorder=10 + rank, edge_width=0.2 * scale)


# ----------------------------------------------------------------------------- labels
def chip(color, value, size, fontsize):
    """Small colored box for one capability; ``value`` (an amount) is printed inside."""
    w = size * (1.25 if value is not None else 1.0)
    area = DrawingArea(w, size, 0, 0)
    area.add_artist(FancyBboxPatch((0.4, 0.4), w - 0.8, size - 0.8, boxstyle="round,pad=0,rounding_size=1.2",
                                   facecolor=color, edgecolor=INK, linewidth=0.5))
    if value is not None:
        area.add_artist(Text(w / 2, size / 2 - 0.2, str(value), ha="center", va="center",
                             fontsize=fontsize, fontweight="bold", color=INK))
    return area


def callout(ax, anchor, content, zorder, fixed, offset, pad, bboxprops, line_width, line_gap):
    """AnnotationBbox around ``content``. Fixed callouts sit on the anchor; the others start
    ``offset`` points above it with a leader line (ending ``line_gap`` points short of the
    anchor) and are moved by ``place_labels``."""
    ab = AnnotationBbox(content, anchor, xybox=(0, 0) if fixed else (0, offset), xycoords="data",
                        boxcoords="offset points", frameon=True, pad=pad, zorder=zorder, bboxprops=bboxprops,
                        arrowprops=None if fixed else dict(arrowstyle="-", color=INK, lw=line_width,
                                                           shrinkA=0, shrinkB=line_gap))
    ab.fixed = fixed
    ax.add_artist(ab)
    return ab


def draw_labels(ax, scene, fontsize):
    """Full labels: robots (fixed, below the robot) and tasks (placed), each with chips."""
    view, data = scene.view, scene.data
    size = fontsize * 1.25

    def label(anchor, title, chips, face, text_color, zorder, fixed=False):
        parts = [TextArea(title, textprops=dict(fontsize=fontsize, fontweight="bold", color=text_color))]
        parts += [chip(c, v, size, fontsize * 0.82) for c, v in chips]
        box = HPacker(children=parts, align="center", pad=0, sep=fontsize * 0.3)
        props = dict(boxstyle="round,pad=0.25,rounding_size=0.3", facecolor=face, edgecolor=INK, linewidth=0.7)
        return callout(ax, anchor, box, zorder, fixed, fontsize * 2.2, 0.28, props, 0.7, 1.5)

    labels = []
    for r in np.flatnonzero(scene.robot_shown):
        caps = [(CAPABILITY_COLORS[c % 5], None) for c in np.flatnonzero(data["abilities"][data["agent_species"][r]])]
        labels.append(label(robot_label_anchor(scene, r), scene.robot_label(r), caps, species_color(scene, r),
                            "white", 30, fixed=True))
    for t, p in enumerate(scene.tasks):
        req = data["requirements"][t]
        caps = [(CAPABILITY_COLORS[c % 5], int(req[c])) for c in np.flatnonzero(req)]
        labels.append(label(view.project(p), f"T{t + 1}", caps, "white", INK, 25))
    place_labels(ax, labels, fontsize, obstacles=robot_boxes(ax, scene))


def draw_compact_labels(ax, scene, fontsize):
    """Compact labels for narrow paper panels: T# and robot text only, no chips."""
    props = dict(boxstyle="round,pad=0.12,rounding_size=0.2", facecolor="white", edgecolor="none", alpha=0.85)

    def tag(anchor, text, color, zorder, fixed=False):
        area = TextArea(text, textprops=dict(fontsize=fontsize, fontweight="bold", color=color))
        return callout(ax, anchor, area, zorder, fixed, fontsize * 1.2, 0.12, props, 0.35, 1.0)

    labels = [tag(robot_label_anchor(scene, r), scene.robot_label(r), species_color(scene, r), 31, fixed=True)
              for r in np.flatnonzero(scene.robot_shown)]
    labels += [tag(scene.view.project(p), f"T{t + 1}", INK, 26) for t, p in enumerate(scene.tasks)]
    place_labels(ax, labels, fontsize * 0.55, obstacles=robot_boxes(ax, scene))


def robot_boxes(ax, scene):
    """Screen (display) boxes around every drawn robot, so labels avoid covering them."""
    boxes = []
    half = 0.32 * ROBOT_SCALE
    for r in np.flatnonzero(scene.robot_shown):
        corners = np.array([[dx, dy, dz] for dx in (-half, half) for dy in (-half, half)
                            for dz in (0.0, 0.75 * ROBOT_SCALE)]) + scene.robot_pos[r]
        xy = ax.transData.transform(scene.view.project(corners))
        boxes.append(Bbox([xy.min(axis=0), xy.max(axis=0)]))
    return boxes


def place_labels(ax, labels, fontsize, obstacles=()):
    """Greedy label placement. Fixed labels stay where they are; every other label tries
    offsets on rings around its anchor (upper side first) and keeps the first that does not
    overlap placed labels, robots, or the frame edge, or else the one with the least overlap."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = fig.dpi / 72.0
    base = fontsize * 2.2
    angles = [90, 60, 120, 30, 150, 0, 180, 330, 210, 270]
    candidates = [(0.0, base)]
    for radius in (1.0, 1.6, 2.3, 3.1, 4.0):
        candidates += [(radius * base * 1.25 * math.cos(math.radians(a)),
                        radius * base * math.sin(math.radians(a))) for a in angles]
    frame = ax.get_window_extent(renderer)

    def overlap(box, others):
        total = 0.0
        for o in others:
            w = min(box.x1, o.x1) - max(box.x0, o.x0)
            h = min(box.y1, o.y1) - max(box.y0, o.y0)
            if w > 0 and h > 0:
                total += w * h
        return total

    def outside(box):
        w = max(0.0, frame.x0 - box.x0) + max(0.0, box.x1 - frame.x1)
        h = max(0.0, frame.y0 - box.y0) + max(0.0, box.y1 - frame.y1)
        return (w + h) * 1e3

    placed = [ab.get_window_extent(renderer) for ab in labels if ab.fixed]
    blocked = list(obstacles)
    for ab in labels:
        if ab.fixed:
            continue
        ext = ab.get_window_extent(renderer)
        x0, y0 = ab.xybox
        best, best_cost = None, math.inf
        for dx, dy in candidates:
            shifted = ext.translated((dx - x0) * to_px, (dy - y0) * to_px)
            cost = (overlap(shifted.expanded(1.06, 1.18), placed) + 0.5 * overlap(shifted, blocked)
                    + outside(shifted))
            if cost < best_cost:
                best, best_cost = (dx, dy, shifted), cost
            if cost == 0.0:
                break
        ab.xybox = best[:2]
        placed.append(best[2])


def capability_key(fig, data, x, y, fontsize):
    """Key for the capability chips, anchored by its lower-left corner at figure fraction (x, y)."""
    size = fontsize * 1.25
    parts = [TextArea("Capabilities:", textprops=dict(fontsize=fontsize, fontweight="bold", color=INK))]
    for c in range(data["abilities"].shape[1]):
        parts.append(chip(CAPABILITY_COLORS[c % 5], None, size, fontsize * 0.82))
        parts.append(TextArea(f"C{c + 1}", textprops=dict(fontsize=fontsize, color=INK)))
    row = HPacker(children=parts, align="center", pad=0, sep=fontsize * 0.45)
    note = TextArea("# in boxes = amount required",
                    textprops=dict(fontsize=fontsize * 0.9, color="#4a5059", style="italic"))
    box = VPacker(children=[row, note], align="left", pad=0, sep=fontsize * 0.35)
    fig.add_artist(AnnotationBbox(box, (x, y), xycoords="figure fraction", box_alignment=(0, 0),
                                  frameon=False, pad=0))


# ----------------------------------------------------------------------------- frames
def header_style(scene, i):
    """Title text and box colors for x^(i); the final plan x^(0) gets a light green box."""
    makespan = scene.cost(i)
    step = f"Denoising Step $\\mathit{{i}}$ = {i}"
    if i == 0:
        return f"{step}  |  Final Makespan: {makespan:.1f}", FINAL_FILL, FINAL_EDGE
    return f"{step}  |  Best Makespan: {makespan:.1f}", "white", INK


def draw_frame(ax, scene, i, scale=1.0, labels="full", fontsize=7.0, header="full"):
    """Draw the scene for x^(i) into ``ax``.

    labels: "full" (callouts with chips), "compact" (T#/R# text), or "none".
    header: "full" (title box inside the axes), "panel" (short axes title), or "none".
    scale: multiplier for line widths and marker sizes.
    """
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(*scene.xlim)
    ax.set_ylim(*(scene.ylim if header == "full" else scene.ylim_tight))
    ax.axis("off")
    draw_board(ax, scene)
    draw_rings(ax, scene, scale)
    draw_routes(ax, scene, i, scale, np.random.default_rng(i))
    draw_tasks(ax, scene, scale)
    draw_robots(ax, scene, scale)
    if header == "full":
        text, fill, edge = header_style(scene, i)
        ax.text(0.5, 0.985, text, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize * 1.5,
                fontweight="bold", color=INK,
                bbox=dict(boxstyle="square,pad=0.35", facecolor=fill, edgecolor=edge, linewidth=1.0))
    elif header == "panel":
        ax.set_title(f"$i = {i}$   makespan {scene.cost(i):.1f}", fontsize=fontsize * 1.15, color=INK, pad=2)
    if labels == "full":
        draw_labels(ax, scene, fontsize)
    elif labels == "compact":
        draw_compact_labels(ax, scene, fontsize)


def _tint(color, amount):
    """Mix a color with white (amount = 0 keeps it, 1 gives white)."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def draw_schedule(gax, scene, i, horizon, fontsize):
    """Schedule of the plan x^(i): one row per robot (grouped by species) with travel, wait,
    and task bars on a fixed time axis [0, horizon]."""
    data = scene.data
    gax.clear()
    segments, makespan = schedule(data, *scene.plan(i))
    species = data["agent_species"]
    rows = sorted(range(len(species)), key=lambda r: (species[r], r))
    row_of = {r: j for j, r in enumerate(rows)}
    h = 0.66
    # Data units per point along x, to size and space the task labels.
    width_pt = gax.get_window_extent().width * 72.0 / gax.figure.dpi
    per_pt = horizon * 1.02 / max(width_pt, 1.0)
    small = fontsize * 0.62
    last_end = {}  # row -> right edge (data units) of the last label placed above that row
    for r, kind, t0, t1, t in sorted(segments, key=lambda seg: seg[2]):
        if t1 - t0 <= 1e-9:
            continue
        y = row_of[r]
        col = species_color(scene, r)
        if kind == "work":
            gax.barh(y, t1 - t0, left=t0, height=h, color=col, edgecolor=INK, linewidth=0.4, zorder=3)
            label = f"T{t + 1}"
            inside_w = len(label) * fontsize * 0.72 * 0.62 * per_pt
            if t1 - t0 > inside_w * 1.15:  # fits inside the bar
                gax.text((t0 + t1) / 2, y, label, ha="center", va="center", fontsize=fontsize * 0.72,
                         fontweight="bold", color="white", zorder=4)
            else:  # too narrow: small label above the bar, moved right if it would collide
                w = len(label) * small * 0.62 * per_pt
                x = max((t0 + t1) / 2, last_end.get(y, -np.inf) + 0.6 * w)
                gax.text(x, y - h / 2 - 0.03, label, ha="center", va="bottom", fontsize=small,
                         fontweight="bold", color=col, zorder=4)
                if abs(x - (t0 + t1) / 2) > 1e-9:  # tick from the bar to its moved label
                    gax.plot([(t0 + t1) / 2, x], [y - h / 2, y - h / 2 - 0.03], color=col, lw=0.4, zorder=4)
                last_end[y] = x + 0.5 * w
        elif kind == "wait":
            gax.barh(y, t1 - t0, left=t0, height=h * 0.62, facecolor="white", edgecolor="#8a919b",
                     hatch="////", linewidth=0.4, zorder=2)
        else:  # travel to a task or back to the depot
            gax.barh(y, t1 - t0, left=t0, height=h * 0.32, color=_tint(col, 0.55), linewidth=0, zorder=1)

    gax.axvline(makespan, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
    near_end = makespan > 0.85 * horizon  # keep the label inside the chart
    gax.text(makespan, -0.62, f"makespan {makespan:.1f}", ha="right" if near_end else "center", va="bottom",
             fontsize=fontsize * 0.85, fontweight="bold", color=INK, clip_on=False)
    gax.set_xlim(0, horizon * 1.02)
    gax.set_ylim(len(rows) - 0.4, -0.6)
    gax.set_yticks(range(len(rows)))
    gax.set_yticklabels([f"R{r + 1}" for r in rows], fontsize=fontsize * 0.9, fontweight="bold")
    for tick, r in zip(gax.get_yticklabels(), rows):
        tick.set_color(species_color(scene, r))
        tick.set_horizontalalignment("left")  # R1..R9 share a left edge
    gax.tick_params(axis="x", labelsize=fontsize * 1.0625, length=2, colors=INK)
    gax.tick_params(axis="y", length=0, pad=fontsize * 0.9 * 1.6 + 4)
    gax.set_xlabel("Time", fontsize=fontsize * 1.1875, color=INK)
    for side in ("top", "right", "left"):
        gax.spines[side].set_visible(False)
    gax.grid(axis="x", color="#e3e6ea", lw=0.5, zorder=0)
    handles = [Patch(facecolor="#9aa1ab", edgecolor=INK, linewidth=0.4, label="Task"),
               Patch(facecolor="white", edgecolor="#8a919b", hatch="////", linewidth=0.4, label="Wait"),
               Patch(facecolor="#d5d9de", linewidth=0, label="Travel")]
    gax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False,
               fontsize=fontsize * 1.0625, handlelength=1.6, columnspacing=1.2)


# ----------------------------------------------------------------------------- outputs
def choose_panels(cfg, M):
    """Iterates i shown in the paper panels, from x^(M) to x^(0)."""
    if cfg.panels == "auto":
        steps = [M, int(round(M * 0.6)), int(round(M * 0.25)), 0]
    else:
        steps = [int(s) for s in str(cfg.panels).split(",")]
    for s in steps:
        if not 0 <= s <= M:
            raise ValueError(f"panel iterate {s} outside 0..{M}")
    return steps


def save_figure(fig, cfg, name, formats):
    paths = []
    for fmt in formats:
        path = Path(cfg.out_dir) / f"{name}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def save_panels(cfg, scene, stem, formats):
    """Paper figure: a row of panels at the chosen iterates, with an arrow underneath."""
    steps = choose_panels(cfg, scene.M)
    n = len(steps)
    panel_w = cfg.width / n
    aspect = (scene.ylim_tight[1] - scene.ylim_tight[0]) / (scene.xlim[1] - scene.xlim[0])
    fig_h = panel_w * aspect + 0.62
    fig, axes = plt.subplots(1, n, figsize=(cfg.width, fig_h))
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.16 / fig_h, bottom=0.42 / fig_h, wspace=0.02)
    for j, (ax, i) in enumerate(zip(np.atleast_1d(axes), steps)):
        show = cfg.panel_labels == "all" or (cfg.panel_labels == "first" and j == 0)
        draw_frame(ax, scene, i, scale=max(panel_w / 4.0, 0.45), labels="compact" if show else "none",
                   fontsize=4.2, header="panel")
    fig.add_artist(FancyArrowPatch((0.03, 0.30 / fig_h), (0.97, 0.30 / fig_h), transform=fig.transFigure,
                                   arrowstyle="-|>,head_length=6,head_width=3.5", color="#c9ced6", lw=2.2))
    fig.text(0.5, 0.05 / fig_h, r"Denoising step $i$:  $M \rightarrow 0$", ha="center", va="bottom",
             fontsize=7, fontweight="bold", color=INK)
    return save_figure(fig, cfg, stem, formats)


def save_final_frame(cfg, scene, stem, formats):
    """One large, fully labeled frame of the final plan x^(0)."""
    aspect = (scene.ylim[1] - scene.ylim[0]) / (scene.xlim[1] - scene.xlim[0])
    fig = plt.figure(figsize=(cfg.width, cfg.width * aspect))
    draw_frame(fig.add_axes([0, 0, 1, 1]), scene, 0, scale=1.2, labels="full", fontsize=7.0, header="full")
    return save_figure(fig, cfg, f"{stem}_final", formats)


def save_animation(cfg, scene, stem, fmt):
    """Animation over x^(M), ..., x^(0): title box on top, scene on the left, and (with
    ``--gantt``) the schedule of the current plan on the right. The final plan is held."""
    frames = list(range(scene.M, 0, -cfg.frame_stride)) + [0]
    frames += [0] * int(cfg.fps * 1.5)
    aspect = (scene.ylim[1] - scene.ylim[0]) / (scene.xlim[1] - scene.xlim[0])
    schedule_w = SCHEDULE_WIDTH if cfg.gantt else 0.0
    total = SCENE_WIDTH + schedule_w
    fig = plt.figure(figsize=(total, SCENE_HEIGHT * aspect), dpi=cfg.gif_dpi)
    ax = fig.add_axes([0.005, 0.01, SCENE_WIDTH / total - 0.01, SUBTITLE_Y - 0.02])
    ax.set_anchor("N")  # directly under the sub-title
    subtitle = dict(ha="left", va="bottom", fontsize=13.8, fontweight="bold", color=INK)
    gax = None
    if cfg.gantt:
        fig.text(0.02, SUBTITLE_Y, "Current Plan Visualization", **subtitle)
        left = (SCENE_WIDTH + 0.05) / total
        gax = fig.add_axes([left, 0.215, (schedule_w - 0.30) / total, 0.565])
        horizon = max(schedule(scene.data, *scene.plan(i))[1] for i in range(scene.M + 1))
        # The schedule's sub-title starts at the left edge of its R# labels.
        draw_schedule(gax, scene, scene.M, horizon, fontsize=8.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x0 = min(t.get_window_extent(renderer).x0 for t in gax.get_yticklabels())
        fig.text(fig.transFigure.inverted().transform((x0, 0))[0], SUBTITLE_Y, "Schedule of the Current Plan",
                 **subtitle)
    capability_key(fig, scene.data, 0.02, 0.025, fontsize=8.0)
    title = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=13, fontweight="bold", color=INK,
                     bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=INK, linewidth=1.2))

    images = []
    for i in frames:
        draw_frame(ax, scene, i, scale=1.0, labels="full", fontsize=6.5, header="none")
        if gax is not None:
            draw_schedule(gax, scene, i, horizon, fontsize=8.0)
        text, fill, edge = header_style(scene, i)
        title.set_text(text)
        title.get_bbox_patch().set_facecolor(fill)
        title.get_bbox_patch().set_edgecolor(edge)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    path = Path(cfg.out_dir) / f"{stem}.{fmt}"
    if fmt == "gif":
        save_gif(images, path, cfg.fps, cfg.gif_colors)
    else:
        save_mp4(images, path, cfg.fps)
    return path


def save_gif(images, path, fps, colors):
    """Compact GIF: one shared adaptive palette without dithering (so faint lines do not
    turn into noise), held frames merged into longer ones, and Pillow's optimizer."""
    from PIL import Image
    frames, durations = [], []
    step_ms = round(1000 / fps)
    for img in images:
        if frames and np.array_equal(img, frames[-1]):
            durations[-1] += step_ms
            continue
        frames.append(img)
        durations.append(step_ms)
    sample = np.concatenate([frames[j] for j in np.linspace(0, len(frames) - 1, 6).astype(int)], axis=0)
    palette = Image.fromarray(sample).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quantized = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=durations, loop=0,
                      optimize=True, disposal=1)


def save_mp4(images, path, fps):
    """H.264 MP4 of the rendered frames (needs ffmpeg on the PATH)."""
    h, w = images[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    shown = ax.imshow(images[0])
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, path, dpi=100):
        for img in images:
            shown.set_data(img)
            writer.grab_frame()
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=value, help=HELP[name])
        else:
            kind = Path if isinstance(value, Path) else type(value)
            p.add_argument(flag, default=value, type=kind, choices=CHOICES.get(name), help=HELP[name])
    p.add_argument("--rerun", action="store_true", help="Ignore the cached MBD run and recompute it.")
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    data = load_or_run(cfg)
    scene = Scene(data, V.View(cfg.azimuth, cfg.elevation), cfg.depot_style)
    stem = f"denoising_{cfg.scenario}_{cfg.configuration}_env{cfg.instance}_seed{cfg.seed}"
    outputs = []
    if cfg.output in ("pdf", "svg", "all"):
        formats = {"pdf": ["pdf"], "svg": ["svg"], "all": ["pdf", "png"]}[cfg.output]
        outputs += save_panels(cfg, scene, stem, formats)
        outputs += save_final_frame(cfg, scene, stem, formats)
    if cfg.output in ("gif", "mp4", "all"):
        outputs.append(save_animation(cfg, scene, stem, "gif" if cfg.output == "all" else cfg.output))
    print(f"final makespan {scene.cost(0):.3f}")
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
