"""Vector pseudo-3D drawing for ``visualize_denoising.py``.

Everything is drawn with matplotlib polygons and lines projected through one
orthographic camera (``View``), so the output stays vector (PDF, SVG) and the
viewing angle is set in a single place: ``View(azimuth, elevation)``.

World frame: x and y span the ground (the benchmark's unit square scaled by
``WORLD_SCALE``), z points up. Robots are built from boxes, prisms and ellipsoids
in a local frame (x forward, y left, z up), about 0.6 units long, and shaded with
a fixed directional light. ``ROBOT_BUILDERS`` maps a design name to its builder.
"""

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb

WORLD_SCALE = 10.0  # one benchmark unit = 10 world units
DARK = "#3a3f47"  # joints, wheels, motors
LIGHT = "#d9dee4"  # rotor discs, grippers
LIGHT_DIR = np.array([-0.35, -0.55, 1.0]) / np.linalg.norm([-0.35, -0.55, 1.0])
AMBIENT, DIFFUSE = 0.58, 0.42


# --------------------------------------------------------------------------- camera
@dataclass
class View:
    """Orthographic camera. ``azimuth`` rotates the ground about z, ``elevation``
    tilts it (90 = top-down). Angles in degrees."""

    azimuth: float = 50.0
    elevation: float = 35.0

    def __post_init__(self):
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        self.e_u = np.array([math.cos(az), -math.sin(az), 0.0])
        self.e_v = np.array([math.sin(az) * math.sin(el), math.cos(az) * math.sin(el), math.cos(el)])
        self.toward = np.cross(self.e_u, self.e_v)  # unit vector pointing at the camera

    def project(self, points):
        """(..., 3) world points -> (..., 2) screen points."""
        points = np.asarray(points, dtype=float)
        return np.stack([points @ self.e_u, points @ self.e_v], axis=-1)

    def depth(self, points):
        """Larger is farther from the camera (draw first)."""
        return -(np.asarray(points, dtype=float) @ self.toward)


def ground(xy, z=0.0):
    """Benchmark coordinates (..., 2) in [0, 1] -> world points (..., 3)."""
    xy = np.asarray(xy, dtype=float) * WORLD_SCALE
    return np.concatenate([xy, np.full(xy.shape[:-1] + (1,), z)], axis=-1)


# --------------------------------------------------------------------------- primitives
class Mesh:
    """A list of planar faces, each with a base color and opacity."""

    def __init__(self):
        self.faces, self.colors, self.alphas = [], [], []

    def add(self, verts, color, alpha=1.0):
        self.faces.append(np.asarray(verts, dtype=float))
        self.colors.append(to_rgb(color))
        self.alphas.append(alpha)

    def extend(self, other):
        self.faces += other.faces
        self.colors += other.colors
        self.alphas += other.alphas
        return self

    def transformed(self, rotation, offset):
        """New mesh with every vertex mapped to ``rotation @ v + offset``."""
        out = Mesh()
        out.faces = [f @ rotation.T + offset for f in self.faces]
        out.colors, out.alphas = list(self.colors), list(self.alphas)
        return out


def _rot_z(yaw):
    """Rotation matrix about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box(center, size, color, alpha=1.0):
    """Axis-aligned box with the given center and (dx, dy, dz) size."""
    cx, cy, cz = center
    dx, dy, dz = np.asarray(size) / 2
    v = np.array([[cx + sx * dx, cy + sy * dy, cz + sz * dz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    # vertex index = 4*(sx>0) + 2*(sy>0) + (sz>0); faces listed counter-clockwise from outside
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    m = Mesh()
    for q in quads:
        m.add(v[list(q)], color, alpha)
    return m


def _frame(axis):
    """Unit axis and two unit vectors perpendicular to it."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(axis, helper)
    a /= np.linalg.norm(a)
    return axis, a, np.cross(axis, a)


def link(p0, p1, radius, color, sides=8, caps=True, alpha=1.0):
    """Prism (cylinder approximation) from p0 to p1."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    axis, a, b = _frame(p1 - p0)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = [radius * (math.cos(t) * a + math.sin(t) * b) for t in ang]
    m = Mesh()
    for k in range(sides):
        r0, r1 = ring[k], ring[(k + 1) % sides]
        m.add([p0 + r0, p0 + r1, p1 + r1, p1 + r0], color, alpha)
    if caps:
        m.add([p1 + r for r in ring], color, alpha)
        m.add([p0 + r for r in ring[::-1]], color, alpha)
    return m


def disc(center, radius, color, height=0.012, sides=16, alpha=1.0):
    """Thin horizontal cylinder (rotor discs)."""
    c = np.asarray(center, dtype=float)
    return link(c - [0, 0, height / 2], c + [0, 0, height / 2], radius, color, sides, True, alpha)


def ellipsoid(center, radii, color, lat=6, lon=10, alpha=1.0):
    """Low-poly ellipsoid (joints, heads)."""
    c, r = np.asarray(center, dtype=float), np.asarray(radii, dtype=float)
    th = np.linspace(0, np.pi, lat + 1)
    ph = np.linspace(0, 2 * np.pi, lon + 1)
    pts = np.array([[c + r * [math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]
                     for p in ph] for t in th])
    m = Mesh()
    for i in range(lat):
        for j in range(lon):
            quad = [pts[i, j], pts[i + 1, j], pts[i + 1, j + 1], pts[i, j + 1]]
            m.add(quad, color, alpha)
    return m


# --------------------------------------------------------------------------- robots
# Each builder takes the species color and returns a Mesh in the local frame
# (x forward, y left, z up, standing on z = 0).

def _rotor_frame(body_z, color, arm=0.26, accent=DARK):
    """Quadrotor body, arms, motors and translucent rotor discs at height ``body_z``."""
    m = Mesh()
    m.extend(box((0, 0, body_z), (0.20, 0.20, 0.08), color))
    m.extend(box((0.0, 0, body_z + 0.055), (0.10, 0.10, 0.03), accent))
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        tip = np.array([sx * arm * 0.7071, sy * arm * 0.7071, body_z])
        m.extend(link((0, 0, body_z), tip, 0.022, accent, sides=6))
        m.extend(link(tip, tip + [0, 0, 0.05], 0.035, accent, sides=8))
        m.extend(disc(tip + [0, 0, 0.065], 0.13, LIGHT, alpha=0.55))
    return m


def drone(color, hover=0.55):
    """Quadrotor with landing skids, hovering at ``hover``."""
    m = _rotor_frame(hover, color)
    for sy in (1, -1):  # landing skids
        m.extend(link((0.06, sy * 0.07, hover - 0.04), (0.06, sy * 0.10, hover - 0.14), 0.012, DARK, 5, False))
        m.extend(link((-0.06, sy * 0.07, hover - 0.04), (-0.06, sy * 0.10, hover - 0.14), 0.012, DARK, 5, False))
        m.extend(link((-0.11, sy * 0.10, hover - 0.14), (0.11, sy * 0.10, hover - 0.14), 0.013, DARK, 5))
    return m


def drone_manipulator(color, hover=0.62):
    """Quadrotor with a two-link arm and gripper below it."""
    m = _rotor_frame(hover, color)
    elbow = np.array([0.10, 0.0, hover - 0.20])
    wrist = np.array([0.20, 0.0, hover - 0.33])
    m.extend(link((0, 0, hover - 0.04), elbow, 0.025, color))
    m.extend(link(elbow, wrist, 0.022, color))
    m.extend(ellipsoid(elbow, (0.035, 0.035, 0.035), DARK, 4, 8))
    for sy in (1, -1):
        m.extend(box(wrist + [0.03, sy * 0.03, -0.04], (0.02, 0.015, 0.08), DARK))
    return m


def mobile_manipulator(color):
    """Four-wheeled base with a two-link arm and gripper."""
    m = Mesh()
    m.extend(box((0, 0, 0.13), (0.46, 0.32, 0.12), color))
    m.extend(box((-0.06, 0, 0.205), (0.22, 0.24, 0.03), DARK))
    for sx in (0.15, -0.15):
        for sy in (0.17, -0.17):
            m.extend(link((sx, sy - 0.03, 0.075), (sx, sy + 0.03, 0.075), 0.075, DARK, 10))
    base = np.array([0.08, 0.0, 0.19])
    shoulder = base + [0, 0, 0.10]
    elbow = shoulder + [-0.05, 0, 0.22]
    wrist = elbow + [0.22, 0, -0.04]
    m.extend(link(base, shoulder, 0.055, DARK, 10))
    m.extend(link(shoulder, elbow, 0.032, color))
    m.extend(link(elbow, wrist, 0.028, color))
    m.extend(ellipsoid(elbow, (0.042, 0.042, 0.042), DARK, 4, 8))
    for sy in (1, -1):
        m.extend(box(wrist + [0.05, sy * 0.035, -0.02], (0.07, 0.016, 0.03), DARK))
    return m


def quadruped(color):
    """Legged robot: body, head and four two-segment legs."""
    m = Mesh()
    hip_z = 0.34
    m.extend(box((0, 0, hip_z + 0.02), (0.46, 0.20, 0.12), color))
    m.extend(box((0.27, 0, hip_z + 0.05), (0.10, 0.14, 0.09), color))
    m.extend(box((0.32, 0, hip_z + 0.06), (0.012, 0.10, 0.04), DARK))
    for sx in (0.17, -0.17):
        for sy in (0.12, -0.12):
            hip = np.array([sx, sy, hip_z])
            knee = hip + [0.05 if sx < 0 else -0.05, 0, -0.17]
            foot = np.array([sx, sy, 0.02])
            m.extend(ellipsoid(hip, (0.04, 0.04, 0.04), DARK, 4, 8))
            m.extend(link(hip, knee, 0.028, color))
            m.extend(link(knee, foot, 0.022, DARK))
    return m


def humanoid(color):
    """Biped with torso, visor, arms and legs."""
    m = Mesh()
    for sy in (0.07, -0.07):
        m.extend(link((0, sy, 0.36), (0.02, sy, 0.19), 0.035, color))
        m.extend(link((0.02, sy, 0.19), (0, sy, 0.03), 0.03, DARK))
        m.extend(box((0.03, sy, 0.02), (0.11, 0.06, 0.035), DARK))
    m.extend(box((0, 0, 0.39), (0.14, 0.22, 0.08), DARK))
    m.extend(box((0, 0, 0.53), (0.16, 0.26, 0.22), color))
    m.extend(box((0.075, 0, 0.55), (0.02, 0.14, 0.10), LIGHT))
    for sy in (0.155, -0.155):
        m.extend(ellipsoid((0, sy, 0.62), (0.045, 0.045, 0.045), DARK, 4, 8))
        m.extend(link((0, sy, 0.62), (0.04, sy + np.sign(sy) * 0.02, 0.48), 0.028, color))
        m.extend(link((0.04, sy + np.sign(sy) * 0.02, 0.48), (0.10, sy + np.sign(sy) * 0.02, 0.38), 0.024, DARK))
    m.extend(link((0, 0, 0.64), (0, 0, 0.68), 0.03, DARK))
    m.extend(ellipsoid((0, 0, 0.75), (0.075, 0.07, 0.08), color, 6, 10))
    m.extend(box((0.065, 0, 0.76), (0.02, 0.10, 0.04), DARK))
    return m


ROBOT_BUILDERS = {
    "drone": drone,
    "mobile_manipulator": mobile_manipulator,
    "quadruped": quadruped,
    "humanoid": humanoid,
    "drone_manipulator": drone_manipulator,
}
AERIAL = {"drone", "drone_manipulator"}  # designs that get a ground shadow


# --------------------------------------------------------------------------- drawing
def draw_mesh(ax, view, mesh, zorder, edge_width=0.25):
    """Back-face cull, shade, depth-sort and draw one mesh as a PolyCollection."""
    polys, faces, depths = [], [], []
    for verts, color, alpha in zip(mesh.faces, mesh.colors, mesh.alphas):
        normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        n = np.linalg.norm(normal)
        if n < 1e-12:
            continue
        normal /= n
        facing = normal @ view.toward
        if alpha >= 1.0 and facing <= 0:  # opaque back face
            continue
        if facing < 0:
            normal = -normal
        shade = AMBIENT + DIFFUSE * max(0.0, float(normal @ LIGHT_DIR))
        rgb = np.clip(np.asarray(color) * shade, 0, 1)
        polys.append(view.project(verts))
        faces.append((*rgb, alpha))
        depths.append(view.depth(verts.mean(axis=0)))
    if not polys:
        return
    order = np.argsort(depths)[::-1]
    fc = [faces[i] for i in order]
    ec = [(*np.clip(np.asarray(f[:3]) * 0.72, 0, 1), min(1.0, f[3] + 0.2)) for f in fc]
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=fc, edgecolors=ec,
                                     linewidths=edge_width, zorder=zorder))


def place(mesh, position, yaw=0.0, scale=1.0):
    """Mesh rotated by ``yaw`` about z, scaled, and moved to ``position`` (world)."""
    return mesh.transformed(_rot_z(yaw) * scale, np.asarray(position, dtype=float))


def ground_ellipse(view, center, radius, n=64):
    """Screen points of a horizontal circle (radius in world units) around ``center``."""
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t),
                    np.full_like(t, center[2] if len(center) > 2 else 0.0)], axis=-1)
    return view.project(pts)


def draw_shadow(ax, view, position, radius, zorder, alpha=0.18):
    """Soft ground shadow under ``position``."""
    xy = ground_ellipse(view, (position[0], position[1], 0.005), radius)
    ax.fill(xy[:, 0], xy[:, 1], color="black", alpha=alpha, lw=0, zorder=zorder)


def bezier(p0, p1, bend, n=24):
    """Quadratic curve on the ground from p0 to p1 (world, 3D); ``bend`` is the
    sideways offset of the control point as a fraction of the leg length."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    side = np.array([-d[1], d[0], 0.0])
    ctrl = (p0 + p1) / 2 + bend * side
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def draw_curves(ax, view, curves, colors, widths, zorder, capstyle="round"):
    """Project world-space polylines and draw them as one LineCollection."""
    if not curves:
        return
    segs = [view.project(c) for c in curves]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=zorder,
                                     capstyle=capstyle, joinstyle="round"))
