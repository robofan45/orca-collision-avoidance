"""Regenerate the report's figures and results: python scripts/make_figures.py

Writes docs/figures/*.png and docs/results.json: the robots' paths, the
closest distance and mean speed over time, an ablation of the keep-right
angle, a crowd-size study, and a check of ORCA's reciprocity guarantee.
"""
import contextlib
import io
import itertools
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orca_collision_avoidance import orca as m

FIGURES = ROOT / "docs" / "figures"
m.show_animation = False
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})
FALLBACKS = {"count": 0}
_least_violating = m.least_violating_velocity


def counting_fallback(*args, **kwargs):
    FALLBACKS["count"] += 1
    return _least_violating(*args, **kwargs)


m.least_violating_velocity = counting_fallback


def simulate(n_robots=10, keep_right_deg=5.0):
    m.N_ROBOTS, m.KEEP_RIGHT = n_robots, math.radians(keep_right_deg)
    FALLBACKS["count"] = 0
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        trajectory, velocities, goals = m.main()
    runtime = time.perf_counter() - t0
    steps = len(trajectory) - 1
    closest = min(np.linalg.norm(trajectory[:, i] - trajectory[:, j], axis=1).min()
                  for i, j in itertools.combinations(range(n_robots), 2))
    finished = np.linalg.norm(goals - trajectory[-1], axis=1).max() <= m.GOAL_TOLERANCE
    return trajectory, velocities, goals, {
        "finished": bool(finished), "time": steps * m.DT, "closest_m": float(closest),
        "fallback_solves": FALLBACKS["count"], "ms_per_step": 1000 * runtime / max(steps, 1)}


def figure_paths(trajectory, goals):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i in range(trajectory.shape[1]):
        color = plt.cm.tab10.colors[i % 10]
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], "-", color=color, lw=1.2)
        ax.plot(*trajectory[0, i], "o", color=color, ms=4)
        ax.plot(*goals[i], "x", color=color, ms=5, mew=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig(FIGURES / "fig1_paths.png")
    plt.close(fig)


def figure_distance_speed(trajectory, velocities):
    n = trajectory.shape[1]
    t = np.arange(len(trajectory)) * m.DT
    closest = np.min([np.linalg.norm(trajectory[:, i] - trajectory[:, j], axis=1)
                      for i, j in itertools.combinations(range(n), 2)], axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    a1.plot(t, closest, color="tab:blue")
    a1.axhline(2 * m.ROBOT_RADIUS, ls="--", color="tab:red", lw=1, label="touching (2r)")
    a1.set_xlabel("time [s]")
    a1.set_ylabel("closest pair [m]")
    a1.legend(fontsize=7)
    a2.plot(t, np.linalg.norm(velocities, axis=2).mean(axis=1), color="tab:purple")
    a2.axhline(m.MAX_SPEED, ls=":", color="0.4")
    a2.set_xlabel("time [s]")
    a2.set_ylabel("mean speed [m/s]")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_distance_speed.png")
    plt.close(fig)


def figure_deadlock(stalled, flowing):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    for ax, (trajectory, title) in zip(axes, ((stalled, "no keep-right: stalls"),
                                              (flowing, "keep right 5 deg: flows"))):
        for i in range(trajectory.shape[1]):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], "-",
                    color=plt.cm.tab10.colors[i % 10], lw=1)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_deadlock.png")
    plt.close(fig)


def reciprocity_check(pairs=2000, samples=30):
    rng = np.random.default_rng(0)
    checked, worst = 0, math.inf
    for _ in range(pairs):
        pos_a, pos_b = rng.uniform(-3.0, 3.0, size=(2, 2))
        if np.linalg.norm(pos_b - pos_a) <= 2 * m.ROBOT_RADIUS:
            continue
        vel_a, vel_b = rng.uniform(-1.5, 1.5, size=(2, 2))
        (pa, da), (pb, db) = m.orca_line(pos_a, vel_a, pos_b, vel_b), m.orca_line(pos_b, vel_b, pos_a, vel_a)
        for new_a, new_b in rng.uniform(-2.0, 2.0, size=(samples, 2, 2)):
            if m.det(da, new_a - pa) >= 0.0 and m.det(db, new_b - pb) >= 0.0:
                rel_p, rel_v = pos_b - pos_a, new_b - new_a
                s = rel_v @ rel_v
                t = 0.0 if s == 0 else min(max(-(rel_p @ rel_v) / s, 0.0), m.TIME_HORIZON)
                worst = min(worst, float(np.linalg.norm(rel_p + rel_v * t)))
                checked += 1
    return {"velocity_pairs_checked": checked, "closest_approach_m": worst,
            "required_m": 2 * m.ROBOT_RADIUS}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    trajectory, velocities, goals, demo = simulate()
    figure_paths(trajectory, goals)
    figure_distance_speed(trajectory, velocities)
    stalled = simulate(keep_right_deg=0.0)
    figure_deadlock(stalled[0], trajectory)
    keep_right = {f"{deg:g}": simulate(keep_right_deg=deg)[3] for deg in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)}
    crowd = {str(n): simulate(n_robots=n)[3] for n in (4, 8, 10, 16, 24)}
    results = {"demo": demo, "keep_right_ablation": keep_right, "crowd_size": crowd,
               "reciprocity": reciprocity_check()}
    m.N_ROBOTS, m.KEEP_RIGHT = 10, math.radians(5.0)
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
