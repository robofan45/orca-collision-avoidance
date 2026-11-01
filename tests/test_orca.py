"""
ORCA: ten robots swap sides of a circle without ever overlapping. The
half-planes are checked against ORCA's guarantee itself, and the linear
programs against a general-purpose optimiser and a case worked out by hand.
"""
import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from orca_collision_avoidance import orca as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def test_every_robot_reaches_its_goal(run):
    trajectory, _, goals = run
    assert np.linalg.norm(goals - trajectory[-1], axis=1).max() \
        <= m.GOAL_TOLERANCE
    assert (len(trajectory) - 1) * m.DT < 20.0  # observed 11.8 s


def test_robots_never_overlap(run):
    trajectory, _, _ = run
    closest = min(np.linalg.norm(trajectory[:, i] - trajectory[:, j],
                                 axis=1).min()
                  for i, j in itertools.combinations(range(m.N_ROBOTS), 2))
    assert closest > 2 * m.ROBOT_RADIUS - 1e-6  # observed: they just touch


def test_speeds_stay_within_the_limit(run):
    _, velocities, _ = run
    assert np.linalg.norm(velocities, axis=2).max() <= m.MAX_SPEED + 1e-9


def closest_approach(rel_pos, rel_vel, horizon):
    """Smallest |rel_pos + rel_vel t| for t in [0, horizon]."""
    speed_sq = rel_vel @ rel_vel
    t = 0.0 if speed_sq == 0.0 else np.clip(-(rel_pos @ rel_vel) / speed_sq,
                                            0.0, horizon)
    return np.linalg.norm(rel_pos + rel_vel * t)


def allowed(line, velocity):
    point, direction = line
    return m.det(direction, velocity - point) >= 0.0


def test_velocities_from_both_half_planes_never_collide():
    """ORCA's guarantee: if robot a picks any velocity in its half-plane
    and robot b any velocity in its own, they stay apart for the whole
    time horizon."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        pos_a, pos_b = rng.uniform(-3.0, 3.0, size=(2, 2))
        if np.linalg.norm(pos_b - pos_a) <= 2 * m.ROBOT_RADIUS:
            continue
        vel_a, vel_b = rng.uniform(-1.5, 1.5, size=(2, 2))
        line_a = m.orca_line(pos_a, vel_a, pos_b, vel_b)
        line_b = m.orca_line(pos_b, vel_b, pos_a, vel_a)
        for new_a, new_b in rng.uniform(-2.0, 2.0, size=(30, 2, 2)):
            if allowed(line_a, new_a) and allowed(line_b, new_b):
                checked += 1
                assert closest_approach(pos_b - pos_a, new_b - new_a,
                                        m.TIME_HORIZON) \
                    >= 2 * m.ROBOT_RADIUS - 1e-9
    assert checked > 1000


def random_feasible_lines(rng, n, radius):
    """n half-planes that all contain an interior point of the disc."""
    inner = rng.uniform(-0.3, 0.3, size=2) * radius
    lines = []
    for angle in rng.uniform(0.0, 2.0 * np.pi, size=n):
        direction = np.array([np.cos(angle), np.sin(angle)])
        left = np.array([-direction[1], direction[0]])
        point = inner - rng.uniform(0.05, 1.0) * left \
            + rng.normal() * direction
        lines.append((point, direction))
    return lines


def test_linear_program_matches_a_general_optimiser():
    rng = np.random.default_rng(1)
    radius = m.MAX_SPEED
    for _ in range(50):
        lines = random_feasible_lines(rng, int(rng.integers(1, 6)), radius)
        target = rng.uniform(-2.5, 2.5, size=2)
        failure, velocity = m.solve_velocity(lines, radius, target)
        assert failure == len(lines)
        constraints = [{"type": "ineq", "fun": lambda v: radius ** 2 - v @ v}]
        constraints += [{"type": "ineq",
                         "fun": lambda v, p=p, d=d: m.det(d, v - p)}
                        for p, d in lines]
        expected = minimize(lambda v: np.sum((v - target) ** 2),
                            np.zeros(2), method="SLSQP",
                            constraints=constraints,
                            options={"ftol": 1e-14, "maxiter": 300}).x
        np.testing.assert_allclose(velocity, expected, atol=1e-5)


def test_crowd_fallback_splits_the_violation_evenly():
    """No velocity has both vy >= 1 and vy <= -1; the least bad choice
    violates both by 1, at vy = 0."""
    lines = [(np.array([0.0, 1.0]), np.array([1.0, 0.0])),  # vy >= 1
             (np.array([0.0, -1.0]), np.array([-1.0, 0.0]))]  # vy <= -1
    failure, velocity = m.solve_velocity(lines, m.MAX_SPEED,
                                         np.array([0.5, 0.2]))
    assert failure == 1
    velocity = m.least_violating_velocity(lines, failure, m.MAX_SPEED,
                                          velocity)
    assert velocity[1] == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(velocity) <= m.MAX_SPEED + 1e-9

