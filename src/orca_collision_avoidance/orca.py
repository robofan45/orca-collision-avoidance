"""

Optimal Reciprocal Collision Avoidance (ORCA)

author: Mahim Trivedi

Ten robots stand on a circle and each must reach the point opposite, so all
of them head for the middle at once. Every step each robot chooses its new
velocity on its own, from the others' positions and velocities:
1. for every neighbour it finds the velocities that would collide with that
   neighbour within TIME_HORIZON seconds (the velocity obstacle), and keeps
   the half-plane of velocities that avoids it, where each of the two
   robots does half of the avoiding;
2. among the velocities inside all half-planes and within its top speed, it
   takes the one closest to its preferred velocity (straight at its goal),
   with a small linear program in two dimensions.
When no velocity fits every half-plane (a dense crowd), it takes the
velocity that violates them the least.

Robots placed in perfect symmetry choose perfectly symmetric velocities and
stall in a ring around the middle, none willing to step aside. Turning every
preferred velocity a few degrees to the right (keep right) breaks the tie,
and the crowd flows through like a roundabout.

Reference:
- van den Berg, Guy, Lin and Manocha, "Reciprocal n-Body Collision
  Avoidance", Robotics Research: The 14th International Symposium ISRR,
  Springer, 2011.
- RVO2 Library, the authors' reference implementation:
  https://gamma.cs.unc.edu/RVO2/

"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

N_ROBOTS = 10
CIRCLE_RADIUS = 8.0  # [m] the robots start on this circle
ROBOT_RADIUS = 0.5  # [m]
MAX_SPEED = 1.5  # [m/s]
TIME_HORIZON = 3.0  # [s] how far ahead collisions with others are avoided
DT = 0.1  # [s]
GOAL_TOLERANCE = 0.05  # [m]
MAX_TIME = 40.0  # [s]
KEEP_RIGHT = np.radians(5.0)  # [rad] turn of every preferred velocity
EPSILON = 1e-9

show_animation = True

# A half-plane of allowed velocities is a boundary line, given by a point on
# it and a unit direction; the allowed side is to the left of the direction.
Line = tuple[np.ndarray, np.ndarray]


def det(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product: positive when b points to the left of a."""
    return float(a[0] * b[1] - a[1] * b[0])


def orca_line(pos_a: np.ndarray, vel_a: np.ndarray, pos_b: np.ndarray,
              vel_b: np.ndarray) -> Line:
    """Half-plane of velocities for robot a that avoid robot b for
    TIME_HORIZON seconds, with a taking half of the change needed."""
    rel_pos = pos_b - pos_a
    rel_vel = vel_a - vel_b
    dist_sq = rel_pos @ rel_pos
    radius = 2.0 * ROBOT_RADIUS
    if dist_sq > radius ** 2:
        # w: from the centre of the velocity obstacle's cut-off circle to the
        # relative velocity
        w = rel_vel - rel_pos / TIME_HORIZON
        w_len_sq = w @ w
        dot = w @ rel_pos
        if dot < 0.0 and dot ** 2 > radius ** 2 * w_len_sq:
            # the nearest way out of the obstacle is across the cut-off circle
            w_len = math.sqrt(w_len_sq)
            unit_w = w / w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (radius / TIME_HORIZON - w_len) * unit_w
        else:
            # ... or across one of the two legs of the cone
            leg = math.sqrt(dist_sq - radius ** 2)
            if det(rel_pos, w) > 0.0:  # left leg
                direction = np.array([
                    rel_pos[0] * leg - rel_pos[1] * radius,
                    rel_pos[0] * radius + rel_pos[1] * leg]) / dist_sq
            else:  # right leg
                direction = -np.array([
                    rel_pos[0] * leg + rel_pos[1] * radius,
                    -rel_pos[0] * radius + rel_pos[1] * leg]) / dist_sq
            u = (rel_vel @ direction) * direction - rel_vel
    else:
        # already overlapping: separate within one time step
        w = rel_vel - rel_pos / DT
        w_len = math.sqrt(w @ w)
        unit_w = w / w_len
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (radius / DT - w_len) * unit_w
    # u is the smallest change of the relative velocity that leaves the
    # obstacle; robot a makes half of it
    return vel_a + 0.5 * u, direction


def solve_on_line(lines: list[Line], k: int, radius: float,
                  target: np.ndarray, direction_mode: bool
                  ) -> np.ndarray | None:
    """The best velocity on boundary line k that satisfies lines[:k] and
    |v| <= radius: the point closest to target or, in direction_mode, the
    point furthest along the direction target. None if there is none."""
    point, direction = lines[k]
    dot = float(point @ direction)
    discriminant = dot ** 2 + radius ** 2 - float(point @ point)
    if discriminant < 0.0:
        return None  # the speed limit circle misses this line
    # the part of the line inside the speed limit circle: t_left..t_right
    t_left = -dot - math.sqrt(discriminant)
    t_right = -dot + math.sqrt(discriminant)
    for other_point, other_direction in lines[:k]:
        denominator = det(direction, other_direction)
        numerator = det(other_direction, point - other_point)
        if abs(denominator) <= EPSILON:
            if numerator < 0.0:
                return None  # parallel, with no allowed velocity in common
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)  # the other line cuts off the right
        else:
            t_left = max(t_left, t)  # ... or the left
        if t_left > t_right:
            return None
    if direction_mode:
        t = t_right if target @ direction > 0.0 else t_left
    else:
        t = min(max(float(direction @ (target - point)), t_left), t_right)
    return point + t * direction


def solve_velocity(lines: list[Line], radius: float, target: np.ndarray,
                   direction_mode: bool = False) -> tuple[int, np.ndarray]:
    """The velocity closest to target that satisfies every half-plane and
    |v| <= radius, adding the half-planes one at a time: while the current
    answer satisfies the next one, it stays the answer; otherwise the new
    answer lies on that half-plane's boundary. Returns the index of the
    first half-plane that cannot be met (len(lines) if all are) and the
    answer so far."""
    if direction_mode:
        result = target * radius
    elif target @ target > radius ** 2:
        result = target / np.linalg.norm(target) * radius
    else:
        result = target.copy()
    for k, (point, direction) in enumerate(lines):
        if det(direction, point - result) > 0.0:  # result is outside line k
            on_line = solve_on_line(lines, k, radius, target, direction_mode)
            if on_line is None:
                return k, result
            result = on_line
    return len(lines), result


def least_violating_velocity(lines: list[Line], first_failure: int,
                             radius: float,
                             result: np.ndarray) -> np.ndarray:
    """For a crowd with no velocity inside every half-plane: the velocity
    that minimises the largest violation, found by moving all half-plane
    boundaries outwards at equal speed until they meet."""
    distance = 0.0  # largest violation so far
    for i in range(first_failure, len(lines)):
        point_i, dir_i = lines[i]
        if det(dir_i, point_i - result) <= distance:
            continue
        # lines along which line i and an earlier line j are violated equally
        projected: list[Line] = []
        for point_j, dir_j in lines[:i]:
            determinant = det(dir_i, dir_j)
            if abs(determinant) <= EPSILON:
                if dir_i @ dir_j > 0.0:
                    continue  # same direction: line i already decides
                new_point = 0.5 * (point_i + point_j)
            else:
                new_point = point_i + det(dir_j, point_i - point_j) \
                    / determinant * dir_i
            new_direction = dir_j - dir_i
            projected.append((new_point,
                              new_direction / np.linalg.norm(new_direction)))
        failure, candidate = solve_velocity(
            projected, radius, np.array([-dir_i[1], dir_i[0]]),
            direction_mode=True)
        if failure == len(projected):  # otherwise keep the previous result
            result = candidate
        distance = det(dir_i, point_i - result)
    return result


def preferred_velocity(position: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """At the goal at full speed, turned KEEP_RIGHT to the right; within one
    step of the goal, exactly onto it."""
    to_goal = goal - position
    distance = float(np.linalg.norm(to_goal))
    if distance < MAX_SPEED * DT:
        return to_goal / DT
    c, s = math.cos(-KEEP_RIGHT), math.sin(-KEEP_RIGHT)
    return np.array([[c, -s], [s, c]]) @ to_goal / distance * MAX_SPEED


def new_velocity(i: int, positions: np.ndarray, velocities: np.ndarray,
                 preferred: np.ndarray) -> np.ndarray:
    """ORCA velocity of robot i; the nearest neighbours' half-planes come
    first, so in a crowd the farthest ones are the ones bent."""
    others = sorted((j for j in range(len(positions)) if j != i),
                    key=lambda j: float(np.sum((positions[j]
                                                - positions[i]) ** 2)))
    lines = [orca_line(positions[i], velocities[i], positions[j],
                       velocities[j]) for j in others]
    failure, velocity = solve_velocity(lines, MAX_SPEED, preferred)
    if failure < len(lines):
        velocity = least_violating_velocity(lines, failure, MAX_SPEED,
                                            velocity)
    return velocity


def plot_frame(positions: np.ndarray, velocities: np.ndarray,
               goals: np.ndarray, trails: np.ndarray,
               time: float) -> None:  # pragma: no cover
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                       else None])
    for i, (p, v, goal) in enumerate(zip(positions, velocities, goals)):
        color = f"C{i}"
        plt.plot(trails[:, i, 0], trails[:, i, 1], "-", color=color, lw=1,
                 alpha=0.6)
        plt.plot(goal[0], goal[1], "x", color=color, ms=8, mew=2)
        plt.gca().add_patch(Circle((p[0], p[1]), ROBOT_RADIUS, color=color))
        plt.arrow(p[0], p[1], v[0], v[1], head_width=0.2, color="k")
    plt.axis((-CIRCLE_RADIUS - 1.5, CIRCLE_RADIUS + 1.5,
              -CIRCLE_RADIUS - 1.5, CIRCLE_RADIUS + 1.5))
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.title(f"ORCA: {len(positions)} robots swap sides  t = {time:4.1f} s"
              "\ncrosses: goals, arrows: velocities")
    plt.pause(0.001)


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(__file__ + " start!!")
    angles = 2.0 * math.pi * np.arange(N_ROBOTS) / N_ROBOTS
    positions = CIRCLE_RADIUS * np.column_stack([np.cos(angles),
                                                 np.sin(angles)])
    goals = -positions
    velocities = np.zeros_like(positions)
    trajectory, speeds = [positions], [velocities]
    time = 0.0

    while (np.linalg.norm(goals - positions, axis=1).max() > GOAL_TOLERANCE
           and time < MAX_TIME):
        preferred = np.array([preferred_velocity(p, g)
                              for p, g in zip(positions, goals)])
        # every robot decides from the same snapshot, then all move
        velocities = np.array([new_velocity(i, positions, velocities,
                                            preferred[i])
                               for i in range(N_ROBOTS)])
        positions = positions + velocities * DT
        time += DT
        trajectory.append(positions)
        speeds.append(velocities)

        if show_animation:  # pragma: no cover
            plot_frame(positions, velocities, goals, np.array(trajectory),
                       time)

    if show_animation:  # pragma: no cover
        plt.show()

    return np.array(trajectory), np.array(speeds), goals


if __name__ == '__main__':
    main()
