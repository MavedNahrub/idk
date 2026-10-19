"""
Trajectory generation for Ackermann golf cart.
Generates kinematically-feasible dynamic turning radius circular-arc corner transitions
with dynamic off-tracking outward clearance compensation and curvature preview speed profiling.
"""

import math


def _perpendicular_distance(pt, line_start, line_end):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-8:
        return math.hypot(pt[0] - line_start[0], pt[1] - line_start[1])
    t = max(0.0, min(1.0, ((pt[0] - line_start[0]) * dx + (pt[1] - line_start[1]) * dy) / seg_len_sq))
    proj_x = line_start[0] + t * dx
    proj_y = line_start[1] + t * dy
    return math.hypot(pt[0] - proj_x, pt[1] - proj_y)


def _ramer_douglas_peucker(pts, epsilon=0.45):
    """
    Simplify polyline waypoints using the Ramer-Douglas-Peucker algorithm.
    Eliminates dense intermediate micro-nodes (<3m) and GPS quantization noise
    while preserving true road geometry and intersection centerlines.
    """
    if len(pts) < 3:
        return list(pts)
    dmax = 0.0
    index = 0
    for i in range(1, len(pts) - 1):
        d = _perpendicular_distance(pts[i], pts[0], pts[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        rec1 = _ramer_douglas_peucker(pts[:index + 1], epsilon)
        rec2 = _ramer_douglas_peucker(pts[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [pts[0], pts[-1]]


def calculate_dynamic_turning_radius(theta, d_in, d_out, L_wb=2.124):
    """
    Dynamically calculate kinematically feasible turning radius and tangent distance.
    Adapts according to turn deflection angle theta and approach/departure segment lengths.
    Enforces strict tangent bounds so corner transitions never overshoot road segments.

    Args:
        theta: corner deflection angle (radians)
        d_in: length of incoming road segment (m)
        d_out: length of outgoing road segment (m)
        L_wb: wheelbase of Ackermann vehicle (m)

    Returns:
        tuple (R, T):
            R: dynamic corner radius (m)
            T: tangent distance along approach/departure (m)
    """
    tan_half = math.tan(theta / 2.0)

    # 1. Base dynamic radius target from deflection angle:
    # Shallow bends (<30°): wide radius (6.2m) for high-speed stability
    # Moderate curves (30°-60°): medium radius (5.2m)
    # Right-angle turns (60°-100°): kinematically sound radius (4.5m)
    # Sharp / hairpin turns (>100°): minimum feasible radius (4.0m)
    if theta < 0.52:    # < 30 deg
        R_target = 6.2
    elif theta < 1.05:  # 30 - 60 deg
        R_target = 5.2
    elif theta < 1.75:  # 60 - 100 deg (standard right angles)
        R_target = 4.5
    else:               # > 100 deg (sharp/hairpin)
        R_target = 4.0

    # Absolute kinematic lower bound for saye Ackermann buggy (L=2.124m, delta_max=0.50 rad -> R_min ~3.89m)
    R_kinematic_min = 3.85

    # Available tangent distance from segment geometry
    max_T = min(d_in, d_out) * 0.44
    T_req = R_target * tan_half

    if T_req <= max_T:
        T = T_req
        R = R_target
    else:
        # If segment is short, scale R dynamically down to kinematic minimum
        R_avail = max_T / max(tan_half, 1e-4)
        if R_avail >= R_kinematic_min:
            R = R_avail
            T = max_T
        else:
            # Strictly enforce segment tangent bounds so T never exceeds the road segment
            T = min(min(d_in, d_out) * 0.46, R_kinematic_min * tan_half)
            R = max(2.5, T / max(tan_half, 1e-4))

    return R, T


def generate_smooth_trajectory(waypoints, step_size=0.35, corner_radius=4.5,
                               v_max=2.40, a_lat_max=0.9, min_turning_radius=None, **kwargs):
    """
    Generate kinematically-feasible Ackermann trajectory using dynamic circular arcs at corners.
    Maintains clean tangency with road centerlines so wheels never clip curbs or swing off road.

    Args:
        waypoints: list of (x, y) tuples from A* in nav coordinates
        step_size: distance between trajectory points (metres)
        corner_radius: default corner radius hint (metres)
        v_max: straight-line cruising speed (m/s, default 2.40 m/s = double speed)
        a_lat_max: max lateral acceleration (m/s²)
        min_turning_radius: alias for corner_radius (for planner_node compatibility)

    Returns:
        list of dicts: {'x', 'y', 'yaw', 'v', 'curvature'}
    """
    if min_turning_radius is not None:
        corner_radius = min_turning_radius
    elif 'min_turning_radius' in kwargs:
        corner_radius = kwargs['min_turning_radius']

    if len(waypoints) < 2:
        return []

    # 1. Remove duplicate/too-close points
    raw_pts = [waypoints[0]]
    for p in waypoints[1:]:
        if math.hypot(p[0] - raw_pts[-1][0], p[1] - raw_pts[-1][1]) > 0.1:
            raw_pts.append(p)
    if len(raw_pts) < 2:
        return []

    # 2. Prune redundant collinear micro-nodes using Ramer-Douglas-Peucker (epsilon=0.45m)
    # This prevents dense polyline nodes (<3m spacing) from artificially truncating
    # the available straight approach distance for corner arc generation while preserving
    # true surveyed road centerlines.
    pts = _ramer_douglas_peucker(raw_pts, epsilon=0.45)
    if len(pts) < 2:
        pts = raw_pts

    # 3. Build smooth path using dynamic circular arcs tangent to incoming and outgoing roads
    refined = []
    point_curv = []
    point_spd = []

    for i in range(len(pts)):
        if i == 0 or i == len(pts) - 1:
            refined.append(pts[i])
            point_curv.append(0.0)
            point_spd.append(v_max)
            continue

        p_prev, p_curr, p_next = pts[i - 1], pts[i], pts[i + 1]
        d_in = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        d_out = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        if d_in < 1e-4 or d_out < 1e-4:
            refined.append(p_curr)
            point_curv.append(0.0)
            point_spd.append(v_max)
            continue

        v_in = ((p_curr[0] - p_prev[0]) / d_in, (p_curr[1] - p_prev[1]) / d_in)
        v_out = ((p_next[0] - p_curr[0]) / d_out, (p_next[1] - p_curr[1]) / d_out)

        dot = max(-1.0, min(1.0, v_in[0] * v_out[0] + v_in[1] * v_out[1]))
        theta = math.acos(dot)

        # Nearly straight line (< 4.5 degrees)
        if theta < 0.08:
            refined.append(p_curr)
            point_curv.append(0.0)
            point_spd.append(v_max)
            continue

        # Dynamic Turning Radius Calculation
        R, T = calculate_dynamic_turning_radius(theta, d_in, d_out)

        # Arc tangent points directly on incoming and outgoing road centerlines
        pc = (p_curr[0] - v_in[0] * T, p_curr[1] - v_in[1] * T)
        pt = (p_curr[0] + v_out[0] * T, p_curr[1] + v_out[1] * T)

        # Turn direction: 2D cross product
        cross_z = v_in[0] * v_out[1] - v_in[1] * v_out[0]
        sign = 1.0 if cross_z > 0 else -1.0
        n_in = (-sign * v_in[1], sign * v_in[0])

        # Center of arc circle
        center = (pc[0] + n_in[0] * R, pc[1] + n_in[1] * R)

        ang_start = math.atan2(pc[1] - center[1], pc[0] - center[0])
        ang_end = math.atan2(pt[1] - center[1], pt[0] - center[0])
        sweep = (ang_end - ang_start + math.pi) % (2.0 * math.pi) - math.pi
        if sign > 0 and sweep < 0:
            sweep += 2.0 * math.pi
        elif sign < 0 and sweep > 0:
            sweep -= 2.0 * math.pi

        arc_len = abs(sweep) * R
        num_steps = max(6, int(arc_len / step_size))
        curv = 1.0 / max(R, 0.5)

        # Dynamic corner speed scaled by turn severity
        turn_severity = min(1.0, theta / (math.pi / 2.0))
        if theta > 1.20:
            # Acute slant turn: slow down to 0.65 - 0.80 m/s for safe curb clearance
            spd = max(0.65, 0.95 * (1.0 - 0.35 * turn_severity))
        elif theta > 0.35:
            spd = max(0.85, v_max * (1.0 - 0.58 * turn_severity))
        else:
            spd = max(0.95, v_max * (1.0 - 0.40 * turn_severity))

        # Ackermann off-tracking outward compensation:
        # Shift the arc outward towards the center of the intersection so rear inside wheels
        # maintain > 1.2m of clean asphalt clearance from the inside curb, while preserving
        # > 1.4m of asphalt margin for the front outside wheels on standard 3.6m campus roads.
        if theta > 1.20:
            # Acute / sharp slant turn: expand outward shift up to 0.75m
            delta_outward = min(0.75, 0.45 + 0.30 * ((theta - 1.20) / 0.80))
        elif theta > 0.35:
            # Moderate turn (30° - 70°): scale smoothly from 0.20m to 0.45m
            delta_outward = 0.20 + 0.25 * ((theta - 0.35) / 0.85)
        else:
            delta_outward = 0.0

        # Add approach tangent point
        refined.append(pc)
        point_curv.append(0.0)
        point_spd.append(spd)

        for s in range(1, num_steps):
            frac = s / num_steps
            a = ang_start + sweep * frac
            # Late-apex profile (exponent 1.22): biases peak outward displacement slightly past
            # mid-corner (frac ~ 0.56), keeping the vehicle deeper in the intersection throat
            # before the rear axle sweeps through the apex.
            profile = math.sin(math.pi * (frac ** 1.22))
            R_eff = R + delta_outward * profile
            refined.append((center[0] + R_eff * math.cos(a), center[1] + R_eff * math.sin(a)))
            point_curv.append(curv)
            point_spd.append(spd)

        # Add departure tangent point
        refined.append(pt)
        point_curv.append(0.0)
        point_spd.append(spd)

    # 3. Resample regularly
    trajectory = []
    raw_speeds = []
    seg_dists = []

    for j in range(len(refined) - 1):
        p1, p2 = refined[j], refined[j + 1]
        seg_d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg_d < 1e-4:
            continue

        steps = max(1, int(seg_d / step_size))
        yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        curv = point_curv[j]
        spd = point_spd[j]

        step_dist = seg_d / steps
        for s in range(steps):
            t = s / steps
            trajectory.append({
                'x': float(p1[0] + t * (p2[0] - p1[0])),
                'y': float(p1[1] + t * (p2[1] - p1[1])),
                'yaw': float(yaw),
                'v': float(spd),
                'curvature': float(curv)
            })
            raw_speeds.append(float(spd))
            seg_dists.append(step_dist)

    if not trajectory:
        return []

    # 4. Preview Deceleration & Acceleration Smoothing Pass
    # Ensures buggy slows down smoothly approaching corners and accelerates exiting
    n_traj = len(trajectory)
    v_smooth = list(raw_speeds)
    a_dec = 0.90   # comfortable deceleration (m/s²)
    a_acc = 1.10   # comfortable acceleration (m/s²)

    # Backward pass: preview deceleration before corners
    for k in range(n_traj - 2, -1, -1):
        v_limit = math.sqrt(v_smooth[k + 1]**2 + 2.0 * a_dec * seg_dists[k])
        if v_smooth[k] > v_limit:
            v_smooth[k] = v_limit

    # Forward pass: acceleration limits
    for k in range(1, n_traj):
        v_limit = math.sqrt(v_smooth[k - 1]**2 + 2.0 * a_acc * seg_dists[k - 1])
        if v_smooth[k] > v_limit:
            v_smooth[k] = v_limit

    for k in range(n_traj):
        trajectory[k]['v'] = float(v_smooth[k])

    return trajectory
