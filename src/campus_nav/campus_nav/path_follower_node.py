"""
Ackermann Pure Pursuit Path Follower with Dynamic Orientation Recognition & 4-Wheel Edge Guard
IIT Ropar Campus Autonomous Navigation System

Controls saye buggy along planned paths.
Features:
- Dynamic relative lane orientation recognition
- Perpendicular (~90°) orientation detection with forward motion inhibition
- Auto-correction recovery maneuver (reverse and realign parallel to lane)
- Double-speed cruising (2.40 m/s nominal, 3.00 m/s max) with corner preview deceleration
- True rear-axle Pure Pursuit with active restorative heading alignment
- Multi-wheel road boundary guard preventing curbs clipping
"""

import math
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from campus_nav.coord_bridge import gz_to_nav, yaw_from_quaternion, nav_to_gz, yaw_nav_to_gz
from campus_nav.osm_loader import CampusMap
import os


def _normalize(angle):
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _find_default_osm():
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('campus_nav')
        candidates.extend([
            os.path.join(pkg_share, 'data', 'campus_with_junctions_and_stops.osm'),
            os.path.join(pkg_share, 'data', 'campus.osm')
        ])
    except Exception:
        pass
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(curr_dir, '..', '..', '..'))
    candidates.extend([
        os.path.join(curr_dir, '..', 'data', 'campus_with_junctions_and_stops.osm'),
        os.path.join(curr_dir, '..', 'data', 'campus.osm'),
        os.path.join(repo_root, 'campus_with_junctions_and_stops.osm'),
        os.path.join(repo_root, 'campus.osm')
    ])
    return next((p for p in candidates if os.path.exists(p)), '')


class PathFollowerNode(Node):

    WHEELBASE = 2.124   # metres (from model.xacro: front -0.4011, rear +1.7234)
    MAX_STEER = 0.50    # rad (~28.6°), within the 30° physical joint limit

    # Vehicle geometry relative to base_link (in vehicle frame where front is -Y)
    FRONT_AXLE_OFFSET = 0.4011  # metres forward of base_link
    REAR_AXLE_OFFSET  = 1.7234  # metres behind base_link
    HALF_TRACK        = 0.389   # metres (half of wheel track width)

    # Road boundary thresholds (road width = 6.0m, half-width = 3.0m)
    WHEEL_WARN_DIST      = 2.40  # wheel is 60cm from asphalt edge: slight speed penalty
    WHEEL_HARD_DIST      = 2.75  # wheel is 25cm from asphalt edge: proportional steer centering blend
    WHEEL_EMERGENCY_DIST = 3.10  # wheel at edge of curb
    WHEEL_CRITICAL_DIST  = 3.50  # wheel fell off curb (>3.5m from center): snap back to road centerline

    def __init__(self):
        super().__init__('path_follower_node')

        from rcl_interfaces.msg import ParameterDescriptor
        dyn_desc = ParameterDescriptor(dynamic_typing=True)

        # ── Parameters ────────────────────────────────────────────────────────
        # Doubled speed requirements: 2.40 m/s cruise (from 1.20 m/s), up to 3.00 m/s
        self.declare_parameter('cruise_speed',     2.40, dyn_desc)  # m/s
        self.declare_parameter('max_speed',        3.00, dyn_desc)  # m/s
        self.declare_parameter('min_speed',        0.80, dyn_desc)  # m/s (corners)
        self.declare_parameter('gate_radius',      1.30, dyn_desc)  # m — gate advance
        self.declare_parameter('goal_tolerance',   1.80, dyn_desc)  # m — arrival
        self.declare_parameter('lookahead',        3.00, dyn_desc)  # m — straightaways at 2.4m/s
        self.declare_parameter('corner_lookahead', 2.10, dyn_desc)  # m — accurate corner arc tracking
        self.declare_parameter('control_rate_hz',  20.0, dyn_desc)

        self.v_cruise         = float(self.get_parameter('cruise_speed').value)
        self.v_max            = float(self.get_parameter('max_speed').value)
        self.v_min            = float(self.get_parameter('min_speed').value)
        self.gate_radius      = float(self.get_parameter('gate_radius').value)
        self.goal_tol         = float(self.get_parameter('goal_tolerance').value)
        self.lookahead        = float(self.get_parameter('lookahead').value)
        self.corner_lookahead = float(self.get_parameter('corner_lookahead').value)
        hz                    = float(self.get_parameter('control_rate_hz').value)

        # ── Load campus road map for boundary enforcement ──────────────────────
        self.osm_path = _find_default_osm()
        self.campus = None
        self._road_segments = []
        if self.osm_path:
            try:
                self.campus = CampusMap(self.osm_path)
                self._build_road_segments()
                self.get_logger().info(
                    f"Road boundary enforcement ACTIVE — {len(self._road_segments)} road segments loaded.")
            except Exception as e:
                self.get_logger().warn(f"Could not load road map for boundary enforcement: {e}")

        # ── State ──────────────────────────────────────────────────────────────
        self.path_gz: list = []       # [(x, y, speed), ...] in Gazebo coords
        self.active_idx: int = 0      # current waypoint index
        self.vx: float = None         # vehicle Gazebo X (base_link)
        self.vy: float = None         # vehicle Gazebo Y (base_link)
        self.yaw: float = 0.0         # vehicle PHYSICAL heading (rad) — corrected
        self.following: bool = False
        self.arrived: bool = False
        self._got_gz_world_pose: bool = False
        self._last_snap_time: float = 0.0

        # State machine: 'FORWARD' or 'RECOVERY_REVERSE'
        self.mode: str = 'FORWARD'
        self.recovery_start_time: float = 0.0
        self.recovery_start_x: float = 0.0
        self.recovery_start_y: float = 0.0
        self.recovery_exit_time: float = 0.0
        self.recovery_cooldown_until: float = 0.0

        # ── Publishers ─────────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/campus/pursuit_status', 10)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(Path,        '/campus/target_path',  self.path_callback,        10)
        self.create_subscription(TFMessage,   '/campus/gazebo_poses', self.gazebo_pose_callback, 10)
        self.create_subscription(Odometry,    '/model/saye/odometry', self.odom_callback,        10)
        self.create_subscription(PoseStamped, '/campus/vehicle_pose', self.pose_callback,        10)

        self.timer = self.create_timer(1.0 / hz, self.control_loop)

        self.get_logger().info(
            f"Pure Pursuit follower active | cruise={self.v_cruise}m/s | "
            f"lookahead={self.lookahead}m (corners={self.corner_lookahead}m) | {hz}Hz")

    # ── Road segment distance checks ──────────────────────────────────────────

    def _build_road_segments(self):
        """Pre-compute road segments as (x1, y1, x2, y2) in nav coords for fast distance checks."""
        if not self.campus:
            return
        for way in self.campus.road_ways:
            for i in range(len(way) - 1):
                u, v = way[i], way[i + 1]
                if u in self.campus.xy_nodes and v in self.campus.xy_nodes:
                    x1, y1 = self.campus.xy_nodes[u]
                    x2, y2 = self.campus.xy_nodes[v]
                    self._road_segments.append((x1, y1, x2, y2))

    def _distance_to_road(self, nav_x, nav_y):
        """Compute the minimum distance from (nav_x, nav_y) to the nearest road segment."""
        if not self._road_segments:
            return 0.0, nav_x, nav_y

        best_dist = float('inf')
        best_px, best_py = nav_x, nav_y

        for x1, y1, x2, y2 in self._road_segments:
            dx = x2 - x1
            dy = y2 - y1
            L2 = dx * dx + dy * dy
            if L2 < 1e-4:
                continue
            t = max(0.0, min(1.0, ((nav_x - x1) * dx + (nav_y - y1) * dy) / L2))
            px = x1 + t * dx
            py = y1 + t * dy
            d = math.hypot(nav_x - px, nav_y - py)
            if d < best_dist:
                best_dist = d
                best_px = px
                best_py = py

        return best_dist, best_px, best_py

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn("Path has < 2 poses — ignoring.")
            return

        self.path_gz = [
            (p.pose.position.x, p.pose.position.y, max(self.v_min, p.pose.position.z))
            for p in msg.poses
        ]
        self.active_idx = 0
        self.mode = 'FORWARD'

        if self.vx is not None:
            # Find closest waypoint ahead of rear axle
            x_rear = self.vx - self.REAR_AXLE_OFFSET * math.cos(self.yaw)
            y_rear = self.vy - self.REAR_AXLE_OFFSET * math.sin(self.yaw)
            self._realign_active_waypoint(x_rear, y_rear)
            self.following = True
        else:
            self.following = False

        self.arrived = False
        n = len(self.path_gz)
        self.get_logger().info(
            f"Path received: {n} waypoints. Start gate: {self.active_idx}. Following: {self.following}")

    def gazebo_pose_callback(self, msg: TFMessage):
        """Receive ground-truth Gazebo world pose."""
        for i, t in enumerate(msg.transforms):
            cid = t.child_frame_id.strip('/')
            if cid == 'saye' or cid.endswith('saye') or (not cid and i == 0):
                self._got_gz_world_pose = True
                self.vx = t.transform.translation.x
                self.vy = t.transform.translation.y
                # Saye front is -Y in base_link frame (front knuckle y=-0.40, rear wheel y=+1.72)
                yaw_model = yaw_from_quaternion(t.transform.rotation)
                self.yaw = _normalize(yaw_model - math.pi / 2.0)
                self._activate_if_waiting()
                break

    def odom_callback(self, msg: Odometry):
        if self._got_gz_world_pose:
            return
        p = msg.pose.pose.position
        self.vx = p.x
        self.vy = p.y
        yaw_model = yaw_from_quaternion(msg.pose.pose.orientation)
        self.yaw = _normalize(yaw_model - math.pi / 2.0)
        self._activate_if_waiting()

    def pose_callback(self, msg: PoseStamped):
        if self._got_gz_world_pose:
            return
        nav_x = msg.pose.position.x
        nav_y = msg.pose.position.y
        self.vx, self.vy = nav_to_gz(nav_x, nav_y)
        nav_yaw = yaw_from_quaternion(msg.pose.orientation)
        self.yaw = yaw_nav_to_gz(nav_yaw)
        self._activate_if_waiting()

    def _activate_if_waiting(self):
        if self.path_gz and not self.following and not self.arrived:
            x_rear = self.vx - self.REAR_AXLE_OFFSET * math.cos(self.yaw)
            y_rear = self.vy - self.REAR_AXLE_OFFSET * math.sin(self.yaw)
            self._realign_active_waypoint(x_rear, y_rear)
            self.following = True
            self.get_logger().info(
                f"Pose available — starting path tracking from gate {self.active_idx}.")

    def _realign_active_waypoint(self, x_rear, y_rear):
        """Find the closest waypoint on the path near the vehicle."""
        if not self.path_gz:
            return
        n = len(self.path_gz)
        best_d = float('inf')
        best_i = self.active_idx
        search_start = max(0, self.active_idx - 80)
        search_end = min(n, self.active_idx + 40)
        for i in range(search_start, search_end):
            px, py, _ = self.path_gz[i]
            d = math.hypot(px - x_rear, py - y_rear)
            if d < best_d:
                best_d = d
                best_i = i
        if best_d > 6.0:
            for i in range(n):
                px, py, _ = self.path_gz[i]
                d = math.hypot(px - x_rear, py - y_rear)
                if d < best_d:
                    best_d = d
                    best_i = i
        self.active_idx = best_i

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.following or self.arrived:
            return
        if self.vx is None or not self.path_gz:
            return

        vx, vy, yaw = self.vx, self.vy, self.yaw
        n = len(self.path_gz)

        # ── 1. Calculate positions of front axle, rear axle, and all 4 wheels ─
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Front & rear axle centers
        x_front = vx + self.FRONT_AXLE_OFFSET * cos_y
        y_front = vy + self.FRONT_AXLE_OFFSET * sin_y
        x_rear  = vx - self.REAR_AXLE_OFFSET  * cos_y
        y_rear  = vy - self.REAR_AXLE_OFFSET  * sin_y

        # Lateral unit vector (pointing to vehicle's left)
        lx, ly = -sin_y, cos_y

        wheels = {
            'fl': (x_front + self.HALF_TRACK * lx, y_front + self.HALF_TRACK * ly),
            'fr': (x_front - self.HALF_TRACK * lx, y_front - self.HALF_TRACK * ly),
            'rl': (x_rear  + self.HALF_TRACK * lx, y_rear  + self.HALF_TRACK * ly),
            'rr': (x_rear  - self.HALF_TRACK * lx, y_rear  - self.HALF_TRACK * ly),
        }

        # ── 2. Check arrival at goal ──────────────────────────────────────────
        gx, gy, _ = self.path_gz[-1]
        dist_to_goal = math.hypot(gx - x_rear, gy - y_rear)
        if dist_to_goal < self.goal_tol:
            self._stop()
            self.arrived = True
            self.following = False
            self._publish_status("ARRIVED")
            self.get_logger().info(f"✅ ARRIVED at destination ({dist_to_goal:.1f}m).")
            return

        # ── 3. Multi-Wheel Road Boundary Guard ─────────────────────────────────
        worst_wheel_name = 'center'
        worst_wheel_dist = 0.0
        worst_correction_steer = 0.0
        speed_penalty = 1.0

        if self.campus and self._road_segments:
            check_points = {'center': (vx, vy), **wheels}
            for name, (wx, wy) in check_points.items():
                n_x, n_y = gz_to_nav(wx, wy)
                w_dist, r_nx, r_ny = self._distance_to_road(n_x, n_y)
                if w_dist > worst_wheel_dist:
                    worst_wheel_dist = w_dist
                    worst_wheel_name = name
                    # Correction steering towards nearest road point
                    r_gx, r_gy = nav_to_gz(r_nx, r_ny)
                    ang_to_road = math.atan2(r_gy - wy, r_gx - wx)
                    worst_correction_steer = _normalize(ang_to_road - yaw)

            # Instant snap-back failsafe if any wheel fell off the curb (> 3.3m)
            if worst_wheel_dist > self.WHEEL_CRITICAL_DIST:
                now_sec = time.time()
                if now_sec - self._last_snap_time > 2.0:
                    self._last_snap_time = now_sec
                    self.get_logger().error(
                        f"🚫 Wheel {worst_wheel_name} off road ({worst_wheel_dist:.2f}m)! Snapping back to road centerline!")
                    from campus_nav.teleport_buggy import find_parallel_road_yaw, teleport_gz
                    nav_cx, nav_cy = gz_to_nav(vx, vy)
                    _, r_cx, r_cy = self._distance_to_road(nav_cx, nav_cy)
                    safe_gx, safe_gy = nav_to_gz(r_cx, r_cy)
                    safe_yaw, _, _ = find_parallel_road_yaw(safe_gx, safe_gy)
                    teleport_gz(safe_gx, safe_gy, yaw=safe_yaw)
                    self.mode = 'FORWARD'
                    return

            elif worst_wheel_dist > self.WHEEL_EMERGENCY_DIST:
                speed_penalty = 0.20
            elif worst_wheel_dist > self.WHEEL_HARD_DIST:
                speed_penalty = 0.40
            elif worst_wheel_dist > self.WHEEL_WARN_DIST:
                t = (worst_wheel_dist - self.WHEEL_WARN_DIST) / (self.WHEEL_HARD_DIST - self.WHEEL_WARN_DIST)
                speed_penalty = 1.0 - 0.3 * t

        # ── 4. Determine Lane Orientation and Orientation Error ───────────────
        idx_forward = min(n - 1, self.active_idx + 3)
        if idx_forward > self.active_idx:
            dx_lane = self.path_gz[idx_forward][0] - self.path_gz[self.active_idx][0]
            dy_lane = self.path_gz[idx_forward][1] - self.path_gz[self.active_idx][1]
        elif self.active_idx > 0:
            dx_lane = self.path_gz[self.active_idx][0] - self.path_gz[self.active_idx - 1][0]
            dy_lane = self.path_gz[self.active_idx][1] - self.path_gz[self.active_idx - 1][1]
        else:
            dx_lane, dy_lane = 1.0, 0.0

        psi_lane = math.atan2(dy_lane, dx_lane)
        # Orientation error relative to lane (-pi to pi)
        # Positive: vehicle heading is rotated counter-clockwise (to the left) of lane
        # Negative: vehicle heading is rotated clockwise (to the right) of lane
        orientation_error = _normalize(yaw - psi_lane)
        orient_err_deg = math.degrees(orientation_error)

        # ── 5. Wrong Orientation Detection & Auto-Correction State Machine ────
        now_time = time.time()
        if self.mode == 'FORWARD':
            in_cooldown = now_time < self.recovery_cooldown_until
            # Do not re-trigger reverse during cooldown (prevents forward-reverse chatter loops)
            if not in_cooldown:
                # Normal 90-degree intersections produce natural heading differences
                # up to 80-90° between vehicle heading and upcoming path tangents.
                # Auto-correction reverse is an emergency maneuver for when the vehicle is ACTUALLY
                # driving off the road (wheel near curb > 2.85m AND pointed outward) or spun backwards (> 125°).
                is_heading_offroad = (worst_wheel_dist > 2.85) and (abs(orientation_error) > math.radians(50.0))
                is_facing_backwards = abs(orientation_error) > math.radians(125.0)

                if is_heading_offroad or is_facing_backwards:
                    self.mode = 'RECOVERY_REVERSE'
                    self.recovery_start_time = now_time
                    self.recovery_start_x = vx
                    self.recovery_start_y = vy
                    self.get_logger().warn(
                        f"⚠️ OFF-ROAD DRIFT DETECTED ({orient_err_deg:.1f}° to lane, wheel={worst_wheel_name} {worst_wheel_dist:.2f}m)! "
                        f"Executing auto-correction reverse maneuver (min 2.5m)..."
                    )

        if self.mode == 'RECOVERY_REVERSE':
            elapsed_rec = now_time - self.recovery_start_time
            dist_reversed = math.hypot(vx - self.recovery_start_x, vy - self.recovery_start_y)

            # Rewind active waypoint continuously while reversing along assigned path
            self._realign_active_waypoint(x_rear, y_rear)

            # Exit criteria for SUFFICIENT REVERSE:
            # 1. Must take SUFFICIENT reverse: at least 2.5 meters back OR at least 2.8s
            # 2. Vehicle must be well-aligned parallel to lane (|orientation_error| < 15°)
            # 3. All wheels must be safely inside the road (worst_wheel_dist < 1.70m)
            # 4. Or safety timeout (5.5s)
            sufficient_reverse = (dist_reversed >= 2.5 and elapsed_rec >= 2.2) or (elapsed_rec >= 3.8)
            well_aligned = abs(orientation_error) < math.radians(15.0)
            wheels_safe = worst_wheel_dist < 1.70
            timed_out = elapsed_rec > 5.5

            if (sufficient_reverse and well_aligned and wheels_safe) or timed_out:
                self.mode = 'FORWARD'
                self.recovery_exit_time = now_time
                self.recovery_cooldown_until = now_time + 4.0  # 4.0s forward tracking grace period
                self.get_logger().info(
                    f"✅ SUFFICIENT REVERSE COMPLETED: reversed {dist_reversed:.2f}m in {elapsed_rec:.1f}s "
                    f"(orient err: {orient_err_deg:.1f}°). Resuming forward lane tracking along assigned path."
                )
                self._stop()
                self._realign_active_waypoint(x_rear, y_rear)
                return
            else:
                # Decisive controlled reverse speed
                rev_speed = -0.85  # m/s

                # Smooth proportional counter-steering to align heading with lane:
                rev_steer = max(-self.MAX_STEER, min(self.MAX_STEER, 2.0 * orientation_error))
                angular_z = (math.tan(rev_steer) / self.WHEELBASE) * rev_speed

                twist = Twist()
                twist.linear.x = rev_speed
                twist.angular.z = angular_z
                self.cmd_pub.publish(twist)

                self._publish_status(
                    f"REVERSING ({dist_reversed:.1f}m/2.5m, {elapsed_rec:.1f}s) | "
                    f"Err:{orient_err_deg:.1f}° | Steer:{math.degrees(rev_steer):.1f}°"
                )
                return

        # ── 6. Normal Forward Tracking: Advance Active Waypoint ───────────────
        while self.active_idx < n - 1:
            px, py, _ = self.path_gz[self.active_idx]
            if math.hypot(px - x_rear, py - y_rear) < self.gate_radius:
                self.active_idx += 1
            else:
                break

        # ── 7. Adaptive Lookahead ─────────────────────────────────────────────
        effective_lookahead = self.lookahead
        look_steps = min(8, n - 1 - self.active_idx)
        if look_steps > 1:
            p_now = self.path_gz[self.active_idx]
            p_future = self.path_gz[self.active_idx + look_steps]
            heading_now = math.atan2(
                self.path_gz[self.active_idx + 1][1] - p_now[1],
                self.path_gz[self.active_idx + 1][0] - p_now[0]
            )
            heading_future = math.atan2(
                p_future[1] - self.path_gz[self.active_idx + look_steps - 1][1],
                p_future[0] - self.path_gz[self.active_idx + look_steps - 1][0]
            )
            heading_delta = abs(_normalize(heading_future - heading_now))
            if heading_delta > 0.22:  # Curve / intersection ahead
                effective_lookahead = self.corner_lookahead

        # Find lookahead target from rear axle
        target_idx = self.active_idx
        for j in range(self.active_idx, n - 1):
            px, py, _ = self.path_gz[j]
            d = math.hypot(px - x_rear, py - y_rear)
            if d >= effective_lookahead:
                target_idx = j
                break
        else:
            target_idx = n - 1

        tx, ty, target_spd = self.path_gz[target_idx]

        # ── 8. Pure Pursuit + Restorative Heading Alignment Steering ──────────
        dx = tx - x_rear
        dy = ty - y_rear
        L_target = max(0.5, math.hypot(dx, dy))

        alpha = _normalize(math.atan2(dy, dx) - yaw)
        curvature = 2.0 * math.sin(alpha) / L_target
        pp_steer = math.atan(curvature * self.WHEELBASE)

        # Restorative heading alignment: gentle centering on straightaways
        # In sharp turns, Pure Pursuit governs the arc without early 90-degree steering saturation
        lane_heading_diff = _normalize(psi_lane - yaw)
        if abs(lane_heading_diff) < math.radians(25.0):
            steer_angle = pp_steer + 0.15 * lane_heading_diff
        else:
            steer_angle = pp_steer

        # ── 9. Active Edge Guard Shield ───────────────────────────────────────
        if worst_wheel_dist > self.WHEEL_HARD_DIST and worst_correction_steer != 0.0:
            # Smooth proportional blend back towards road center to prevent curb crossing
            blend = min(0.60, (worst_wheel_dist - self.WHEEL_HARD_DIST) / (self.WHEEL_EMERGENCY_DIST - self.WHEEL_HARD_DIST))
            steer_angle = (1.0 - blend) * steer_angle + blend * worst_correction_steer

        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

        # ── 10. Speed Calculation with Orientation Inhabitation ───────────────
        # 1. Base cruise speed with steering reduction
        steer_ratio = abs(steer_angle) / self.MAX_STEER
        speed = self.v_cruise * (1.0 - 0.45 * steer_ratio)
        speed = max(self.v_min, min(self.v_max, speed))

        # 2. Curvature preview speed limit from trajectory
        speed = min(speed, target_spd)

        # 3. Corner speed control: slow down gracefully for sharp turns, NEVER stall
        # Always maintains positive drive to power through turns
        orient_factor = max(0.45, math.cos(orientation_error))
        speed = max(self.v_min, speed * orient_factor)

        # 4. Multi-wheel edge penalty
        speed *= speed_penalty

        # 5. Post-recovery forward transition: cap speed gently for the first 2s after recovery
        if now_time < self.recovery_cooldown_until:
            time_since_exit = now_time - self.recovery_exit_time
            if time_since_exit < 2.0:
                speed = min(speed, 1.25)

        # ── 11. Publish /cmd_vel ──────────────────────────────────────────────
        angular_z = (math.tan(steer_angle) / self.WHEELBASE) * speed
        twist = Twist()
        twist.linear.x  = speed
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

        # ── 12. Status ────────────────────────────────────────────────────────
        pct = 100.0 * self.active_idx / max(1, n - 1)
        self._publish_status(
            f"Gate {self.active_idx}/{n-1} ({pct:.0f}%) | "
            f"Spd:{speed:.2f}m/s | OrientErr:{orient_err_deg:.1f}° | Steer:{math.degrees(steer_angle):.1f}° | "
            f"Edge:{worst_wheel_name}:{worst_wheel_dist:.2f}m"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _publish_status(self, text: str):
        msg = String()
        msg.data = f"[follower] {text}"
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop()
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
