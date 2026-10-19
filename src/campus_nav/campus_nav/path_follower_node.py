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
from std_msgs.msg import String, Float32
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
    HALF_TRACK        = 0.389   # metres (half of wheel track width to hub center)
    TIRE_WIDTH        = 0.250   # metres (physical tire width)
    TIRE_OUTER_TRACK  = HALF_TRACK + (TIRE_WIDTH / 2.0)  # 0.514m to outer tire contact shoulder

    def __init__(self):
        super().__init__('path_follower_node')

        from rcl_interfaces.msg import ParameterDescriptor
        dyn_desc = ParameterDescriptor(dynamic_typing=True)

        # ── Parameters ────────────────────────────────────────────────────────
        # Cruise at 2.40 m/s, up to 3.00 m/s
        self.declare_parameter('cruise_speed',        2.40, dyn_desc)  # m/s
        self.declare_parameter('max_speed',           3.00, dyn_desc)  # m/s
        self.declare_parameter('min_speed',           0.80, dyn_desc)  # m/s (corners)
        self.declare_parameter('gate_radius',         1.30, dyn_desc)  # m — gate advance
        self.declare_parameter('goal_tolerance',      1.80, dyn_desc)  # m — arrival
        self.declare_parameter('lookahead',           3.00, dyn_desc)  # m — straightaways at 2.4m/s
        self.corner_lookahead = 2.10
        self.declare_parameter('corner_lookahead',    2.10, dyn_desc)  # m — accurate corner arc tracking
        self.declare_parameter('safe_edge_distance',  0.50, dyn_desc)  # m — safe margin to road curb
        self.declare_parameter('road_half_width',     1.85, dyn_desc)  # m — physical IIT Ropar road half-width (3.7m total)
        self.declare_parameter('control_rate_hz',     20.0, dyn_desc)

        self.v_cruise         = float(self.get_parameter('cruise_speed').value)
        self.v_max            = float(self.get_parameter('max_speed').value)
        self.v_min            = float(self.get_parameter('min_speed').value)
        self.gate_radius      = float(self.get_parameter('gate_radius').value)
        self.goal_tol         = float(self.get_parameter('goal_tolerance').value)
        self.lookahead        = float(self.get_parameter('lookahead').value)
        self.corner_lookahead = float(self.get_parameter('corner_lookahead').value)
        self.safe_edge_dist   = float(self.get_parameter('safe_edge_distance').value)
        self.road_half_width  = float(self.get_parameter('road_half_width').value)
        hz                    = float(self.get_parameter('control_rate_hz').value)

        # Multi-level edge thresholds (in terms of distance from road centerline to outer tire)
        # road_half_width is 1.85m. Outer tire centered is at 0.514m.
        self.WHEEL_BUFFER_DIST      = self.road_half_width - (self.safe_edge_dist + 0.15)  # ~1.20m: soft buffer begins
        self.WHEEL_CORRECT_DIST     = self.road_half_width - self.safe_edge_dist           # ~1.35m: active edge repulsion starts
        self.WHEEL_HARD_DIST        = self.road_half_width - 0.25                          # ~1.60m: dominant restorative steering
        self.WHEEL_EMERGENCY_DIST   = self.road_half_width - 0.10                          # ~1.75m: emergency counter-steer away from edge
        self.WHEEL_CRITICAL_DIST    = self.road_half_width + 0.15                          # ~2.00m: outer tire breached curb failsafe snapback

        # Edge guard emergency thresholds (in terms of clearance from asphalt edge)
        self.EDGE_EMERGENCY_CLEARANCE = -0.20  # m: only trigger failsafe snap if wheel breached deep into grass (< -0.20m)
        self.EDGE_DANGER_CLEARANCE    = 0.30   # m: below this → hard edge-guard override + crawl speed 0.35 m/s
        self.EDGE_WARNING_CLEARANCE   = 0.55   # m: below this → dominant edge guard, speed cap

        # ── Load campus road map for boundary enforcement ──────────────────────
        self.osm_path = _find_default_osm()
        self.campus = None
        self._road_segments = []
        if self.osm_path:
            try:
                self.campus = CampusMap(self.osm_path)
                self._build_road_segments()
                self.get_logger().info(
                    f"Road boundary enforcement ACTIVE — {len(self._road_segments)} road segments loaded. Safe edge clearance: {self.safe_edge_dist}m")
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

        # Camera visual edge detection feedback
        self.camera_lane_bias: float = 0.0
        self.last_camera_bias_time: float = 0.0
        self.visual_edge_dist: float = 999.0

        # Traction loss & stuck watchdog
        self._last_stuck_check_time: float = 0.0
        self._last_stuck_x: float = 0.0
        self._last_stuck_y: float = 0.0
        self._stuck_seconds: float = 0.0

        # ── Publishers ─────────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/campus/pursuit_status', 10)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(Path,        '/campus/target_path',  self.path_callback,        10)
        self.create_subscription(TFMessage,   '/campus/gazebo_poses', self.gazebo_pose_callback, 10)
        self.create_subscription(Odometry,    '/model/saye/odometry', self.odom_callback,        10)
        self.create_subscription(PoseStamped, '/campus/vehicle_pose', self.pose_callback,        10)
        self.create_subscription(Float32,     '/campus/lane_correction', self.camera_correction_callback, 10)
        self.create_subscription(Float32,     '/campus/visual_edge_dist', self.visual_edge_dist_callback, 10)

        self.timer = self.create_timer(1.0 / hz, self.control_loop)

        self.get_logger().info(
            f"Pure Pursuit follower active | cruise={self.v_cruise}m/s | "
            f"lookahead={self.lookahead}m (corners={self.corner_lookahead}m) | edge_buffer={self.safe_edge_dist}m | {hz}Hz")

    def _now(self) -> float:
        """Return current simulation time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def camera_correction_callback(self, msg: Float32):
        self.camera_lane_bias = msg.data
        self.last_camera_bias_time = self._now()

    def visual_edge_dist_callback(self, msg: Float32):
        self.visual_edge_dist = msg.data

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

    def _distance_to_active_route_gz(self, gx: float, gy: float):
        """
        Compute distance and closest point from (gx, gy) in Gazebo world coords
        to the local active route waypoints [active_idx - 3 : active_idx + 8].
        Prevents snapping to perpendicular crossroads at intersections.
        """
        if not self.path_gz:
            return None, None, None
        n = len(self.path_gz)
        if n < 2:
            return None, None, None

        start_i = max(0, self.active_idx - 3)
        end_i = min(n - 1, self.active_idx + 8)

        best_dist = float('inf')
        best_px, best_py = gx, gy

        for i in range(start_i, end_i):
            x1, y1, _ = self.path_gz[i]
            x2, y2, _ = self.path_gz[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            L2 = dx * dx + dy * dy
            if L2 < 1e-4:
                continue
            t = max(0.0, min(1.0, ((gx - x1) * dx + (gy - y1) * dy) / L2))
            px = x1 + t * dx
            py = y1 + t * dy
            d = math.hypot(gx - px, gy - py)
            if d < best_dist:
                best_dist = d
                best_px = px
                best_py = py

        return best_dist, best_px, best_py

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

    def _teleport_to_waypoint(self, waypoint_idx: int):
        """Teleport buggy directly onto the route centerline with correct forward alignment."""
        if not self.path_gz:
            return
        from campus_nav.teleport_buggy import teleport_gz
        n = len(self.path_gz)
        idx = max(0, min(n - 1, waypoint_idx))
        gx, gy, _ = self.path_gz[idx]
        next_idx = min(n - 1, idx + 2)
        if next_idx > idx:
            dx = self.path_gz[next_idx][0] - gx
            dy = self.path_gz[next_idx][1] - gy
            lane_heading = math.atan2(dy, dx)
        else:
            lane_heading = self.yaw
        # Saye physical front is along lane_heading. Gazebo model_yaw = lane_heading + pi/2
        model_yaw = _normalize(lane_heading + math.pi / 2.0)
        teleport_gz(gx, gy, yaw=model_yaw)
        now_time = self._now()
        self.mode = 'FORWARD'
        self.recovery_exit_time = now_time
        self.recovery_cooldown_until = now_time + 4.0
        self._stuck_seconds = 0.0
        self._last_stuck_check_time = now_time
        self._last_stuck_x = gx
        self._last_stuck_y = gy
        self._last_snap_time = now_time
        self._stop()
        self._realign_active_waypoint(gx, gy)

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
            'fl': (x_front + self.TIRE_OUTER_TRACK * lx, y_front + self.TIRE_OUTER_TRACK * ly),
            'fr': (x_front - self.TIRE_OUTER_TRACK * lx, y_front - self.TIRE_OUTER_TRACK * ly),
            'rl': (x_rear  + self.TIRE_OUTER_TRACK * lx, y_rear  + self.TIRE_OUTER_TRACK * ly),
            'rr': (x_rear  - self.TIRE_OUTER_TRACK * lx, y_rear  - self.TIRE_OUTER_TRACK * ly),
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

        check_points = {'center': (vx, vy), **wheels}
        for name, (wx, wy) in check_points.items():
            w_dist, r_gx, r_gy = self._distance_to_active_route_gz(wx, wy)
            if w_dist is None and self.campus and self._road_segments:
                n_x, n_y = gz_to_nav(wx, wy)
                w_dist, r_nx, r_ny = self._distance_to_road(n_x, n_y)
                r_gx, r_gy = nav_to_gz(r_nx, r_ny)

            if w_dist is not None and w_dist > worst_wheel_dist:
                worst_wheel_dist = w_dist
                worst_wheel_name = name
                # Correction steering towards nearest road centerline point (mapped to Ackermann angle)
                ang_to_road = math.atan2(r_gy - wy, r_gx - wx)
                alpha_road = _normalize(ang_to_road - yaw)
                delta_road = math.atan(2.0 * self.WHEELBASE * math.sin(alpha_road) / max(1.0, self.lookahead))
                worst_correction_steer = max(-self.MAX_STEER, min(self.MAX_STEER, delta_road))

        # Clearance from road asphalt edge: distance = road_half_width - wheel_distance_to_centerline
        worst_edge_clearance = max(0.0, self.road_half_width - worst_wheel_dist)

        # Instant snap-back failsafe if any wheel fell off the curb (> WHEEL_CRITICAL_DIST)
        if worst_wheel_dist > self.WHEEL_CRITICAL_DIST and self.path_gz:
            now_sec = self._now()
            if now_sec - self._last_snap_time > 2.0:
                self.get_logger().error(
                    f"🚫 Wheel {worst_wheel_name} off road ({worst_wheel_dist:.2f}m, clearance {worst_edge_clearance:.2f}m)! Snapping back to route centerline!")
                self._teleport_to_waypoint(self.active_idx)
                return

        # Proactive speed penalty as road edge approaches (< safe_edge_dist = 0.50m)
        if worst_edge_clearance < self.safe_edge_dist:
            urgency = (self.safe_edge_dist - worst_edge_clearance) / max(0.1, self.safe_edge_dist)
            speed_penalty = max(0.35, 1.0 - 0.65 * urgency)

        # ── 4. Sequential Segment Progress Tracking ───────────────────────────
        # Evaluates progress strictly along the active segment [active_idx, active_idx + 1].
        # In a single 50ms cycle at 2.4 m/s, the buggy moves at most 0.12m.
        # Advancing at most +1 (or +2 when catching up) guarantees the buggy traverses
        # every single waypoint of the corner arc in order, preventing corner-skipping,
        # orientation error spikes, and corner freezes.
        advance_count = 0
        while self.active_idx < n - 1 and advance_count < 2:
            p1 = self.path_gz[self.active_idx]
            p2 = self.path_gz[self.active_idx + 1]
            dx_seg = p2[0] - p1[0]
            dy_seg = p2[1] - p1[1]
            L2 = dx_seg * dx_seg + dy_seg * dy_seg
            if L2 < 1e-4:
                self.active_idx += 1
                advance_count += 1
                continue

            # Normalized projection along segment for rear axle and vehicle center
            t_rear = ((x_rear - p1[0]) * dx_seg + (y_rear - p1[1]) * dy_seg) / L2
            t_center = ((vx - p1[0]) * dx_seg + (vy - p1[1]) * dy_seg) / L2
            d_p1 = math.hypot(x_rear - p1[0], y_rear - p1[1])
            d_p2 = math.hypot(x_rear - p2[0], y_rear - p2[1])

            # Advance if rear axle has progressed along the segment or vehicle is closer to p2
            if t_rear > 0.40 or t_center > 0.70 or (d_p2 < d_p1 and t_rear > 0.05):
                self.active_idx += 1
                advance_count += 1
            else:
                break

        # ── 5. Determine Lane Orientation and Heading Preview ─────────────────
        idx_forward = min(n - 1, self.active_idx + 2)
        if idx_forward > self.active_idx:
            dx_lane = self.path_gz[idx_forward][0] - self.path_gz[self.active_idx][0]
            dy_lane = self.path_gz[idx_forward][1] - self.path_gz[self.active_idx][1]
        elif self.active_idx > 0:
            dx_lane = self.path_gz[self.active_idx][0] - self.path_gz[self.active_idx - 1][0]
            dy_lane = self.path_gz[self.active_idx][1] - self.path_gz[self.active_idx - 1][1]
        else:
            dx_lane, dy_lane = 1.0, 0.0

        psi_lane = math.atan2(dy_lane, dx_lane)
        orientation_error = _normalize(yaw - psi_lane)
        orient_err_deg = math.degrees(orientation_error)

        # Curvature preview / upcoming turn detection
        look_steps = min(8, n - 1 - self.active_idx)
        heading_delta = 0.0
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
            heading_delta = _normalize(heading_future - heading_now)

        is_actively_turning = (abs(heading_delta) > 0.15) or (abs(orientation_error) > math.radians(25.0))

        # ── 5. Wrong Orientation & Traction-Loss Watchdog State Machine ───────
        now_time = self._now()
        if self.mode == 'FORWARD':
            # Traction loss & stuck watchdog: detect if buggy is stopped while commanding speed
            if self._last_stuck_check_time == 0.0:
                self._last_stuck_check_time = now_time
                self._last_stuck_x = vx
                self._last_stuck_y = vy

            in_cooldown = now_time < self.recovery_cooldown_until
            dt_stuck = now_time - self._last_stuck_check_time
            if dt_stuck >= 1.0:
                dist_moved = math.hypot(vx - self._last_stuck_x, vy - self._last_stuck_y)
                self._last_stuck_check_time = now_time
                self._last_stuck_x = vx
                self._last_stuck_y = vy

                if not in_cooldown and not is_actively_turning and dist_moved < 0.10 and abs(self.v_cruise) > 0.5:
                    self._stuck_seconds += dt_stuck
                    if self._stuck_seconds >= 2.5:
                        self.get_logger().warn(
                            f"⚠️ BUGGY TRACTION OBSTRUCTION DETECTED (moved only {dist_moved:.2f}m in 2.5s)! "
                            f"Executing auto-recovery reverse maneuver to clear road edge obstruction..."
                        )
                        self.mode = 'RECOVERY_REVERSE'
                        self.recovery_start_time = now_time
                        self.recovery_start_x = vx
                        self.recovery_start_y = vy
                        self._stuck_seconds = 0.0
                else:
                    self._stuck_seconds = 0.0
            # Do not trigger reverse during cooldown or while actively negotiating a turn
            if not in_cooldown and not is_actively_turning:
                is_heading_offroad = (worst_edge_clearance < 0.25) and (abs(orientation_error) > math.radians(45.0))
                is_facing_backwards = abs(orientation_error) > math.radians(145.0)

                if is_heading_offroad or is_facing_backwards:
                    self.mode = 'RECOVERY_REVERSE'
                    self.recovery_start_time = now_time
                    self.recovery_start_x = vx
                    self.recovery_start_y = vy
                    self.get_logger().warn(
                        f"⚠️ OFF-ROAD DRIFT DETECTED ({orient_err_deg:.1f}° to lane, wheel={worst_wheel_name} clearance={worst_edge_clearance:.2f}m)! "
                        f"Executing auto-correction reverse maneuver..."
                    )

            # ── 5b. EMERGENCY EDGE FAILSAFE: Snap only if deeply off-road with cooldown ───
            if worst_edge_clearance < self.EDGE_EMERGENCY_CLEARANCE and worst_edge_clearance < 900.0 and self.path_gz:
                if (now_time - self._last_snap_time) > 4.0:
                    self._last_snap_time = now_time
                    self.get_logger().error(
                        f"🚫 EDGE EMERGENCY: wheel {worst_wheel_name} clearance {worst_edge_clearance:.2f}m < {self.EDGE_EMERGENCY_CLEARANCE}m! "
                        f"Snapping to route centerline."
                    )
                    self._teleport_to_waypoint(self.active_idx)
                    return

        if self.mode == 'RECOVERY_REVERSE':
            elapsed_rec = now_time - self.recovery_start_time
            dist_reversed = math.hypot(vx - self.recovery_start_x, vy - self.recovery_start_y)

            # High-centered / curb-trap failsafe: if wedged on curb and unable to reverse
            if elapsed_rec >= 2.5 and dist_reversed < 0.10 and self.path_gz:
                self.get_logger().error(
                    f"⚠️ HIGH-CENTERED CURB TRAP DETECTED (dist_reversed={dist_reversed:.2f}m in {elapsed_rec:.1f}s)! "
                    f"Snapping buggy to route centerline to clear obstruction."
                )
                self._teleport_to_waypoint(self.active_idx)
                return

            # Find reverse target waypoint behind vehicle along the road path
            rev_target_idx = max(0, self.active_idx - 5)
            rev_tx, rev_ty, _ = self.path_gz[rev_target_idx]

            # Vector from rear axle to reverse target
            dx_rev = rev_tx - x_rear
            dy_rev = rev_ty - y_rear
            L_rev = max(1.2, math.hypot(dx_rev, dy_rev))

            # Direction of rear axle heading in reverse (yaw + pi)
            rear_heading = _normalize(yaw + math.pi)
            angle_to_rev_target = math.atan2(dy_rev, dx_rev)
            alpha_rev = _normalize(angle_to_rev_target - rear_heading)

            # Ackermann reverse steering curvature
            curv_rev = 2.0 * math.sin(alpha_rev) / L_rev
            raw_rev_steer = -math.atan(curv_rev * self.WHEELBASE)
            # Gentle steering range (max 22 degrees) to prevent front bumper swing and tyre scrub
            max_rev_angle = math.radians(22.0)
            rev_steer = max(-max_rev_angle, min(max_rev_angle, raw_rev_steer))

            # Controlled reverse crawl speed (-0.65 m/s)
            rev_speed = -0.65
            angular_z = (math.tan(rev_steer) / self.WHEELBASE) * rev_speed

            # Exit criteria
            sufficient_reverse = (dist_reversed >= 1.6 and elapsed_rec >= 1.8) or (elapsed_rec >= 3.2)
            well_aligned = abs(orientation_error) < math.radians(25.0)
            wheels_safe = worst_edge_clearance > 0.35
            timed_out = elapsed_rec > 4.0

            if (sufficient_reverse and well_aligned and wheels_safe) or timed_out:
                self.mode = 'FORWARD'
                self.recovery_exit_time = now_time
                self.recovery_cooldown_until = now_time + 3.0
                self._stuck_seconds = 0.0
                self.get_logger().info(
                    f"✅ REVERSE MANEUVER COMPLETED: reversed {dist_reversed:.2f}m in {elapsed_rec:.1f}s "
                    f"(orient err: {orient_err_deg:.1f}°). Resuming forward path tracking."
                )
                self._stop()
                self._realign_active_waypoint(x_rear, y_rear)
                return
            else:
                twist = Twist()
                twist.linear.x = rev_speed
                twist.angular.z = angular_z
                self.cmd_pub.publish(twist)

                self._publish_status(
                    f"REVERSING ({dist_reversed:.1f}m/1.6m, {elapsed_rec:.1f}s) | "
                    f"Err:{orient_err_deg:.1f}° | Steer:{math.degrees(rev_steer):.1f}°"
                )
                return

        # ── 6. Adaptive Lookahead ─────────────────────────────────────────────
        # For acute / slant turns (abs(heading_delta) > 0.35 rad ~20°), reduce lookahead
        # down to 1.40m so the pure pursuit point hugs the outer late-apex arc rather
        # than chord-cutting across the corner early.
        if abs(heading_delta) > 0.35:
            turn_sev = min(1.0, (abs(heading_delta) - 0.35) / 0.65)
            effective_lookahead = max(1.40, self.corner_lookahead - 0.70 * turn_sev)
        elif abs(heading_delta) > 0.18:
            effective_lookahead = self.corner_lookahead
        else:
            effective_lookahead = self.lookahead

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

        # ── 7. Pure Pursuit + Stanley Cross-Track Steering ────────────────────
        dx = tx - x_rear
        dy = ty - y_rear
        L_target = max(0.5, math.hypot(dx, dy))

        alpha = _normalize(math.atan2(dy, dx) - yaw)
        curvature = 2.0 * math.sin(alpha) / L_target
        pp_steer = math.atan(curvature * self.WHEELBASE)

        # Front-axle lateral cross-track error to current route segment
        # Normal to the left of the lane: (-sin(psi_lane), cos(psi_lane))
        # e_fa > 0: vehicle is to the left of the lane (needs right steer -)
        e_fa = (x_front - self.path_gz[self.active_idx][0]) * (-math.sin(psi_lane)) + \
               (y_front - self.path_gz[self.active_idx][1]) * math.cos(psi_lane)

        # Stanley cross-track assistance (active centering, prevents corner understeer push)
        stanley_steer = -math.atan(0.40 * e_fa / max(0.8, abs(self.v_cruise)))
        stanley_steer = max(-0.12, min(0.12, stanley_steer))

        # Restorative heading alignment on straightaways
        lane_heading_diff = _normalize(psi_lane - yaw)
        if abs(lane_heading_diff) < math.radians(25.0) and abs(heading_delta) < 0.15:
            steer_angle = pp_steer + stanley_steer + 0.10 * lane_heading_diff
        else:
            steer_angle = pp_steer + stanley_steer

        # ── 8. Active Edge Guard Shield (Turn-Aware Defense) ───────────────────
        # CRITICAL: During intentional turns, the inside wheel naturally nears the inside curb.
        # The Edge Guard must NEVER counter-steer away from an active turn!
        # If an inside wheel approaches the curb: ease steer slightly (0.78x-0.90x) and crawl at 0.55 m/s.
        # If an outside wheel approaches the outer curb: maintain full steer into turn and crawl at 0.55 m/s.
        edge_guard_active = False
        corner_edge_crawling = False
        if worst_edge_clearance < 900.0 and worst_correction_steer != 0.0:
            is_turning = abs(pp_steer) > 0.08 or abs(heading_delta) > 0.18
            opposes_turn = is_turning and ((worst_correction_steer * pp_steer) < 0.0)

            if opposes_turn:
                # Inside wheel is close to inside curb during an acute / slant turn
                edge_guard_active = True
                corner_edge_crawling = True
                if worst_edge_clearance < 0.30:
                    steer_angle = pp_steer * 0.78
                elif worst_edge_clearance < 0.45:
                    steer_angle = pp_steer * 0.84
                else:
                    steer_angle = pp_steer * 0.90
            elif is_turning:
                # Outside wheel is approaching outer curb during a turn (understeer)
                if worst_edge_clearance < self.safe_edge_dist:
                    corner_edge_crawling = True
                    steer_angle = math.copysign(self.MAX_STEER, pp_steer)
                    edge_guard_active = True
            else:
                # Normal straightaway edge guard
                if worst_edge_clearance < self.EDGE_DANGER_CLEARANCE:
                    steer_angle = worst_correction_steer
                    edge_guard_active = True
                elif worst_edge_clearance < self.safe_edge_dist:
                    urgency = (self.safe_edge_dist - worst_edge_clearance) / max(0.1, self.safe_edge_dist - self.EDGE_DANGER_CLEARANCE)
                    blend = min(1.0, 0.80 + 0.20 * urgency)
                    steer_angle = (1.0 - blend) * steer_angle + blend * worst_correction_steer
                    edge_guard_active = True
                elif worst_edge_clearance < self.EDGE_WARNING_CLEARANCE:
                    urgency = (self.EDGE_WARNING_CLEARANCE - worst_edge_clearance) / max(0.1, self.EDGE_WARNING_CLEARANCE - self.safe_edge_dist)
                    blend = min(0.80, 0.30 + 0.50 * urgency)
                    steer_angle = (1.0 - blend) * steer_angle + blend * worst_correction_steer
                    edge_guard_active = True
                elif worst_wheel_dist > self.WHEEL_BUFFER_DIST:
                    urgency = max(0.0, min(1.0, (worst_wheel_dist - self.WHEEL_BUFFER_DIST) /
                                           max(0.1, self.WHEEL_EMERGENCY_DIST - self.WHEEL_BUFFER_DIST)))
                    blend = min(0.30, urgency * 0.30)
                    steer_angle = (1.0 - blend) * steer_angle + blend * worst_correction_steer

        # Fuse camera-based visual edge correction only on straightaways (intersections cause visual artifacts)
        if abs(heading_delta) < 0.15 and abs(pp_steer) < 0.10:
            if (self._now() - self.last_camera_bias_time < 0.6) and abs(self.camera_lane_bias) > 0.01:
                steer_angle += 0.30 * self.camera_lane_bias

        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

        # ── 10. Speed Calculation with Pre-Corner Deceleration & Edge Caps ─────
        # 1. Base cruise speed with steering reduction
        steer_ratio = abs(steer_angle) / self.MAX_STEER
        speed = self.v_cruise * (1.0 - 0.45 * steer_ratio)
        speed = max(self.v_min, min(self.v_max, speed))

        # 2. Curvature preview speed limit from trajectory
        speed = min(speed, target_spd)

        # 3. Corner speed control: smooth pre-corner deceleration for sharp turns
        if abs(heading_delta) > 0.18:
            turn_sev = min(1.0, abs(heading_delta) / 0.80)
            target_corner_speed = max(0.95, 2.0 * (1.0 - 0.52 * turn_sev))
            speed = min(speed, target_corner_speed)

        orient_factor = max(0.45, math.cos(orientation_error))
        speed = max(self.v_min, speed * orient_factor)

        # 4. Multi-wheel edge penalty
        speed *= speed_penalty

        # 5. HARD SPEED CAP based on edge proximity
        if worst_edge_clearance < 900.0:
            if worst_edge_clearance < self.EDGE_DANGER_CLEARANCE:
                speed = min(speed, 0.40)  # Crawl speed in danger zone
            elif worst_edge_clearance < self.safe_edge_dist:
                speed = min(speed, 0.75)  # Slow in active repulsion zone
            elif worst_edge_clearance < self.EDGE_WARNING_CLEARANCE:
                speed = min(speed, 1.20)  # Moderate speed in warning zone

        # 6. Preemptive speed cap: if heading diverges from lane AND near edge (< safe_edge_dist)
        if worst_edge_clearance < self.safe_edge_dist and abs(orientation_error) > math.radians(20.0):
            speed = min(speed, 0.60)

        # 7. Turn-edge crawl clamp: slow to 0.55 m/s if wheels near curb during turn
        if corner_edge_crawling:
            speed = min(speed, 0.55)

        # 8. Post-recovery forward transition: cap speed gently for the first 2s after recovery
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
            f"Edge:{worst_wheel_name}:{worst_edge_clearance:.2f}m"
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
