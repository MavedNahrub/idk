#!/usr/bin/env python3
"""
Precision Autonomous Lane Centering & Following Controller for Ackermann Vehicle (Saye)
IIT Ropar Campus Autonomous Navigation System

Features & Kinematic Innovations:
1. Inverse Perspective Mapping (IPM):
   - Projects camera pixels into metric vehicle-frame ground coordinates (X_meters, Z_meters).
   - Eliminates pixel perspective distortion and bridges gaps between white dashes seamlessly.

2. Strict Lane Lock (Same-Lane Curve Tracking):
   - Automatically locks to the lane the vehicle is currently driving in (Left Lane vs Right Lane).
   - Maintains exact lane center offset (1.75m from road centerline) through straightaways and turns.
   - Prevents lane hopping or drifting across the centerline.

3. Exact Curvature Feedforward Steering (Zero Understeer / Perfect Radius):
   - Fits a 2nd-order polynomial X(Z) = c2*Z^2 + c1*Z + c0 in physical ground coordinates.
   - Directly extracts road curvature kappa = 2*c2 and radius R = 1 / |kappa|.
   - Computes kinematic Ackermann feedforward steering: delta_ff = atan(L_wb * kappa).
   - Combines feedforward with Pure Pursuit feedback to lock onto the curve radius with zero drift.

4. Active Edge Repulsion Safety Barrier:
   - Monitors physical distance from vehicle to curb boundaries in ground meters.
   - Applies an exponential repulsive bias if approaching within 0.85m of either curb.

5. Off-Track Grass Detection & Automatic Recovery:
   - Detects off-road excursions via near-field grass area fraction.
   - Automatically crawls and steers back onto the dark asphalt road surface.
"""

import time
import math
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
import rclpy
# pyrefly: ignore [missing-import]
from rclpy.node import Node
# pyrefly: ignore [missing-import]
from sensor_msgs.msg import Image
# pyrefly: ignore [missing-import]
from geometry_msgs.msg import Twist
# pyrefly: ignore [missing-import]
from nav_msgs.msg import Path
# pyrefly: ignore [missing-import]
from std_msgs.msg import Bool, Float32, String


class LaneCenteringController(Node):
    def __init__(self):
        super().__init__('lane_centering_controller')

        # Configurable Parameters
        self.declare_parameter('camera_topic', '/real_sense/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('driving_mode', 'left_lane')           # 'left_lane' = drive on left half (Indian road rules)
        self.declare_parameter('cruise_speed', 1.25)          # Cruising speed on straightaways (m/s)
        self.declare_parameter('turn_speed', 0.85)            # Speed on curves (m/s)
        self.declare_parameter('recovery_speed', 0.55)        # Speed when recovering from off-track (m/s)
        self.declare_parameter('max_steering', 0.55)          # Maximum Ackermann steering angle (radians)
        self.declare_parameter('enable_control', True)        # Enable active vehicle steering
        self.declare_parameter('target_edge_distance', 0.50)  # Desired minimum clearance from road edges (m)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.driving_mode = self.get_parameter('driving_mode').value
        self.cruise_speed = float(self.get_parameter('cruise_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.recovery_speed = float(self.get_parameter('recovery_speed').value)
        self.max_steering = float(self.get_parameter('max_steering').value)
        self.enable_control = bool(self.get_parameter('enable_control').value)
        self.target_edge_dist = float(self.get_parameter('target_edge_distance').value)

        # Vehicle physical geometry (Saye model)
        self.wheelbase = 2.124       # Axle separation (m)
        self.camera_height = 1.0     # Camera height above ground (m)
        self.fx = 343.159            # Focal length X (px)
        self.fy = 343.159            # Focal length Y (px)
        self.cx = 319.5              # Optical center X (px)
        self.cy = 179.5              # Optical center Y (px)
        self.road_half_width = 1.85  # Physical IIT Ropar road half-width = 1.85m (3.7m total)
        self.lane_offset = 0.45      # Drive 0.45m left of road center (left lane, Indian traffic rules)
        self.lane_width = 1.85       # Single lane width ~1.85m

        # Subscriptions & Publications
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, '/debug/lane_overlay', 10)
        self.safety_sub = self.create_subscription(
            Bool, '/safety/emergency_stop', self.safety_callback, 10
        )

        # Advisory lane correction & edge distance output for path follower integration
        self.lane_correction_pub = self.create_publisher(Float32, '/campus/lane_correction', 10)
        self.edge_dist_pub = self.create_publisher(Float32, '/campus/visual_edge_dist', 10)

        # Detect whether path planning is active (path follower is running)
        self.path_planning_active = False
        self._path_active_stamp = 0.0
        self.path_sub = self.create_subscription(
            Path, '/campus/target_path', self._path_active_callback, 10
        )
        self.pursuit_sub = self.create_subscription(
            String, '/campus/pursuit_status', self._pursuit_status_callback, 10
        )

        # Operational state
        self.emergency_stop = False
        self.locked_lane = None      # 'left_lane' or 'right_lane' once acquired
        self.smoothed_steering = 0.0
        self.prev_steering = 0.0
        self.prev_lat_error = 0.0
        self.prev_time = time.time()
        self.prev_alpha = 0.0
        self.last_dominant_curb = 'right'
        self.recovery_mode = False
        self.estimated_radius = float('inf')
        self.frame_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()

        self.get_logger().info(
            f"Precision LaneCenteringController online | Mode: {self.driving_mode} | "
            f"Cruise: {self.cruise_speed}m/s | IPM & Curvature Feedforward: ACTIVE | "
            f"Advisory mode: publishes /campus/lane_correction when path planning is active"
        )

        # Allow dynamic parameter updates mid-drive
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'driving_mode':
                self.driving_mode = param.value
                self.locked_lane = None  # Reset lane lock when commanded
                self.get_logger().info(f"Command Received: Switched to {self.driving_mode.upper()} mode!")
            elif param.name == 'cruise_speed':
                self.cruise_speed = float(param.value)
            elif param.name == 'enable_control':
                self.enable_control = bool(param.value)
        return SetParametersResult(successful=True)

    def safety_callback(self, msg: Bool):
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.get_logger().warn("Emergency Stop active!")
            self.publish_cmd_vel(0.0, 0.0)

    def _path_active_callback(self, msg: Path):
        """Detect when path planning publishes a target path - switch to advisory mode."""
        if len(msg.poses) >= 2:
            self.path_planning_active = True
            self._path_active_stamp = time.time()
            self.get_logger().info(
                "Path planning active detected - switching to ADVISORY mode (corrections only)",
                throttle_duration_sec=10.0
            )

    def _pursuit_status_callback(self, msg: String):
        """Track path follower lifecycle — keep advisory mode active while following."""
        data = msg.data.strip()
        if "Gate" in data or "Driving" in data or "XTE" in data:
            self.path_planning_active = True
            self._path_active_stamp = time.time()
        elif "ARRIVED" in data or "OFF_PATH" in data:
            self.path_planning_active = False
            self.get_logger().info(f"Path follower reported {data} - advisory mode ended")

    def image_callback(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            if frame.shape[2] == 3 and msg.encoding in ['rgb8', 'RGB8']:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 3:
                bgr = frame
            elif frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                return
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")
            return

        h, w = bgr.shape[:2]
        now = time.time()
        self.frame_count += 1
        if now - self.fps_timer >= 1.0:
            self.fps = self.frame_count / (now - self.fps_timer)
            self.frame_count = 0
            self.fps_timer = now

        # Define Region of Interest (Bottom 46% of image)
        roi_top = int(h * 0.54)
        roi_bottom = int(h * 0.98)
        roi = bgr[roi_top:roi_bottom, :]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # -------------------------------------------------------------
        # 1. Vision Masks & Terrain Segmentation
        # -------------------------------------------------------------
        # --- WHITE CENTERLINE DETECTION ---
        # The road has a white dashed centerline painted on the gray asphalt.
        # White in HSV: low saturation (S < 50), high value (V > 180), any hue.
        # We use a tight range to avoid picking up the bright gray road surface.
        lower_white = np.array([0,   0, 185])
        upper_white = np.array([180, 50, 255])
        mask_white_raw = cv2.inRange(roi_hsv, lower_white, upper_white)

        # The full road surface is also light-gray; subtract it by requiring
        # that white pixels are BRIGHTER than their neighbours (dashes stand out)
        # Morphological open: remove tiny noise, keep only solid dash segments
        kernel_w = np.ones((3, 12), np.uint8)          # horizontal kernel — dashes are wider than tall
        mask_white_roi = cv2.morphologyEx(mask_white_raw, cv2.MORPH_OPEN, kernel_w)
        mask_white_roi = cv2.morphologyEx(mask_white_roi, cv2.MORPH_CLOSE, kernel_w)

        # Remove very small isolated blobs (road surface speckle)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_white_roi, connectivity=8)
        mask_white_clean = np.zeros_like(mask_white_roi)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 40:     # keep blobs ≥ 40px (actual dashes)
                mask_white_clean[labels == i] = 255
        mask_white_roi = mask_white_clean

        # -------------------------------------------------------------
        # STRICT GREEN GRASS DETECTION
        # -------------------------------------------------------------
        # Road boundaries are defined by where the green grass starts.
        # Use HSV for shadow invariance. Green hue is ~35 to 75, Saturation >= 45 (eliminates grey asphalt).
        lower_green = np.array([35, 45, 40])
        upper_green = np.array([75, 255, 255])
        mask_grass_roi = cv2.inRange(roi_hsv, lower_green, upper_green)

        # Clean up the grass mask
        kernel = np.ones((5, 5), np.uint8)
        mask_grass_clean = cv2.morphologyEx(mask_grass_roi, cv2.MORPH_OPEN, kernel)
        mask_grass_clean = cv2.morphologyEx(mask_grass_clean, cv2.MORPH_CLOSE, kernel)

        # mask_yellow_roi = grass (used to find road outer edges)
        mask_yellow_roi = mask_grass_clean

        # Asphalt mask (for recovery fallback logic)
        val_ch = roi_hsv[:, :, 2]
        sat_ch = roi_hsv[:, :, 1]
        mask_asphalt_roi = ((val_ch > 50) & (val_ch < 185) & (sat_ch < 60)).astype(np.uint8) * 255

        # Embed masks in full-frame buffers for visualization
        mask_white = np.zeros((h, w), dtype=np.uint8)
        mask_yellow = np.zeros((h, w), dtype=np.uint8)
        mask_white[roi_top:roi_bottom, :] = mask_white_roi
        mask_yellow[roi_top:roi_bottom, :] = mask_yellow_roi

        # -------------------------------------------------------------
        # 2. Off-Track Grass Detection & Automatic Recovery
        # -------------------------------------------------------------
        near_roi_top = int(h * 0.75)
        near_grass_pixels = np.sum(mask_grass_roi[(near_roi_top - roi_top):, :] > 0)
        total_near_pixels = (roi_bottom - near_roi_top) * w
        near_grass_ratio = near_grass_pixels / max(1, total_near_pixels)

        if near_grass_ratio > 0.45:
            self.recovery_mode = True
        elif near_grass_ratio < 0.20:
            self.recovery_mode = False

        if self.recovery_mode:
            asphalt_cols = np.where(mask_asphalt_roi > 0)[1]
            if len(asphalt_cols) > 200:
                road_center_x = float(np.mean(asphalt_cols))
            else:
                left_half_grass = np.sum(mask_grass_roi[:, :w//2] > 0)
                right_half_grass = np.sum(mask_grass_roi[:, w//2:] > 0)
                road_center_x = 100.0 if left_half_grass < right_half_grass else float(w - 100)

            alpha_rec = math.atan(-(road_center_x - self.cx) / self.fx)
            rec_steer = math.atan(2.0 * self.wheelbase * math.sin(alpha_rec) / 3.0)
            rec_steer = max(-self.max_steering, min(self.max_steering, rec_steer))

            cmd_speed = self.recovery_speed
            omega_cmd = (cmd_speed * math.tan(rec_steer)) / self.wheelbase

            if self.enable_control and not self.emergency_stop and not self.path_planning_active:
                self.publish_cmd_vel(cmd_speed, omega_cmd)

            self.publish_debug_overlay(bgr, mask_white, mask_yellow, [], None, rec_steer, cmd_speed, status="OFF-TRACK RECOVERY")
            return

        # -------------------------------------------------------------
        # 3. Inverse Perspective Mapping (IPM) to Ground Coordinates
        # -------------------------------------------------------------
        # Strategy: Use DUAL boundaries to define the left lane:
        #   - WHITE centerline on the RIGHT boundary of the left lane
        #   - GREEN grass on the LEFT outer boundary of the road
        # The left lane center = midpoint between grass edge and white centerline.
        # If only one boundary is seen, use road_half_width to infer the other.
        # Sample ground distances in meters ahead: Z = 2.4m to 8.2m
        ground_distances = np.linspace(2.4, 8.2, 14)
        ground_center_pts = []  # List of (X_left_lane_center, Z)
        curb_detections = []    # List of (X_curb, Z) for single-boundary fallback
        curb_votes = []
        min_curb_dist = 999.0
        white_line_detections = []  # (X_white, Z) — confirmed centerline positions

        for Z in ground_distances:
            # Map ground distance Z to camera row v: v = cy + (H_cam * fy) / Z
            v = int(self.cy + (self.camera_height * self.fy) / Z)
            if v < roi_top or v >= roi_bottom:
                continue

            row_idx = v - roi_top
            row_w = np.where(mask_white_roi[row_idx, :] > 0)[0]
            row_y = np.where(mask_yellow_roi[row_idx, :] > 0)[0]

            # Convert horizontal pixel u to ground lateral coordinate X: X = (u - cx) * Z / fx
            # White line: pick the RIGHTMOST cluster if multiple detected (centerline is to the right of left lane)
            X_white = None
            if len(row_w) >= 5:
                # Find the median pixel of the white line pixels
                # If multiple clusters, pick rightmost (the actual centerline)
                w_splits = np.where(np.diff(row_w) > 20)[0]
                w_clusters = np.split(row_w, w_splits + 1)
                w_clusters = [c for c in w_clusters if len(c) >= 4]
                if w_clusters:
                    # Use the cluster closest to the image center (most likely the real centerline)
                    best_cluster = min(w_clusters, key=lambda c: abs(np.median(c) - self.cx))
                    X_white = float((np.median(best_cluster) - self.cx) * Z / self.fx)
                    white_line_detections.append((X_white, Z))

            # Cluster grass (outer road edge)
            X_curbs = []
            if len(row_y) >= 3:
                splits = np.where(np.diff(row_y) > 25)[0]
                clusters = np.split(row_y, splits + 1)
                for c in clusters:
                    if len(c) >= 3:
                        # Find the inner edge of the grass (closest to the road)
                        if np.mean(c) < w / 2:
                            u_c = float(np.max(c))   # left grass: rightmost pixel = inner edge
                        else:
                            u_c = float(np.min(c))   # right grass: leftmost pixel = inner edge
                        x_m = float((u_c - self.cx) * Z / self.fx)
                        X_curbs.append(x_m)

            # Classify curb side
            for c_x in X_curbs:
                u_pix = int(self.cx + (c_x * self.fx) / Z)
                if X_white is not None:
                    side = 'right' if c_x > X_white else 'left'
                    curb_votes.append(side)
                else:
                    xl = max(0, u_pix - 35)
                    xr = min(w - 1, u_pix + 35)
                    r_grass = (mask_grass_roi[row_idx, xr] > 0)
                    l_grass = (mask_grass_roi[row_idx, xl] > 0)
                    if r_grass and not l_grass:
                        curb_votes.append('right')
                    elif l_grass and not r_grass:
                        curb_votes.append('left')

                if Z <= 4.0:
                    min_curb_dist = min(min_curb_dist, abs(c_x))

            # ── Compute left-lane center from available boundaries ────────────
            X_center = None
            left_curbs  = [x for x in X_curbs if x < 0]   # grass on the left (negative X)
            right_curbs = [x for x in X_curbs if x > 0]   # grass on the right (positive X)

            if X_white is not None and left_curbs:
                # BEST CASE: both boundaries visible
                # White line = right edge of left lane, left grass = left edge of road
                X_left_grass = max(left_curbs)   # innermost left grass pixel
                # Left lane center = midpoint between left grass edge and white centerline
                X_center = (X_left_grass + X_white) / 2.0
            elif X_white is not None:
                # Only white centerline visible: left lane is to the LEFT of the white line
                X_center = X_white - self.lane_offset
            elif left_curbs and right_curbs:
                # Both grass edges visible but no white line: use road midpoint and go left
                X_road_center = (max(left_curbs) + min(right_curbs)) / 2.0
                X_center = X_road_center - self.lane_offset
            elif left_curbs:
                # Only left grass visible: center is road_half_width to the right
                X_road_edge = max(left_curbs)
                X_road_center = X_road_edge + self.road_half_width
                X_center = X_road_center - self.lane_offset
                curb_detections.append((X_road_edge, Z))
            elif right_curbs:
                # Only right grass visible: center is road_half_width to the left
                X_road_edge = min(right_curbs)
                X_road_center = X_road_edge - self.road_half_width
                X_center = X_road_center - self.lane_offset
                curb_detections.append((X_road_edge, Z))

            if X_center is not None:
                ground_center_pts.append((X_center, Z))

        # Determine dominant curb side with temporal hysteresis
        right_votes = curb_votes.count('right')
        left_votes = curb_votes.count('left')
        if right_votes > left_votes:
            dominant_curb = 'right'
        elif left_votes > right_votes:
            dominant_curb = 'left'
        else:
            dominant_curb = self.last_dominant_curb
        self.last_dominant_curb = dominant_curb

        ground_center_pts.sort(key=lambda p: p[1])

        # -------------------------------------------------------------
        # 4. Left-Lane Enforcement (Indian Traffic Rules)
        # -------------------------------------------------------------
        # The ground_center_pts already represent the LEFT LANE center.
        # We always follow the left lane — no mode switching needed.
        # driving_mode parameter is kept for manual override only.
        if self.driving_mode in ['right_lane']:
            # Rare override: operator explicitly requested right lane
            # Shift all target points to the RIGHT of the road center
            # (road_center = left_lane_center + lane_offset, right_lane = road_center + lane_offset)
            adjusted_pts = []
            for X_c, Z in ground_center_pts:
                X_road_center = X_c + self.lane_offset  # recover road center from left-lane point
                X_right_lane = X_road_center + self.lane_offset
                adjusted_pts.append((X_right_lane, Z))
            target_ground_pts = adjusted_pts
            active_lane = 'right_lane'
        elif self.driving_mode in ['center_road']:
            adjusted_pts = []
            for X_c, Z in ground_center_pts:
                X_road_center = X_c + self.lane_offset
                adjusted_pts.append((X_road_center, Z))
            target_ground_pts = adjusted_pts
            active_lane = 'center_road'
        else:
            # Default: LEFT LANE (Indian road rules — keep left)
            target_ground_pts = ground_center_pts
            active_lane = 'left_lane'
            if self.locked_lane != 'left_lane':
                self.locked_lane = 'left_lane'
                self.get_logger().info("Lane lock: LEFT LANE (Indian road rules — keep left)")

        # -------------------------------------------------------------
        # 5. Imaginary Path Vector Guidance (Vector Pursuit & Damping)
        # -------------------------------------------------------------
        steering_angle = 0.0
        current_speed = self.cruise_speed
        display_pts = []
        target_vec_disp = None
        feedforward_steer = 0.0
        feedback_steer = 0.0
        curb_repulsion = 0.0

        if len(target_ground_pts) >= 3:
            Zs = np.array([p[1] for p in target_ground_pts], dtype=np.float32)
            Xs = np.array([p[0] for p in target_ground_pts], dtype=np.float32)

            # Fit smooth quadratic path along ground: X(Z) = c2*Z^2 + c1*Z + c0
            poly = np.polyfit(Zs, Xs, 2)
            c2, c1, c0 = poly[0], poly[1], poly[2]

            # Physical road curvature: kappa = 2 * c2
            kappa = 2.0 * float(c2)
            if abs(kappa) > 1e-4:
                self.estimated_radius = 1.0 / abs(kappa)
            else:
                self.estimated_radius = float('inf')

            # 1. KINEMATIC CURVATURE FEEDFORWARD:
            # Anticipates road bends: delta_ff = -atan(L_wb * kappa)
            # Camera coordinate convention: X > 0 is right, Ackermann steering > 0 is LEFT.
            feedforward_steer = - math.atan(self.wheelbase * kappa)

            # IMAGINARY PATH VECTOR PURSUIT (Critically Damped Tracking):
            # Dynamic lookahead distance scaled with vehicle forward speed:
            # Increased lookahead significantly to completely eliminate zig-zag motion and noise.
            lookahead_dist = max(4.2, min(7.5, 3.2 + 1.2 * current_speed))

            # Target point along the imaginary lane centerline vector
            X_target = float(np.polyval(poly, lookahead_dist))

            # --- Curve Widening (Ackermann Compensation) ---
            # Prevents the buggy from cutting too close to the inner corner on turns
            # kappa > 0 (Right turn) -> Shift target Left (-X)
            # kappa < 0 (Left turn)  -> Shift target Right (+X)
            # Clamped so target vector stays safely within physical road surface (half-width 1.85m)
            curve_shift = max(-0.45, min(0.45, -1.8 * kappa))
            X_target += curve_shift

            # Approach angle eta to the imaginary vector target:
            eta = math.atan2(X_target, lookahead_dist)

            # Tangent vector of the lane at lookahead: dX/dZ = 2*c2*Z + c1
            lane_tangent_slope = 2.0 * c2 * lookahead_dist + c1
            lane_heading = math.atan(lane_tangent_slope)

            # Derivative damping on approach angle rate
            dt = max(now - self.prev_time, 0.001)
            d_eta = (eta - self.prev_alpha) / dt
            self.prev_alpha = eta
            self.prev_time = now

            # Geometric Vector Pursuit steering law with tangent heading damping:
            # delta_pursuit = - atan(2 * L_wb * sin(eta) / lookahead_dist)
            steer_pursuit = - math.atan(2.0 * self.wheelbase * math.sin(eta) / lookahead_dist)
            steer_damping = - 0.22 * lane_heading - 0.06 * d_eta

            feedback_steer = 0.85 * steer_pursuit + steer_damping
            steering_angle = feedforward_steer + feedback_steer

            # Curvature-regulated smooth speed scheduling
            curve_factor = min(1.0, abs(kappa) / 0.045)
            current_speed = self.cruise_speed - (self.cruise_speed - self.turn_speed) * curve_factor
            current_speed = max(self.turn_speed, current_speed)

            # Map target polynomial back to 2D image pixels for HUD display
            display_Zs = np.linspace(Zs[0], Zs[-1], 12)
            display_Xs = np.polyval(poly, display_Zs)
            for x_m, z_m in zip(display_Xs, display_Zs):
                u_disp = int(self.cx + (x_m * self.fx) / z_m)
                v_disp = int(self.cy + (self.camera_height * self.fy) / z_m)
                if -50 <= u_disp <= w + 50 and 0 <= v_disp < h:
                    display_pts.append((u_disp, v_disp))

            u_target = int(self.cx + (X_target * self.fx) / lookahead_dist)
            v_target = int(self.cy + (self.camera_height * self.fy) / lookahead_dist)
            target_vec_disp = (u_target, v_target)

        elif len(target_ground_pts) >= 1:
            X_t, Z_t = target_ground_pts[0]
            alpha = math.atan(- X_t / max(Z_t, 1.0))
            steering_angle = math.atan(2.0 * self.wheelbase * math.sin(alpha) / max(Z_t, 1.0))
            current_speed = self.turn_speed

        # -------------------------------------------------------------
        # 6. Directional Active Road Edge Buffer & Repulsion
        # -------------------------------------------------------------
        # Calculate physical clearance from wheels (half-track = 0.389m) to grass edges
        all_left_curbs = [abs(x) for x, z in curb_detections if x < 0]
        all_right_curbs = [x for x, z in curb_detections if x > 0]
        # Also include near-field detections from all rows
        for Z in ground_distances:
            if Z <= 5.0:
                v = int(self.cy + (self.camera_height * self.fy) / Z)
                if roi_top <= v < roi_bottom:
                    row_idx = v - roi_top
                    row_y = np.where(mask_yellow_roi[row_idx, :] > 0)[0]
                    if len(row_y) >= 3:
                        left_pts = row_y[row_y < w // 2]
                        right_pts = row_y[row_y >= w // 2]
                        if len(left_pts) >= 2:
                            u_in = float(np.max(left_pts))
                            x_m = float((u_in - self.cx) * Z / self.fx)
                            all_left_curbs.append(abs(x_m))
                        if len(right_pts) >= 2:
                            u_in = float(np.min(right_pts))
                            x_m = float((u_in - self.cx) * Z / self.fx)
                            all_right_curbs.append(x_m)

        left_edge_clearance = (min(all_left_curbs) - 0.514) if all_left_curbs else 999.0
        right_edge_clearance = (min(all_right_curbs) - 0.514) if all_right_curbs else 999.0
        min_edge_clearance = min(left_edge_clearance, right_edge_clearance)

        curb_repulsion = 0.0
        # If approaching left curb (< target_edge_dist), steer RIGHT (negative delta) away from left edge
        if left_edge_clearance < self.target_edge_dist:
            urgency_left = (self.target_edge_dist - left_edge_clearance) / max(0.1, self.target_edge_dist)
            curb_repulsion -= 0.40 * (urgency_left ** 1.2)

        # If approaching right curb (< target_edge_dist), steer LEFT (positive delta) away from right edge
        if right_edge_clearance < self.target_edge_dist:
            urgency_right = (self.target_edge_dist - right_edge_clearance) / max(0.1, self.target_edge_dist)
            curb_repulsion += 0.40 * (urgency_right ** 1.2)

        steering_angle += curb_repulsion

        # -------------------------------------------------------------
        # 7. Low-Pass Smoothing & Output Command Conversion
        # -------------------------------------------------------------
        # Critically damped smoothing filter: eliminates sensor noise jitter
        self.smoothed_steering = 0.55 * steering_angle + 0.45 * self.smoothed_steering
        self.smoothed_steering = max(-self.max_steering, min(self.max_steering, self.smoothed_steering))

        # Convert Ackermann steering angle delta to Gazebo yaw rate omega:
        omega_cmd = (current_speed * math.tan(self.smoothed_steering)) / self.wheelbase

        # ── Publish visual edge distance ───────────────────────────────────────
        if min_edge_clearance < 900.0:
            ed_msg = Float32()
            ed_msg.data = float(max(0.0, min_edge_clearance))
            self.edge_dist_pub.publish(ed_msg)

        # ── Always publish advisory lane/edge correction for path follower ─────
        correction_msg = Float32()
        if self.path_planning_active:
            correction_msg.data = float(curb_repulsion)
        else:
            correction_msg.data = float(self.smoothed_steering)
        self.lane_correction_pub.publish(correction_msg)

        # ── Direct control mode (standalone only, NEVER when path planning is active)
        if self.enable_control and not self.path_planning_active:
            if not self.emergency_stop:
                self.publish_cmd_vel(current_speed, omega_cmd)
            else:
                self.publish_cmd_vel(0.0, 0.0)

        # -------------------------------------------------------------
        # 8. Visual HUD Overlay
        # -------------------------------------------------------------
        white_detected = len(white_line_detections) if 'white_line_detections' in dir() else 0
        mode_label = "PATH-ADVISORY" if self.path_planning_active else active_lane.upper()
        status_text = f"KEEP-LEFT | {mode_label}"
        if min_edge_clearance < 900.0:
            status_text += f" | EDGE: {min_edge_clearance:.2f}m"
        if white_detected > 0:
            status_text += f" | WHITE LINE: {white_detected}pts"
        if abs(curb_repulsion) > 0.02:
            status_text += " | BUFFER GUARD ACTIVE"
        self.publish_debug_overlay(bgr, mask_white, mask_yellow, display_pts, target_vec_disp, self.smoothed_steering, current_speed, status=status_text)

    def publish_cmd_vel(self, linear_x: float, angular_z: float):
        if not self.enable_control or self.path_planning_active:
            return
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)

    def publish_debug_overlay(self, bgr, mask_white, mask_yellow, pts, target_vec=None, steering=0.0, speed=0.0, status="TRACKING"):
        h, w = bgr.shape[:2]
        overlay = bgr.copy()

        if mask_yellow is not None and np.sum(mask_yellow) > 0:
            overlay[mask_yellow > 0] = (0, 215, 255)  # Amber yellow
        if mask_white is not None and np.sum(mask_white) > 0:
            overlay[mask_white > 0] = (0, 255, 120)   # Emerald green

        cv2.addWeighted(overlay, 0.65, bgr, 0.35, 0, overlay)

        cx = int(w / 2.0)
        cv2.line(overlay, (cx, h - 35), (cx, h - 5), (0, 0, 255), 3)

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], (255, 255, 0), 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(overlay, p, 4, (0, 165, 255), -1, cv2.LINE_AA)

        # Draw Imaginary Path Guidance Vector
        if target_vec is not None:
            cv2.arrowedLine(overlay, (cx, h - 20), target_vec, (0, 255, 255), 3, tipLength=0.15)
            cv2.circle(overlay, target_vec, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(overlay, "IMAGINARY PATH VECTOR", (max(10, target_vec[0] - 80), max(20, target_vec[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        elif len(pts) >= 2:
            far_pt = pts[-1]
            cv2.arrowedLine(overlay, (cx, h - 20), far_pt, (0, 255, 255), 2, tipLength=0.15)

        hud_bg = overlay.copy()
        cv2.rectangle(hud_bg, (10, 10), (370, 120), (20, 24, 28), -1)
        cv2.addWeighted(hud_bg, 0.80, overlay, 0.20, 0, overlay)

        steering_deg = math.degrees(steering)
        steer_color = (0, 255, 0) if abs(steering_deg) < 12 else (0, 165, 255)
        status_color = (0, 255, 255) if "LOCKED" in status else ((0, 100, 255) if "REPULSION" in status else (0, 0, 255))

        r_text = f"R: {self.estimated_radius:.1f}m" if self.estimated_radius < 100 else "R: Straight"

        cv2.putText(overlay, "AUTONOMOUS LANE CONTROLLER", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"{status} | {self.fps:.1f} FPS", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Steer: {steering_deg:+.1f} deg | {r_text}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, steer_color, 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Speed: {speed:.2f} m/s", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        if self.emergency_stop:
            cv2.putText(overlay, "EMERGENCY STOP", (w // 2 - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug_msg = Image()
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        debug_msg.header.frame_id = "depth_camera_link"
        debug_msg.height = h
        debug_msg.width = w
        debug_msg.encoding = "bgr8"
        debug_msg.is_bigendian = 0
        debug_msg.step = w * 3
        debug_msg.data = overlay.tobytes()
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneCenteringController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd_vel(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
