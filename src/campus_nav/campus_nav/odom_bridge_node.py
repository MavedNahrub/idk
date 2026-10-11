"""
Odometry Bridge & Vehicle Visualization Node

Responsibilities:
  1. Subscribes to /model/saye/odometry (ground truth Gazebo odometry from ros_gz_bridge).
  2. Converts Gazebo coordinates (gz_x, gz_y) -> planner coordinates (nav_x, nav_y).
  3. Publishes /campus/vehicle_pose (PoseStamped) in map frame for navigation tracking.
  4. Publishes /campus/vehicle_marker (MarkerArray) -> 3D Orange Buggy with white canopy,
     yellow heading arrow, and label, visible in RViz!
  5. Void Recovery Supervisor: If the buggy ever drops below Z = -0.5m, it automatically
     catches the buggy, zeroes velocity, and safely respawns it onto the nearest road.
"""

import math
import subprocess
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from campus_nav.coord_bridge import (
    gz_to_nav, nav_to_gz, yaw_from_quaternion, quaternion_from_yaw,
    yaw_gz_to_nav, yaw_nav_to_gz
)
from campus_nav.osm_loader import CampusMap
import os


WORLD_NAME = 'campus_world'
MODEL_NAME = 'saye'
ROAD_SURFACE_Z = 0.60  # Road mesh is at 0.30m + wheel radius 0.28m


class OdomBridgeNode(Node):
    def __init__(self):
        super().__init__('odom_bridge_node')

        self.declare_parameter('model_name', MODEL_NAME)
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value

        # Load map for void recovery road snapping
        candidates_osm = []
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('campus_nav')
            candidates_osm.extend([
                os.path.join(pkg_share, 'data', 'campus_with_junctions_and_stops.osm'),
                os.path.join(pkg_share, 'data', 'campus.osm')
            ])
        except Exception:
            pass
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(curr_dir, '..', '..', '..'))
        candidates_osm.extend([
            os.path.join(curr_dir, '..', 'data', 'campus_with_junctions_and_stops.osm'),
            os.path.join(curr_dir, '..', 'data', 'campus.osm'),
            os.path.join(repo_root, 'campus_with_junctions_and_stops.osm'),
            os.path.join(repo_root, 'campus.osm')
        ])
        default_osm = next((p for p in candidates_osm if os.path.exists(p)), '')
        self.campus = CampusMap(default_osm)

        # Publishers
        self.vehicle_pose_pub = self.create_publisher(
            PoseStamped, '/campus/vehicle_pose', 10)
        self.odom_pub = self.create_publisher(
            Odometry, '/odom', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/campus/vehicle_marker', 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.abort_pub = self.create_publisher(
            Path, '/campus/target_path', 10)

        # TF Broadcaster: connects map -> base_footprint -> base_link -> lidar_link
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        # Ground truth world poses directly from Gazebo Harmonic
        self.gz_pose_sub = self.create_subscription(
            TFMessage, '/campus/gazebo_poses', self.gazebo_poses_callback, 10)

        # Fallback local odometry from Gazebo plugin
        self.odom_sub = self.create_subscription(
            Odometry, '/model/saye/odometry', self.odometry_callback, 10)

        self.last_gz_x = None
        self.last_gz_y = None
        self.last_gz_z = 0.60
        self.last_yaw = 0.0
        self.last_recovery_time = 0.0
        self._got_gz_world_pose = False

        self.get_logger().info(
            f"OdomBridgeNode ready. Tracking '{self.model_name}' via ground-truth world poses & odometry.")

    def gazebo_poses_callback(self, msg: TFMessage):
        """Receive ground-truth Gazebo world pose directly from Gazebo simulation engine."""
        for i, t in enumerate(msg.transforms):
            cid = t.child_frame_id.strip('/')
            # Match by child_frame_id, or if empty string, index 0 is always the root model 'saye'
            if cid == self.model_name or cid.endswith(self.model_name) or (not cid and i == 0):
                self._got_gz_world_pose = True
                tr = t.transform.translation
                rot = t.transform.rotation
                self._process_pose(tr.x, tr.y, tr.z, rot)
                break

    def odometry_callback(self, msg: Odometry):
        """Process local odometry from Gazebo (fallback only if world pose not yet received)."""
        if self._got_gz_world_pose:
            return
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self._process_pose(pos.x, pos.y, pos.z, ori)

    def _process_pose(self, gz_x, gz_y, gz_z, orientation_q):
        yaw = yaw_from_quaternion(orientation_q)
        self.last_gz_x = gz_x
        self.last_gz_y = gz_y
        self.last_gz_z = gz_z
        self.last_yaw = yaw

        # ── 1. Void & Off-Road Recovery Supervisor ──────────────────────────────
        # Catch falling into void (Z < -0.5) or wandering into grass (> 3.0m off-road)
        nav_x, nav_y = gz_to_nav(gz_x, gz_y)
        now_sec = time.time()
        is_in_void = gz_z < -0.5
        road_dist = self.campus.distance_to_road_segment(nav_x, nav_y) if self.campus else 0.0
        is_deep_offroad = road_dist > 3.8


        if (is_in_void or is_deep_offroad) and (now_sec - self.last_recovery_time > 2.0):
            self.last_recovery_time = now_sec
            reason = "fell below road" if is_in_void else f"wandered {road_dist:.1f}m off road"
            self.get_logger().warn(f"🚨 Buggy {reason}! Catching and respawning onto nearest road lane...")
            self._recover_from_void(gz_x, gz_y)
            return

        # ── 2. Convert to planner coords ────────────────────────────────────────
        now = self.get_clock().now().to_msg()

        # Buggy's physical heading: saye front is -Y in base_link (model_yaw - pi/2)
        phys_gz = (yaw - math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi
        nav_yaw = yaw_gz_to_nav(phys_gz)

        qx, qy, qz, qw = quaternion_from_yaw(nav_yaw)

        # ── 3. Publish /campus/vehicle_pose ─────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = nav_x
        pose_msg.pose.position.y = nav_y
        pose_msg.pose.position.z = gz_z
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.vehicle_pose_pub.publish(pose_msg)

        # ── 4. Publish /odom (Gazebo frame) ─────────────────────────────────────
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = f'{self.model_name}/base_link'
        odom_msg.pose.pose.position.x = gz_x
        odom_msg.pose.pose.position.y = gz_y
        odom_msg.pose.pose.position.z = gz_z
        odom_msg.pose.pose.orientation = orientation_q
        self.odom_pub.publish(odom_msg)

        # ── 4b. Broadcast TF: map -> odom & map -> world ──────────────────────
        # IIT Ropar Gazebo world to planner metric coordinate transform:
        # nav_x = 760.49 - gz_y, nav_y = 718.37 + gz_x (Rotation = +90 deg around Z)
        t_odom = TransformStamped()
        t_odom.header.stamp = now
        t_odom.header.frame_id = 'map'
        t_odom.child_frame_id = 'odom'
        t_odom.transform.translation.x = 760.49
        t_odom.transform.translation.y = 718.37
        t_odom.transform.translation.z = 0.0
        t_odom.transform.rotation.x = 0.0
        t_odom.transform.rotation.y = 0.0
        t_odom.transform.rotation.z = 0.7071067811865475
        t_odom.transform.rotation.w = 0.7071067811865476
        self.tf_broadcaster.sendTransform(t_odom)

        # Also broadcast aliases for world and saye/odom
        t_world = TransformStamped()
        t_world.header.stamp = now
        t_world.header.frame_id = 'map'
        t_world.child_frame_id = 'world'
        t_world.transform = t_odom.transform
        self.tf_broadcaster.sendTransform(t_world)

        t_saye_odom = TransformStamped()
        t_saye_odom.header.stamp = now
        t_saye_odom.header.frame_id = 'map'
        t_saye_odom.child_frame_id = 'saye/odom'
        t_saye_odom.transform = t_odom.transform
        self.tf_broadcaster.sendTransform(t_saye_odom)

        t_nav = TransformStamped()
        t_nav.header.stamp = now
        t_nav.header.frame_id = 'map'
        t_nav.child_frame_id = 'nav_base_link'
        t_nav.transform.translation.x = nav_x
        t_nav.transform.translation.y = nav_y
        t_nav.transform.translation.z = gz_z
        t_nav.transform.rotation.x = qx
        t_nav.transform.rotation.y = qy
        t_nav.transform.rotation.z = qz
        t_nav.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t_nav)

        # ── 5. Publish 3D High-Contrast Vehicle Marker for RViz ────────────────
        self._publish_vehicle_markers(now, nav_x, nav_y, gz_z, nav_yaw, qx, qy, qz, qw)

    def _publish_vehicle_markers(self, stamp, nav_x, nav_y, gz_z, nav_yaw, qx, qy, qz, qw):
        """Constructs and publishes a 3D high-visibility buggy marker in RViz in Fixed Frame 'map'."""
        ma = MarkerArray()

        # Buggy chassis center is ~0.60m behind the front axle origin along heading
        cx = nav_x - 0.60 * math.cos(nav_yaw)
        cy = nav_y - 0.60 * math.sin(nav_yaw)
        cz = gz_z + 0.40

        # 1. Main Chassis (Fluorescent Orange)
        body = Marker()
        body.header.frame_id = 'map'
        body.header.stamp = stamp
        body.ns = 'buggy'
        body.id = 0
        body.type = Marker.CUBE
        body.action = Marker.ADD
        body.pose.position.x = cx
        body.pose.position.y = cy
        body.pose.position.z = cz
        body.pose.orientation.x = qx
        body.pose.orientation.y = qy
        body.pose.orientation.z = qz
        body.pose.orientation.w = qw
        body.scale.x = 2.4  # Length along driving heading
        body.scale.y = 1.3  # Width lateral to driving heading
        body.scale.z = 0.5  # Height
        body.color.r = 1.0
        body.color.g = 0.35
        body.color.b = 0.0
        body.color.a = 0.95
        ma.markers.append(body)

        # 2. Canopy Roof (Clean White)
        roof = Marker()
        roof.header.frame_id = 'map'
        roof.header.stamp = stamp
        roof.ns = 'buggy'
        roof.id = 1
        roof.type = Marker.CUBE
        roof.action = Marker.ADD
        roof.pose.position.x = cx
        roof.pose.position.y = cy
        roof.pose.position.z = gz_z + 1.30
        roof.pose.orientation.x = qx
        roof.pose.orientation.y = qy
        roof.pose.orientation.z = qz
        roof.pose.orientation.w = qw
        roof.scale.x = 1.7  # Length
        roof.scale.y = 1.2  # Width
        roof.scale.z = 0.08 # Thickness
        roof.color.r = 0.95
        roof.color.g = 0.95
        roof.color.b = 0.95
        roof.color.a = 0.95
        ma.markers.append(roof)

        # 3. Heading Arrow (Bright Yellow, points straight forward through front bumper)
        arrow = Marker()
        arrow.header.frame_id = 'map'
        arrow.header.stamp = stamp
        arrow.ns = 'buggy'
        arrow.id = 2
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [
            Point(x=nav_x, y=nav_y, z=gz_z + 0.70),
            Point(x=nav_x + 2.0 * math.cos(nav_yaw), y=nav_y + 2.0 * math.sin(nav_yaw), z=gz_z + 0.70)
        ]
        arrow.scale.x = 0.25 # Shaft diameter
        arrow.scale.y = 0.50 # Head diameter
        arrow.scale.z = 0.40 # Head length
        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        ma.markers.append(arrow)

        # 4. Text Label
        label = Marker()
        label.header.frame_id = 'map'
        label.header.stamp = stamp
        label.ns = 'buggy'
        label.id = 3
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = nav_x
        label.pose.position.y = nav_y
        label.pose.position.z = gz_z + 1.85
        label.scale.z = 1.0
        label.text = "saye Buggy"
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        ma.markers.append(label)

        self.marker_pub.publish(ma)

    def _recover_from_void(self, current_gz_x, current_gz_y):
        """Emergency catch: buggy fell below the road. Respawn onto nearest road."""
        self.get_logger().warn(
            f"🚨 Buggy fell below road (Z={self.last_gz_z:.1f}m)! Catching and respawning onto nearest road...")

        # 1. Immediately zero velocity and clear path follower
        self.cmd_pub.publish(Twist())
        self.abort_pub.publish(Path())

        # 2. Find nearest road node to last known location
        nav_x, nav_y = gz_to_nav(current_gz_x, current_gz_y)
        nearest_nid, d = self.campus.snap_to_road(nav_x, nav_y)

        if nearest_nid:
            target_nav_x, target_nav_y = self.campus.xy_nodes[nearest_nid]
            respawn_gz_x, respawn_gz_y = nav_to_gz(target_nav_x, target_nav_y)
        else:
            respawn_gz_x, respawn_gz_y = 15.02, 43.00
            target_nav_x, target_nav_y = gz_to_nav(respawn_gz_x, respawn_gz_y)

        # Orient buggy towards connected road neighbor (inwards towards campus)
        # CRITICAL: saye front is -Y → model_yaw = physical_heading + π/2
        respawn_yaw_gz = 1.5708 + math.pi / 2.0  # default + correction
        if nearest_nid and nearest_nid in self.campus.road_graph and self.campus.road_graph[nearest_nid]:
            nbr = next(iter(self.campus.road_graph[nearest_nid]))
            nbr_x, nbr_y = self.campus.xy_nodes[nbr]
            nav_angle = math.atan2(nbr_y - target_nav_y, nbr_x - target_nav_x)
            phys_gz = yaw_nav_to_gz(nav_angle)
            respawn_yaw_gz = (phys_gz + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi

        qz = math.sin(respawn_yaw_gz / 2.0)
        qw = math.cos(respawn_yaw_gz / 2.0)

        # 3. Teleport buggy to road height (Z = 0.60m)
        self.get_logger().info(f"Respawning saye at Gazebo ({respawn_gz_x:.1f}, {respawn_gz_y:.1f}, Z={ROAD_SURFACE_Z})")
        req = (
            f'name: "{self.model_name}", '
            f'position: {{x: {respawn_gz_x:.3f}, y: {respawn_gz_y:.3f}, z: {ROAD_SURFACE_Z:.3f}}}, '
            f'orientation: {{x: 0.0, y: 0.0, z: {qz:.4f}, w: {qw:.4f}}}'
        )
        cmd = [
            'gz', 'service',
            '-s', f'/world/{WORLD_NAME}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req
        ]
        subprocess.run(cmd, capture_output=True, text=True)


def main(args=None):
    rclpy.init(args=args)
    node = OdomBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
