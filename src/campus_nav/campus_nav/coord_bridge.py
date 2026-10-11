"""
Coordinate bridge between campus_nav planner frame (OSM metric) and Gazebo world frame.

Accurately calibrated via alignment of IIT Ropar map1.obj road mesh (roads_service)
and OSM road network:
    gz_x = nav_y - 718.37
    gz_y = 760.49 - nav_x

Inverse:
    nav_x = 760.49 - gz_y
    nav_y = gz_x + 718.37

Heading / Yaw relationship:
    dx_gz = dy_nav
    dy_gz = -dx_nav
    yaw_gz = yaw_nav - pi/2
    yaw_nav = yaw_gz + pi/2
"""

import math

T_X = 760.49
T_Y = 718.37


def gz_to_nav(gz_x, gz_y):
    """Convert Gazebo world coordinates to campus_nav planner coordinates."""
    nav_x = T_X - gz_y
    nav_y = gz_x + T_Y
    return nav_x, nav_y


def nav_to_gz(nav_x, nav_y):
    """Convert campus_nav planner coordinates to Gazebo world coordinates."""
    gz_x = nav_y - T_Y
    gz_y = T_X - nav_x
    return gz_x, gz_y


def yaw_nav_to_gz(yaw_nav):
    """Convert heading from nav (map) frame to Gazebo world frame."""
    return (yaw_nav - math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi


def yaw_gz_to_nav(yaw_gz):
    """Convert heading from Gazebo world frame to nav (map) frame."""
    return (yaw_gz + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi


def convert_path_nav_to_gz(path_msg):
    """
    Convert a nav_msgs/Path from planner coords to Gazebo coords (in-place).
    Returns the same path_msg for convenience.
    """
    for pose in path_msg.poses:
        gx, gy = nav_to_gz(pose.pose.position.x, pose.pose.position.y)
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        yaw_n = yaw_from_quaternion(pose.pose.orientation)
        yaw_g = yaw_nav_to_gz(yaw_n)
        qx, qy, qz, qw = quaternion_from_yaw(yaw_g)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
    return path_msg


def convert_path_gz_to_nav(path_msg):
    """
    Convert a nav_msgs/Path from Gazebo coords to planner coords (in-place).
    Returns the same path_msg for convenience.
    """
    for pose in path_msg.poses:
        nx, ny = gz_to_nav(pose.pose.position.x, pose.pose.position.y)
        pose.pose.position.x = nx
        pose.pose.position.y = ny
        yaw_g = yaw_from_quaternion(pose.pose.orientation)
        yaw_n = yaw_gz_to_nav(yaw_g)
        qx, qy, qz, qw = quaternion_from_yaw(yaw_n)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
    return path_msg


def yaw_from_quaternion(q):
    """Extract yaw angle from a quaternion (geometry_msgs Quaternion or similar)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    """Return (x, y, z, w) quaternion tuple from a yaw angle."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

