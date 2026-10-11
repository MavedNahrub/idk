import os
# pyrefly: ignore [missing-import]
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
# pyrefly: ignore [missing-import]
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# pyrefly: ignore [missing-import]
from launch.conditions import IfCondition
# pyrefly: ignore [missing-import]
from launch.launch_description_sources import PythonLaunchDescriptionSource
# pyrefly: ignore [missing-import]
from launch.substitutions import LaunchConfiguration
# pyrefly: ignore [missing-import]
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('saye_bringup')

    # Arguments
    enable_yolop_arg = DeclareLaunchArgument(
        'enable_yolop',
        default_value='false',
        description='Enable YOLOP panoptic neural perception node.'
    )
    driving_mode_arg = DeclareLaunchArgument(
        'driving_mode',
        default_value='center_road',
        description='Lane following mode: center_road, left_lane, or right_lane.'
    )
    cruise_speed_arg = DeclareLaunchArgument(
        'cruise_speed',
        default_value='1.5',
        description='Target cruising speed in m/s.'
    )

    # 1. Base Gazebo Simulation & Vehicle Bringup
    sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'saye_spawn.launch.py')
        )
    )

    # 2. Real-time Lane Centering & Steering Controller
    lane_controller_node = Node(
        package='saye_control',
        executable='lane_centering_controller.py',
        name='lane_centering_controller',
        output='screen',
        parameters=[{
            'camera_topic': '/real_sense/image_raw',
            'cmd_vel_topic': '/cmd_vel',
            'driving_mode': LaunchConfiguration('driving_mode'),
            'cruise_speed': LaunchConfiguration('cruise_speed'),
            'enable_control': True
        }]
    )

    # 3. YOLOP Panoptic Perception Engine (Optional / Parallel)
    yolop_node = Node(
        package='saye_control',
        executable='yolop_perception_node.py',
        name='yolop_perception_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_yolop')),
        parameters=[{
            'camera_topic': '/real_sense/image_raw',
            'device': 'cuda:0',
            'enable_safety_stop': True
        }]
    )

    return LaunchDescription([
        enable_yolop_arg,
        driving_mode_arg,
        cruise_speed_arg,
        sim_bringup,
        lane_controller_node,
        yolop_node
    ])
