""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: wrist_3_link -> flange """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "wrist_3_link",
                "--child-frame-id",
                "flange",
                "--x",
                "0.021",
                "--y",
                "-0.039",
                "--z",
                "0.275",
                "--qx",
                "0",
                "--qy",
                "0",
                "--qz",
                "0",
                "--qw",
                "1",
                # "--roll",
                # "-0",
                # "--pitch",
                # "0",
                # "--yaw",
                # "-0",
            ],
        ),
    ]
    return LaunchDescription(nodes)
