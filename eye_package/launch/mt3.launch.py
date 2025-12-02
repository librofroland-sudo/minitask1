from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ------------------ 启动 eye_node ------------------
        Node(
            package='eye_package',           # ← 修改为你的包名
            executable='eye_node',        # ← 你的眼睛节点文件名
            name='eye_node',              
            output='screen'
        ),

        # ------------------ 启动 nav_node ------------------
        Node(
            package='eye_package',           # ← 如果 nav_node 也在 eye_node 包中，不需要改
            executable='nav_node',        # ← 你的导航节点文件名
            name='nav_node',
            output='screen'
        )

    ])