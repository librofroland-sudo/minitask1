import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
import random
import time

from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker


class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # ------------------------ Nav2 Action Client ------------------------
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        # 目标记录
        self.green_point = None
        self.red_point = None

        # 探索相关
        self.map_data = None          # 地图
        self.scan_data = None         # 激光雷达
        self.last_explore_time = 0.0  # 避免频繁探索
        self.explore_interval = 5.0   # 每 5 秒生成一次探索点

        # 安全距离
        self.declare_parameter("safe_distance", 0.6)
        self.safe_distance = self.get_parameter("safe_distance").value

        # RViz 标记
        self.marker_pub = self.create_publisher(Marker, "object_markers", 10)
        self.found_objects = []

        # 订阅激光雷达
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # 订阅地图
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)

        # 订阅绿色与红色目标
        self.create_subscription(Point, "/green_point", self.green_callback, 10)
        self.create_subscription(Point, "/red_point", self.red_callback, 10)

        # 探索定时器
        self.create_timer(1.0, self.exploration_loop)

        self.get_logger().info("nav_node smart version started. Waiting for objects...")

    # ------------------------ 基础回调 ------------------------
    def scan_callback(self, msg):
        self.scan_data = msg

    def map_callback(self, msg):
        self.map_data = msg

    def green_callback(self, msg):
        self.green_point = msg
        self.get_logger().info(f"Received GREEN point: ({msg.x:.2f}, {msg.y:.2f})")
        self.navigate_to_object(msg, "GREEN")

    def red_callback(self, msg):
        self.red_point = msg
        self.get_logger().info(f"Received RED point: ({msg.x:.2f}, {msg.y:.2f})")
        self.navigate_to_object(msg, "RED")

    # ------------------------ 真实物体导航 ------------------------
    def navigate_to_object(self, msg, color_name):
        dist = math.hypot(msg.x, msg.y)
        if dist > self.safe_distance:
            scale = (dist - self.safe_distance) / dist
            x = msg.x * scale
            y = msg.y * scale
        else:
            x, y = msg.x, msg.y

        # 标记
        self.publish_marker(msg.x, msg.y, color_name)
        self.send_nav_goal(x, y, color_name)

    # ------------------------ RViz 标记 ------------------------
    def publish_marker(self, x, y, color_name):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "found"
        marker.id = len(self.found_objects)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.2

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        if color_name == "GREEN":
            marker.color.g = 1.0
        else:
            marker.color.r = 1.0

        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    # ------------------------ 导航命令 ------------------------
    def send_nav_goal(self, x, y, label):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Nav2 server not available!")
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Navigating to {label}: ({x:.2f}, {y:.2f})")

        self.nav_client.send_goal_async(goal)

    # =====================================================================
    #                     智能探索：主循环
    # =====================================================================
    def exploration_loop(self):
        # 如果检测到目标 → 停止探索
        if self.green_point or self.red_point:
            return

        # 保持一定时间间隔，避免探索太频繁
        if time.time() - self.last_explore_time < self.explore_interval:
            return

        # 需要激光雷达 + 地图支持
        if self.map_data is None or self.scan_data is None:
            return

        # --------- Step 1：前沿探索（找未知区域） ---------
        frontier_goal = self.find_frontier_goal()
        if frontier_goal:
            self.get_logger().info(f"Exploring frontier: {frontier_goal}")
            self.send_nav_goal(frontier_goal[0], frontier_goal[1], "EXPLORE-FRONTIER")
            self.last_explore_time = time.time()
            return

        # --------- Step 2：如果没有前沿 → 壁跟随探索 ---------
        wall_goal = self.follow_wall()
        if wall_goal:
            self.get_logger().info(f"Following wall: {wall_goal}")
            self.send_nav_goal(wall_goal[0], wall_goal[1], "FOLLOW-WALL")
            self.last_explore_time = time.time()
            return

        # --------- Step 3：随机探索（兜底方案） ---------
        rx, ry = self.random_explore_point()
        self.get_logger().info(f"Random exploring: ({rx:.2f}, {ry:.2f})")
        self.send_nav_goal(rx, ry, "RANDOM")
        self.last_explore_time = time.time()

    # =====================================================================
    #                   Step 1：前沿探索（智能探索核心）
    # =====================================================================
    def find_frontier_goal(self):
        """
        前沿 frontier 的定义：
        - 地图像素 == -1（未知区域）
        - 其邻居中至少有一个是已知（0 或 100）

        我们从地图中找离机器人最近的前沿目标。
        """
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin = self.map_data.info.origin

        data = self.map_data.data

        frontiers = []

        for i in range(len(data)):
            if data[i] != -1:  # 未知区域
                continue

            x = i % width
            y = i // width

            # 检查周围是否存在已知区域（形成前沿）
            neighbors = [
                data[i + 1] if i + 1 < len(data) else None,
                data[i - 1] if i - 1 >= 0 else None,
                data[i + width] if i + width < len(data) else None,
                data[i - width] if i - width >= 0 else None
            ]

            if any(n == 0 for n in neighbors):  # 有空地邻居 → 前沿
                wx = origin.position.x + x * resolution
                wy = origin.position.y + y * resolution
                frontiers.append((wx, wy))

        if not frontiers:
            return None

        # 返回最靠近机器人的前沿
        frontiers.sort(key=lambda p: math.hypot(p[0], p[1]))
        return frontiers[0]

    # =====================================================================
    #                   Step 2：壁跟随（沿墙走）
    # =====================================================================
    def follow_wall(self):
        """
        使用激光雷达寻找最近的墙壁方向并沿着墙移动。
        """
        ranges = self.scan_data.ranges
        angle_increment = self.scan_data.angle_increment
        angle_min = self.scan_data.angle_min

        # 找到最近的墙
        min_dist = min(ranges)
        min_index = ranges.index(min_dist)
        wall_angle = angle_min + min_index * angle_increment

        # 与墙平行方向探索（+90°）
        explore_angle = wall_angle + math.pi/2

        dist = random.uniform(0.8, 1.2)
        x = math.cos(explore_angle) * dist
        y = math.sin(explore_angle) * dist

        return (x, y)

    # =====================================================================
    #                   Step 3：随机探索（兜底）
    # =====================================================================
    def random_explore_point(self):
        angle = random.uniform(0, 2*math.pi)
        radius = random.uniform(1.0, 2.0)
        return math.cos(angle)*radius, math.sin(angle)*radius



def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
