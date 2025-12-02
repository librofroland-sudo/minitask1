import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
import math
import time


class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # ------------------------ Nav2 Action Client ------------------------
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        # 记录最新坐标
        self.green_point = None
        self.red_point = None

        # 防止频繁导航
        self.last_nav_time = 0
        self.nav_interval_sec = 3.0  # 3秒内不重复发送

        # --------------------- 订阅绿色目标坐标 ---------------------
        self.create_subscription(
            Point,
            "/green_point",
            self.green_callback,
            10
        )

        # --------------------- 订阅红色目标坐标 ---------------------
        self.create_subscription(
            Point,
            "/red_point",
            self.red_callback,
            10
        )

        self.get_logger().info("nav_node 已启动，正在监听 green_point 与 red_point ...")

    # ======================== 回调：绿色目标 ========================
    def green_callback(self, msg):
        self.green_point = msg
        self.get_logger().info(f"收到绿色坐标: ({msg.x:.2f}, {msg.y:.2f})")

        self.try_navigate()

    # ======================== 回调：红色目标 ========================
    def red_callback(self, msg):
        self.red_point = msg
        self.get_logger().info(f"收到红色坐标: ({msg.x:.2f}, {msg.y:.2f})")

        self.try_navigate()

    # ======================== 自动选择最近目标 ========================
    def try_navigate(self):

        now = time.time()
        if now - self.last_nav_time < self.nav_interval_sec:
            return  # 防止短时间内重复调用导航

        target = None

        # 两者都存在 → 选最近
        if self.green_point and self.red_point:
            dg = math.sqrt(self.green_point.x**2 + self.green_point.y**2)
            dr = math.sqrt(self.red_point.x**2 + self.red_point.y**2)

            if dg <= dr:
                target = ("绿色", self.green_point)
            else:
                target = ("红色", self.red_point)

        # 只有绿色
        elif self.green_point:
            target = ("绿色", self.green_point)

        # 只有红色
        elif self.red_point:
            target = ("红色", self.red_point)

        # 没有目标
        else:
            return

        color, point = target
        self.send_goal(point.x, point.y, color)
        self.last_nav_time = now

    # ======================== 发送导航 Goal ========================
    def send_goal(self, x, y, color_name):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Nav2 action server 未启动！")
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()

        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = "map"

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0  # 朝向不重要

        self.get_logger().info(f"🚀 导航到{color_name}目标点: ({x:.2f}, {y:.2f})")

        # 发送异步 goal
        self.nav_client.send_goal_async(goal)


# ============================ 启动入口 ============================
def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
