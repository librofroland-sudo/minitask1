import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Point      # ← 修改：引入 Point 消息
from nav2_msgs.action import NavigateToPose


class NavToPoseClient(Node):
    def __init__(self):
        super().__init__("nav_to_pose_client")

        # 创建 action client
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        # ---------------------- 订阅绿色物体坐标 ----------------------
        # eye_node 会发布 geometry_msgs/Point
        # topic 名称需要和 eye_node 完全一致
        self.subscription = self.create_subscription(
            Point,
            '/green_point',      # ← 修改：你要监控的 topic
            self.point_callback,
            10
        )
        self.get_logger().info("nav_node 已启动，等待 eye_node 坐标...")
        # -------------------------------------------------------------

    # ---------------------- 新增：Point 回调 ------------------------
    def point_callback(self, msg: Point):
        """收到 eye_node 发布的绿色物体坐标时调用"""
        self.get_logger().info(
            f"收到绿色物体坐标: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}"
        )

        # 调用导航函数
        self.send_goal(msg.x, msg.y)
    # -------------------------------------------------------------

    # ---------------------- 修改：加入参数 (x, y) -------------------
    def send_goal(self, x, y):
        """发送导航目标到 Nav2"""
        self._action_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()

        # 必须设置 header
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = "map"   # 全局坐标系

        # 使用 green object 的 (x, y)
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # orientation（必须有）
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"发送导航目标: x={x:.2f}, y={y:.2f}")
        self._action_client.send_goal_async(goal_msg)
    # -------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = NavToPoseClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()