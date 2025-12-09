#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from nav2_msgs.action import FollowWaypoints

def yaw_to_quat(yaw: float):
    return (0.0, 0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0))

class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # -------- markers -------
        self.marker_pub = self.create_publisher(Marker, "/object_markers", 10)
        self.marker_id = 0

        # -------- detect objects -------
        self.create_subscription(Point, "/green_point", self.green_cb, 10)
        self.create_subscription(Point, "/red_point", self.red_cb, 10)

        # -------- FollowWaypoints action -------
        self.client = ActionClient(self, FollowWaypoints, "/follow_waypoints")
        self.get_logger().info("Waiting for FollowWaypoints server...")
        self.client.wait_for_server()
        self.get_logger().info("FollowWaypoints ready.")

        # -------- waypoint list -------
        self.waypoints = [
            (2.0, -3.0, 0.0),
            (5.0,  1.0, 0.0),
            (5.0, -3.0, 0.0),
            (0.0,  4.5, 0.0),
            (1.5,  3.0, 0.0),
        ]

        # small delay, send after nav2 fully active
        self.create_timer(1.0, self.send_all_waypoints)

    # ----------- object markers ----------
    def publish_marker(self, p, color):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "objects"
        m.id = self.marker_id
        self.marker_id += 1

        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.3

        m.pose.position.x = p.x
        m.pose.position.y = p.y
        m.pose.position.z = 0.1
        m.color.a = 1.0
        if color == "GREEN":
            m.color.g = 1.0
        else:
            m.color.r = 1.0

        self.marker_pub.publish(m)

    def green_cb(self, msg):
        self.get_logger().info(f"Green: {msg.x:.2f}, {msg.y:.2f}")
        self.publish_marker(msg, "GREEN")

    def red_cb(self, msg):
        self.get_logger().info(f"Red: {msg.x:.2f}, {msg.y:.2f}")
        self.publish_marker(msg, "RED")

    # ----------- send all waypoints ----------
    def send_all_waypoints(self):
        goal = FollowWaypoints.Goal()
        poses = []

        for (x, y, yaw) in self.waypoints:
            qx, qy, qz, qw = yaw_to_quat(yaw)

            p = PoseStamped()
            p.header.frame_id = "map"
            p.header.stamp = self.get_clock().now().to_msg()

            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.x = qx
            p.pose.orientation.y = qy
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw

            poses.append(p)

        goal.poses = poses

        self.get_logger().info("Sending waypoint list...")
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("FollowWaypoints goal rejected!")
            return

        self.get_logger().info("FollowWaypoints goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.print_result)

    def print_result(self, future):
        result = future.result().result
        self.get_logger().info(f"Finished waypoint route, nav error codes: {result.failed_waypoints}")

def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
