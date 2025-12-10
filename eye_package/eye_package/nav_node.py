#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# ---------- 四元数 → yaw ----------
def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # ================== 控制与可视化 ==================
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(Marker, "/object_markers", 10)
        self.marker_id = 0

        # ================== 里程计 ==================
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # ================== 激光 ==================
        self.have_scan = False
        self.laser_ranges = None
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        # ================== 红绿点 ==================
        self.create_subscription(Point, "/green_point", self.green_cb, 10)
        self.create_subscription(Point, "/red_point", self.red_cb, 10)

        # ================== 路径点 ==================
        self.waypoints = [
            (-1.0, -1.0),
            (-1.6, 0.5),
            (1.0, -3.0),
            (4.8, 2.0),
            (4.8, -1.0),
            (-3.0, 4.0),
            (-0.3, 4.3),
            (3.5, 3.8),
        ]
        self.goal_index = 0

        # ================== 势场法参数 ==================
        self.k_att = 1.2      # 吸引力
        self.k_rep = 0.5      # 排斥力
        self.d0 = 0.4         # 排斥力有效距离
        self.v_max = 0.2     # 最大前进速度
        self.w_max = 1.8      # 最大旋转速度（rad/s）

        # ====== 强化“到达路径点”的判定 ======
        self.goal_tolerance = 0.20        # 到达判定半径（米）
        self.goal_stable_cycles = 20      # 连续多少次（20×0.05s=1秒）在圈内才算到达
        self.at_goal_count = 0            # 当前已经连续在圈内的次数

        # 主循环 20Hz
        self.timer = self.create_timer(0.05, self.main_loop)

        self.get_logger().info("APF NavNode started.")

    # ---------------- 里程计 ----------------
    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

    # ---------------- 激光 ----------------
    def scan_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, 5.0)
        self.laser_ranges = ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.have_scan = True

    # -------------- 红绿点标记 ----------------
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
        self.publish_marker(msg, "GREEN")

    def red_cb(self, msg):
        self.publish_marker(msg, "RED")

    # ============================================================
    #                     主循环
    # ============================================================
    def main_loop(self):
        if self.goal_index >= len(self.waypoints):
            self.stop_robot()
            return

        if not self.have_scan:
            self.stop_robot()
            return

        gx, gy = self.waypoints[self.goal_index]

        dx = gx - self.robot_x
        dy = gy - self.robot_y
        dist = math.hypot(dx, dy)

        # ------------ 强化：到达路径点判定 ------------
        if dist < self.goal_tolerance:
            self.at_goal_count += 1
        else:
            self.at_goal_count = 0

        if self.at_goal_count >= self.goal_stable_cycles:
            self.get_logger().info(
                f"Arrived waypoint {self.goal_index} (dist={dist:.2f}), rotating then going to next."
            )
            self.rotate_one_circle()
            self.goal_index += 1
            self.at_goal_count = 0
            return

        # ------- 吸引力 -------
        goal_angle = math.atan2(dy, dx)
        goal_angle_robot = (goal_angle - self.robot_yaw + math.pi) % (2*math.pi) - math.pi

        F_att_x = self.k_att * math.cos(goal_angle_robot)
        F_att_y = self.k_att * math.sin(goal_angle_robot)

        # ------- 排斥力 -------
        F_rep_x, F_rep_y = self.compute_repulsive_force()

        # ------- 合力 -------
        Fx = F_att_x + F_rep_x
        Fy = F_att_y + F_rep_y

        force_angle = math.atan2(Fy, Fx)
        forward_component = math.cos(force_angle)

        cmd = Twist()
        # 旋转（正 = 左转，负 = 右转）
        cmd.angular.z = max(-self.w_max, min(self.w_max, force_angle))

        # 前进速度
        if forward_component < 0:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = min(self.v_max, self.v_max * forward_component)

        self.cmd_pub.publish(cmd)

    # ============================================================
    #            排斥力（固定偏转 ±90°，不再动态变化）
    # ============================================================
    def compute_repulsive_force(self):
        ranges = self.laser_ranges
        n = len(ranges)

        # 统一角度为 (-pi, pi)，与机器人坐标系对齐
        angles = self.angle_min + np.arange(n) * self.angle_increment
        angles = (angles + math.pi) % (2*math.pi) - math.pi

        Fx, Fy = 0.0, 0.0

        for d, ang in zip(ranges, angles):

            if d < 0.05 or d >= self.d0:
                continue

            # ===== 排斥力强度 =====
            diff = 1.0/d - 1.0/self.d0
            mag = self.k_rep * diff / (d*d + 1e-6)

            # 固定偏转 ±90°（π/2）
            turn = math.pi / 2   # 90°

            # 注意：你说左边是负角度，右边是正角度
            # 左侧障碍（ang<0）→ 应该向右转（+90°）
            # 右侧障碍（ang>0）→ 应该向左转（-90°）
            if ang < 0:
                rep_angle = ang + turn     # 右转
            else:
                rep_angle = ang - turn     # 左转

            fx = mag * math.cos(rep_angle)
            fy = mag * math.sin(rep_angle)

            Fx += fx
            Fy += fy

        return Fx, Fy

    # ============================================================
    #      到达目标后：静止 1 秒 → 再旋转 1 圈
    # ============================================================
    def rotate_one_circle(self):
        # ---------- 第一步：静止 ----------
        stop_cmd = Twist()
        t_end_pause = self.get_clock().now().nanoseconds + int(1.0 * 1e9)  # 静止 1 秒
        while self.get_clock().now().nanoseconds < t_end_pause:
            self.cmd_pub.publish(stop_cmd)

        # ---------- 第二步：旋转一圈（慢速） ----------
        cmd = Twist()
        cmd.angular.z = math.radians(90)    # 90°/s
        duration = 4.0                       # 旋转 4 秒

        t_end_spin = self.get_clock().now().nanoseconds + int(duration * 1e9)
        while self.get_clock().now().nanoseconds < t_end_spin:
            self.cmd_pub.publish(cmd)
            
        stop_cmd = Twist()
        t_end_pause = self.get_clock().now().nanoseconds + int(1.0 * 1e9)
        while self.get_clock().now().nanoseconds < t_end_pause:
            self.cmd_pub.publish(stop_cmd)

        # ---------- 结束 ----------
        self.stop_robot()

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
