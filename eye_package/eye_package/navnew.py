import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# Convert quaternion to yaw angle
def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # Initialization for control and visualization
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(Marker, "/object_markers", 10)
        self.marker_id = 0

        # Initialization for odometry parameters
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # Initialization for laser data parameters
        self.have_scan = False
        self.laser_ranges = None
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        # Initialization for red/green point subscription
        self.create_subscription(Point, "/green_point", self.green_cb, 10)
        self.create_subscription(Point, "/red_point", self.red_cb, 10)

        # Initialization for waypoint list
        self.waypoints = [
            (-1.5, -0.7),
            (1.0, -3.0),
            (4.8, 2.0),
            (5.2, -0.3),
            (-0.2, 4.3),
            (3.63, 3.3),
        ]
        self.goal_index = 0

        # Artificial Potential Field (APF) parameter configuration
        self.k_att = 1.2      # Attractive force coefficient
        self.k_rep = 0.6      # Repulsive force coefficient
        self.d0 = 0.47         # Effective distance of repulsive force
        self.v_max = 0.25      # Maximum forward speed
        self.w_max = 1.0      # Maximum rotation speed (rad/s)

        # Waypoint arrival judgment parameter configuration
        self.goal_tolerance = 0.20        # Arrival judgment radius (meters)
        self.goal_stable_cycles = 20      # Threshold of consecutive cycles within the judgment circle (20×0.05s=1 second)
        self.at_goal_count = 0            # Current number of consecutive cycles within the judgment circle

        # Main loop 20Hz
        self.timer = self.create_timer(0.05, self.main_loop)

        self.get_logger().info("APF NavNode started.")

    # Odometry callback function
    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

    # Laser scan data callback function
    def scan_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, 5.0)
        self.laser_ranges = ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.have_scan = True

    # Red/green point marker publishing function
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

    # Main loop processing function
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

        # Waypoint arrival judgment logic
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

        # Calculate attractive force
        goal_angle = math.atan2(dy, dx)
        goal_angle_robot = (goal_angle - self.robot_yaw + math.pi) % (2*math.pi) - math.pi

        F_att_x = self.k_att * math.cos(goal_angle_robot)
        F_att_y = self.k_att * math.sin(goal_angle_robot)

        # Calculate repulsive force
        F_rep_x, F_rep_y = self.compute_repulsive_force()

        # Calculate resultant force
        Fx = F_att_x + F_rep_x
        Fy = F_att_y + F_rep_y

        force_angle = math.atan2(Fy, Fx)
        forward_component = math.cos(force_angle)

        cmd = Twist()
        # Rotation speed control (positive = turn left, negative = turn right)
        cmd.angular.z = max(-self.w_max, min(self.w_max, force_angle))

        # Forward speed control
        if forward_component < 0:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = min(self.v_max, self.v_max * forward_component)

        self.cmd_pub.publish(cmd)

    # Repulsive force calculation function (fixed deflection ±90°)
    def compute_repulsive_force(self):
        ranges = self.laser_ranges
        n = len(ranges)

        # Unify angles to (-pi, pi), aligned with robot coordinate system
        angles = self.angle_min + np.arange(n) * self.angle_increment
        angles = (angles + math.pi) % (2*math.pi) - math.pi

        Fx, Fy = 0.0, 0.0

        for d, ang in zip(ranges, angles):

            if d < 0.05 or d >= self.d0:
                continue

            # Calculate repulsive force magnitude
            diff = 1.0/d - 1.0/self.d0
            mag = self.k_rep * diff / (d*d + 1e-6)

            # Fixed deflection ±90° (π/2)
            turn = math.pi / 2   # 90°

            # Left obstacle (ang<0) → turn right (+90°)
            # Right obstacle (ang>0) → turn left (-90°)
            if ang < 0:
                rep_angle = ang + turn     # Turn right
            else:
                rep_angle = ang - turn     # Turn left

            fx = mag * math.cos(rep_angle)
            fy = mag * math.sin(rep_angle)

            Fx += fx
            Fy += fy

        return Fx, Fy

    # Action to execute after reaching target: stop for 1 second → rotate one circle → stop for another 1 second
    def rotate_one_circle(self):
        # Stop for 1 second
        stop_cmd = Twist()
        t_end_pause = self.get_clock().now().nanoseconds + int(1.0 * 1e9)  # Stop for 1 second
        while self.get_clock().now().nanoseconds < t_end_pause:
            self.cmd_pub.publish(stop_cmd)

        # Rotate one circle (speed 90°/s, duration 4 seconds)
        cmd = Twist()
        cmd.angular.z = math.radians(60)    # 60°/s
        duration = 6.0                       # Rotate for 4 seconds

        t_end_spin = self.get_clock().now().nanoseconds + int(duration * 1e9)
        while self.get_clock().now().nanoseconds < t_end_spin:
            self.cmd_pub.publish(cmd)
            
        # Stop for another 1 second
        stop_cmd = Twist()
        t_end_pause = self.get_clock().now().nanoseconds + int(1.0 * 1e9)  # Stop for 1 second
        while self.get_clock().now().nanoseconds < t_end_pause:
            self.cmd_pub.publish(stop_cmd)

        # Stop the robot
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
