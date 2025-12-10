#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node

from geometry_msgs.msg import Quaternion, PointStamped, Point
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point

from rclpy.duration import Duration
from rclpy.time import Time

import numpy as np
import cv2
import pyrealsense2 as rs


# ---------- 四元数转欧拉角 ----------
def quaternion_to_rpy(q: Quaternion):
    x, y, z, w = q.x, q.y, q.z, q.w
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ============================================================
#                        主类
# ============================================================
class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('eye_node')

        # CvBridge
        self.br = CvBridge()

        # TF Buffer + Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 订阅深度和彩色图
        self.create_subscription(Image, '/camera_depth/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera_depth/depth/image_raw', self.dimage_callback, 10)

        # 相机内参
        self.create_subscription(CameraInfo, '/camera_depth/camera_info', self.ins_callback, 10)

        # Odom
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # 发布物体坐标
        self.green_pub = self.create_publisher(Point, 'green_point', 10)
        self.red_pub = self.create_publisher(Point, 'red_point', 10)

        # 状态变量
        self.image = None
        self.dimage = None
        self.ins = None
        self.orientation = 0.0

        # 定时器 5Hz
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info("eye_node 启动成功，等待相机数据...")


    # -------------------- 回调们 --------------------
    def odom_callback(self, msg):
        _, _, self.orientation = quaternion_to_rpy(msg.pose.pose.orientation)

    def ins_callback(self, msg):
        self.ins = msg

    def image_callback(self, msg):
        self.image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def dimage_callback(self, msg):
        self.dimage = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')


    # -------------------- 修复后的 TF 获取函数 --------------------
    def tf_from_cam_to_map(self):

        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                'camera_rgb_optical_frame',
                Time(),                           # 最新时间戳
                timeout=Duration(seconds=0.5)
            )
            return tf

        except Exception as e:
            self.get_logger().warn(f"TF 转换失败: {e}")
            return None


    # ============================================================
    #                     图像处理 + 坐标发布
    # ============================================================
    def timer_callback(self):

        if self.image is None or self.dimage is None or self.ins is None:
            return

        frame = self.image
        depth_frame = self.dimage

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ----------------- 绿色检测 -----------------
        lower_green = np.array([45, 70, 20])
        upper_green = np.array([75, 255, 180])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # ----------------- 红色检测（双区间） -----------------
        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 150, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # 去噪
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        # ----------------- 找最大圆形目标 -----------------
        def find_target(mask):

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best = None
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue

                circularity = 4 * math.pi * (area / (peri * peri))
                if circularity < 0.3:
                    continue

                if area > best_area:
                    best_area = area
                    best = cnt

            if best is None:
                return None

            M = cv2.moments(best)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            return (cx, cy)

        centroid_green = find_target(mask_green)
        centroid_red = find_target(mask_red)


        # ----------------- 计算世界坐标并发布 -----------------
        def process_color(centroid, pub, color_name):

            if centroid is None:
                return None

            cx, cy = centroid

            depth_val = float(depth_frame[cy, cx])
            if depth_val <= 0.0 or math.isnan(depth_val):
                return None

            cam = self.ins

            # RealSense 内参写法
            intr = rs.intrinsics()
            intr.width = cam.width
            intr.height = cam.height
            intr.ppx = cam.k[2]
            intr.ppy = cam.k[5]
            intr.fx = cam.k[0]
            intr.fy = cam.k[4]
            intr.model = rs.distortion.none
            intr.coeffs = list(cam.d)

            p3d = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)

            point_cam = PointStamped()
            point_cam.header.frame_id = 'camera_rgb_optical_frame'
            point_cam.point.x = p3d[0]
            point_cam.point.y = p3d[1]
            point_cam.point.z = p3d[2]

            # 获取 TF
            tf = self.tf_from_cam_to_map()
            if tf is None:
                return None

            point_world = do_transform_point(point_cam, tf)

            msg = Point()
            msg.x = point_world.point.x
            msg.y = point_world.point.y
            msg.z = point_world.point.z
            pub.publish(msg)

            self.get_logger().info(
                f"{color_name} 目标 → 世界坐标: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})"
            )

            return (cx, cy)


        gp = process_color(centroid_green, self.green_pub, "绿色")
        rp = process_color(centroid_red, self.red_pub, "红色")

        # 可视化
        debug = frame.copy()
        if gp: cv2.circle(debug, gp, 6, (0, 255, 0), -1)
        if rp: cv2.circle(debug, rp, 6, (0, 0, 255), -1)

        cv2.imshow("camera", debug)
        cv2.imshow("green_mask", mask_green)
        cv2.imshow("red_mask", mask_red)
        cv2.waitKey(1)


# ============================================================
#                         main()
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
