import rclpy
import math
from rclpy.node import Node

from geometry_msgs.msg import Quaternion, PointStamped, Point
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point

import numpy as np
import cv2
import pyrealsense2 as rs


# ---------- 工具：四元数转 yaw ----------
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


class EyeNode(Node):
    def __init__(self):
        super().__init__("eye_node")

        # ---- CvBridge 提前初始化 ----
        self.br = CvBridge()

        # ---- TF：camera_rgb_optical_frame → map ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- 订阅相机图像与深度 ----
        self.create_subscription(Image, "/camera_depth/image_raw", self.image_callback, 10)
        self.create_subscription(Image, "/camera_depth/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/camera_depth/camera_info", self.caminfo_callback, 10)

        # 可选：订阅 odom（不一定用得上）
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # 当前帧数据
        self.image = None
        self.depth = None
        self.camera_info = None

        # ---- 发布绿色/红色物体世界坐标 ----
        self.green_pub = self.create_publisher(Point, "/green_point", 10)
        self.red_pub = self.create_publisher(Point, "/red_point", 10)

        # 定时器：图像处理主循环（5Hz）
        self.timer = self.create_timer(0.2, self.process)

        self.get_logger().info("eye_node started, waiting for camera data...")

    # ----------------- 回调：里程计（可选） -----------------
    def odom_callback(self, msg):
        # 这里暂时不使用，只是保留接口
        pass

    # ----------------- 回调：相机内参 -----------------
    def caminfo_callback(self, msg: CameraInfo):
        self.camera_info = msg

    # ----------------- 回调：RGB 图像 -----------------
    def image_callback(self, msg: Image):
        self.image = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # ----------------- 回调：深度图像 -----------------
    def depth_callback(self, msg: Image):
        # encoding: 32FC1 → float32, 单位：米
        self.depth = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # ----------------- 获取 TF：camera → map -----------------
    def tf_cam_to_map(self):
        try:
            return self.tf_buffer.lookup_transform(
                "map",
                "camera_rgb_optical_frame",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    # ----------------- 主循环：颜色检测 + 深度 + TF -----------------
    def process(self):
        if self.image is None or self.depth is None or self.camera_info is None:
            return

        frame = self.image
        depth_frame = self.depth

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ============ 绿色垃圾桶（较深的绿） ============
        # 根据实际画面可以稍微调节
        green_lower = np.array([45, 70, 20])
        green_upper = np.array([75, 255, 200])
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        # ============ 红色消防栓（分两段红） ============
        red_lower1 = np.array([0, 150, 50])
        red_upper1 = np.array([8, 255, 255])
        red_lower2 = np.array([172, 150, 50])
        red_upper2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)

        # 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        # 找最大连通区域 + 圆形度过滤（减少误检）
        def find_centroid(mask, min_area=800, min_circularity=0.2):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_cnt = None
            best_area = 0.0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue

                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity < min_circularity:
                    continue

                if area > best_area:
                    best_area = area
                    best_cnt = cnt

            if best_cnt is None:
                return None

            M = cv2.moments(best_cnt)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)

        green_c = find_centroid(mask_green)
        red_c = find_centroid(mask_red)

        # 将像素点 + 深度 → map 世界坐标，并发布
        self.handle_color_target(green_c, depth_frame, self.green_pub, "GREEN", frame)
        self.handle_color_target(red_c, depth_frame, self.red_pub, "RED", frame)

        # 调试显示（如果有图形界面）
        debug = frame.copy()
        if green_c:
            cv2.circle(debug, green_c, 6, (0, 255, 0), -1)
        if red_c:
            cv2.circle(debug, red_c, 6, (0, 0, 255), -1)

        cv2.imshow("camera_debug", debug)
        cv2.imshow("green_mask", mask_green)
        cv2.imshow("red_mask", mask_red)
        cv2.waitKey(1)

    # ----------------- 颜色目标处理：像素 → 深度 → 相机坐标 → map 坐标 -----------------
    def handle_color_target(self, centroid, depth_frame, pub: rclpy.node.Publisher, color_name: str, frame_bgr):
        if centroid is None:
            return

        cx, cy = centroid

        # 取该点深度
        depth_value = float(depth_frame[cy, cx])
        if depth_value <= 0.0 or math.isnan(depth_value) or math.isinf(depth_value):
            return

        cam_info = self.camera_info
        intr = rs.intrinsics()
        intr.width = cam_info.width
        intr.height = cam_info.height
        intr.ppx = cam_info.k[2]
        intr.ppy = cam_info.k[5]
        intr.fx = cam_info.k[0]
        intr.fy = cam_info.k[4]
        intr.model = rs.distortion.none
        intr.coeffs = list(cam_info.d)

        # 像素 → 相机坐标（米）
        p3d = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_value)

        point_cam = PointStamped()
        point_cam.header.frame_id = "camera_rgb_optical_frame"
        point_cam.point.x = p3d[0]
        point_cam.point.y = p3d[1]
        point_cam.point.z = p3d[2]

        tf = self.tf_cam_to_map()
        if tf is None:
            return

        point_world = do_transform_point(point_cam, tf)

        msg = Point()
        msg.x = point_world.point.x
        msg.y = point_world.point.y
        msg.z = point_world.point.z
        pub.publish(msg)

        self.get_logger().info(
            f"{color_name} OBJ in map frame: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = EyeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
