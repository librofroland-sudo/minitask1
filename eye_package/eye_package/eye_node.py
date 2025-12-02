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
#                        主类：ImageSubscriber
# ============================================================
class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('eye_node')

        # 必须提前初始化 CvBridge，否则回调会崩溃
        self.br = CvBridge()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------- 订阅图像 -----------
        self.create_subscription(Image, '/camera_depth/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera_depth/depth/image_raw', self.dimage_callback, 10)

        # 相机内参
        self.create_subscription(CameraInfo, '/camera_depth/camera_info', self.ins_callback, 10)

        # 里程计（可选）
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # 初始状态
        self.image = None
        self.dimage = None
        self.ins = None

        # ----------- 发布绿色 + 红色世界坐标 -----------
        self.green_pub = self.create_publisher(Point, 'green_point', 10)
        self.red_pub = self.create_publisher(Point, 'red_point', 10)

        # 定时器（5Hz）
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info(" eye_node 启动成功，等待图像...")

    # ------------------ 回调区 ------------------
    def odom_callback(self, msg):
        _, _, self.orientation = quaternion_to_rpy(msg.pose.pose.orientation)

    def ins_callback(self, data):
        self.ins = data

    def image_callback(self, data):
        self.image = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

    def dimage_callback(self, data):
        self.dimage = self.br.imgmsg_to_cv2(data, desired_encoding='passthrough')

    # ------------------ 获取 TF ------------------
    def tf_from_cam_to_map(self):
        try:
            return self.tf_buffer.lookup_transform(
                'map', 'camera_rgb_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f"TF 转换失败: {e}")
            return None

    # ============================================================
    #                     颜色检测 + 深度 + 发布世界坐标
    # ============================================================
    def timer_callback(self):

        # 等待图像
        if self.image is None or self.dimage is None or self.ins is None:
            return

        frame = self.image
        depth_frame = self.dimage

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ======================================================
        #  🟩 深绿色垃圾桶 ——严格版 HSV
        # ======================================================
        lower_green = np.array([45, 70, 20])
        upper_green = np.array([75, 255, 180])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # ======================================================
        #  🔴 红色消防栓 —— 红色分两段
        # ======================================================
        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 150, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # -------- 形态学降噪 --------
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        # ======================================================
        #  函数：找最大面积 + 圆形度过滤（重要！防止误检）
        # ======================================================
        def find_target(mask):

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt = None
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:     # 最小面积过滤
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                if circularity < 0.3:
                    continue   # 非圆形全部排除

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

        centroid_green = find_target(mask_green)
        centroid_red = find_target(mask_red)

        # ======================================================
        #  函数：像素 → 深度 → 相机坐标 → map 坐标 → 发布
        # ======================================================
        def process_color(centroid, pub, color_name):

            if centroid is None:
                return

            cx, cy = centroid
            depth_val = float(depth_frame[cy, cx])

            # 深度过滤
            if depth_val <= 0.0 or math.isnan(depth_val) or math.isinf(depth_val):
                return

            # 相机内参
            cam = self.ins
            intr = rs.intrinsics()
            intr.width = cam.width
            intr.height = cam.height
            intr.ppx = cam.k[2]
            intr.ppy = cam.k[5]
            intr.fx = cam.k[0]
            intr.fy = cam.k[4]
            intr.model = rs.distortion.none
            intr.coeffs = list(cam.d)

            # 像素 → 相机坐标（米）
            p3d = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)

            point_cam = PointStamped()
            point_cam.header.frame_id = 'camera_rgb_optical_frame'
            point_cam.point.x = p3d[0]
            point_cam.point.y = p3d[1]
            point_cam.point.z = p3d[2]

            tf = self.tf_from_cam_to_map()
            if tf is None:
                return

            point_world = do_transform_point(point_cam, tf)

            # 发布
            msg = Point()
            msg.x = point_world.point.x
            msg.y = point_world.point.y
            msg.z = point_world.point.z
            pub.publish(msg)

            self.get_logger().info(
                f"检测到 {color_name} 物体 → 世界坐标: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})"
            )

            return (cx, cy)

        # ============= 分别处理绿色与红色 ================
        g_px = process_color(centroid_green, self.green_pub, "绿色")
        r_px = process_color(centroid_red, self.red_pub, "红色")

        # ============= Debug 显示（画圆） ================
        debug = frame.copy()
        if g_px: cv2.circle(debug, g_px, 6, (0, 255, 0), -1)
        if r_px: cv2.circle(debug, r_px, 6, (0, 0, 255), -1)

        cv2.imshow("camera", debug)
        cv2.imshow("green_mask", mask_green)
        cv2.imshow("red_mask", mask_red)
        cv2.waitKey(1)


# ============================================================
#                         启动入口
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
