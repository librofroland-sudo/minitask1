import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, PointStamped, Point   # ★ 新增 Point 用于发布
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np
import cv2
import pyrealsense2 as rs

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

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscription_image = self.create_subscription(
            Image, '/camera_depth/image_raw', self.image_callback, 10)

        self.subscription_dimage = self.create_subscription(
            Image, '/camera_depth/depth/image_raw', self.dimage_callback, 10)

        self.subscription_int = self.create_subscription(
            CameraInfo, '/camera_depth/camera_info', self.ins_callback, 10)

        self.subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        self.ins = None
        self.image = []
        self.dimage = []

        self.br = CvBridge()

        # ★★★ 新增：发布绿色物体世界坐标的 Publisher
        self.point_pub = self.create_publisher(Point, 'green_point', 10)

        self.timer = self.create_timer(0.2, self.timer_callback)

    def odom_callback(self, msg):
        self.location = (msg.pose.pose.position.x,
                         msg.pose.pose.position.y,
                         msg.pose.pose.position.z)
        _, _, self.orientation = quaternion_to_rpy(msg.pose.pose.orientation)

    def ins_callback(self, data):
        self.ins = data

    def tf_from_cam_to_map(self):
        from_frame = 'camera_rgb_optical_frame'
        to_frame = 'map'
        now = rclpy.time.Time()

        try:
            tf = self.tf_buffer.lookup_transform(
                to_frame, from_frame, now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return tf
        except:
            return None

    def image_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.image = current_frame

    def dimage_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.dimage = current_frame

    def timer_callback(self):

        current_frame = self.image
        depth_frame = self.dimage

        if isinstance(current_frame, list) or self.ins is None:
            return

        # ---- 绿色检测 ----
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        depths = []

        min_area = 500

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            depth_value = depth_frame[cy, cx] if depth_frame is not None else 0

            if depth_value == 0 or math.isnan(depth_value):
                continue

            centroids.append((cx, cy))
            depths.append(float(depth_value))

        if len(centroids) == 0:
            cv2.imshow("camera", current_frame)
            cv2.imshow("mask", mask)
            cv2.waitKey(1)
            return

        # ---- 深度相机内参 ----
        cameraInfo = self.ins
        _intrinsics = rs.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.k[2]
        _intrinsics.ppy = cameraInfo.k[5]
        _intrinsics.fx = cameraInfo.k[0]
        _intrinsics.fy = cameraInfo.k[4]
        _intrinsics.model = rs.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.d]

        # ---- 取最大的绿色物体（第一个 contour）----
        (cx, cy) = centroids[0]
        depth = depths[0]

        # ---- 像素转相机坐标 ----
        point_3d = rs.rs2_deproject_pixel_to_point(_intrinsics, [cx, cy], depth)

        point_cam = PointStamped()
        point_cam.header.frame_id = 'camera_rgb_optical_frame'
        point_cam.point.x = point_3d[0]
        point_cam.point.y = point_3d[1]
        point_cam.point.z = point_3d[2]

        # ---- 坐标转换（相机 → map）----
        tf = self.tf_from_cam_to_map()
        if tf is None:
            return

        point_world = do_transform_point(point_cam, tf)

        world_x = point_world.point.x
        world_y = point_world.point.y
        world_z = point_world.point.z

        print("GREEN OBJ WORLD:", world_x, world_y, world_z)

        # ---- ★★★ 新增：发布世界坐标 Point ----
        msg = Point()
        msg.x = world_x
        msg.y = world_y
        msg.z = world_z
        self.point_pub.publish(msg)      # ★ 发布绿色物体坐标

        # ---- Debug 显示 ----
        debug_img = current_frame.copy()
        cv2.circle(debug_img, (cx, cy), 6, (0, 0, 255), -1)

        cv2.imshow("camera", debug_img)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
