import rclpy
from rclpy.node import Node

import math
import random
import time

from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker


# ---------- 工具：四元数转 yaw ----------
def quaternion_to_yaw(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================
#                      A* 全局规划器
# ============================================================
class AStarPlanner:
    def __init__(self, occ_grid: OccupancyGrid, extra_obstacles=None):
        self.grid = occ_grid.data
        self.width = occ_grid.info.width
        self.height = occ_grid.info.height
        self.resolution = occ_grid.info.resolution
        self.origin_x = occ_grid.info.origin.position.x
        self.origin_y = occ_grid.info.origin.position.y

        self.extra_occ = set()
        if extra_obstacles:
            for (ox, oy) in extra_obstacles:
                gx, gy = self.world_to_grid(ox, oy)
                self.extra_occ.add((gx, gy))

    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin_x
        y = gy * self.resolution + self.origin_y
        return x, y

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_occupied(self, gx, gy):
        if not self.in_bounds(gx, gy):
            return True
        if (gx, gy) in self.extra_occ:
            return True

        idx = gy * self.width + gx
        v = self.grid[idx]
        return v >= 50

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, sx, sy, gx, gy):
        start = self.world_to_grid(sx, sy)
        goal = self.world_to_grid(gx, gy)

        if self.is_occupied(*start) or self.is_occupied(*goal):
            return None

        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda o: f_score.get(o, float("inf")))
            if current == goal:
                path = self.reconstruct(came_from, current)
                return [self.grid_to_world(gx, gy) for (gx, gy) in path]

            open_set.remove(current)
            cx, cy = current

            for nx, ny in [
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1),
                (cx + 1, cy + 1),
                (cx - 1, cy - 1),
                (cx + 1, cy - 1),
                (cx - 1, cy + 1),
            ]:
                if self.is_occupied(nx, ny):
                    continue

                new_g = g_score[current] + 1
                if new_g < g_score.get((nx, ny), 999999):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = new_g
                    f_score[(nx, ny)] = new_g + self.heuristic((nx, ny), goal)
                    if (nx, ny) not in open_set:
                        open_set.append((nx, ny))

        return None

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


# ============================================================
#                      主节点：NavNode
# ============================================================
class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # ==================== Publisher ====================
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.object_marker_pub = self.create_publisher(Marker, "object_markers", 10)
        self.obstacle_marker_pub = self.create_publisher(Marker, "obstacle_markers", 10)
        self.path_marker_pub = self.create_publisher(Marker, "planned_path", 10)
        self.found_marker_pub = self.create_publisher(Marker, "found_targets", 10)

        # ==================== Subscriber ====================
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Point, "/green_point", self.green_callback, 10)
        self.create_subscription(Point, "/red_point", self.red_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.pose_callback, 10)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)

        # ==================== 状态变量 ====================
        self.scan = None
        self.map_data = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.current_target = None       # geometry_msgs/Point
        self.current_target_label = None # 'green' 或 'red'

        self.detected_green = []   # [(x, y), ...]
        self.detected_red = []     # [(x, y), ...]
        self.visited_targets = []  # [(label, x, y), ...]
        self.obstacles = []        # [(x, y), ...]

        self.global_path = None
        self.current_wp_index = 0

        # ==================== 模式 ====================
        self.MODE_STARTUP = -1     # 启动观察模式
        self.MODE_EXPLORE = 0
        self.MODE_FOLLOW_PATH = 1
        self.MODE_GOTO_DIRECT = 2
        self.MODE_FINISHED = 3

        self.mode = self.MODE_STARTUP  # 初始模式为 STARTUP

        # startup 参数
        self.start_time = time.time()
        self.startup_delay = 2.0  # 启动等待 2 秒

        # 速度 / 距离参数（调慢一点 + 墙体“增厚”少一点）
        self.safe_distance = 0.5          # 目标安全距离（稍微靠近一点）
        self.obstacle_threshold = 0.4     # 只把 0.4m 内的障碍记录到地图，减小“增厚”效果

        self.last_obstacle_mark_time = 0.0
        self.max_objects = 6

        # 控制循环（10Hz）
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("nav_node started with 2s startup observation delay.")

    # ---------------- 回调函数 ----------------
    def scan_callback(self, msg):
        self.scan = msg

    def pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

    def map_callback(self, msg):
        self.map_data = msg

    # ----------- 绿色 / 红色目标回调 -----------
    def green_callback(self, msg: Point):
        self.handle_detected_target(msg, "green")

    def red_callback(self, msg: Point):
        self.handle_detected_target(msg, "red")

    def handle_detected_target(self, point: Point, label: str):
        """
        处理新检测到的目标：
        - 如果这个目标已经在 visited_targets 中（同颜色 + 坐标接近），直接忽略
        - 否则加入检测列表，并在当前没有目标时将其设为当前目标
        - 尝试用 A* 规划路径，规划失败则用直接追
        """
        x, y = point.x, point.y

        # 已经访问过的目标，不再追
        if self.is_target_visited(label, x, y):
            return

        # 记录为新的检测目标（去重）
        container = self.detected_green if label == "green" else self.detected_red
        if self.add_if_new(container, x, y, min_dist=0.3):
            self.publish_object_marker(x, y, label)

        # 当前没有正在追的目标，且比赛还没结束，才切换目标
        if self.current_target is None and self.mode != self.MODE_FINISHED:
            self.current_target = point
            self.current_target_label = label
            self.plan_global_path_if_possible()

    # =============== STARTUP 2 秒观察模式 ===============
    def handle_startup(self):
        if time.time() - self.start_time < self.startup_delay:
            stop = Twist()
            self.cmd_pub.publish(stop)
            return True  # 仍在 startup 阶段
        else:
            print("Startup observation complete → switching to EXPLORE.")
            self.mode = self.MODE_EXPLORE
            return False

    # ====================================================
    #                    控制主循环
    # ====================================================
    def control_loop(self):
        if self.scan is None:
            return

        # ----------- 启动等待阶段 -----------
        if self.mode == self.MODE_STARTUP:
            if self.handle_startup():
                return

        # ---- 原控制逻辑 ----
        self.detect_and_mark_obstacles()

        # 所有目标都找完
        if len(self.visited_targets) >= self.max_objects:
            self.mode = self.MODE_FINISHED

        if self.mode == self.MODE_FINISHED:
            self.do_finished_behavior()
            return
        elif self.mode == self.MODE_EXPLORE:
            self.do_explore()
        elif self.mode == self.MODE_FOLLOW_PATH:
            self.follow_global_path()
        elif self.mode == self.MODE_GOTO_DIRECT:
            self.goto_target_direct()

    # ==================== 工具函数 ====================

    def get_range_at_angle(self, angle_rad):
        """
        从 LaserScan 中取出指定角度方向的距离（机器人坐标系，0 前方）
        """
        if self.scan is None:
            return None

        ang_min = self.scan.angle_min
        ang_inc = self.scan.angle_increment
        index = int((angle_rad - ang_min) / ang_inc)

        if index < 0 or index >= len(self.scan.ranges):
            return None

        d = self.scan.ranges[index]
        if math.isinf(d) or math.isnan(d):
            return None
        return d

    def add_if_new(self, container, x, y, min_dist=0.2):
        """
        容器里保存 (x, y)，若离已有点都大于 min_dist，就加入并返回 True
        否则返回 False
        """
        for (ix, iy) in container:
            if math.hypot(ix - x, iy - y) < min_dist:
                return False
        container.append((x, y))
        return True

    def is_target_visited(self, label, x, y, tol=0.4):
        """
        判断这个颜色 + 坐标附近的目标是否已经“找到”过。
        """
        for (lbl, tx, ty) in self.visited_targets:
            if lbl == label and math.hypot(tx - x, ty - y) < tol:
                return True
        return False

    def plan_global_path_if_possible(self):
        """
        尝试基于当前 map + obstacles + 目标，规划 A* 路径。
        成功：进入 FOLLOW_PATH
        失败：进入 GOTO_DIRECT
        """
        if self.map_data is None or self.current_target is None:
            self.global_path = None
            self.current_wp_index = 0
            self.mode = self.MODE_GOTO_DIRECT
            return

        planner = AStarPlanner(self.map_data, extra_obstacles=self.obstacles)
        path = planner.plan(
            self.robot_x,
            self.robot_y,
            self.current_target.x,
            self.current_target.y,
        )

        if path is None:
            print("A* planning failed, using direct mode.")
            self.global_path = None
            self.current_wp_index = 0
            self.mode = self.MODE_GOTO_DIRECT
        else:
            print(f"A* planning success, {len(path)} waypoints.")
            self.global_path = path
            self.current_wp_index = 0
            self.mode = self.MODE_FOLLOW_PATH
            self.publish_path_marker(path)

    # ==================== Marker 发布 ====================

    def publish_object_marker(self, x, y, label):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "objects"
        marker.id = int(time.time() * 1000) % 1000000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        if label == "green":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        marker.color.a = 1.0

        self.object_marker_pub.publish(marker)

    def publish_obstacle_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = int(time.time() * 1000) % 1000000
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0

        self.obstacle_marker_pub.publish(marker)

    def publish_path_marker(self, path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.points = []
        for (x, y) in path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)

    def publish_found_marker(self, x, y, label):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "found"
        marker.id = int(time.time() * 1000) % 1000000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.2
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        # 用黄色表示“已找到”
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.found_marker_pub.publish(marker)

    # ==================== 障碍检测 + 记录 ====================

    def detect_and_mark_obstacles(self):
        """
        使用激光雷达 + amcl 位姿，在 map 中记录障碍物位置，
        并在 RViz 画出来，同时影响下次 A* 规划。
        """
        if self.scan is None:
            return

        now = time.time()
        if now - self.last_obstacle_mark_time < 1.0:
            return

        ranges = self.scan.ranges
        ang_min = self.scan.angle_min
        ang_inc = self.scan.angle_increment
        n = len(ranges)

        found_new = False
        step = max(1, n // 36)  # 每约 10 度采样一次

        for i in range(0, n, step):
            d = ranges[i]
            if math.isinf(d) or math.isnan(d):
                continue
            if d > self.obstacle_threshold:
                continue  # 只记比较近的障碍，避免墙体“增厚”太夸张

            angle = ang_min + i * ang_inc
            world_angle = self.robot_yaw + angle

            ox = self.robot_x + d * math.cos(world_angle)
            oy = self.robot_y + d * math.sin(world_angle)

            if self.add_if_new(self.obstacles, ox, oy, min_dist=0.2):
                self.publish_obstacle_marker(ox, oy)
                found_new = True

        if found_new:
            print(f"Obstacle count: {len(self.obstacles)}")
            self.last_obstacle_mark_time = now

    # ==================== 模式 0：探索 ====================

    def do_explore(self):
        """
        探索模式：
        - 激光避障
        - 沿墙行驶
        - 加随机扰动遍历更多区域
        速度整体比之前略慢，更稳一点。
        """
        cmd = Twist()

        d_front = self.get_range_at_angle(0.0)
        d_left = self.get_range_at_angle(+math.pi / 2)
        d_right = self.get_range_at_angle(-math.pi / 2)

        if d_front is None:
            d_front = 10.0
        if d_left is None:
            d_left = 10.0
        if d_right is None:
            d_right = 10.0

        if d_front < 0.35:
            cmd.linear.x = -0.06   # 慢一点后退
            cmd.angular.z = -0.9
            print("EMERGENCY: Obstacle ahead in explore mode → backing & turning.")
        elif d_front < 0.7:
            cmd.linear.x = 0.0
            cmd.angular.z = -0.8
            print("Avoiding front obstacle in explore mode → turning right.")
        elif d_left < 0.4:
            cmd.linear.x = 0.10    # 比原来慢
            cmd.angular.z = -0.4
        elif d_right < 0.4:
            cmd.linear.x = 0.10
            cmd.angular.z = 0.4
        else:
            cmd.linear.x = 0.15    # 原来 0.20，整体慢一点
            cmd.angular.z = random.uniform(-0.25, 0.25)

        self.cmd_pub.publish(cmd)

    # ==================== 模式 1：跟随全局路径 ====================

    def follow_global_path(self):
        """
        跟随 A* 规划的 waypoint。
        同时保持“优先避障”。
        线速度整体调慢。
        """
        if self.global_path is None or self.current_wp_index >= len(self.global_path):
            print("Global path finished or invalid, switch to direct mode.")
            self.global_path = None
            self.current_wp_index = 0
            self.mode = self.MODE_GOTO_DIRECT
            return

        wx, wy = self.global_path[self.current_wp_index]

        d_front = self.get_range_at_angle(0.0)
        if d_front is None:
            d_front = 10.0

        cmd = Twist()

        # 紧急避障
        if d_front < 0.35:
            cmd.linear.x = -0.08   # 慢一点后退
            cmd.angular.z = -0.9
            print("🔥 EMERGENCY while following path → backing up.")
            self.cmd_pub.publish(cmd)
            return

        # 一般避障
        if d_front < 0.6:
            cmd.linear.x = 0.0
            cmd.angular.z = -0.7
            print("⚠️ Avoiding obstacle while following path.")
            self.cmd_pub.publish(cmd)
            return

        dx = wx - self.robot_x
        dy = wy - self.robot_y
        dist = math.hypot(dx, dy)
        angle_world = math.atan2(dy, dx)
        angle_robot = angle_world - self.robot_yaw

        # waypoint 已经接近 → 切到下一个
        if dist < 0.3:
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.global_path):
                print("Reached final waypoint of global path, switching to direct target tracking.")
                self.global_path = None
                self.current_wp_index = 0
                self.mode = self.MODE_GOTO_DIRECT
            return

        cmd.angular.z = 1.5 * angle_robot

        # 线速度整体调慢
        if abs(angle_robot) < math.pi / 6:
            cmd.linear.x = 0.15   # 原来 0.20
        elif abs(angle_robot) < math.pi / 3:
            cmd.linear.x = 0.08   # 原来 0.10
        else:
            cmd.linear.x = 0.0

        if d_front < 0.45:
            cmd.linear.x = 0.0

        self.cmd_pub.publish(cmd)

    # ==================== 模式 2：直接追目标 ====================

    def goto_target_direct(self):
        """
        无法规划全局路径时，使用简单的“朝目标旋转 + 前进”逻辑，
        仍然带有“优先避障”和“到达后记录 + 继续探索”行为。
        找到一个目标后：到安全距离 self.safe_distance 就记为“已找到”，
        加入 visited_targets，并且以后不再追这个目标。
        """
        if self.current_target is None or self.current_target_label is None:
            self.mode = self.MODE_EXPLORE
            return

        tx = self.current_target.x
        ty = self.current_target.y

        dx = tx - self.robot_x
        dy = ty - self.robot_y
        distance = math.hypot(dx, dy)
        angle_world = math.atan2(dy, dx)
        angle_robot = angle_world - self.robot_yaw

        d_front = self.get_range_at_angle(0.0)
        if d_front is None:
            d_front = 10.0

        # ① 紧急避障
        if d_front < 0.35:
            cmd = Twist()
            cmd.linear.x = -0.08   # 比原来慢
            cmd.angular.z = 0.9 if angle_robot > 0 else -0.9
            print("🔥 EMERGENCY while going to target → backing up!")
            self.cmd_pub.publish(cmd)
            return

        # ② 一般避障
        if d_front < 0.6:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.7 if angle_robot > 0 else -0.7
            print("⚠️ Avoiding obstacle before pursuing target.")
            self.cmd_pub.publish(cmd)
            return

        # ③ 到达目标附近 → 标记 FOUND + 记录 + 继续找其它目标
        if distance < self.safe_distance:
            print(f"🎯 TARGET FOUND: {self.current_target_label} at distance {distance:.2f} m.")

            # 记录“已找到”的目标（以后不再追这个点）
            self.visited_targets.append(
                (self.current_target_label, tx, ty)
            )
            # 在 RViz 标记“已找到目标”
            self.publish_found_marker(tx, ty, self.current_target_label)

            stop = Twist()
            self.cmd_pub.publish(stop)

            # 清空当前目标，恢复为探索寻找其他目标
            self.current_target = None
            self.current_target_label = None
            self.global_path = None
            self.current_wp_index = 0
            self.mode = self.MODE_EXPLORE
            return

        # ④ 正常朝目标移动（线速度调慢一点）
        cmd = Twist()
        cmd.angular.z = 1.5 * angle_robot

        if abs(angle_robot) < math.pi / 6:
            cmd.linear.x = 0.15   # 原来 0.20
        elif abs(angle_robot) < math.pi / 3:
            cmd.linear.x = 0.08   # 原来 0.10
        else:
            cmd.linear.x = 0.0

        if d_front < 0.45:
            cmd.linear.x = 0.0

        self.cmd_pub.publish(cmd)

    # ==================== 模式 3：全部完成 ====================

    def do_finished_behavior(self):
        """
        所有目标都已“找到”：
        - 不再主动探索
        - 若有障碍靠近，做简单避障
        """
        d_front = self.get_range_at_angle(0.0)
        if d_front is None:
            d_front = 10.0

        cmd = Twist()
        if d_front < 0.4:
            cmd.linear.x = -0.05
            cmd.angular.z = 0.8
            print("FINISHED mode: avoiding obstacle while staying in place.")
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
