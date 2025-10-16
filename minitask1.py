class SquareDriver(Node):
    def __init__(self):
        super().__init__('square_driver')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("SquareDriver 节点已启动，开始方形运动...")
        self.drive_square()

    def drive_square(self):
        msg = Twist()
        l = Vector3()
        a = Vector3()

        for i in range(4):  # 共 4 条边
            self.get_logger().info(f"------ 第 {i+1} 条边开始 ------")

            # ① 前进 1 秒
            l.x = 1.0; a.z = 0.0
            msg.linear = l
            msg.angular = a
            self.publisher_.publish(msg)
            self.get_logger().info("前进中（1s）...")
            time.sleep(1.0)

            # ② 停止 1 秒
            l.x = 0.0; a.z = 0.0
            msg.linear = l
            msg.angular = a
            self.publisher_.publish(msg)
            self.get_logger().info("停止中（1s）...")
            time.sleep(1.0)

            # ③ 转弯 90°（约 1 秒）
            l.x = 0.0; a.z = 1.57  # 1.57 rad/s ≈ 90°/s
            msg.linear = l
            msg.angular = a
            self.publisher_.publish(msg)
            self.get_logger().info("转弯中（1s）...")
            time.sleep(1.0)

            # ④ 停止 1 秒
            l.x = 0.0; a.z = 0.0
            msg.linear = l
            msg.angular = a
            self.publisher_.publish(msg)
            self.get_logger().info("停止中（1s）...")
            time.sleep(1.0)

        # 最后确保完全停止
        l.x = 0.0; a.z = 0.0
        msg.linear = l
        msg.angular = a
        self.publisher_.publish(msg)
        self.get_logger().info("方形运动结束，机器人已停止。")


def main(args=None):
    rclpy.init(args=args)
    node = SquareDriver()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
