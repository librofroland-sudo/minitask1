import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry

class Minitask1(Node):

    def __init__(self):
        super().__init__('minitask1')
        #create the publisher
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # 定时器频率（回调频繁些以保证切换及时）
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 任务参数：每次前进 0.5s，转 90° 0.5s
        self.forward_time = 0.5   # 前进持续时间（秒）
        self.turn_time = 0.5      # 转弯持续时间（秒）
        self.linear_speed = 1.0   # 前进线速度（m/s）
        # 需要在 0.5s 内转 90° -> 角速度 = (pi/2) / 0.5 = pi rad/s
        self.angular_speed = (math.pi / 2.0) / self.turn_time
        
         # 启动时间（秒）
        self.start_time = self._now_seconds()
        self.total_duration = 4.0  # 总共运行 4 秒（4 条边）
        
        def _now_seconds(self):
            return self.get_clock().now().nanoseconds / 1e9
        
        #create the subscriber
        #self.subscription = self.create_subscription(
        #    Odometry,
        #    'odom',
        #    self.odom_callback,
        #    10)
        #self.subscription  # prevent unused variable warning

    #publish a message every 0.5 seconds
    def timer_callback(self):
        #create new message of type Twist
        msg = Twist()
        #create linear component
        l = Vector3()
        l.x = lin_x
        l.y = 0.0
        l.z = 0.0
        #create angular component
        a = Vector3()
        a.x = 0.0
        a.y = 0.0
        a.z = ang_z

        #set message linear and angular
        msg.linear = l
        msg.angular = a
        #publish message
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing Twist...')

 #   def odom_callback(self, msg):
 #       location = msg.pose.pose.position
 #       self.get_logger().info('Robot is at: "%s"' % location)

    def timer_callback(self):
        now = self._now_seconds()
        t = now - self.start_time

        # 如果超过总时长，停车并取消定时器
        if t >= self.total_duration:
            self.publish_twist(0.0, 0.0)
            self.get_logger().info('已达到总时长 4.0s，停止发布运动指令。')
            self.timer.cancel()
            return

        # 每 1.0s 为一周期：前 0.5s 前进，后 0.5s 原地转
        phase = t % (self.forward_time + self.turn_time)  # 等于 t % 1.0
        if phase < self.forward_time:
            # 前进阶段
            self.publish_twist(self.linear_speed, 0.0)
        else:
            # 转弯阶段（原地旋转）
            self.publish_twist(0.0, self.angular_speed)

    

def main(args=None):
    rclpy.init(args=args)
    node = Minitask1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 结束前确保停车
        node.publish_twist(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        
'''def main(args=None):
    rclpy.init(args=args)

    mt = Minitask1()

    rclpy.spin(mt)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mt.destroy_node()
    rclpy.shutdown()'''


if __name__ == '__main__':
    main()
