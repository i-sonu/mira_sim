import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')

        self.thruster_pub = self.create_publisher(
            Float64MultiArray,
            '/bluerov2/thrusters',
            10
        )

        self.command_sub = self.create_subscription(
            Float64MultiArray,
            '/motion_command',
            self.command_callback,
            10
        )

        self.MAX_THRUST = 20.0

        # --- motion intent ---
        self.surge = 0.0   # forward (+), backward (-)
        self.yaw   = 0.0    # left (+), right (-)
        self.heave = 0.0   # up (+), down (-)
        self.pitch = 0.0   # nose up (+), nose down (-)
        self.roll = 0.0   # roll right (+), roll left (-)
        self.sway = 0.0   # right (+), left (-)





        # publish at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_thrusters)

        self.get_logger().info('Motion controller started')



    def clamp(self, value, limit):
        return max(-limit, min(limit, value))



    def publish_thrusters(self):
        msg = Float64MultiArray()

        msg.data = [
            self.clamp(+self.surge + self.yaw + self.sway, self.MAX_THRUST),
            self.clamp(+self.surge - self.yaw - self.sway, self.MAX_THRUST),
            self.clamp(-self.surge - self.yaw + self.sway, self.MAX_THRUST),
            self.clamp(-self.surge + self.yaw - self.sway, self.MAX_THRUST),

            self.clamp(self.heave + self.pitch + self.roll, self.MAX_THRUST),
            self.clamp(self.heave + self.pitch - self.roll, self.MAX_THRUST),
            self.clamp(self.heave - self.pitch + self.roll, self.MAX_THRUST),
            self.clamp(self.heave - self.pitch - self.roll, self.MAX_THRUST)
        ]

        self.thruster_pub.publish(msg)

    def command_callback(self, msg):
        self.surge = msg.data[0]
        self.yaw   = msg.data[1]
        self.sway  = msg.data[2]
        self.heave = msg.data[3]
        self.pitch = msg.data[4]
        self.roll  = msg.data[5]



def main():
    rclpy.init()
    node = MotionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
