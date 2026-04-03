#Note: this is created only to check if the motion_controller code 
# is working, as well as to debug if the thrusters and motion
# are working




import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class MissionScript(Node):
    def __init__(self):
        super().__init__('mission_script')

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/motion_command',
            10
        )

        self.get_logger().info('Mission script started')
        self.run_mission()

    def send_command(self, surge, yaw, sway, heave, pitch, roll):
        msg = Float64MultiArray()
        msg.data = [surge, yaw, sway, heave, pitch, roll]
        self.pub.publish(msg)

    def run_mission(self):

        def run_for(duration, surge, yaw, sway, heave, pitch, roll):
            start = time.time()
            while time.time() - start < duration:
                self.send_command(surge, yaw, sway, heave, pitch, roll)
                time.sleep(0.1)  # 10 Hz

        self.get_logger().info('Shutting down motors...')
        run_for(10.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0)


        # Stop
        for _ in range(10):
            self.send_command(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.1)

        self.get_logger().info('Motors Shut down.')


def main():
    rclpy.init()
    node = MissionScript()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()