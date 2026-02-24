#!/usr/bin/env python3
"""
ArduPilot SITL <-> ROS2 Bridge

Observed frame situation (confirmed from live topic data):
──────────────────────────────────────────────────────────
odom.frame_id = "world_ned"
  • Position and linear velocity are ALREADY in NED — pass through as-is.
  • Angular velocity in odom.twist is body-frame, NED-consistent.

imu.frame_id = "bluerov2/imu_filter"
  • linear_acceleration: body-frame, Z ≈ +9.81 when flat
    → gravity on +Z is correct for ArduPilot's NED body frame. Pass as-is.
  • angular_velocity: body-frame, NED-consistent. Pass as-is.
  • orientation quaternion: reports ~180° roll when the vehicle is flat.
    The sim IMU body frame is rotated 180° around X vs ArduPilot's expected
    NED body frame. Fix: q_corrected = q * q_rot180x before converting to Euler.

Home latch:
  The first odom position is recorded as the home origin. All subsequent
  positions are sent as deltas so ArduPilot sees (0, 0, 0) at startup.
"""

import json
import socket
import struct

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_quaternion, quaternion_multiply


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITL_MAGIC       = 18458
SITL_RECV_FORMAT = 'HHI16H'
SITL_RECV_SIZE   = struct.calcsize(SITL_RECV_FORMAT)
NUM_THRUSTERS    = 8

# 180° rotation about body X axis [x, y, z, w] — cancels the sim's frame offset
_Q_ROT180X = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pwm_to_normalized(pwm: int) -> float:
    """Map PWM [1100-1900] -> [-1, 1], centre at 1500."""
    return (pwm - 1500) / 400.0


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_ned_euler(qx, qy, qz, qw):
    """
    Convert an orientation quaternion from the sim (which has a 180 deg X-axis
    offset) to NED Euler angles [roll, pitch, yaw] expected by ArduPilot.

    Steps:
      1. Apply 180 deg X correction:  q_fixed = q_raw * q_rot180x
      2. Extract ZYX Euler angles from the corrected quaternion.
    """
    q_raw = np.array([qx, qy, qz, qw])
    q_fixed = quaternion_multiply(q_raw, _Q_ROT180X)
    roll, pitch, yaw = euler_from_quaternion(q_fixed)
    return (wrap_pi(roll), wrap_pi(pitch), wrap_pi(yaw))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ArduSimPatch(Node):

    def __init__(self, node_name: str, namespace: str, sitl_port: int):
        super().__init__(node_name, namespace=namespace)

        self.ns = self.get_namespace().lstrip('/')
        self.get_logger().info(
            f"ArduSim patch starting — namespace: '{self.ns}', port: {sitl_port}"
        )

        # ROS subscribers
        self.create_subscription(Imu,      'imu',      self._cb_imu,  1)
        self.create_subscription(Odometry, 'odometry', self._cb_odom, 1)

        # ROS publishers
        self.pub_pwm = self.create_publisher(Float64MultiArray, 'thrusters', 1)

        # State
        self.imu:  Imu      = None
        self.odom: Odometry = None
        self.home_pos: tuple = None   # NED home latch (x, y, z)

        # UDP socket — same socket for recvfrom and sendto SITL
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', sitl_port))
        self.sock.settimeout(1.0)

        # 50 Hz main loop
        self.create_timer(1.0 / 50.0, self._loop)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _cb_imu(self, msg: Imu):
        self.imu = msg

    def _cb_odom(self, msg: Odometry):
        self.odom = msg

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def _loop(self):
        if self.imu is None or self.odom is None:
            self.get_logger().info(
                f"Waiting — IMU: {'OK' if self.imu else 'missing'}, "
                f"Odom: {'OK' if self.odom else 'missing'}",
                once=False,
            )
            return

        self.get_logger().info("All data sources ready.", once=True)

        # --- Receive PWM packet from ArduPilot SITL ---
        try:
            data, sitl_addr = self.sock.recvfrom(256)
        except socket.timeout:
            self.get_logger().warn(
                "Timeout waiting for SITL packet — is ArduPilot SITL running?",
                once=False,
            )
            return
        except Exception as ex:
            self.get_logger().error(f"Socket error: {ex}")
            return

        # Validate
        if len(data) != SITL_RECV_SIZE:
            self.get_logger().warn(
                f"Bad packet size: got {len(data)}, expected {SITL_RECV_SIZE}"
            )
            return

        decoded = struct.unpack(SITL_RECV_FORMAT, data)
        magic, _frame_rate_hz, _frame_count, *pwm_raw = decoded

        if magic != SITL_MAGIC:
            self.get_logger().warn(f"Bad magic: {magic}, expected {SITL_MAGIC}")
            return

        # --- Publish normalised PWM to ROS ---
        pwm_norm = [pwm_to_normalized(pwm_raw[i]) for i in range(NUM_THRUSTERS)]
        self.pub_pwm.publish(Float64MultiArray(data=pwm_norm))

        # -------------------------------------------------------------------
        # Build state for ArduPilot.
        # Odom is world_ned, IMU body-frame is NED-consistent (Z ≈ +9.81 at rest).
        # Only the orientation quaternion needs fixing (180 deg X-axis offset).
        # -------------------------------------------------------------------

        # IMU — pass through directly, already NED body frame
        accel = (
            self.imu.linear_acceleration.x,
            self.imu.linear_acceleration.y,
            self.imu.linear_acceleration.z,
        )
        gyro = (
            self.imu.angular_velocity.x,
            self.imu.angular_velocity.y,
            self.imu.angular_velocity.z,
        )

        # Position — world_ned, latch home on first reading to zero the origin
        raw_pos = (
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        )
        if self.home_pos is None:
            self.home_pos = raw_pos
            self.get_logger().info(
                f"Home latched: N={raw_pos[0]:.3f} E={raw_pos[1]:.3f} D={raw_pos[2]:.3f}"
            )
        position = (
            raw_pos[0] - self.home_pos[0],
            raw_pos[1] - self.home_pos[1],
            raw_pos[2] - self.home_pos[2],
        )

        # Attitude — correct the 180 deg X-axis sim offset, then extract NED Euler
        q = self.odom.pose.pose.orientation
        attitude = quaternion_to_ned_euler(q.x, q.y, q.z, q.w)

        # Velocity — linear (world NED) + angular (body NED), pass through
        velocity = (
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
            self.odom.twist.twist.angular.x,
            self.odom.twist.twist.angular.y,
            self.odom.twist.twist.angular.z,
        )

        # Timestamp
        t = self.get_clock().now().to_msg()
        timestamp = t.sec + t.nanosec * 1e-9

        # --- Assemble and send JSON to ArduPilot ---
        payload = {
            "timestamp": timestamp,
            "imu": {
                "gyro":       gyro,
                "accel_body": accel,
            },
            "position": position,
            "attitude": attitude,
            "velocity": velocity,
        }
        json_str = "\n" + json.dumps(payload, separators=(',', ':')) + "\n"
        self.sock.sendto(json_str.encode('ascii'), sitl_addr)

        self.get_logger().debug(
            f"att r={np.degrees(attitude[0]):+.1f}deg "
            f"p={np.degrees(attitude[1]):+.1f}deg "
            f"y={np.degrees(attitude[2]):+.1f}deg | "
            f"pos N={position[0]:+.2f} E={position[1]:+.2f} D={position[2]:+.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    node = ArduSimPatch(
        node_name = 'ardusim_patch',
        namespace = 'bluerov2',
        sitl_port = 9002,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()