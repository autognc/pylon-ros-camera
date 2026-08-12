#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.time import Time
from theo_msgs.msg import TheoCode


import numpy as np
import cv2
import os
from datetime import datetime, timezone


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        self.lock_ = threading.Lock()
        # --- Parameters ---
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('save_directory', '/tmp/captured_images')
        self.declare_parameter('save_rate_hz', 1.0)
        self.declare_parameter('image_prefix', 'frame')
        # 'header' = camera stamp | 'wall' = system clock | 'both' = log both
        self.declare_parameter('timestamp_source', 'header')
        self.declare_parameter('skip_duplicates', True)
        self.declare_parameter('trigger_topic', '/start_capture')
        self.declare_parameter('broker_topic', '/external/broker_robotic_topic')
        self.declare_parameter('broadcast_code', 4)
        self.declare_parameter('is_compressed_image', True)

        self.image_topic         = self.get_parameter('image_topic').value
        self.broker_topic        = self.get_parameter('broker_topic').value
        self.broadcast_code      = self.get_parameter('broadcast_code').value
        self.save_dir            = self.get_parameter('save_directory').value
        self.save_rate_hz        = self.get_parameter('save_rate_hz').value
        self.image_prefix        = self.get_parameter('image_prefix').value
        self.timestamp_source    = self.get_parameter('timestamp_source').value
        self.skip_duplicates     = self.get_parameter('skip_duplicates').value
        self.is_compressed_image = self.get_parameter('is_compressed_image').value


        # --- Setup ---
        os.makedirs(self.save_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.last_saved_seq = None   # used for duplicate detection
        self.saved_count = 1
        self.capture_active = False

        # --- Subscriber ---
        qos_imager = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        if self.is_compressed_image:
            self.subscription = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_callback,
                qos_imager
            )
        else:
            self.subscription = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                qos_imager
            )
        
        qos_brokerage = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.subscription_brokerage = self.create_subscription(
            TheoCode,
            self.broker_topic,
            self.broker_callback,
            qos_brokerage
        )


        # --- Save timer ---
        save_period = 1.0 / self.save_rate_hz
        self.save_period_ns = save_period * 1e9
        self.latest_savetime_msg = self.get_clock().now().to_msg()



        self.get_logger().info(
            f"ImageSaverNode started\n"
            f"  Topic      : {self.image_topic}\n"
            f"  Save dir   : {self.save_dir}\n"
            f"  Rate       : {self.save_rate_hz} Hz\n"
            f"  Timestamp  : {self.timestamp_source}\n"
            f"  Skip dupes : {self.skip_duplicates}"
        )                    

    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        # Ignore image if saver is inactive
        if not self.capture_active:
            return 
        

        savetime_now_msg =  msg.header.stamp
        dt = savetime_now_msg.nanosec - self.latest_savetime_msg.nanosec + ( savetime_now_msg.sec - self.latest_savetime_msg.sec ) * 1e9
        # self.get_logger().info(f"dt: {dt}")
        if( dt < self.save_period_ns ):
            return
        else:
            self.latest_savetime_msg = savetime_now_msg

        
        # --- Timestamps ---
        header_dt, header_ns = self._ros_stamp_to_datetime(msg.header.stamp)


        if self.timestamp_source == 'header':
            file_dt, file_ns = header_dt, header_ns
        elif self.timestamp_source == 'wall':
            now_stamp = self.get_clock().now().to_msg()
            file_dt, file_ns = self._ros_stamp_to_datetime(now_stamp)
        else:  # 'both' — use header for filename, log wall too
            file_dt, file_ns = header_dt, header_ns

        # --- Build filename ---
        #ms = (file_ns % 1_000_000_000) // 1_000_000
        #timestamp_str = file_dt.strftime('%Y%m%d_%H%M%S') + f'_{ms:03d}ms'
        filename = f'{self.image_prefix}_{self.saved_count}_{file_ns}.png'
        filepath = os.path.join(self.save_dir, filename)

        # --- Convert and save ---
        try:
            if self.is_compressed_image:
                np_arr   = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        if not cv2.imwrite(filepath, cv_image):
            self.get_logger().error(f'Failed to write: {filepath}')
            return

        self.saved_count += 1

        log = (
            f'[{self.saved_count}] Saved: {filename} | Camera stamp : {file_ns} [Sec_NanoSec]'
        )
        if self.timestamp_source == 'both':
            log += f'\n  Wall clock   : {file_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]} UTC'
        self.get_logger().info(log)

    def broker_callback(self, msg: TheoCode):
        if int( msg.code ) == self.broadcast_code:
       	    self.capture_active = True
        else:
            self.capture_active = False

    # ------------------------------------------------------------------
    def _ros_stamp_to_datetime(self, stamp) -> tuple[datetime, int]:
        """Return (UTC datetime, nanoseconds) from a ROS stamp."""
        total_ns = f'{stamp.sec}_{stamp.nanosec}'
        dt = datetime.fromtimestamp(stamp.sec + stamp.nanosec * 1e-9, tz=timezone.utc)
        return dt, total_ns

    # ------------------------------------------------------------------

# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'Shutting down. Total saved: {node.saved_count}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
