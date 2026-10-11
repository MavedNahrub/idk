#!/usr/bin/env python3
"""
YOLOP Panoptic Driving Perception ROS 2 Node for Saye Ackermann Vehicle
Performs 3-in-1 real-time perception:
1. Lane Line Segmentation (/yolop/lane_lines)
2. Drivable Area Segmentation (/yolop/drivable_area)
3. Object & Obstacle Detection with Emergency Stop Safety (/safety/emergency_stop)
4. Comprehensive Panoptic Overlay (/yolop/panoptic_overlay)
"""

import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class YOLOPPerceptionNode(Node):
    def __init__(self):
        super().__init__('yolop_perception_node')

        self.declare_parameter('camera_topic', '/real_sense/image_raw')
        self.declare_parameter('device', 'cuda:0')          # 'cuda:0' or 'cpu'
        self.declare_parameter('conf_thres', 0.30)          # Detection confidence threshold
        self.declare_parameter('iou_thres', 0.45)           # NMS IoU threshold
        self.declare_parameter('enable_safety_stop', True)  # Emergency stop trigger for frontal obstacles

        self.camera_topic = self.get_parameter('camera_topic').value
        self.device_str = self.get_parameter('device').value
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)
        self.enable_safety_stop = bool(self.get_parameter('enable_safety_stop').value)

        # Publishers
        self.overlay_pub = self.create_publisher(Image, '/yolop/panoptic_overlay', 10)
        self.drivable_pub = self.create_publisher(Image, '/yolop/drivable_area', 10)
        self.lane_lines_pub = self.create_publisher(Image, '/yolop/lane_lines', 10)
        self.safety_pub = self.create_publisher(Bool, '/safety/emergency_stop', 10)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 5
        )

        self.get_logger().info("Initializing YOLOP Panoptic Neural Network...")
        self.model = None
        self.device = None
        self.torch = None
        self.init_yolop()

        # Telemetry
        self.frame_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()

    def init_yolop(self):
        try:
            import torch
            self.torch = torch
            
            if self.device_str.startswith('cuda') and torch.cuda.is_available():
                self.device = torch.device(self.device_str)
                self.get_logger().info(f"Using NVIDIA GPU Acceleration: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                self.get_logger().info("Using CPU Inference")

            # Load YOLOP via PyTorch Hub
            self.get_logger().info("Loading hustvl/yolop pretrained model from PyTorch Hub...")
            self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Warmup inference pass
            dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

            self.get_logger().info("YOLOP Panoptic Engine successfully loaded and ready!")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize YOLOP: {e}")
            self.model = None

    def image_callback(self, msg: Image):
        if self.model is None or self.torch is None:
            return

        # Decode ROS Image to OpenCV BGR
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            if frame.shape[2] == 3 and msg.encoding in ['rgb8', 'RGB8']:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 3:
                bgr = frame
            elif frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                return
        except Exception as e:
            self.get_logger().error(f"Decode error: {e}")
            return

        orig_h, orig_w = bgr.shape[:2]

        # Telemetry calculation
        self.frame_count += 1
        t_now = time.time()
        if t_now - self.fps_timer >= 1.0:
            self.fps = self.frame_count / (t_now - self.fps_timer)
            self.frame_count = 0
            self.fps_timer = t_now

        # 1. Preprocess: Resize to 640x640 with ImageNet Normalization
        t_start = time.time()
        resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to (img/255 - mean) / std
        tensor_img = rgb_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor_img = (tensor_img - mean) / std
        tensor_img = np.transpose(tensor_img, (2, 0, 1)) # HWC -> CHW
        tensor_img = np.expand_dims(tensor_img, axis=0) # Add batch dim: (1, 3, 640, 640)

        input_tensor = self.torch.from_numpy(tensor_img).to(self.device)

        # 2. Forward Inference
        with self.torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.model(input_tensor)

        infer_time_ms = (time.time() - t_start) * 1000.0

        # 3. Postprocess Drivable Area & Lane Lines
        # Drivable area mask (Class 1 = Drivable)
        da_pred = self.torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        da_mask = cv2.resize(da_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Lane line mask (Threshold binary sigmoid or argmax)
        ll_pred = self.torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        ll_mask = cv2.resize(ll_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 4. Obstacle Detection & Collision Safety
        obstacle_detected = False
        det_boxes = []

        if isinstance(det_out, list) and len(det_out) > 0:
            det_boxes = det_out[0].cpu().numpy() if hasattr(det_out[0], 'cpu') else []

        # Check safety corridor in front of vehicle (bottom central 40% of image)
        corridor_x_min = int(orig_w * 0.30)
        corridor_x_max = int(orig_w * 0.70)
        corridor_y_min = int(orig_h * 0.65)

        for box in det_boxes:
            if len(box) >= 6:
                x1, y1, x2, y2, conf, cls_id = box[:6]
                if conf >= self.conf_thres:
                    # Scale coordinates from 640x640 to orig dimensions
                    x1 = int(x1 * orig_w / 640.0)
                    y1 = int(y1 * orig_h / 640.0)
                    x2 = int(x2 * orig_w / 640.0)
                    y2 = int(y2 * orig_h / 640.0)

                    # Check intersection with collision corridor
                    if (x2 > corridor_x_min and x1 < corridor_x_max and y2 > corridor_y_min):
                        obstacle_detected = True

        # Publish Emergency Stop status
        if self.enable_safety_stop:
            stop_msg = Bool()
            stop_msg.data = bool(obstacle_detected)
            self.safety_pub.publish(stop_msg)

        # 5. Composite Panoptic Visual Overlay
        overlay = bgr.copy()

        # Green tint for Drivable Area
        drivable_color = np.zeros_like(bgr)
        drivable_color[da_mask > 0] = [0, 220, 0] # Bright Green
        cv2.addWeighted(overlay, 1.0, drivable_color, 0.35, 0, overlay)

        # Crimson Red tint for Lane Lines
        overlay[ll_mask > 0] = [20, 20, 240]

        # Draw detected bounding boxes
        for box in det_boxes:
            if len(box) >= 6:
                x1, y1, x2, y2, conf, cls_id = box[:6]
                if conf >= self.conf_thres:
                    x1 = int(x1 * orig_w / 640.0)
                    y1 = int(y1 * orig_h / 640.0)
                    x2 = int(x2 * orig_w / 640.0)
                    y2 = int(y2 * orig_h / 640.0)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    label = f"Obj: {conf:.2f}"
                    cv2.putText(overlay, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # HUD Telemetry Banner
        hud_bg = overlay.copy()
        cv2.rectangle(hud_bg, (10, 10), (360, 95), (18, 22, 26), -1)
        cv2.addWeighted(hud_bg, 0.75, overlay, 0.25, 0, overlay)

        cv2.putText(overlay, "YOLOP PANOPTIC PERCEPTION", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Inference: {infer_time_ms:.1f} ms | FPS: {self.fps:.1f}", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        
        status_text = "STATUS: SAFE PATH" if not obstacle_detected else "WARNING: OBSTACLE IN PATH"
        status_color = (0, 255, 0) if not obstacle_detected else (0, 0, 255)
        cv2.putText(overlay, status_text, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Device: {self.device.type.upper()}", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        # Publish composite panoptic overlay
        overlay_msg = Image()
        overlay_msg.header.stamp = self.get_clock().now().to_msg()
        overlay_msg.header.frame_id = "depth_camera_link"
        overlay_msg.height = orig_h
        overlay_msg.width = orig_w
        overlay_msg.encoding = "bgr8"
        overlay_msg.is_bigendian = 0
        overlay_msg.step = orig_w * 3
        overlay_msg.data = overlay.tobytes()
        self.overlay_pub.publish(overlay_msg)

        # Publish binary drivable area mask
        da_msg = Image()
        da_msg.header.stamp = overlay_msg.header.stamp
        da_msg.header.frame_id = "depth_camera_link"
        da_msg.height = orig_h
        da_msg.width = orig_w
        da_msg.encoding = "mono8"
        da_msg.is_bigendian = 0
        da_msg.step = orig_w
        da_msg.data = (da_mask * 255).tobytes()
        self.drivable_pub.publish(da_msg)

        # Publish binary lane line mask
        ll_msg = Image()
        ll_msg.header.stamp = overlay_msg.header.stamp
        ll_msg.header.frame_id = "depth_camera_link"
        ll_msg.height = orig_h
        ll_msg.width = orig_w
        ll_msg.encoding = "mono8"
        ll_msg.is_bigendian = 0
        ll_msg.step = orig_w
        ll_msg.data = (ll_mask * 255).tobytes()
        self.lane_lines_pub.publish(ll_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOPPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
