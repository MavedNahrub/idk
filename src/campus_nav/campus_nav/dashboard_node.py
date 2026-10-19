#!/usr/bin/env python3
"""
Autonomous Vehicle Live Lane Perception & Mission Telemetry Dashboard Node
IIT Ropar Campus Autonomous Navigation System

Features:
- High-performance multithreaded HTTP server on port 8080
- Live MJPEG streaming for:
    /stream/lane       - Real-time algorithmic lane detection HUD overlay
    /stream/raw        - Raw RealSense onboard camera feed
    /stream/birdseye   - Real-time Top-Down Bird's-Eye (BEV) Metric Lane Radar
- Server-Sent Events (SSE) & REST API for live telemetry at 20Hz:
    /api/telemetry, /api/stream_telemetry
- Interactive Mission Dispatcher API:
    /api/mission, /api/estop
- Serves the responsive web application from web_dashboard/
"""

import os
import sys
import time
import math
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge

from campus_nav.osm_loader import CampusMap


def _find_web_dir():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(curr_dir, '..', 'web_dashboard'),
        os.path.join(curr_dir, 'web_dashboard'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory('campus_nav')
        candidates.append(os.path.join(pkg, 'web_dashboard'))
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(os.path.join(c, 'index.html')):
            return os.path.abspath(c)
    return curr_dir


def _find_default_osm():
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory('campus_nav')
        candidates.extend([
            os.path.join(pkg, 'data', 'campus_with_junctions_and_stops.osm'),
            os.path.join(pkg, 'data', 'campus.osm')
        ])
    except Exception:
        pass
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(curr_dir, '..', 'data', 'campus_with_junctions_and_stops.osm'),
        os.path.join(curr_dir, '..', 'data', 'campus.osm')
    ])
    return next((p for p in candidates if os.path.exists(p)), '')


class DashboardNode(Node):

    def __init__(self):
        super().__init__('dashboard_node')

        self.declare_parameter('port', 8080)
        self.port = int(self.get_parameter('port').value)

        self.bridge = CvBridge()
        self.web_dir = _find_web_dir()
        self.get_logger().info(f"Serving Dashboard static assets from: {self.web_dir}")

        # Map and Stops
        self.osm_path = _find_default_osm()
        self.stops = [
            "Ravi Back", "Chemistry Block", "SAB C", "Satluj Front",
            "Satluj Back", "Beas Front", "Chenab Front", "Main Gate", "Library"
        ]
        if self.osm_path:
            try:
                campus = CampusMap(self.osm_path)
                if campus.stops:
                    self.stops = sorted(list(campus.stops.keys()))
            except Exception as e:
                self.get_logger().warn(f"Could not load stop names: {e}")

        # Shared Live Telemetry State (Thread-Safe)
        self.lock = threading.Lock()
        self.latest_lane_jpeg = None
        self.latest_raw_jpeg = None
        self.latest_bev_jpeg = None

        self.telemetry = {
            'timestamp': time.time(),
            'speed': 0.0,
            'speed_kmh': 0.0,
            'steer_angle_deg': 0.0,
            'steer_angle_rad': 0.0,
            'orientation_err_deg': 0.0,
            'edge_wheel': 'rr',
            'edge_dist': 0.60,
            'edge_clearance': 2.40,
            'mode': 'FORWARD',
            'is_autocorrecting': False,
            'dynamic_radius': 4.2,
            'gate_current': 0,
            'gate_total': 1367,
            'gate_percent': 0.0,
            'checkpoint_str': 'Awaiting Mission Request...',
            'checkpoint_dist': 0.0,
            'checkpoint_eta': 0,
            'pose': {'x': 0.0, 'y': 0.0, 'yaw_deg': 0.0},
            'stops': self.stops
        }

        # Subscribers
        self.create_subscription(Image, '/debug/lane_overlay', self.lane_overlay_callback, 5)
        self.create_subscription(Image, '/real_sense/image_raw', self.raw_image_callback, 5)
        self.create_subscription(String, '/campus/pursuit_status', self.pursuit_status_callback, 10)
        self.create_subscription(String, '/campus/nav_status', self.nav_status_callback, 10)
        self.create_subscription(PoseStamped, '/campus/vehicle_pose', self.vehicle_pose_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers
        self.mission_pub = self.create_publisher(String, '/campus/mission', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Periodic Bird's-Eye View Generation Timer (20Hz)
        self.update_bev_radar()
        self.timer_bev = self.create_timer(0.05, self.update_bev_radar)

        # Start Multithreaded HTTP Server in background daemon thread
        self.server = None
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        self.get_logger().info(f"🚀 SAYE Live Dashboard running at: http://localhost:{self.port}")

    # ── ROS Callbacks ─────────────────────────────────────────────────────────

    def lane_overlay_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _, jpeg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self.lock:
                self.latest_lane_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().warn(f"Error encoding lane overlay: {e}")

    def raw_image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _, jpeg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.latest_raw_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().warn(f"Error encoding raw image: {e}")

    def pursuit_status_callback(self, msg: String):
        txt = msg.data
        with self.lock:
            self.telemetry['timestamp'] = time.time()
            if 'AUTO-CORRECTING' in txt:
                self.telemetry['mode'] = 'RECOVERY_REVERSE'
                self.telemetry['is_autocorrecting'] = True
                # Parse: Err:-54.1° | Rev:-0.65m/s | Steer:-28.6°
                for part in txt.split('|'):
                    part = part.strip()
                    if part.startswith('Err:'):
                        try:
                            val = float(part.replace('Err:', '').replace('°', ''))
                            self.telemetry['orientation_err_deg'] = val
                        except ValueError:
                            pass
                    elif part.startswith('Rev:'):
                        try:
                            val = float(part.replace('Rev:', '').replace('m/s', ''))
                            self.telemetry['speed'] = abs(val)
                            self.telemetry['speed_kmh'] = abs(val) * 3.6
                        except ValueError:
                            pass
                    elif part.startswith('Steer:'):
                        try:
                            val = float(part.replace('Steer:', '').replace('°', ''))
                            self.telemetry['steer_angle_deg'] = val
                            self.telemetry['steer_angle_rad'] = math.radians(val)
                        except ValueError:
                            pass
            else:
                self.telemetry['mode'] = 'FORWARD'
                self.telemetry['is_autocorrecting'] = False
                # Parse: Gate 16/1367 (1%) | Spd:2.37m/s | OrientErr:0.3° | Steer:-0.8° | Edge:fl:0.53m
                for part in txt.split('|'):
                    part = part.strip()
                    if 'Gate' in part:
                        try:
                            # Gate 16/1367 (1%)
                            gp = part.split('(')[0].replace('[follower]', '').replace('Gate', '').strip()
                            curr, tot = gp.split('/')
                            self.telemetry['gate_current'] = int(curr)
                            self.telemetry['gate_total'] = int(tot)
                            pct_str = part.split('(')[1].replace('%)', '').strip()
                            self.telemetry['gate_percent'] = float(pct_str)
                        except Exception:
                            pass
                    elif part.startswith('Spd:'):
                        try:
                            val = float(part.replace('Spd:', '').replace('m/s', ''))
                            self.telemetry['speed'] = val
                            self.telemetry['speed_kmh'] = val * 3.6
                        except ValueError:
                            pass
                    elif part.startswith('OrientErr:'):
                        try:
                            val = float(part.replace('OrientErr:', '').replace('°', ''))
                            self.telemetry['orientation_err_deg'] = val
                        except ValueError:
                            pass
                    elif part.startswith('Steer:'):
                        try:
                            val = float(part.replace('Steer:', '').replace('°', ''))
                            self.telemetry['steer_angle_deg'] = val
                            self.telemetry['steer_angle_rad'] = math.radians(val)
                        except ValueError:
                            pass
                    elif part.startswith('Edge:'):
                        try:
                            # Edge:fl:0.85m
                            tokens = part.split(':')
                            if len(tokens) >= 3:
                                w_name = tokens[1]
                                w_clearance = float(tokens[2].replace('m', ''))
                                self.telemetry['edge_wheel'] = w_name
                                self.telemetry['edge_clearance'] = w_clearance
                                self.telemetry['edge_dist'] = max(0.0, 3.0 - w_clearance)
                        except Exception:
                            pass

    def nav_status_callback(self, msg: String):
        txt = msg.data.strip()
        with self.lock:
            self.telemetry['checkpoint_str'] = txt
            # e.g.: Checkpoint 2/11 | Turn LEFT at 3-way T/Y Junction in 34m | ETA: 15s
            for part in txt.split('|'):
                part = part.strip()
                if 'in ' in part and 'm' in part:
                    try:
                        d_str = part.split('in ')[1].split('m')[0]
                        self.telemetry['checkpoint_dist'] = float(d_str)
                    except Exception:
                        pass
                if 'ETA:' in part:
                    try:
                        e_str = part.replace('ETA:', '').replace('s', '').strip()
                        self.telemetry['checkpoint_eta'] = int(e_str)
                    except Exception:
                        pass

    def vehicle_pose_callback(self, msg: PoseStamped):
        with self.lock:
            p = msg.pose.position
            self.telemetry['pose']['x'] = round(p.x, 2)
            self.telemetry['pose']['y'] = round(p.y, 2)
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            yaw = 2.0 * math.atan2(qz, qw)
            self.telemetry['pose']['yaw_deg'] = round(math.degrees(yaw), 1)

    def cmd_vel_callback(self, msg: Twist):
        pass

    # ── Synthetic Bird's-Eye View (BEV) Radar Generator ────────────────────────

    def update_bev_radar(self):
        """
        Renders a top-down metric Bird's-Eye View (BEV) of the road, left-lane envelope,
        vehicle chassis, steered front wheels, dynamic cornering trajectory, and clearance.
        """
        w, h = 460, 520
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (12, 17, 24)  # Deep dark obsidian slate

        scale = 22.0  # px per meter (shows ~20m ahead, ~3m behind)
        cx = w // 2
        cy_rear = h - 80  # Rear axle origin

        with self.lock:
            speed = self.telemetry['speed']
            steer_deg = self.telemetry['steer_angle_deg']
            steer_rad = self.telemetry['steer_angle_rad']
            orient_err = self.telemetry['orientation_err_deg']
            edge_dist = self.telemetry['edge_dist']
            edge_wheel = self.telemetry['edge_wheel']
            is_rec = self.telemetry['is_autocorrecting']

        # 1. Road Geometry (6.0m total width, 3.0m half width)
        road_half_px = int(3.0 * scale)
        asphalt_l = cx - road_half_px
        asphalt_r = cx + road_half_px

        # Grass Borders
        cv2.rectangle(img, (0, 0), (asphalt_l, h), (18, 38, 22), -1)
        cv2.rectangle(img, (asphalt_r, 0), (w, h), (18, 38, 22), -1)

        # Asphalt Road Surface
        cv2.rectangle(img, (asphalt_l, 0), (asphalt_r, h), (32, 38, 48), -1)

        # White Curbs
        cv2.line(img, (asphalt_l, 0), (asphalt_l, h), (200, 210, 220), 3)
        cv2.line(img, (asphalt_r, 0), (asphalt_r, h), (200, 210, 220), 3)

        # Dashed Road Centerline
        for y in range(0, h, 28):
            cv2.line(img, (cx, y), (cx, y + 16), (240, 240, 240), 2)

        # Left Lane Target Line (1.5m left of center)
        lane_target_x = cx - int(1.5 * scale)
        cv2.line(img, (lane_target_x, 0), (lane_target_x, h), (0, 220, 255), 1, cv2.LINE_AA)

        # 2. Dynamic Trajectory Arc Ahead of Buggy
        if abs(steer_rad) > 0.02:
            R_curve = 2.124 / math.tan(steer_rad)
            R_px = abs(R_curve) * scale
            arc_center_x = cx - int(R_curve * scale)
            arc_center_y = cy_rear
            axes = (int(R_px), int(R_px))
            # Draw arc
            if R_curve > 0:  # Turning Left
                cv2.ellipse(img, (arc_center_x, arc_center_y), axes, 0, 0, -45, (0, 242, 254), 2, cv2.LINE_AA)
            else:  # Turning Right
                cv2.ellipse(img, (arc_center_x, arc_center_y), axes, 0, 180, 225, (0, 242, 254), 2, cv2.LINE_AA)
        else:
            # Straight projection line
            cv2.line(img, (cx, cy_rear), (cx, 30), (0, 242, 254), 2, cv2.LINE_AA)

        # 3. Vehicle Chassis Geometry
        L_wb_px = int(2.124 * scale)  # Wheelbase
        W_trk_px = int(0.778 * scale) # Wheel track
        half_w = W_trk_px // 2

        cy_front = cy_rear - L_wb_px

        # Chassis Body Rectangle
        cv2.rectangle(img, (cx - half_w - 6, cy_front - 12), (cx + half_w + 6, cy_rear + 14), (20, 30, 45), -1)
        cv2.rectangle(img, (cx - half_w - 6, cy_front - 12), (cx + half_w + 6, cy_rear + 14), (56, 189, 248), 2)

        # Front Heading Arrow
        cv2.arrowedLine(img, (cx, cy_rear - 10), (cx, cy_front - 24), (0, 242, 254), 2, tipLength=0.3)

        # Rear Wheels (Fixed Parallel)
        rw_w, rw_h = 8, 18
        cv2.rectangle(img, (cx - half_w - rw_w//2, cy_rear - rw_h//2), (cx - half_w + rw_w//2, cy_rear + rw_h//2), (200, 200, 200), -1)
        cv2.rectangle(img, (cx + half_w - rw_w//2, cy_rear - rw_h//2), (cx + half_w + rw_w//2, cy_rear + rw_h//2), (200, 200, 200), -1)

        # Front Wheels (Ackermann Steered)
        def draw_steered_wheel(center_x, center_y, angle_deg):
            rect = ((center_x, center_y), (rw_w, rw_h), -angle_deg)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(img, [box], 0, (0, 255, 180), -1)

        draw_steered_wheel(cx - half_w, cy_front, steer_deg)
        draw_steered_wheel(cx + half_w, cy_front, steer_deg)

        # 4. Wheel Clearance Alert Bubbles
        d_clearance = max(0.1, 3.0 - edge_dist)
        col_clearance = (16, 185, 129) if d_clearance > 1.2 else ((245, 158, 11) if d_clearance > 0.8 else (239, 68, 68))
        cv2.circle(img, (cx + half_w + 14, cy_rear), int(d_clearance * 6), col_clearance, 1, cv2.LINE_AA)

        # 5. Telemetry HUD Overlay Banner on BEV Canvas
        overlay_txt1 = f"BEV RADAR | 22 px/m | SPD: {speed:.2f}m/s"
        cv2.putText(img, overlay_txt1, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        overlay_txt2 = f"STEER: {steer_deg:+.1f} deg | ORIENT_ERR: {orient_err:+.1f} deg"
        cv2.putText(img, overlay_txt2, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)

        if is_rec:
            cv2.putText(img, "STATE: AUTO-CORRECTION REVERSE", (14, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"STATE: LANE TRACKING | CURB CLEAR: {d_clearance:.2f}m", (14, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)

        # Encode to JPEG
        _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            self.latest_bev_jpeg = jpeg.tobytes()

    # ── Multithreaded HTTP Server ─────────────────────────────────────────────

    def _run_server(self):
        node = self

        class DashboardHTTPHandler(BaseHTTPRequestHandler):

            def log_message(self, format, *args):
                # Suppress noisy HTTP access logs
                pass

            def do_HEAD(self):
                self.do_GET()

            def do_GET(self):
                parsed = urlparse(self.path)
                p = parsed.path

                # 1. Video Streams (MJPEG)
                if p in ['/stream/lane', '/stream/raw', '/stream/birdseye']:
                    self._handle_mjpeg_stream(p)
                    return

                # 2. REST API / Telemetry JSON
                if p == '/api/telemetry':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    with node.lock:
                        payload = dict(node.telemetry)
                    self.wfile.write(json.dumps(payload).encode('utf-8'))
                    return

                # 3. Server-Sent Events (SSE) Stream
                if p == '/api/stream_telemetry':
                    self._handle_sse_stream()
                    return

                # 4. Static Web Assets
                if p == '/' or p == '/index.html':
                    self._serve_file(os.path.join(node.web_dir, 'index.html'), 'text/html')
                elif p == '/style.css':
                    self._serve_file(os.path.join(node.web_dir, 'style.css'), 'text/css')
                elif p == '/app.js':
                    self._serve_file(os.path.join(node.web_dir, 'app.js'), 'application/javascript')
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                parsed = urlparse(self.path)
                p = parsed.path

                if p == '/api/mission':
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        start = data.get('start', 'Ravi Back')
                        goal = data.get('goal', 'Chemistry Block')
                        mission_str = f"{start}|{goal}"
                        node.mission_pub.publish(String(data=mission_str))
                        node.get_logger().info(f"🌐 Mission Dispatched from Web Dashboard: '{mission_str}'")

                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps({'success': True, 'mission': mission_str}).encode('utf-8'))
                    except Exception as e:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
                    return

                if p == '/api/estop':
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        stop = data.get('stop', True)
                        if stop:
                            node.cmd_pub.publish(Twist())
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps({'success': True, 'estop': stop}).encode('utf-8'))
                    except Exception as e:
                        self.send_response(400)
                        self.end_headers()
                    return

                self.send_response(404)
                self.end_headers()

            def _serve_file(self, filepath, mime_type):
                if not os.path.exists(filepath):
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(filepath, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', mime_type)
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(content)

            def _handle_mjpeg_stream(self, path):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                last_sent = None
                try:
                    while True:
                        with node.lock:
                            if path == '/stream/lane':
                                frame = node.latest_lane_jpeg or node.latest_raw_jpeg or node.latest_bev_jpeg
                            elif path == '/stream/raw':
                                frame = node.latest_raw_jpeg or node.latest_lane_jpeg
                            elif path == '/stream/birdseye':
                                if node.latest_bev_jpeg is None:
                                    node.update_bev_radar()
                                frame = node.latest_bev_jpeg or node.latest_lane_jpeg
                            else:
                                frame = node.latest_lane_jpeg

                        if frame:
                            header = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                            self.wfile.write(header)
                            self.wfile.write(frame)
                            self.wfile.write(b'\r\n')
                        time.sleep(0.05)  # 20 FPS stream
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _handle_sse_stream(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                try:
                    while True:
                        with node.lock:
                            payload = dict(node.telemetry)
                        msg = f"data: {json.dumps(payload)}\n\n"
                        self.wfile.write(msg.encode('utf-8'))
                        self.wfile.flush()
                        time.sleep(0.05)  # 20Hz
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            self.server = ThreadingHTTPServer(('0.0.0.0', self.port), DashboardHTTPHandler)
            self.server.serve_forever()
        except Exception as e:
            self.get_logger().error(f"HTTP Server error on port {self.port}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
