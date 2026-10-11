#!/usr/bin/env python3
"""
Route Execution Monitor
=======================
Monitors whether the buggy correctly executes its A*-planned route
node-by-node and reaches the final destination.

Subscribes to:
  /campus/nav_status       — checkpoint + direction updates from planner
  /campus/pursuit_status   — gate idx, XTE, speed, steer from path follower
  /campus/vehicle_pose     — vehicle position in map (nav) coords

Usage (inside Docker container):
  python3 /root/colcon_ws/src/ackermann-vehicle-gzsim-ros2/src/campus_nav/campus_nav/route_monitor.py
"""

# pyrefly: ignore [missing-import]
import rclpy
# pyrefly: ignore [missing-import]
from rclpy.node import Node
# pyrefly: ignore [missing-import]
from std_msgs.msg import String
# pyrefly: ignore [missing-import]
from geometry_msgs.msg import PoseStamped
import time
import math
import os
import sys

# ANSI colours
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"
DIM  = "\033[2m"


class RouteMonitor(Node):
    def __init__(self):
        super().__init__('route_monitor')

        # ── state ──────────────────────────────────────────────────
        self.nav_status   = "Waiting for route..."
        self.pursuit_status = "Waiting for follower..."
        self.pose_x = None
        self.pose_y = None

        # XTE tracking
        self.xte_samples   = []
        self.max_xte        = 0.0
        self.gate_history   = []
        self.last_gate      = -1
        self.arrived        = False
        self.mission_started_at = None

        # Checkpoint tracking
        self.checkpoints_passed = 0
        self.total_gates        = 0
        self.current_speed      = 0.0
        self.barrier_level      = "NOMINAL"

        # Log file
        log_dir = "/tmp/campus_nav_logs"
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = f"{log_dir}/route_test_{ts}.log"
        self.log_file = open(self.log_path, "w")
        self._log(f"Route Execution Test started at {ts}")
        self._log(f"Monitoring: /campus/nav_status | /campus/pursuit_status | /campus/vehicle_pose")
        self._log("-" * 72)

        # ── subscriptions ──────────────────────────────────────────
        self.create_subscription(String,      '/campus/nav_status',    self._nav_cb,     10)
        self.create_subscription(String,      '/campus/pursuit_status',self._pursuit_cb, 10)
        self.create_subscription(PoseStamped, '/campus/vehicle_pose',  self._pose_cb,    10)

        # Print HUD at 2 Hz
        self.create_timer(0.5, self._print_hud)

        self.get_logger().info(f"Route Monitor active. Log: {self.log_path}")

    # ── callbacks ──────────────────────────────────────────────────

    def _nav_cb(self, msg: String):
        self.nav_status = msg.data
        self._log(f"[NAV]     {msg.data}")

        if "Active Route:" in msg.data and self.mission_started_at is None:
            self.mission_started_at = time.time()

        if "checkpoint" in msg.data.lower() or "Checkpoint" in msg.data:
            self.checkpoints_passed += 1

    def _pursuit_cb(self, msg: String):
        raw = msg.data
        self.pursuit_status = raw

        # Parse: "Gate X/Y (ZZ%) | XTE:A.BBm | Spd:C.DDm/s | Steer:EE.F | Goal:NNNm"
        try:
            # Gate progress
            if "Gate " in raw:
                g_part = raw.split("Gate ")[1].split(" ")[0]
                gate_idx, total = g_part.split("/")
                gate_idx  = int(gate_idx)
                total_g   = int(total)
                self.total_gates = total_g

                if gate_idx != self.last_gate:
                    if self.last_gate >= 0 and gate_idx > self.last_gate:
                        self.gate_history.append((time.time(), gate_idx, total_g))
                        self._log(f"[GATE]    Advanced to gate {gate_idx}/{total_g}")
                    self.last_gate = gate_idx

            # XTE
            if "XTE:" in raw:
                xte_str = raw.split("XTE:")[1].split("m")[0]
                xte = float(xte_str)
                self.xte_samples.append(xte)
                if xte > self.max_xte:
                    self.max_xte = xte
                    if xte > 0.70:
                        self._log(f"[XTE!]   High XTE={xte:.2f}m at gate {self.last_gate}/{self.total_gates}")

            # Speed
            if "Spd:" in raw:
                spd_str = raw.split("Spd:")[1].split("m/s")[0]
                self.current_speed = float(spd_str)

            # Barrier level
            if "<" in raw and ">" in raw:
                bl = raw.split("<")[1].split(">")[0]
                if bl != self.barrier_level:
                    self.barrier_level = bl
                    self._log(f"[BARRIER] Entered {bl} zone at gate {self.last_gate}")

        except Exception:
            pass

        # Arrival
        if "ARRIVED" in raw and not self.arrived:
            self.arrived = True
            elapsed = time.time() - self.mission_started_at if self.mission_started_at else 0
            avg_xte = sum(self.xte_samples) / max(1, len(self.xte_samples))
            self._log("=" * 72)
            self._log(f"[ARRIVED] Buggy reached destination!")
            self._log(f"          Total gates traversed : {self.last_gate}/{self.total_gates}")
            self._log(f"          Elapsed time          : {int(elapsed)}s")
            self._log(f"          Avg XTE               : {avg_xte:.3f} m")
            self._log(f"          Max XTE               : {self.max_xte:.3f} m")
            self._log(f"          XTE samples collected : {len(self.xte_samples)}")
            self._log(f"          Checkpoints acked     : {self.checkpoints_passed}")
            self._log(f"          PASS: {self.max_xte < 1.10}")
            self._log("=" * 72)

    def _pose_cb(self, msg: PoseStamped):
        self.pose_x = msg.pose.position.x
        self.pose_y = msg.pose.position.y

    # ── HUD ────────────────────────────────────────────────────────

    def _print_hud(self):
        os.system("clear")
        avg_xte = sum(self.xte_samples[-20:]) / max(1, min(20, len(self.xte_samples)))
        elapsed = int(time.time() - self.mission_started_at) if self.mission_started_at else 0
        pct = 100 * self.last_gate / max(1, self.total_gates)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = GRN + "X" * filled + DIM + "." * (bar_len - filled) + RST

        # XTE colour
        xte_col = GRN if avg_xte < 0.35 else (YLW if avg_xte < 0.70 else RED)

        print(f"{BLD}{CYN}=========================================================={RST}")
        print(f"{BLD}{CYN}     CAMPUS ROUTE EXECUTION MONITOR{RST}")
        print(f"{BLD}{CYN}=========================================================={RST}")
        print()
        print(f"  {BLD}Route Progress:{RST}  [{bar}]  {pct:.0f}%")
        print(f"  {BLD}Gate          :{RST}  {self.last_gate} / {self.total_gates}")
        print(f"  {BLD}Elapsed       :{RST}  {elapsed}s")
        print()
        print(f"  {BLD}Speed         :{RST}  {self.current_speed:.2f} m/s")
        print(f"  {BLD}Avg XTE (20pt):{RST}  {xte_col}{avg_xte:.3f} m{RST}  (max: {self.max_xte:.3f} m)")
        barrier_col = GRN if self.barrier_level == "NOMINAL" else (YLW if "REPULSION" in self.barrier_level else RED)
        print(f"  {BLD}Barrier Level :{RST}  {barrier_col}{self.barrier_level}{RST}")
        print()
        pose_str = f"({self.pose_x:.1f}, {self.pose_y:.1f})" if self.pose_x else "waiting..."
        print(f"  {BLD}Vehicle Pos   :{RST}  {pose_str}  (map/nav frame)")
        print()
        print(f"  {BLD}Nav Status    :{RST}  {self.nav_status[:70]}")
        print(f"  {BLD}Follower      :{RST}  {self.pursuit_status[:70]}")
        print()
        print(f"  {DIM}Log: {self.log_path}{RST}")
        if self.arrived:
            avg_xte_final = sum(self.xte_samples) / max(1, len(self.xte_samples))
            grade = "PASS" if self.max_xte < 1.10 else "FAIL"
            print(f"\n  {BLD}{GRN}MISSION COMPLETE - Buggy arrived!{RST}")
            print(f"  Avg XTE: {avg_xte_final:.3f}m  Max XTE: {self.max_xte:.3f}m  [{grade}]\n")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def __del__(self):
        try:
            self.log_file.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = RouteMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
