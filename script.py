import airsim
import cv2
import time
import math
import csv
import datetime
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
YOLO_MODEL_PATH = "yolov8n.pt" 
SAFE_DISTANCE = 5.0             # Obstacle avoidance trigger distance
TARGET_ALTITUDE = -3.0          # Target height (5m up, negative in NED)
FLIGHT_SPEED = 8.0
INVESTIGATE_SPEED = 1.0
VEHICLE_NAME = "Drone1"         # Must match your settings.json
CSV_FILENAME = "mission_report.csv"


if __name__ == "__main__":
    points = [
        airsim.Vector3r(5.15, 27.08, -10),
        airsim.Vector3r(104, -42, -20)
    ]

    drone = DroneMission()
    try:
        drone.fly_mission(points)
    except KeyboardInterrupt:
        # Even if interrupted, try to save what we found
        drone.save_mission_report()
        drone.client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
