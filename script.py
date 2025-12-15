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

class DroneMission:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
        self.client.armDisarm(True, vehicle_name=VEHICLE_NAME)

        print(f"Loading YOLO model...")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.person_class_id = 0 
        
        # List to store detection data: [Time, X, Y, Z, Count]
        self.detection_logs = []
    def get_position(self):
        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        return state.kinematics_estimated.position

    def get_yaw(self):
        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        orientation = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        return yaw

    def get_ground_distance(self):
        """Reads distance sensor or falls back to altimeter."""
        try:
            dist_data = self.client.getDistanceSensorData(distance_sensor_name="Distance", vehicle_name=VEHICLE_NAME)
            return dist_data.distance
        except Exception:
            z = self.get_position().z_val
            return -z 

    def get_front_depth(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
        ], vehicle_name=VEHICLE_NAME)
        
        if not responses: return 100.0

        depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape(responses[0].height, responses[0].width)
        depth_img[depth_img > 100] = 100 
        
        h, w = depth_img.shape
        center_h, center_w = int(h//2), int(w//2)
        crop = 50
        center_region = depth_img[center_h-crop:center_h+crop, center_w-crop:center_w+crop]
        return np.min(center_region)

    def manage_altitude(self):
        pos = self.get_position()
        current_z = pos.z_val
        ground_dist = self.get_ground_distance()

        if current_z < (TARGET_ALTITUDE - 0.5): 
            if ground_dist > 3.5:
                self.client.moveByVelocityZAsync(0, 0, TARGET_ALTITUDE, 0.5, vehicle_name=VEHICLE_NAME).join()
        elif current_z > (TARGET_ALTITUDE + 0.5):
            self.client.moveToZAsync(TARGET_ALTITUDE, 1, vehicle_name=VEHICLE_NAME).join()
            
    def find_alternate_route(self):
        print("Obstacle detected! Finding alternate route...")
        self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=VEHICLE_NAME).join()
        
        original_yaw_deg = math.degrees(self.get_yaw())
        search_angles = [30, -30, 60, -60, 90, -90, 120, -120]
        path_found = False
        safe_heading_rad = 0
        
        for angle in search_angles:
            target_yaw = original_yaw_deg + angle
            if target_yaw > 180: target_yaw -= 360
            elif target_yaw < -180: target_yaw += 360
            
            self.client.rotateToYawAsync(target_yaw, 1, vehicle_name=VEHICLE_NAME).join()
            if self.get_front_depth() > SAFE_DISTANCE:
                path_found = True
                safe_heading_rad = math.radians(target_yaw)
                break
        
        if path_found:
            vx = math.cos(safe_heading_rad) * FLIGHT_SPEED
            vy = math.sin(safe_heading_rad) * FLIGHT_SPEED
            self.client.moveByVelocityZAsync(vx, vy, TARGET_ALTITUDE, 1, vehicle_name=VEHICLE_NAME).join()
        else:
            print("Blocked! Ascending...")
            current_z = self.get_position().z_val
            new_z = current_z - 3.0 
            self.client.moveToZAsync(new_z, 2, vehicle_name=VEHICLE_NAME).join()
            
            yaw = self.get_yaw()
            vx = math.cos(yaw) * FLIGHT_SPEED
            vy = math.sin(yaw) * FLIGHT_SPEED
            self.client.moveByVelocityZAsync(vx, vy, new_z, 2, vehicle_name=VEHICLE_NAME).join()

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
