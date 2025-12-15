# Autonomous Disaster Response Drone

A simulation project for autonomous decision-making and path planning in disaster scenarios.

## Project Description
This system controls a drone in the **Microsoft AirSim** environment. It is designed to fly autonomously toward a target location while performing two critical tasks:
1.  **Survivor Detection:** Identifying humans using deep learning.
2.  **Obstacle Avoidance:** navigating around or over debris using depth mapping.

## Technologies Used
* **Microsoft AirSim**
* **Python**
* **YOLO** (For Person Detection)
* **OpenCV / DepthPlanner** (For Object Detection & Avoidance)

## How It Works
The drone executes a logical loop divided into **0.1-second time steps**:

1.  **Move:** Advance toward the target.
2.  **Scan for Persons:**
    * If a person is detected -> **Slow Down** and search.
    * If no person -> Continue to object detection.
3.  **Scan for Obstacles:**
    * If an obstacle is detected via depth map -> **Check Left/Right**.
    * If Left/Right are blocked -> **Climb Up** and move forward.

## Video Demo
[YouTube Link Pending]

## Contributors

| Name | Student ID |
| :--- | :--- |
| **Chaudhary Darshil** | 206125005 |
| **Chavda Jaydip** | 206125008 |
| **Pratyush Mishara** | 206125021 |
