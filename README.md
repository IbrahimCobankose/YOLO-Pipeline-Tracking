# Autonomous AUV Pipeline Tracking and Control System

## Overview
This project implements an end-to-end computer vision and autonomous navigation system designed for Autonomous Underwater Vehicles (AUVs). Utilizing YOLO for instance segmentation, the system detects pipelines and pipeline nodes in real-time. It goes beyond simple detection by integrating **ByteTrack** for node speed estimation and **MAVLink (pymavlink)** to send dynamic, real-time velocity commands directly to the vehicle's flight controller (ArduSub/ArduPilot).

## Key Features
* **Instance Segmentation:** Real-time pipe and node detection using YOLO.
* **Autonomous Navigation:** Real-time decision-making logic that translates visual data into MAVLink body-frame velocity commands (`vx`, `vy`, `vz`).
* **Node Tracking & Speed Estimation:** Integration of `supervision` and ByteTrack to assign unique IDs to structural nodes and estimate vehicle progression speed based on pixel displacement.
* **Customizable ROI:** Interactive polygon zone definition to limit the vehicle's functional field of view.

## Example Results
The system overlays segmentation masks, highlights the defined tracking zone, prints navigational signals, and displays tracker IDs with speed estimation on the processed frames.

### Scenario 1
**Original Input:**
![Original Frame 1](example%20result/1.png)

**Processed Output:**
![Processed Frame 1](example%20result/2.png)

### Scenario 2
**Original Input:**
![Original Frame 2](example%20result/3.png)

**Processed Output:**
![Processed Frame 2](example%20result/4.png)

## Installation
Ensure you have Python installed. It is highly recommended to use a virtual environment. Install the required dependencies (including `ultralytics`, `supervision`, `opencv-python`, and `pymavlink`) using the provided requirements file:

```bash
pip install -r requirements.txt
```

## System Architecture & Workflow
To implement and run this autonomous tracking system, follow these sequential steps:

### 1. Data Preparation and Labeling
Prepare your dataset by annotating your images or video frames. You must label the targets (e.g., "pipe" and "pipe node") specifically using **Instance Segmentation** mode (creating polygonal masks).

### 2. Model Training
Train the YOLO model with your custom dataset using the training script:
```bash
python train.py
```

### 3. Defining the Region of Interest (Zone)
To establish the optimal tracking boundaries for your camera's field of view, run the polygon mapping tool:
```bash
python polygon_zone.py
```
Interact with the UI to draw your desired zone. Copy the resulting coordinate array outputted in the terminal and update the `raw_pts` variable inside `main.py`.

### 4. Vehicle Connection & Autonomous Tracking
Ensure your AUV or SITL (Software In The Loop) simulation is running and accessible via the defined TCP/UDP port (default is `tcp:127.0.0.1:5762`). 

Run the main execution script. This script will:
1. Establish a MAVLink connection to the AUV.
2. Arm the vehicle and set it to `GUIDED` mode.
3. Process the video feed, track nodes, and continuously send local NED velocity commands to center and follow the pipeline.

```bash
python main.py
```