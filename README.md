# YOLO Pipeline Tracking

## Overview
This project implements a computer vision-based tracking system designed to detect and follow pipelines and pipeline nodes. Utilizing YOLO for instance segmentation, the system processes video feeds to identify structural elements and calculates real-time directional feedback based on a customizable region of interest (ROI).

## Example Results
The system overlays segmentation masks, highlights the defined tracking zone, and prints navigational signals on the processed frames. Below are examples of the raw input frames and their corresponding outputs.

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
Ensure you have Python installed, then install the required dependencies using the provided requirements file.

```bash
pip install -r requirements.txt
```

## Workflow and Usage
To implement and run this tracking system, follow these sequential steps:

### 1. Data Preparation and Labeling
[cite_start]Label the pipes and pipe nodes in your video using Instance Segmentation mode[cite: 2].

### 2. Model Training
[cite_start]Train the model with your dataset using the `train.py` code[cite: 2].
```bash
python train.py
```

### 3. Defining the Region of Interest (Zone)
[cite_start]Determine a zone for your vehicle's viewing range in `polygon_zone.py`[cite: 3]. [cite_start]Use the resulting coordinate outputs for the `raw_pts` variable in `main.py`[cite: 3].
```bash
python polygon_zone.py
```

### 4. Inference and Tracking
[cite_start]Then run `main.py` to get the results[cite: 4].
```bash
python main.py
```
