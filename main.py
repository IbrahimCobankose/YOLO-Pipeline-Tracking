import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv 
from collections import defaultdict, deque 

from auv_controller import AUVController 

WEIGHTS_PATH = 'models/best.pt' 
VIDEO_PATH = 'videos/test_video.mp4' 
OUTPUT_PATH = 'videos/AUV_Pipe_Tracking_System.mp4'

auv = AUVController(connection_address="tcp:127.0.0.1:5762")
auv.arm_and_set_mode("GUIDED")

model = YOLO(WEIGHTS_PATH)
model.to('cuda') 

# --- TRACKER INITIALIZATION ---
# Initialize ByteTrack to assign unique IDs to detected nodes
byte_track = sv.ByteTrack(frame_rate=30, track_activation_threshold=0.15)

# Dictionary to store the Y-coordinates of each node over the last 30 frames
node_y_coordinates = defaultdict(lambda: deque(maxlen=30))

# --- POLYGON ZONE DEFINITION ---
raw_pts = [[530, 824], [611, 609], [843, 595], [892, 825], [531, 824]]
ZONE_POINTS = np.array(raw_pts, np.int32)

#
x_coords = ZONE_POINTS[:, 0]
poly_center_x = (np.min(x_coords) + np.max(x_coords)) // 2

cap = cv2.VideoCapture(VIDEO_PATH)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (w, h))

frame_idx = 0
print(f"Underwater video is being processed... Resolution: {w}x{h}")

while True:
    ret, frame = cap.read()
    
    if not ret:
          print(f"\n✅ Process complete! Total frames processed: {frame_idx}")
          break

    results = model(frame, conf=0.15, imgsz=1024, half=True, verbose=False)[0]
    
    # Convert YOLO results to Supervision Detections format and update tracker
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_track.update_with_detections(detections=detections)

    overlay = frame.copy()
    
    # Default Signals
    signal = "The pipe is missing!"
    signal_color = (128, 128, 128) # Gray
    
    node_detected = False
    pipe_pts = None

    # Draw the sensor area on the image.
    cv2.fillPoly(overlay, [ZONE_POINTS], (200, 200, 200))

    # Existing Instance Segmentation mask drawing
    if results.masks is not None:
        classes = results.boxes.cls.cpu().numpy()
        masks = results.masks.xy
        
        for i, mask_data in enumerate(masks):
            cls = int(classes[i])
            pts = np.array(mask_data, dtype=np.int32)

            if cls == 1: 
                # Pipe Node 
                cv2.fillPoly(overlay, [pts], (0, 0, 255))      
                node_detected = True
            elif cls == 0: 
                # Pipe 
                cv2.fillPoly(overlay, [pts], (255, 0, 0))      
                pipe_pts = pts 

    # --- NODE TRACKING & SPEED ESTIMATION ---
    if detections.tracker_id is not None:
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i]
            xyxy = detections.xyxy[i] # Bounding box coordinates [x1, y1, x2, y2]
            
            # Only track and calculate speed for Pipe Nodes (Class 1)
            if class_id == 1:
                # Calculate the center X and Y of the node
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)
                
                # Store the Y coordinate for historical tracking
                node_y_coordinates[tracker_id].append(center_y)
                
                # If we have enough history (e.g., 10 frames), calculate pixel speed
                if len(node_y_coordinates[tracker_id]) > 10:
                    y_start = node_y_coordinates[tracker_id][0]
                    y_current = node_y_coordinates[tracker_id][-1]
                    
                    # Positive displacement means the node is moving down on the screen
                    # This implies the AUV is moving forward
                    pixel_displacement = y_current - y_start
                    
                    text = f"ID: #{tracker_id} Speed: {pixel_displacement} px/s"
                else:
                    text = f"ID: #{tracker_id}"
                
                # Draw the tracking ID and speed slightly above the node
                cv2.putText(overlay, text, (center_x - 50, center_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- AUV Decision Logic ---
    if node_detected:
        signal = "RISE UP - NODE DETECTED!"
        signal_color = (0, 0, 255) # Red
        # Stop moving forward (vx=0) and rise up (vz=-0.5) when a node is detected
        auv.send_body_velocity(vx=0.0, vy=0.0, vz=-0.5, yaw_rate=0.0)
        
    elif pipe_pts is not None:
        pipe_cx = int(np.mean(pipe_pts[:, 0]))
        margin = 80 
        
        if pipe_cx < (poly_center_x - margin):
            signal = "Move Left"
            signal_color = (0, 255, 255) # Yellow
            # Keep moving forward (vx=0.5) but also slide left (vy=-0.5)
            auv.send_body_velocity(vx=0.5, vy=-0.5, vz=0.0, yaw_rate=0.0)
            
        elif pipe_cx > (poly_center_x + margin):
            signal = "Move Right"
            signal_color = (0, 165, 255) # Orange
            # Keep moving forward (vx=0.5) but also slide right (vy=0.5)
            auv.send_body_velocity(vx=0.5, vy=0.5, vz=0.0, yaw_rate=0.0)
            
        else:
            signal = "Centered - Forward"
            signal_color = (0, 255, 0) # Green
            # Pipe is centered, just move straight forward (vx=1.0)
            auv.send_body_velocity(vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0)
            
    else:
        # Switch to safe mode (hover) if the pipe completely leaves the frame
        auv.send_body_velocity(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Merge overlay with original frame
    output_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # Right corner box for signals
    box_w, box_h = 500, 100 
    box_x, box_y = w - box_w - 30, 30
    
    sub_face = output_frame[box_y:box_y+box_h, box_x:box_x+box_w]
    black_rect = np.zeros(sub_face.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_face, 0.3, black_rect, 0.7, 1.0) 
    output_frame[box_y:box_y+box_h, box_x:box_x+box_w] = res
    
    cv2.putText(output_frame, f"{signal}", (box_x + 20, box_y + 70), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, signal_color, 3)

    out.write(output_frame)
    frame_idx += 1
    
    if frame_idx % 80 == 0: 
        print(f"Processed frame: {frame_idx}")

cap.release()
out.release()

