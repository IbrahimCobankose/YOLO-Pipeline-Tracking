import cv2
from ultralytics import YOLO
import numpy as np

WEIGHTS_PATH = 'models/best.pt' 
VIDEO_PATH = 'videos/test_video.mp4' 
OUTPUT_PATH = 'videos/AUV_Pipe_Tracking_System.mp4'

model = YOLO(WEIGHTS_PATH)
model.to('cuda') 

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
    overlay = frame.copy()
    
    # Default Signals
    signal = "The pipe is missing!"
    signal_color = (128, 128, 128) # Gri
    
    node_detected = False
    pipe_pts = None

    # Draw the sensor area on the image.
    cv2.fillPoly(overlay, [ZONE_POINTS], (200, 200, 200))

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

    # --- AUV Decision Logic ---
    if node_detected:
        signal = "RISE UP - NODE DETECTED!"
        signal_color = (0, 0, 255) # Red
        
    elif pipe_pts is not None:
        pipe_cx = int(np.mean(pipe_pts[:, 0]))
        margin = 80 
        
        if pipe_cx < (poly_center_x - margin):
            signal = "Move Left"
            signal_color = (0, 255, 255) # Yellow
        elif pipe_cx > (poly_center_x + margin):
            signal = "Move Right"
            signal_color = (0, 165, 255) # Orange
        else:
            signal = "Centered - Forward"
            signal_color = (0, 255, 0) # Green

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

