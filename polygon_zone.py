import cv2
import numpy as np

pts = []

def draw_polygon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        print(f"Point {len(pts)} added: {x, y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pts) > 0:
            pts.pop() 
            print("Last point removed.")


video_path = 'videos/test_video.mp4' #Video path
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

# For polygon mapping, we will jump to the 1-minute mark of the video to find a clear frame of the pipe.
frame_number = int(fps * 60)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

scale_w, scale_h = 1920,1080
frame_reshaped = cv2.resize(frame, (scale_w, scale_h))

cv2.namedWindow('Polygon Mapper')
cv2.setMouseCallback('Polygon Mapper', draw_polygon)

print("INSTRUCTIONS")
print("1. Left Click to add points (make as many as you want for a curve)")
print("2. Right Click to undo a point")
print("3. Press 's' to print the final array")
print("4. Press 'q' to quit")

while True:
    img_display = frame_reshaped.copy()
    
    if len(pts) > 1:
        cv2.polylines(img_display, [np.array(pts)], False, (0, 255, 255), 2)
    
    for p in pts:
        cv2.circle(img_display, (p[0], p[1]), 4, (0, 0, 255), -1)
    
    cv2.imshow('Polygon Mapper', img_display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("\n--- COPY THIS INTO YOUR MAIN SCRIPT ---")
        print(f"ZONE_POINTS = np.array({pts}, np.int32)")

        if len(pts) > 2:
            cv2.fillPoly(img_display, [np.array(pts)], (0, 255, 0))
            cv2.imshow('Polygon Mapper', img_display)
            cv2.waitKey(1000)
            
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()