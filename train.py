from ultralytics import YOLO

# Load the model
model = YOLO('yolo11x-seg.pt')

# Start training
model.train(
    data='dataset/data.yaml', 
    epochs=100,
    imgsz=640,
    device=0, 
    project='runs/train', 
    name='pipe_tracking_final_run'
)