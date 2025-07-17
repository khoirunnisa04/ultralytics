from ultralytics import YOLO

# Load model YOLOv8
model = YOLO('yolov8s.pt')  # Bisa diganti: yolov8n.pt, yolov8m.pt, atau custom model

# Training
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov8_kopi',
    optimizer='AdamW',         # Karena dataset kompleks
    workers=4,
    # device=0                   # Gunakan GPU (ubah ke 'cpu' jika tanpa GPU)
)