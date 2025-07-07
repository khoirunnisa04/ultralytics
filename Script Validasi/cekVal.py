from ultralytics import YOLO

# Melakukan tuning hyperparameter
if __name__ == '__main__':
    model = YOLO('runs/detect/yolov8s+convnextv2_/weights/best.pt')  
    model.val(
        data='dataset_buahkopi\data.yaml',
        name="val_YOLOv8s+convnextv2_wd-0.000075"
)
# Tampilkan hasil validasi
# print(results)