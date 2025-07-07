# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO(r"")

# # Run inference on 'bus.jpg' with arguments
# results=model("D:\, save=True, show= True, imgsz=640, conf=0.5)

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk



import os
from ultralytics import YOLO

# Path ke folder images (ubah sesuai lokasi folder kamu)
folder_images = 'dataset_buahkopi/test/images'

# Muat model
model_path = 'runs/detect/ConvNeXtV2_ECA_YOLOv82/weights/best.pt'
model = YOLO(model_path)

# Pastikan folder berisi gambar
files = os.listdir(folder_images)
print("File gambar ditemukan:", files)

# Lakukan inference semua gambar di folder
results = model(folder_images)

# Hitung confidence dari semua deteksi
import numpy as np

confidences = []

for result in results:
    boxes = result.boxes
    if boxes:
        confs = boxes.conf.cpu().numpy()
        confidences.extend(confs)

# Rata-rata confidence keseluruhan
if confidences:
    print(f"Rata-rata confidence seluruh deteksi: {np.mean(confidences):.3f}")
else:
    print("Tidak ada deteksi yang ditemukan.")

# Jika ingin analisis berdasarkan kelas
conf_by_class = {}

for result in results:
    boxes = result.boxes
    if boxes:
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for cls, conf in zip(classes, confs):
            cls = int(cls)
            if cls not in conf_by_class:
                conf_by_class[cls] = []
            conf_by_class[cls].append(conf)

for cls, confs in conf_by_class.items():
    print(f"Class {cls} - Rata-rata confidence: {np.mean(confs):.3f}")