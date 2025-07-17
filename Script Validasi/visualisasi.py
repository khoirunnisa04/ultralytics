
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Path model YOLO
# model_path = 'ultralytics/ultralytics/runs/detect/ConvNeXtV2_6JT3/weights/best.pt'  # Ganti sesuai path model kamu

# # Path gambar
# # image_path = 'dataKopi/dataset_split/test/images/20240324_090801_jpg.rf.13dac4452e0ec73f50b7147eea532edc.jpg'  # Ganti sesuai gambar kamu
# image_path = "dataKopi/dataset_split/test/images/IMG_20240324_130957_jpg.rf.f6139ea33b2fbee9b555b06b8496701d.jpg"

# # Muat model
# model = YOLO(model_path)

# # Lakukan prediksi
# results = model(image_path)

# # Baca gambar asli
# img = cv2.imread(image_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Inisialisasi hitungan kematangan
# count_kematangan = {
#     'matang': 0,
#     'setengah_matang': 0,
#     'mentah': 0,
#     'terlalu_matang': 0
# }

# # Simpan semua koordinat bounding box untuk menentukan posisi teks
# all_boxes = []

# # Deteksi dan gambar bounding box
# for r in results:
#     boxes = r.boxes
#     if boxes:
#         xyxy = boxes.xyxy.cpu().numpy()
#         classes = boxes.cls.cpu().numpy()

#         for box, cls in zip(xyxy, classes):
#             xmin, ymin, xmax, ymax = map(int, box)
#             cls = int(cls)
#             all_boxes.append(ymin)  # save ymin untuk tentukan posisi atas

#             # Tentukan label kategori
#             if cls == 0:
#                 label = 'terlalu_matang'
#                 count_kematangan['terlalu_matang'] += 1
#             elif cls == 1:
#                 label = 'matang'
#                 count_kematangan['matang'] += 1
#             elif cls == 2:
#                 label = 'setengah_matang'
#                 count_kematangan['setengah_matang'] += 1
#             elif cls == 3:
#                 label = 'mentah'
#                 count_kematangan['mentah'] += 1
#             else:
#                 label = 'tidak_diketahui'

#             # Gambar bounding box
#             color = (0, 255, 0)
#             cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), color, 2)
#             # Tulis label di atas bounding box
#             cv2.putText(img_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# # Jika ada bounding box, tentukan posisi teks di atas paling atas
# if all_boxes:
#     ymin_terendah = min(all_boxes)
#     # Tambahkan jumlah di posisi atas tersebut
#     text_jumlah = "Jumlah buah kopi:\n"
#     for kategori, jumlah in count_kematangan.items():
#         text_jumlah += f"{kategori}: {jumlah}\n"

#     # Tulis teks di posisi atas (di atas bounding box tertinggi)
#     y_pos = max(0, ymin_terendah - 20)  # 20 piksel dari bounding box paling atas
#     x_pos = 10
#     for line in text_jumlah.splitlines():
#         cv2.putText(img_rgb, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         y_pos -= 20  # naik ke atas untuk baris berikutnya

# # Tampilkan gambar
# plt.figure(figsize=(10, 10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.title('Deteksi Buah Kopi dan Kematangannya')
# plt.show()

# # Tampilkan jumlah di console juga
# print("Jumlah buah kopi berdasarkan kematangannya:")
# for kategori, jumlah in count_kematangan.items():
#     print(f"{kategori}: {jumlah}")


import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Path model YOLO
model = "runs/detect/TRAIN-YAML/YOLOv8s+ConvNeXtV2/weights/best.pt"

# Path gambar
# image_path = "dataset_buahkopi/test/images/1_20211226_112802_jpg.rf.dd60dd0befd4738129d178b05d0232d2.jpg"
image_path ="dataset_buahkopi/test/images/1_20211209_100741_jpg.rf.ce92d9b3e9149db54d780083bca8d7ac.jpg"
# image_path = "test3.jp
# image_path =  "dataKopi/dataset_split/test/images/1_20211204_125721_jpg.rf.44bcb63c02cac97ab9bd921e42094f11.jpg"

# Muat model
model = YOLO(model)

# Lakukan prediksi
results = model(image_path)

# Baca gambar asli
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Inisialisasi hitungan kematangan
count_kematangan = {
    'matang': 0,
    'setengah_matang': 0,
    'mentah': 0,
    'terlalu_matang': 0
}

# Define colors for each class
colors = {
    'terlalu_matang': (255, 0, 0),  # Red
    'matang': (0, 255, 0),        # Green
    'setengah_matang': (255, 255, 0), # Yellow
    'mentah': (0, 0, 255)         # Blue
}

# Deteksi dan gambar bounding box
for r in results:
    boxes = r.boxes
    if boxes:
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for box, cls in zip(xyxy, classes):
            xmin, ymin, xmax, ymax = map(int, box)
            cls = int(cls)

            # Tentukan label kategori dan warna
            if cls == 0:
                label = 'terlalu_matang'
                count_kematangan['terlalu_matang'] += 1
                color = colors['terlalu_matang']
            elif cls == 1:
                label = 'matang'
                count_kematangan['matang'] += 1
                color = colors['matang']
            elif cls == 2:
                label = 'setengah_matang'
                count_kematangan['setengah_matang'] += 1
                color = colors['setengah_matang']
            elif cls == 3:
                label = 'mentah'
                count_kematangan['mentah'] += 1
                color = colors['mentah']
            else:
                label = 'tidak_diketahui'
                color = (128, 128, 128) # Grey for unknown

            # Gambar bounding box
            cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), color, 2)
            # Tulis label di atas bounding box
            cv2.putText(img_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Tambahkan jumlah di posisi atas kiri gambar
text_jumlah = "Jumlah buah kopi:"
y_pos_start = 30  # Starting Y position for the first line of text
x_pos = 10
font_scale = 0.7
font_thickness = 2
text_color = (0, 0, 0) # White color for the summary text for better contrast

cv2.putText(img_rgb, text_jumlah, (x_pos, y_pos_start), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

y_offset = y_pos_start + 25 # Offset for subsequent lines

for kategori, jumlah in count_kematangan.items():
    line = f"{kategori}: {jumlah}"
    cv2.putText(img_rgb, line, (x_pos, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    y_offset += 25 # Move down for the next line

# Tampilkan gambar
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.title('Deteksi Buah Kopi dan Kematangannya')
plt.show()

# Tampilkan jumlah di console juga
print("Jumlah buah kopi berdasarkan kematangannya:")
for kategori, jumlah in count_kematangan.items():
    print(f"{kategori}: {jumlah}")