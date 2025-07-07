# import optuna
# from ultralytics import YOLO

# def objective(trial):
#     lr0 = trial.suggest_loguniform("lr0", 1e-4, 1e-2)
#     weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
#     momentum = trial.suggest_uniform("momentum", 0.85, 0.99)  # Menambahkan tuning momentum
    
#     model = YOLO("yolov8s+convnextv2.yaml")
#     results = model.train(data='dataset_buahkopi/data.yaml', epochs=2, imgsz=640, batch=8, 
#                           optimizer='AdamW', lr0=lr0, weight_decay=weight_decay, momentum=momentum, 
#                           name='optuna')

#     return results.box.map  # Menggunakan mean Average Precision (mAP) sebagai evaluasi patience=20

# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=5)

#     print("Best lr0:", study.best_params["lr0"])
#     print("Best weight_decay:", study.best_params["weight_decay"])
#     print("Best momentum:", study.best_params["momentum"])


# import optuna
# from ultralytics import YOLO

# def objective(trial):
#     # Hyperparameter suggestions dari Optuna
#     lr0 = trial.suggest_loguniform("lr0", 1e-4, 1e-2)
#     weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
#     momentum = trial.suggest_uniform("momentum", 0.85, 0.95)
#     lrf = trial.suggest_uniform("lrf", 0.01, 0.05)  # Tambahan lrf

#     try:
#         # Load model (pastikan file YAML-nya benar)
#         model = YOLO("yolov8s+convnextv2.yaml")

#         # Train model dengan hyperparameter hasil saran
#         results = model.train(
#             data='dataset_buahkopi/data.yaml',
#             epochs=20,  # Gunakan lebih dari 2 agar hasil Optuna bermakna
#             imgsz=640,
#             batch=8,
#             optimizer='AdamW',
#             lr0=lr0,
#             weight_decay=weight_decay,
#             momentum=momentum,
#             lrf=lrf,
#             name=f"optuna_trial_{trial.number}",
#             verbose=False
#         )

#         # Ambil nilai mAP50-95
#         map50_95 = results.metrics.get('metrics/mAP50-95(B)', 0.0)
#         return map50_95

#     except Exception as e:
#         print(f"Trial {trial.number} gagal: {e}")
#         return 0.0

# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=15)  # Jumlah trial bisa ditambah

#     print("\n✅ Best Parameters Found:")
#     for k, v in study.best_params.items():
#         print(f"{k}: {v:.6f}")


from ultralytics import YOLO

# Inisialisasi model YOLO dengan backbone ConvNeXtV2
model = YOLO("yolov8s+eca.yaml")

# Definisikan ruang pencarian hyperparameter
search_space = {
    "lr0": (1e-4, 3e-3),
    "momentum": (0.85, 0.95),
    "weight_decay": (1e-5, 5e-4)
    # "lr0": (1e-5, 1e-1),  # Learning rate
    # "momentum": (0.6, 0.98),  # Momentum
    # "weight_decay": (0.0, 0.001),  # Weight decay
    # "warmup_epochs": (0.0, 5.0),  # Warmup epochs
    # "box": (0.02, 7.5),  # Box loss weight
    # "cls": (0.2, 4.0),  # Classification loss weight
}

# Lakukan tuning hyperparameter pada dataset COCO8 selama 30 epoch
results = model.tune(
    data="dataset_buahkopi/data.yaml",
    epochs=200,
    iterations=5,
    optimizer="AdamW",
    batch=8,
    space=search_space,
    plots=False,
    save=True,
    val=True,
)

# Tampilkan hasil tuning
print(results)
