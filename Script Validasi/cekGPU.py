import torch

if torch.cuda.is_available():
    print("✅ GPU terdeteksi dan aktif:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU tidak aktif atau tidak terdeteksi.")
