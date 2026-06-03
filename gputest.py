import torch
print(torch.cuda.is_available()) # True dönmeli
print(torch.cuda.get_device_name(0)) # 'NVIDIA GeForce RTX 4050' yazmalı