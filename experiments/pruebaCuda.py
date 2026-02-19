import torch
print(torch.__version__)
print(torch.cuda.is_available())       # esperemos que devuelva True
print(torch.cuda.get_device_name(0))   # Debe mostrar "NVIDIA GeForce RTX 2060"
print(torch.version.cuda)              # 12.6 o 12.8