import torch
print("CUDA 可用？", torch.cuda.is_available())
print("GPU 數量：", torch.cuda.device_count())
print("目前使用的 GPU：", torch.cuda.get_device_name(0))
print("PyTorch CUDA 版本：", torch.version.cuda)
print("PyTorch 版本：", torch.__version__)
