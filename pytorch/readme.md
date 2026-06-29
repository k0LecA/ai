# PyTorch CUDA Setup (GTX 1060 3GB)

Install PyTorch with CUDA 12.6 support:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

By default use:

```bash
python3
```

Test CUDA support:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
```

Expected output:

```python
True
NVIDIA GeForce GTX 1060 3GB
(6, 1)
```
