import torch
print("可用GPU数量:", torch.cuda.device_count())
print("GPU名称:", torch.cuda.get_device_name(0))
print("GPU编号:", torch.cuda.current_device())

'''
python -c "import torch; print('\n'.join([f'CUDA编号 {i}: {torch.cuda.get_device_name(i)}' for i in range(torch.cuda.device_count())]))"
CUDA编号 0: NVIDIA RTX 6000 Ada Generation
CUDA编号 1: NVIDIA RTX 6000 Ada Generation
CUDA编号 2: NVIDIA GeForce RTX 4090
CUDA编号 3: NVIDIA A100-SXM4-80GB
CUDA编号 4: NVIDIA RTX A6000
CUDA编号 5: NVIDIA RTX A6000
