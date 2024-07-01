import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print CUDA toolkit version if CUDA is available
if cuda_available:
    cuda_version = torch.version.cuda
    print(f"CUDA Toolkit Version: {cuda_version}")
else:
    print("CUDA is not available.")

# Print Torch version
torch_version = torch.__version__
print(f"PyTorch Version: {torch_version}")
