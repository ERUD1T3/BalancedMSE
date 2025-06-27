import torch

print("--- PyTorch Device Availability Check ---")

# Check for CUDA (NVIDIA GPUs)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_gpus}")
    for i in range(num_gpus):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    # Try using the first CUDA device
    try:
        cuda_device = torch.device("cuda:0")
        x_cuda = torch.ones(1, device=cuda_device)
        print(f"Successfully created tensor on {cuda_device}: {x_cuda}")
        print("CUDA device seems functional.")
    except Exception as e:
        print(f"Could not create tensor on CUDA device: {e}")
else:
    print("No CUDA devices found.")

print("-" * 20) # Separator

# Check for MPS (Apple Silicon GPUs)
# Check if the attribute exists before checking availability
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS Available: {mps_available}")

if mps_available:
    mps_device = torch.device("mps")
    print(f"Using MPS device: {mps_device}")
    # Try using the MPS device
    try:
        x_mps = torch.ones(1, device=mps_device)
        print(f"Successfully created tensor on {mps_device}: {x_mps}")
        print("MPS device seems functional.")
    except Exception as e:
        print(f"Could not create tensor on MPS device: {e}")
        print("There might be an issue with the MPS backend.")
else:
    # Added check to clarify if MPS backend exists but is not built/found
    if hasattr(torch.backends, "mps") and not torch.backends.mps.is_built():
         print("MPS backend exists but is not built.")
    elif not hasattr(torch.backends, "mps"):
        print("MPS backend attribute does not exist in this PyTorch build.")
    else: # Exists and is built, but not available
        print("MPS device found but not available/functional.")


print("-" * 20) # Separator

# Conclusion
if cuda_available or mps_available:
    print("GPU acceleration (CUDA or MPS) is available.")
else:
    print("No GPU acceleration found. PyTorch will use CPU.")
    cpu_device = torch.device("cpu")
    try:
        x_cpu = torch.ones(1, device=cpu_device)
        print(f"Successfully created tensor on CPU: {x_cpu}")
    except Exception as e:
        print(f"Could not create tensor on CPU: {e}")

print("-----------------------------------------") 