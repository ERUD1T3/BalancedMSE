import torch

# Check for MPS availability
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS (Apple Silicon backend) is available!")
    print(f"Using MPS device: {mps_device}")

    # Attempt to create a tensor on the MPS device to confirm functionality
    try:
        x = torch.ones(1, device=mps_device)
        print("Successfully created a tensor on MPS device:", x)
        print("MPS device seems functional.")
    except Exception as e:
        print(f"Could not create tensor on MPS device: {e}")
        print("There might be an issue with the MPS backend.")
else:
    print("MPS backend is not available on this system.")
