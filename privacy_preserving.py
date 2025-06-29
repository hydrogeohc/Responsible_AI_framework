import torch
import syft as sy

def main():
    # Hook PyTorch to add PySyft functionalities
    hook = sy.TorchHook(torch)
    
    # Create virtual workers
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")

    # Create and share tensor
    x = torch.tensor([1, 2, 3, 4, 5])
    x_shared = x.share(alice, bob)

    print("Original tensor:", x)
    print("Shared tensor:", x_shared)

    # Secure computation: Addition
    y_shared = x_shared + x_shared
    print("Addition result:", y_shared)

    # Secure computation: Multiplication
    z_shared = x_shared * 2
    print("Multiplication result:", z_shared)

    # Reconstruct results
    y = y_shared.get()
    z = z_shared.get()
    print("Reconstructed addition:", y)
    print("Reconstructed multiplication:", z)

if __name__ == "__main__":
    main()