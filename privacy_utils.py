import torch
import syft as sy

def get_virtual_workers():
    hook = sy.TorchHook(torch)
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    return alice, bob

def share_tensor(tensor, workers):
    return tensor.share(*workers)