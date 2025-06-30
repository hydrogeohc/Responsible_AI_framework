import torch
import syft as sy

# Setup virtual workers for secure aggregation
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
secure_worker = sy.VirtualWorker(hook, id="secure_worker")

def secure_aggregate(data_alice, data_bob):
    # Simulate sending data to workers
    data_alice_ptr = torch.tensor(data_alice).send(alice)
    data_bob_ptr = torch.tensor(data_bob).send(bob)
    # Secure aggregation: collect and sum at secure_worker
    result = (data_alice_ptr + data_bob_ptr).get(secure_worker)
    return result.get()

def encrypt_tensor(tensor):
    # Simple additive secret sharing
    return tensor.share(alice, bob, crypto_provider=secure_worker)

def decrypt_tensor(shared_tensor):
    return shared_tensor.get()

def check_access(user_role, resource):
    # Simple access control: only 'admin' can access sensitive resource
    if user_role != "admin":
        raise PermissionError("Access denied to resource: {}".format(resource))
    return True