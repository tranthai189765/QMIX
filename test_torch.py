import torch, numpy as np
print("torch version:", torch.__version__)
print("torch.long =", torch.long)            # phải là torch.int64
print("torch.int64 =", torch.int64)
print("sanity 1:", torch.tensor([1], dtype=torch.long).dtype)
print("sanity 2:", torch.tensor([1]).to(torch.long).dtype)