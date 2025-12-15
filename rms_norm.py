import time

import torch
import torch.nn as nn
from torch.cuda import synchronize
import triton
import triton.language as tl

DEVICE = torch.device("cuda")

@triton.jit
def rms_norm_kernel(
    x_ptr, # pointer to input [B*T, D]
    output_ptr, # pointer to output [B*T, D]
    weight_ptr, # pointer to scale gamma [D]
    eps: float,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * D + tl.arange(0, D)
    mask = offsets < (pid + 1) * D # mask boundaries
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_squares = tl.sum(x * x, axis=0)
    rms = tl.sqrt(sum_squares / D + eps)
    x_norm = x / rms
    weight = tl.load(weight_ptr + tl.arange(0, D), mask=mask)
    out = x_norm * weight
    tl.store(output_ptr + offsets, out, mask=mask)


class RMSNormTriton(nn.Module):
    def __init__(self, d_model: int, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, D)
        output = torch.empty_like(x)

        grid = (B * T,)
        rms_norm_kernel[grid](
            x, output, self.weight, self.eps, D,
            num_warps=4, num_stages=2
        )

        # Reshape output back to [B, T, D]
        return output.view(B, T, D)

class RMSNormTorch(nn.Module):
    def __init__(self, d_model: int, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return self.weight * (
            x / x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        )

# init
B, T, D = 16, 8, 4096
x = torch.randn(B, T, D, dtype=torch.float16, device=DEVICE)

# kernel norm
kernel_norm = RMSNormTriton(D).to(DEVICE)

# warmup kernel
for _ in range(3):
    _ = kernel_norm(x)

# kernel forward pass with timing
synchronize()
kernel_start = time.time()
x_kernel = kernel_norm(x)
synchronize()
kernel_end = time.time()
kernel_time = kernel_end - kernel_start

# naive norm
naive_norm = RMSNormTorch(D).to(DEVICE)

# warmup torch
for _ in range(3):
    _ = naive_norm(x)

# naive forward pass with timing
synchronize()
torch_start = time.time()
x_naive = naive_norm(x)
synchronize()
torch_end = time.time()
torch_time = torch_end - torch_start

print(f"kernel time: {kernel_time}")
print(f"torch  time: {torch_time}")
print(f"speedup: {max(kernel_time, torch_time) / min(kernel_time, torch_time)}")
