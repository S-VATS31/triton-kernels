"""
This program leverages Triton to multiply two N dimensional tensors element-wise.
"""

from typing import Tuple
import time

import torch
import triton
import triton.language as tl

@triton.jit
def element_wise_mul_kernel(
    x_ptr, # pointer to input tensor x
    y_ptr, # pointer to input tensor y
    out_ptr, # pointer to output tensor
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    # set up program id, axis=0 since we flatten tensors to single dim
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # Compute offsets to split work across threads
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N # create boolean mask for out of bounds positions
    # load pointers from VRAM for computation
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # multiply tensors element-wise
    output = x * y
    tl.store(out_ptr + offsets, output, mask=mask)

def mul(
    x: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE=256
):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel() == y.numel()
    # pre-allocate output tensor
    # empty_like() retains shape, numel, dtype, device
    out = torch.empty_like(x)
    # Get number of elements in tensor
    N = x.numel()
    # use ceil div for grid
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,) # must be a tuple
    # launch kernel
    element_wise_mul_kernel[grid](
        x, y, out, N, BLOCK_SIZE
    )
    return out

def benchmark(x: torch.Tensor, y: torch.Tensor, num_warmups: int = 3) -> Tuple[bool, str]:
    # warmup kernel/torch
    for _ in range(num_warmups):
        _ = mul(x, y)
        _ = x * y
    # time kernel
    torch.cuda.synchronize()
    kernel_start = time.time()
    kernel_out = mul(x, y)
    torch.cuda.synchronize()
    kernel_end = time.time()
    # time torch kernel
    torch.cuda.synchronize()
    torch_start = time.time()
    torch_out = x * y
    torch.cuda.synchronize()
    torch_end = time.time()
    # Get actual times
    kernel_time = kernel_end - kernel_start
    torch_time = torch_end - torch_start

    # check if kernel worked correctly
    if not torch.allclose(kernel_out, torch_out):
        return False, "kernel failed, incorrect output."
    else:
        return (
            True,
            f"kernel time: {kernel_time:.6f} | "
            f"torch time: {torch_time:.6f} | "
            f"speedup: {max(kernel_time, torch_time) / min(kernel_time, torch_time)}"
        )

def main():
    x = torch.randn(32, 32, 32, 32, dtype=torch.float16, device="cuda")
    y = torch.randn(32, 32, 32, 32, dtype=torch.float16, device="cuda")
    return benchmark(x, y)

success, info = main()
print(success)
print(info)
