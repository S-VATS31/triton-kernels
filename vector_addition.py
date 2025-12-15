"""
Add 2 same dimension tensors element-wise and benchmark against PyTorch kernel.
"""

import time

import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    A, # pointer to input vector A
    B, # pointer to input vector B
    C, # pointer to output vector C
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Add two single dimension vectors.

    Args:
        A: Pointer to input vector A.
        B: Pointer to input vector B.
        C: Pointer to input vector C.
        N (int): Number of elements in the input vector.
        BLOCK_SIZE (tl.constexpr): Number of elements in each kernel process.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # create offsets to split tiles across GPU threads
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N # create boolean mask for out of bounds positions
    # create input tensors
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    output = a + b # compute output, we will need to allocate a tensor to store the output
    tl.store(C + offsets, output, mask=mask)

def add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Add two input tensors.

    Args:
        A (torch.Tensor): Input tensor to be added.
        B (torch.Tensor): Input tensor to be added.

    Returns:
        torch.Tensor: Output tensor after both tensors are added.
    """
    C = torch.empty_like(A) # pre-allocate output tensor
    assert A.numel() == B.numel() == C.numel()
    assert A.dtype == B.dtype == C.dtype
    assert A.device == B.device == C.device
    N = C.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    kernel[grid](A, B, C, N, BLOCK_SIZE=1024)
    return C

def main(log: bool):
    A = torch.randn(1_000_000, dtype=torch.float32, device="cuda")
    B = torch.randn(1_000_000, dtype=torch.float32, device="cuda")

    # Time torch kernel
    torch.cuda.synchronize()
    torch_start = time.time()
    torch_C = A + B
    torch.cuda.synchronize()
    torch_end = time.time()

    # Time triton kernel
    torch.cuda.synchronize()
    triton_start = time.time()
    triton_C = add(A, B)
    torch.cuda.synchronize()
    triton_end = time.time()

    # Calculate times using start/end
    torch_time = torch_end - torch_start
    triton_time = triton_end - triton_start

    if log:
        print(
            f"torch kernel time: {torch_time:.6f} | "
            f"triton kernel time: {triton_time:.6f}"
        )
        if torch_time < triton_time:
            print(f"torch kernel is {triton_time / torch_time:.4f} times faster.")
        else:
            print(f"triton kernel is {torch_time / triton_time:.4f} times faster")

if __name__ == "__main__":
    main(log=True)
