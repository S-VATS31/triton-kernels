"""Triton kernel to implement SwiGLU as well as a backward pass."""
import time
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import triton
import triton.language as tl

DEVICE = torch.device("cuda")
DTYPE = torch.float32
D = 256
D_FFN = 4 * D
B = 4
T = 128

# SwiGLU definition:
# swiglu(x) = w3 @ (swish(x @ w1 + b1) * (x @ w2 + b2)) + b3
# broken down:
# y1 = swish(x @ w1 + b1)
# y2 = x @ w2 + b2
# y3 = w3 @ (y1 * y2) + b3
# where:
# - y1: [BT, D_FFN]
# - y2: [BT, D_FFN]
# - y3: [BT, D]

# OPTIMIZED SWIGLU FUNCTION IN TRITON (HIGH LEVEL OVERVIEW)
# For SwiGLU we get an input tensor of 3 dims which we flatten to 2 dims for
# classic optimized matrix multiplication. We use 3 different weight matrices
# which act as linear projections. Optional biases are added if given in
# correct shape and dimension size. For extreme optimization, we expect
# all tensors must be contiguous.
#
# step 1. get input tensor [B, T, D] and flatten B, T dims for matmul
# >> x shape = [B, T, D]
# >> x shape = [B*T, D]
#
# step 2. set up weight matrices for SwiGLU
# >> w1 and w2 shape: [D, D_FFN]
# >> w3 shape: [D_FFN, D]
#
# step 3. set up bias vectors for SwiGLU
# >> b1 and b2 shape: [D_FFN]
# >> b3 shape: [D]
#
# step 4. compute SwiGLU(x)
# >> swiglu(x) = w3(swish(w1x) * (w2x)) NOTE: added biases if given
# >> biases are added via b[None, :] as x @ w1 is a 2 dim tensor


@triton.jit
def swiglu_kernel(
    x_ptr,   # [B*T, D]
    out_ptr, # [B*T, D]
    w1_ptr,  # [D, D_FFN]
    w2_ptr,  # [D, D_FFN]
    w3_ptr,  # [D_FFN, D]
    b1_ptr,  # [D_FFN]
    b2_ptr,  # [D_FFN]
    b3_ptr,  # [D]
    stridex_row: tl.constexpr,
    stridex_col: tl.constexpr,
    stridew1_row: tl.constexpr,
    stridew1_col: tl.constexpr,
    stridew2_row: tl.constexpr,
    stridew2_col: tl.constexpr,
    stridew3_row: tl.constexpr,
    stridew3_col: tl.constexpr,
    strideout_row: tl.constexpr,
    strideout_col: tl.constexpr,
    tokens: int, # B*T
    D: int,
    D_FFN: int, # typically 4*D
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_D_FFN: tl.constexpr
):
    # row program id, col program id
    row_pid = tl.program_id(axis=0)
    col_pid = tl.program_id(axis=1)
    x_row_offs = row_pid * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)
    out_row_offs = row_pid * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)
    out_col_offs = col_pid * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)

    # handle biases
    if b1_ptr is not None:
        b1_offs = tl.arange(0, BLOCK_SIZE_D_FFN)
        b2_offs = tl.arange(0, BLOCK_SIZE_D_FFN)
        b3_offs = col_pid * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        # bias masks
        b1_mask = b1_offs < D_FFN
        b2_mask = b2_offs < D_FFN
        b3_mask = b3_offs < D
        # load biases
        b1 = tl.load(b1_ptr + b1_offs, mask=b1_mask, other=0.0)
        b2 = tl.load(b2_ptr + b2_offs, mask=b2_mask, other=0.0)
        b3 = tl.load(b3_ptr + b3_offs, mask=b3_mask, other=0.0)
    else:
        b1, b2, b3 = None, None, None
    # initialize accumulation for final output
    acc_out = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_D), dtype=tl.float32)
    # Loop over D_FFN dimension in blocks
    for d_ffn in range(0, D_FFN, BLOCK_SIZE_D_FFN):
        # Initialize accumulators for x @ w1 and x @ w2
        acc1 = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_D_FFN), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_D_FFN), dtype=tl.float32)
        # Compute x @ w1 and x @ w2 for this D_FFN block
        for d in range(0, D, BLOCK_SIZE_D):
            # iterate by d for tiled matrix multiplication
            x_col_offs = d + tl.arange(0, BLOCK_SIZE_D)
            w1_row_offs = d + tl.arange(0, BLOCK_SIZE_D)
            w2_row_offs = d + tl.arange(0, BLOCK_SIZE_D)
            w1_col_offs = d_ffn + tl.arange(0, BLOCK_SIZE_D_FFN)
            w2_col_offs = d_ffn + tl.arange(0, BLOCK_SIZE_D_FFN)
            x_ptrs = x_ptr + (x_row_offs[:, None] * stridex_row + x_col_offs[None, :] * stridex_col)
            x_mask = (x_row_offs[:, None] < tokens) & (x_col_offs[None, :] < D)
            X = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w1_ptrs = w1_ptr + (w1_row_offs[:, None] * stridew1_row + w1_col_offs[None, :] * stridew1_col)
            w1_mask = (w1_row_offs[:, None] < D) & (w1_col_offs[None, :] < D_FFN)
            W1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
            w2_ptrs = w2_ptr + (w2_row_offs[:, None] * stridew2_row + w2_col_offs[None, :] * stridew2_col)
            w2_mask = (w2_row_offs[:, None] < D) & (w2_col_offs[None, :] < D_FFN)
            W2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)
            acc1 += tl.dot(X, W1) # x @ w1
            acc2 += tl.dot(X, W2) # x @ w2

        # Apply swish and bias to
        if b1 is not None:
            A = swish_kernel(acc1 + b1[None, :]) # swish(x @ w1 + b1)
            B = acc2 + b2[None, :] # x @ w2 + b2
        else:
            A = swish_kernel(acc1) # swish(x @ w1)
            B = acc2 # x @ w2

        y = A * B # [BLOCK_SIZE_TOKENS, BLOCK_SIZE_D_FFN]

        # Now compute y @ w3 for output columns
        w3_row_offs = d_ffn + tl.arange(0, BLOCK_SIZE_D_FFN)
        w3_col_offs = col_pid * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        w3_ptrs = w3_ptr + (w3_row_offs[:, None] * stridew3_row + w3_col_offs[None, :] * stridew3_col)
        w3_mask = (w3_row_offs[:, None] < D_FFN) & (w3_col_offs[None, :] < D)
        W3 = tl.load(w3_ptrs, mask=w3_mask, other=0.0)
        acc_out += tl.dot(y, W3)

    # Apply final bias if given
    if b3 is not None:
        result = acc_out + b3[None, :]
    else:
        result = acc_out

    # Store final output
    out_ptrs = out_ptr + (out_row_offs[:, None] * strideout_row + out_col_offs[None, :] * strideout_col)
    out_mask = (out_row_offs[:, None] < tokens) & (out_col_offs[None, :] < D)
    tl.store(out_ptrs, result, mask=out_mask)

@triton.jit
def swish_kernel(x):
    return x * tl.sigmoid(x) # x * (1 / 1 + exp(-x))

def swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    b3: Optional[torch.Tensor] = None,
    BLOCK_SIZE_TOKENS=32,
    BLOCK_SIZE_D=32,
    BLOCK_SIZE_D_FFN=32
) -> torch.Tensor:
    # weight matrix assertions
    assert w1.dim() == 2
    assert w2.dim() == 2
    assert w3.dim() == 2
    assert w1.is_contiguous()
    assert w2.is_contiguous()
    assert w3.is_contiguous()
    # bias vector assertions
    if b1 is not None and b2 is not None and b3 is not None:
        assert b1.dim() == 1
        assert b2.dim() == 1
        assert b3.dim() == 1
        assert b1.is_contiguous()
        assert b2.is_contiguous()
        assert b3.is_contiguous()
    # input tensor assertions
    assert x.dim() == 3
    assert x.is_contiguous()
    # get input tensor
    B, T, D = x.shape
    x = x.view(B*T, D) # [B*T, D]
    tokens = x.size(0)
    # matmul assertions
    # x: [BT, D], w1 and w2: [D, D_FFN], w3: [D_FFN, D]
    # y = x @ w2, y shape: [BT, D_FFN]
    # z = y @ w3, z shape: [BT, D]
    assert x.size(1) == w1.size(0)
    assert x.size(1) == w2.size(0)
    # pre-allocate empty output tensor with same shape, device, dtype, numel
    output = torch.empty_like(x)
    D_FFN = w1.size(1) # Get D_FFN for kernel
    grid_row = triton.cdiv(tokens, BLOCK_SIZE_TOKENS)
    grid_col = triton.cdiv(D, BLOCK_SIZE_D)
    grid = (grid_row, grid_col)
    swiglu_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        w1_ptr=w1,
        w2_ptr=w2,
        w3_ptr=w3,
        b1_ptr=b1,
        b2_ptr=b2,
        b3_ptr=b3,
        stridex_row=x.stride(0),
        stridex_col=x.stride(1),
        stridew1_row=w1.stride(0),
        stridew1_col=w1.stride(1),
        stridew2_row=w2.stride(0),
        stridew2_col=w2.stride(1),
        stridew3_row=w3.stride(0),
        stridew3_col=w3.stride(1),
        strideout_row=output.stride(0),
        strideout_col=output.stride(1),
        tokens=tokens,
        D=D,
        D_FFN=D_FFN,
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_D_FFN=BLOCK_SIZE_D_FFN
    )
    return output.view(B, T, D)

def init_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    b3: Optional[torch.Tensor] = None,
) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
    # initialize weights with Kaiming Uniform
    w1_init = init.kaiming_uniform_(w1, a=math.sqrt(5))
    w2_init = init.kaiming_uniform_(w2, a=math.sqrt(5))
    w3_init = init.kaiming_uniform_(w3, a=math.sqrt(5))

    # initialize biases same way as nn.Linear
    if b1 is not None and b2 is not None and b3 is not None:
        def init_bias(b, w):
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            return init.uniform_(b, -bound, bound)

        b1_init = init_bias(b1, w1)
        b2_init = init_bias(b2, w2)
        b3_init = init_bias(b3, w3)
    else:
        b1_init, b2_init, b3_init = None, None, None

    return w1_init, w2_init, w3_init, b1_init, b2_init, b3_init
      
class SwiGLU(nn.Module):
    def __init__(
        self,
        D: int,
        D_FFN: int,
    ):
        super().__init__()

        self.w1 = nn.Linear(D, D_FFN, bias=False)
        self.w2 = nn.Linear(D, D_FFN, bias=False)
        self.w3 = nn.Linear(D_FFN, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def benchmark(num_iters: int) -> None:
    x = torch.randn(B, T, D, dtype=DTYPE, device=DEVICE)

    # set up weights for Triton kernel
    w1 = torch.empty((D, D_FFN), device=DEVICE, dtype=DTYPE)
    w2 = torch.empty((D, D_FFN), device=DEVICE, dtype=DTYPE)
    w3 = torch.empty((D_FFN, D), device=DEVICE, dtype=DTYPE)
    w1, w2, w3, _, _, _ = init_weights(w1, w2, w3)

    # initialize torch swiglu (without biases)
    torch_swiglu = SwiGLU(D=D, D_FFN=D_FFN).to(DEVICE)

    # copy Triton weights into torch module
    torch_swiglu.w1.weight.data.copy_(w1.T)
    torch_swiglu.w2.weight.data.copy_(w2.T)
    torch_swiglu.w3.weight.data.copy_(w3.T)

    # warm up
    for _ in range(num_iters):
        _ = swiglu(x, w1, w2, w3, None, None, None)
        _ = torch_swiglu(x)

    # benchmark PyTorch
    torch.cuda.synchronize()
    torch_start_time = time.time()
    for _ in range(num_iters):
        torch_out = torch_swiglu(x)
    torch.cuda.synchronize()
    torch_time = (time.time() - torch_start_time) / num_iters

    # benchmark Triton kernel
    torch.cuda.synchronize()
    triton_start_time = time.time()
    for _ in range(num_iters):
        triton_out = swiglu(x, w1, w2, w3, None, None, None)
    torch.cuda.synchronize()
    triton_time = (time.time() - triton_start_time) / num_iters

    # validity check (single forward pass)
    torch_out = torch_swiglu(x)
    triton_out = swiglu(x, w1, w2, w3, None, None, None)

    # calculate average time
    print(f"averaging time over {num_iters} forward passes...\n")
    print(f"PyTorch SwiGLU Avg. latency: {(torch_time * 1000):.4f} ms")
    print(f"Triton SwiGLU Avg. latency:  {(triton_time * 1000):.4f} ms")

    # calculate speedup based on faster kernel
    if triton_time < torch_time:
        print(f"Triton kernel has a speedup of {(torch_time / triton_time):.4f}x \n")
    else:
        print(f"PyTorch kernel has a speedup of {(triton_time / torch_time):.3f}x \n")

    # print first scalar value from each
    print("Torch first value:", torch_out.flatten()[0].item())
    print("Triton first value:", triton_out.flatten()[0].item())
    print("All close?", torch.allclose(torch_out, triton_out, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    benchmark(1000)
