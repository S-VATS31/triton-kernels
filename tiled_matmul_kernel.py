"""
Matrix multiplication kernel for 2 dimensional matrices.
"""
import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda")

@triton.jit
def matmul_kernel(
    A, # ptr to input tensor -> [M, K]
    B, # ptr to input tensor -> [K, N]
    out_ptr, # ptr to out tensor -> [M, N]
    # M, N, K: matrix shapes - we compute: C = A[M, K] @ B[K, N], where C has shape [M, N]
    M: tl.constexpr, # rows in input matrix
    N: tl.constexpr, # cols in input matrix
    K: tl.constexpr, # reduction dim: C: [M, N] = summation(A[M, K] * B[K, N])
    stride_am: tl.constexpr, stride_ak: tl.constexpr, # strides for A
    stride_bk: tl.constexpr, stride_bn: tl.constexpr, # strides for B
    stride_cm: tl.constexpr, stride_cn: tl.constexpr, # strides for C
    alpha: tl.constexpr, # scaling
    beta: tl.constexpr,  # shifting
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2 program ids for row, col
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    block_start_m = pid_m * BLOCK_M # row start
    block_start_n = pid_n * BLOCK_N # col start
    # get offsets
    m_offs = block_start_m + tl.arange(0, BLOCK_M)
    n_offs = block_start_n + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K) # reduction dim, we will create block_start_k in loop
    # initialize accumulation
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # tiled matmul
    for k in range(0, K, BLOCK_K):
        k_range = k + k_offs
        # Fixed pointer calculations using proper strides
        A_ptrs = A + (m_offs[:, None] * stride_am + k_range[None, :] * stride_ak)
        B_ptrs = B + (k_range[:, None] * stride_bk + n_offs[None, :] * stride_bn)
        # create masks to guard against out of bounds positions
        a_mask = (m_offs[:, None] < M) & (k_range[None, :] < K)
        b_mask = (k_range[:, None] < K) & (n_offs[None, :] < N)
        # load pointers before computation
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        # accumulate tiles
        acc += tl.dot(a, b)

    # no scaling or shifting (alpha=1.0, beta=0.0)
    # C = A @ B
    # A shape: [M, K], B shape: [K, N], C shape: [M, N]
    #
    # scaling + shifting (alpha!=1.0, beta!=0.0)
    # C = alpha * A @ B + beta * C_init
    # C_init: pre-allocated output tensor, not the actual output
    # A shape: [M, K], B shape: [K, N], C shape: [M, N]

    # apply scaling if given
    if alpha != 1.0:
        acc *= alpha # alpha * AB ...

    out_ptrs = out_ptr + (m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn) # output ptr
    # construct full mask for rows/cols
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)

    # apply shifting if given
    if beta != 0.0:
        out_init = tl.load(out_ptrs, mask=out_mask, other=0.0)
        acc += beta * out_init # ... + beta * C_init

    tl.store(out_ptrs, acc, mask=out_mask)

def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    BLOCK_M=32,
    BLOCK_N=32,
    BLOCK_K=32
) -> torch.Tensor:
    # Assertions for input validation
    assert A.size(1) == B.size(0)
    assert A.device == B.device
    assert A.dtype == B.dtype
    assert A.is_contiguous()
    assert B.is_contiguous()

    # Get matrix dimensions
    M, K = A.shape
    K_b, N = B.shape

    # Pre-allocate output tensor
    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Calculate grid dimensions
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)

    # Launch kernel
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        alpha, beta,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return out

def main():
    # Create test matrices
    A = torch.randn(64, 128, dtype=torch.float32, device=DEVICE)
    B = torch.randn(128, 64, dtype=torch.float32, device=DEVICE)

    # Test the kernel
    C_triton = matmul(A, B, alpha=1.0, beta=0.0)

    # Compare with PyTorch
    C_torch = torch.matmul(A, B)

    return C_triton, C_torch

if __name__ == "__main__":
    C_triton, C_torch = main()
    print(C_torch[:1, :1])
    print(C_triton[:1, :1])
