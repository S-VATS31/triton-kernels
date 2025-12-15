"""
Element wise multiplication between 2 [i, j] matrices and benchmark against PyTorch element wise matmul
"""
import time

import torch
import triton
import triton.language as tl

@triton.jit
def element_wise_mul_kernel(
    x_ptr,      # pointer to input matrix x
    y_ptr,      # pointer to input matrix y
    output_ptr, # pointer to output matrix
    N_rows: int,
    N_cols: int,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    # Use 2 program ids for rows/cols
    row_pid = tl.program_id(axis=0)
    col_pid = tl.program_id(axis=1)
    # Get starting position to use for offsets
    block_start_row = row_pid * BLOCK_SIZE_ROW
    block_start_col = col_pid * BLOCK_SIZE_COL
    row_offset = block_start_row + tl.arange(0, BLOCK_SIZE_ROW) # compute row offsets
    col_offset = block_start_col + tl.arange(0, BLOCK_SIZE_COL) # compute col offsets
    # create boolean masks where True=in bounds, False=out of bounds
    row_mask = row_offset < N_rows # rows -> i
    col_mask = col_offset < N_cols # cols -> j
    # mat shape: [i, j], row mask shape: [i], col mask shape: [j]
    # create new mask which takes into account rows/cols -> shape [i, j]
    mask = row_mask[:, None] & col_mask[None, :] # bitwise and (same as torch.bitwise_and())
    # create pointers
    x_ptrs = x_ptr + row_offset[:, None] * N_cols + col_offset[None, :]
    y_ptrs = y_ptr + row_offset[:, None] * N_cols + col_offset[None, :]
    out_ptrs = output_ptr + row_offset[:, None] * N_cols + col_offset[None, :]
    # load pointers onto GPU before computing element-wise mul
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    output = x * y # apply element wise mul
    tl.store(out_ptrs, output, mask=mask)

def multiply(
    x: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE_ROW=128,
    BLOCK_SIZE_COL=128,
) -> torch.Tensor:
    output = torch.empty_like(x) # initialize with same shape, dtype, device, elements
    assert x.shape == y.shape == output.shape
    assert x.dtype == y.dtype == output.dtype
    assert x.device == y.device == output.device
    assert x.numel() == y.numel() == output.numel()
    assert x.dim() == 2, "must be a 2 dimensional matrix for element wise mul"
    N_rows, N_cols = x.shape
    # compute grid rows/cols using ceiling division
    grid_rows = (N_rows + BLOCK_SIZE_ROW - 1) // BLOCK_SIZE_ROW
    grid_cols = (N_cols + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    # launch kernel with 2D grid
    element_wise_mul_kernel[grid_rows, grid_cols](
        x, y, output, N_rows, N_cols, BLOCK_SIZE_ROW, BLOCK_SIZE_COL
    )
    return output

def main():
    x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    y = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    # warmup
    _ = multiply(x, y)
    # triton
    torch.cuda.synchronize()
    k_start = time.time()
    kernel_out = multiply(x, y)
    torch.cuda.synchronize()
    k_end = time.time()
    # torch
    torch.cuda.synchronize()
    t_start = time.time()
    torch_out = x * y
    torch.cuda.synchronize()
    t_end = time.time()

    return kernel_out, torch_out, k_end - k_start, t_end - t_start

kernel_out, torch_out, kernel_time, torch_time  = main()
assert torch.allclose(kernel_out, torch_out)

print(f"kernel time: {kernel_time}")
print(f"torch time: {torch_time}")

if torch_time < kernel_time:
    print(f"torch is {kernel_time / torch_time}x faster")
else:
    print(f"custom kernel is {torch_time / kernel_time}x faster")
    
