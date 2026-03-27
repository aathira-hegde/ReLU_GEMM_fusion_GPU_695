import triton
import torch
import triton.language as tl

@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A = tl.load(A_ptrs, mask=offs_m[:, None] < M, other=0.0)
        B = tl.load(B_ptrs, mask=offs_n[None, :] < N, other=0.0)

        acc += tl.dot(A, B)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def bias_add_kernel(
    C_ptr, bias_ptr,
    M, N,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row = offs // N
    col = offs % N

    mask = (row < M) & (col < N)

    C_ptrs = C_ptr + row * stride_cm + col * stride_cn
    bias = tl.load(bias_ptr + col, mask=col < N, other=0.0)

    val = tl.load(C_ptrs, mask=mask, other=0.0)
    val += bias

    tl.store(C_ptrs, val, mask=mask)

@triton.jit
def relu_kernel(
    C_ptr,
    M, N,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row = offs // N
    col = offs % N

    mask = (row < M) & (col < N)

    C_ptrs = C_ptr + row * stride_cm + col * stride_cn

    val = tl.load(C_ptrs, mask=mask, other=0.0)
    val = tl.maximum(val, 0)

    tl.store(C_ptrs, val, mask=mask)

    

def unfused_gemm_relu(A, B, bias):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # GEMM
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Bias add
    BLOCK_SIZE = 1024
    grid = ((M * N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    bias_add_kernel[grid](
        C, bias,
        M, N,
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # ReLU
    relu_kernel[grid](
        C,
        M, N,
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return C