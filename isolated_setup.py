import torch
import time 
from triton_fused_kernel import fused_gemm_relu
from trition_unfused_kernel import unfused_gemm_relu

def baseline(A, B, bias):
    return unfused_gemm_relu(A, B, bias)

def fused(A, B, bias):
    return fused_gemm_relu(A, B, bias)

def main():
    # Dimensions
    M, K, N = 256, 256, 256

    A = torch.randn((M, K), device='cuda')
    B = torch.randn((K, N), device='cuda')
    bias = torch.randn((N,), device='cuda')

    # Warmup
    for _ in range(10):
        baseline_out = torch.relu(A @ B + bias)

    # Baseline timing
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        baseline_out = unfused_gemm_relu(A, B ,bias)

    torch.cuda.synchronize()
    baseline_time = time.time() - start


    # Fused timing
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        fused_out = fused_gemm_relu(A, B, bias)

    torch.cuda.synchronize()
    fused_time = time.time() - start

    print("Baseline:", baseline_time)
    print("Fused:", fused_time)


if __name__ == "__main__":
    main()