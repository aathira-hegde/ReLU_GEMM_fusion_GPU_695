import torch
import time 
import torch.cuda.profiler as profiler
from triton_fused_kernel import fused_gemm_relu
from trition_unfused_kernel import unfused_gemm_relu


def main():
    # Dimensions
    M, K, N = 256, 256, 256

    A = torch.randn((M, K), device='cuda')
    B = torch.randn((K, N), device='cuda')
    bias = torch.randn((N,), device='cuda')

    # Warmup
    for _ in range(10):
        baseline_out = torch.relu(A @ B + bias)

    for _ in range(10):
        baseline_out = unfused_gemm_relu(A, B ,bias)

    for _ in range(10):
        baseline_out = fused_gemm_relu(A, B, bias)


    # Baseline timing
    torch.cuda.synchronize()
    start = time.time()
    
    torch.cuda.nvtx.range_push("BASELINE")
    for _ in range(50):
        baseline_out = torch.relu(A @ B + bias)
    
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    baseline_time = time.time() - start


    # Baseline timing
    torch.cuda.synchronize()
    start = time.time()
    
    torch.cuda.nvtx.range_push("UNFUSED")
    for _ in range(50):
        baseline_out = unfused_gemm_relu(A, B ,bias)
    
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    unfused_time = time.time() - start


    # Fused timing
    torch.cuda.synchronize()
    start = time.time()

    torch.cuda.nvtx.range_push("FUSED")

    for _ in range(50):
        fused_out = fused_gemm_relu(A, B, bias)

    torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    fused_time = time.time() - start

    print("Baseline:", baseline_time)
    print("Unfused:", unfused_time)
    print("Fused:", fused_time)


if __name__ == "__main__":
    profiler.start()
    main()
    profiler.stop()