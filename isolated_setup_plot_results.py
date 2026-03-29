import torch
import time 
import matplotlib.pyplot as plt
from triton_fused_kernel import fused_gemm_relu
from trition_unfused_kernel import unfused_gemm_relu

def baseline(A, B, bias):
    return torch.relu(A @ B + bias)


def benchmark(func, A, B, bias, iters=50):
    # warmup
    for _ in range(20):
        func(A, B, bias)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        func(A, B, bias)
    torch.cuda.synchronize()

    return (time.time() - start) / iters

def main():
    sizes = [128, 256, 512, 1024, 2048]
    baseline_times = []
    unfused_times = []
    fused_times = []

    for size in sizes:
        M = K = N = size

        A = torch.randn((M, K), device='cuda')
        B = torch.randn((K, N), device='cuda')
        bias = torch.randn((N,), device='cuda')
        baseline_times.append(benchmark(baseline, A, B, bias))
        unfused_times.append(benchmark(unfused_gemm_relu, A, B, bias))
        fused_times.append(benchmark(fused_gemm_relu, A, B, bias))



    plt.plot(sizes, baseline_times, label="PyTorch (cuBLAS)")
    plt.plot(sizes, unfused_times, label="Triton Unfused")
    plt.plot(sizes, fused_times, label="Triton Fused")

    plt.xlabel("Matrix Size (M = N = K)")
    plt.ylabel("Time (seconds)")
    plt.title("GEMM + ReLU Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig('runtime_comparisons_1.png')
    plt.show()


if __name__ == "__main__":
    main()