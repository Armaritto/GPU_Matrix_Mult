import numpy as np
import cupy as cp
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def cpu_operation(A, B, C, repeats=5):
    """Compute AxBxC + A on CPU, averaged over `repeats` runs"""
    times = []
    for _ in range(repeats):
        start = time.time()
        temp = np.dot(A, B)
        result = np.dot(temp, C) + A
        times.append(time.time() - start)
    return sum(times) / repeats

def gpu_operation(A, B, C, repeats=5):
    """Compute AxBxC + A on GPU, averaged over `repeats` runs"""
    try:
        times = []
        for _ in range(repeats):
            start = time.time()
            temp = cp.dot(A, B)
            result = cp.dot(temp, C) + A
            cp.cuda.Stream.null.synchronize()  # Ensure completion
            times.append(time.time() - start)
        return sum(times) / repeats
    except cp.cuda.memory.OutOfMemoryError:
        return None

def benchmark_operations(sizes, repeats=5):
    """Benchmark CPU and GPU performance for different matrix sizes"""
    results = []

    for N in sizes:
        log(f"\n===== Testing matrix size: {N}x{N} =====")

        # Generate random matrices
        log("Generating matrices")
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.random.rand(N, N).astype(np.float32)

        # CPU benchmark
        log("CPU: Starting operation")
        cpu_time = cpu_operation(A, B, C, repeats)
        log(f"CPU avg time: {cpu_time:.2f} s")

        # GPU benchmark
        log("GPU: Starting operation")
        try:
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)
            C_gpu = cp.asarray(C)
            gpu_time = gpu_operation(A_gpu, B_gpu, C_gpu, repeats)
            
            if gpu_time is None:
                log("GPU: Out of memory, skipping...")
            else:
                log(f"GPU avg time: {gpu_time:.2f} s")
        except cp.cuda.memory.OutOfMemoryError:
            gpu_time = None
            log("GPU: Out of memory during matrix allocation")

        # Calculate speedup if GPU succeeded
        speedup = cpu_time / gpu_time if gpu_time else None
        results.append((N, cpu_time, gpu_time, speedup))

    return results

def plot_results(results):
    """Create interactive performance plots (no speedup shown)"""
    sizes = [r[0] for r in results]
    cpu_times = [r[1] for r in results]
    gpu_times = [r[2] if r[2] is not None else None for r in results]

    fig = go.Figure()

    # Add execution time traces
    fig.add_trace(
        go.Scatter(
            x=sizes, y=cpu_times, 
            name="CPU Time", 
            line=dict(color='royalblue', width=3),
            mode='lines+markers'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=sizes, y=gpu_times, 
            name="GPU Time", 
            line=dict(color='limegreen', width=3),
            mode='lines+markers'
        )
    )

    # Update layout
    fig.update_layout(
        title="Matrix Operation Benchmark: AxBxC + A",
        xaxis_title="Matrix Size (NxN)",
        yaxis_title="Execution Time (s)",
        template="plotly_dark",
        hovermode="x unified",
        height=600
    )
    
    fig.write_html("benchmark_results.html")
    fig.show()

def print_summary(results):
    """Print formatted benchmark results with speedup"""
    print("\n" + "="*60)
    print(f"{'MATRIX SIZE':>12} | {'CPU TIME (s)':>12} | {'GPU TIME (s)':>12} | {'SPEEDUP':>12}")
    print("="*60)
    
    for N, cpu, gpu, spd in results:
        gpu_str = f"{gpu:.4f}" if gpu else "OOM"
        spd_str = f"{spd:.2f}x" if spd else "---"
        print(f"{N:12} | {cpu:12.4f} | {gpu_str:>12} | {spd_str:>12}")

if __name__ == "__main__":
    test_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 10000]
    
    log("Starting benchmark...")
    results = benchmark_operations(test_sizes, repeats=5)
    
    plot_results(results)
    print_summary(results)
    
    log("Benchmark completed. Results saved to 'benchmark_results.html'")
