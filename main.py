import numpy as np
import cupy as cp
import time
import plotly.graph_objects as go

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def gpu_dot(A, B):
    try:
        start = time.time()
        C = cp.dot(A, B)
        cp.cuda.Stream.null.synchronize()
        return time.time() - start
    except cp.cuda.memory.OutOfMemoryError:
        return None

def benchmark_matrix_multiplication(sizes):
    results = []

    for N in sizes:
        log(f"\n===== Testing matrix size: {N}x{N} =====")

        log("CPU: Generating matrices")
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        log("CPU: Starting multiplication")
        start_cpu = time.time()
        C = np.dot(A, B)
        cpu_time = time.time() - start_cpu
        log(f"CPU time: {cpu_time:.2f} s")

        log("GPU: Generating matrices")
        try:
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)
            log("GPU: Starting multiplication")
            gpu_time = gpu_dot(A_gpu, B_gpu)
            if gpu_time is None:
                log("GPU: Out of memory, skipping...")
            else:
                log(f"GPU time: {gpu_time:.2f} s")
        except cp.cuda.memory.OutOfMemoryError:
            gpu_time = None
            log("GPU: Out of memory during matrix allocation")

        speedup = cpu_time / gpu_time if gpu_time else None
        results.append((N, cpu_time, gpu_time, speedup))

    return results

def plot_results(results):
    # Prepare data for plotting
    sizes = [r[0] for r in results]
    cpu_times = [r[1] for r in results]
    gpu_times = [r[2] if r[2] is not None else float('nan') for r in results]

    # Create a Plotly figure
    fig = go.Figure()

    # Plot CPU times
    fig.add_trace(go.Scatter(x=sizes, y=cpu_times, mode='lines+markers', name='CPU Time (s)', line=dict(color='blue')))

    # Plot GPU times
    fig.add_trace(go.Scatter(x=sizes, y=gpu_times, mode='lines+markers', name='GPU Time (s)', line=dict(color='green')))

    # Update layout
    fig.update_layout(
        title='Matrix Multiplication Benchmark',
        xaxis_title='Matrix Size (NxN)',
        yaxis_title='Time (s)',
        legend_title='Legend',
        template='plotly_dark'
    )

    fig.show()

def print_speedup(results):
    print("\n==== SUMMARY ====")
    print(f"{'Size':>10} | {'CPU Time':>10} | {'GPU Time':>10} | {'Speedup':>8}")
    print("-" * 46)
    for N, c_time, g_time, spd in results:
        if spd:
            print(f"{N:10d} | {c_time:10.2f} | {g_time if g_time else 'OOM':>10} | {spd:.2f}")
        else:
            print(f"{N:10d} | {c_time:10.2f} | {'OOM':>10} | {'--':>8}")

if __name__ == "__main__":
    # Test sizes: small to large
    test_sizes = [128, 512, 1024, 2048, 4096, 8192, 10000, 12000, 16000]
    results = benchmark_matrix_multiplication(test_sizes)
    plot_results(results)
    print_speedup(results)
