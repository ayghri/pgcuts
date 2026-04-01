import time

import numpy as np
from reference.hypervect_py import hypergeometric_2F1
from scipy.special import hyp2f1


def main() -> None:
    m = 512
    b = 1.0
    c = 2.0
    n_points = 1_000_000

    z = np.linspace(0.0, 1.0, n_points)
    print(z.min(), z.max())

    # Correctness check
    vals = hypergeometric_2F1(-m, b, c, z)
    sc = hyp2f1(-m, b, c, z)

    print("First 5 values (our C++ wrapper):", vals[:5])
    print("First 5 values (SciPy hyp2f1):  ", sc[:5])
    print("L2 difference:", np.linalg.norm(vals - sc))
    print("Max difference:", np.abs(vals - sc).max())
    print("Min difference:", np.abs(vals - sc).min())
    print("Avg difference:", np.abs(vals - sc).mean())

    # Timing benchmark
    n_repeats = 10

    # Warm-up
    _ = hypergeometric_2F1(-m, b, c, z)
    _ = hyp2f1(-m, b, c, z)

    start = time.perf_counter()
    for _ in range(n_repeats):
        vals = hypergeometric_2F1(-m, b, c, z)
    elapsed_ours = (time.perf_counter() - start) / n_repeats

    start = time.perf_counter()
    for _ in range(n_repeats):
        sc = hyp2f1(-m, b, c, z)
    elapsed_scipy = (time.perf_counter() - start) / n_repeats

    print(
        "\nBenchmark over",
        n_points,
        "points (averaged over",
        n_repeats,
        "runs):",
    )
    print(f"  Ours (C++ via ctypes): {elapsed_ours * 1e3:.3f} ms per call")
    print(f"  SciPy hyp2f1:          {elapsed_scipy * 1e3:.3f} ms per call")
    print(f"  Speedup (SciPy / ours): {elapsed_scipy / elapsed_ours:.2f}x")
    print(f"  Time per element (ours): {elapsed_ours / n_points * 1e9:.2f} ns")
    print(
        f"  Time per element (SciPy): {elapsed_scipy / n_points * 1e9:.2f} ns"
    )


if __name__ == "__main__":
    main()
