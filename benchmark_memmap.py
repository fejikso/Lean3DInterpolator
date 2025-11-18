#!/usr/bin/env python3
"""
Benchmark script comparing LeanVolumeInterpolator with
scipy.interpolate.RegularGridInterpolator on a large NumPy memmap volume.

This is intended to mirror the experiment described in the README, where
an oblique slice (here approximated by a large set of random coordinates)
is extracted from a large on-disk volume.
"""

import os
import time
import argparse

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from LeanVolumeInterpolator import LeanVolumeInterpolator


TMP_DIR = "/mnt/vol1/tmp"


def create_memmap_volume(path, shape, dtype=np.float32, reuse_existing=False):
    """
    Create a large 3D memmap volume on disk with the given shape.

    The volume is initialized to zeros to avoid the overhead of writing random
    data across the entire file. For interpolation benchmarks, the actual
    values are less important than access patterns.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    filename = path

    if os.path.exists(filename) and reuse_existing:
        print(f"Reusing existing memmap file: {filename}")
        print(f"  Requested shape: {shape}")
        print(f"  Requested dtype: {dtype}")
        vol = np.memmap(filename, mode="r+", dtype=dtype, shape=shape)
        created_new = False
    else:
        # Remove any previous file
        if os.path.exists(filename):
            os.remove(filename)

        print(f"Creating memmap file: {filename}")
        print(f"  Shape: {shape}")
        print(f"  Dtype: {dtype}")

        vol = np.memmap(filename, mode="w+", dtype=dtype, shape=shape)
        vol[:] = 0  # ensure file is allocated and initialized
        vol.flush()

        file_size = os.path.getsize(filename) / (1024 ** 3)
        print(f"  File size on disk: {file_size:.2f} GB")
        created_new = True

    return vol, filename, created_new


def benchmark_memmap(
    shape=(1300, 900, 2300),
    n_coords=1_000_000,
    n_runs=3,
    dtype=np.float32,
    reuse_existing=False,
    keep_file=False,
    path=None,
):
    """
    Benchmark LeanVolumeInterpolator vs RegularGridInterpolator on a memmap.

    Parameters
    ----------
    shape : tuple
        Shape of the 3D volume (nx, ny, nz).
    n_coords : int
        Number of coordinates to interpolate (approximately a (1000x1000) grid).
    n_runs : int
        Number of benchmark runs to average.
    dtype : numpy.dtype
        Data type for the volume.
    """
    nx, ny, nz = shape

    print("=" * 70)
    print("Memmap Benchmark: LeanVolumeInterpolator vs RegularGridInterpolator")
    print("=" * 70)
    print(f"Volume shape: {shape}")
    print(f"Number of coordinates: {n_coords}")
    if path is None:
        path = os.path.join(TMP_DIR, "lean_memmap_benchmark.dat")
    print(f"Memmap path: {path}")
    print("=" * 70)

    # Create memmap volume
    vol, filename, created_new = create_memmap_volume(
        path, shape=shape, dtype=dtype, reuse_existing=reuse_existing
    )

    try:
        # Generate coordinates uniformly within the volume bounds.
        # This approximates an "oblique slice" workload by using a dense set
        # of arbitrary 3D coordinates.
        rng = np.random.default_rng()
        print("\nGenerating random coordinates...")
        x_coords = rng.uniform(0, nx - 1, n_coords).astype(np.float64)
        y_coords = rng.uniform(0, ny - 1, n_coords).astype(np.float64)
        z_coords = rng.uniform(0, nz - 1, n_coords).astype(np.float64)

        # Grids for SciPy
        x_grid = np.arange(nx)
        y_grid = np.arange(ny)
        z_grid = np.arange(nz)

        runs = n_runs

        print("\nInitializing interpolators...")
        t0 = time.perf_counter()
        lean_interp = LeanVolumeInterpolator(vol, extrap_val=np.nan, dtype=np.float32)
        lean_init_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        scipy_interp = RegularGridInterpolator(
            (x_grid, y_grid, z_grid),
            vol,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        scipy_init_time = time.perf_counter() - t0

        print(f"  ✓ LeanVolumeInterpolator initialized in {lean_init_time*1000:.3f} ms")
        print(f"  ✓ scipy RegularGridInterpolator initialized in {scipy_init_time*1000:.3f} ms")

        coords_scipy = np.column_stack([x_coords, y_coords, z_coords])

        # Warm-up runs (small subset)
        print("\nWarming up (small subset, cached)...")
        _ = lean_interp((x_coords[:10], y_coords[:10], z_coords[:10]))
        _ = scipy_interp(coords_scipy[:10])

        # Optional pause to allow manual cache clearing before SciPy benchmark
        print(
            "\nIf you want cold-cache timings, clear the OS page cache now in another terminal,\n"
            "for example:\n"
            "  echo 3 | sudo tee /proc/sys/vm/drop_caches\n"
        )
        input("Press Enter to start scipy RegularGridInterpolator benchmark...")

        # Benchmark scipy RegularGridInterpolator
        print(f"\nBenchmarking scipy RegularGridInterpolator ({runs} run{'s' if runs != 1 else ''})...")
        scipy_times = []
        for i in range(runs):
            start = time.perf_counter()
            scipy_result = scipy_interp(coords_scipy)
            scipy_times.append(time.perf_counter() - start)
            print(f"  Run {i+1}/{n_runs}: {scipy_times[-1]*1000:.3f} ms")

        scipy_avg_time = np.mean(scipy_times)
        scipy_std_time = np.std(scipy_times)
        scipy_min_time = np.min(scipy_times)

        # Optional pause to allow manual cache clearing before Lean benchmark
        print(
            "\nIf you want cold-cache timings for LeanVolumeInterpolator, clear the OS page cache now in another terminal,\n"
            "for example:\n"
            "  echo 3 | sudo tee /proc/sys/vm/drop_caches\n"
        )
        input("Press Enter to start LeanVolumeInterpolator benchmark...")

        # Benchmark LeanVolumeInterpolator
        print(f"\nBenchmarking LeanVolumeInterpolator ({runs} run{'s' if runs != 1 else ''})...")
        lean_times = []
        for i in range(runs):
            start = time.perf_counter()
            lean_result = lean_interp((x_coords, y_coords, z_coords))
            lean_times.append(time.perf_counter() - start)
            print(f"  Run {i+1}/{n_runs}: {lean_times[-1]*1000:.3f} ms")

        lean_avg_time = np.mean(lean_times)
        lean_std_time = np.std(lean_times)
        lean_min_time = np.min(lean_times)



        # Verify results
        print("\nVerifying results...")
        all_nan = np.isnan(lean_result) & np.isnan(scipy_result)
        both_valid = ~(np.isnan(lean_result) | np.isnan(scipy_result))
        if np.any(both_valid):
            max_diff = np.max(np.abs(lean_result[both_valid] - scipy_result[both_valid]))
            mean_diff = np.mean(np.abs(lean_result[both_valid] - scipy_result[both_valid]))
        else:
            max_diff = 0.0
            mean_diff = 0.0

        n_nan = np.sum(all_nan)

        speedup = scipy_avg_time / lean_avg_time if lean_avg_time > 0 else float("inf")

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print("LeanVolumeInterpolator:")
        print(f"  Average time: {lean_avg_time*1000:.3f} ms ± {lean_std_time*1000:.3f} ms")
        print(f"  Min time:     {lean_min_time*1000:.3f} ms")
        print(f"  Init time:    {lean_init_time*1000:.3f} ms")
        print(f"  Throughput:   {n_coords/lean_avg_time:,.0f} points/sec")
        print()
        print("scipy RegularGridInterpolator:")
        print(f"  Average time: {scipy_avg_time*1000:.3f} ms ± {scipy_std_time*1000:.3f} ms")
        print(f"  Min time:     {scipy_min_time*1000:.3f} ms")
        print(f"  Init time:    {scipy_init_time*1000:.3f} ms")
        print(f"  Throughput:   {n_coords/scipy_avg_time:,.0f} points/sec")
        print()
        print(f"Speedup (Lean / SciPy): {speedup:.2f}x")
        print(f"Results match:          {bool(np.allclose(lean_result[both_valid], scipy_result[both_valid], equal_nan=True))}")
        print(f"Max difference:         {max_diff:.2e}")
        print(f"Mean difference:        {mean_diff:.2e}")
        print(f"NaN count:              {n_nan} (out of {n_coords})")

        print("\nMemmap volume information:")
        print(f"  File path: {filename}")
        print(f"  File size: {os.path.getsize(filename)/(1024**3):.2f} GB")
        print(f"  Logical size in memory: {vol.size * vol.dtype.itemsize/(1024**3):.2f} GB")

    finally:
        # Ensure memmap is closed and file removed
        try:
            del vol
        except Exception:
            pass
        if os.path.exists(filename) and not keep_file:
            os.remove(filename)
            print(f"\nCleaned up memmap file: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LeanVolumeInterpolator vs scipy RegularGridInterpolator "
            "on a large 3D NumPy memmap volume."
        )
    )
    parser.add_argument(
        "path",
        help=(
            "Path to the memmap file to create or reuse. "
            "If the file exists, it can be reused with --reuse."
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the memmap file after the benchmark finishes.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse the existing memmap file at the given path if it exists.",
    )
    parser.add_argument(
        "--coords",
        type=int,
        default=1_000,
        help="Number of coordinates to interpolate (default: 1000).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=16,
        help="Maximum number of benchmark runs (default: 16). "
             "If OS cache cannot be dropped, this is reduced to 1.",
    )

    args = parser.parse_args()

    # Approximate the original README experiment by default:
    # large volume and ~1e6 coordinates (1000x1000 grid).
    benchmark_memmap(
        shape=(1300, 900, 2300),
        n_coords=args.coords,
        n_runs=args.runs,
        dtype=np.uint8,
        reuse_existing=args.reuse,
        keep_file=args.keep,
        path=args.path,
    )


if __name__ == "__main__":
    main()


