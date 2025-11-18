#!/usr/bin/env python3
"""
Benchmark script comparing LeanVolumeInterpolator with scipy.interpolate.RegularGridInterpolator
on large **sparse** 3D volumes.

Sparse volumes are created using the `sparse` library
(`https://pypi.org/project/sparse/`), which supports n-dimensional sparse arrays.

LeanVolumeInterpolator operates directly on the sparse volume, while
RegularGridInterpolator uses a dense version of the same data as a baseline.
This isolates interpolation performance (not conversion cost).
"""

import numpy as np
import time
import sparse
from scipy.interpolate import RegularGridInterpolator
from LeanVolumeInterpolator import SparseLeanVolumeInterpolator


def create_sparse_volume(shape, density=1e-4, dtype=np.float32):
    """
    Create a sparse 3D volume with random non-zero entries using `sparse.COO`.

    Parameters
    ----------
    shape : tuple
        Shape of the 3D volume (nx, ny, nz).
    density : float
        Fraction of non-zero entries in the volume (between 0 and 1).
    dtype : numpy.dtype
        Data type of the non-zero values.

    Returns
    -------
    sparse.COO
        A 3D sparse volume with the given shape and density.
    """
    nx, ny, nz = shape
    total = nx * ny * nz
    nnz = max(1, int(total * density))

    # Use a generator for reproducibility and better randomness handling
    rng = np.random.default_rng()

    # Sample linear indices without replacement, then convert to 3D coordinates
    lin_idx = rng.choice(total, size=nnz, replace=False)
    coords = np.vstack(np.unravel_index(lin_idx, shape))  # shape: (3, nnz)

    data = rng.random(nnz, dtype=dtype)

    vol_sparse = sparse.COO(coords, data, shape=shape)
    return vol_sparse.astype(dtype)


def coo_to_dok(vol_coo):
    """
    Convert a sparse.COO volume to sparse.DOK with the same non-zero pattern.
    """
    return sparse.DOK.from_coo(vol_coo)


def verify_results(lean_result, scipy_result, rtol=1e-5, atol=1e-6):
    """Verify that both interpolators produce similar results."""
    # Handle NaN values
    both_nan = np.isnan(lean_result) & np.isnan(scipy_result)
    both_valid = ~(np.isnan(lean_result) | np.isnan(scipy_result))
    
    if np.any(both_valid):
        max_diff = np.max(np.abs(lean_result[both_valid] - scipy_result[both_valid]))
        mean_diff = np.mean(np.abs(lean_result[both_valid] - scipy_result[both_valid]))
        all_close = np.allclose(lean_result[both_valid], scipy_result[both_valid], rtol=rtol, atol=atol)
        return all_close, max_diff, mean_diff, np.sum(both_nan)
    else:
        return True, 0.0, 0.0, np.sum(both_nan)


def benchmark_interpolation(vol_shape, n_coords, n_runs=5, dtype=np.float32, density=1e-4):
    """
    Benchmark both interpolators on the same task.
    
    Parameters:
        vol_shape: Shape of the 3D volume (nx, ny, nz)
        n_coords: Number of coordinates to interpolate
        n_runs: Number of runs for averaging
        dtype: Data type for the volume
    
    Returns:
        Dictionary with timing results and verification info
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: Sparse volume shape {vol_shape}, {n_coords} coordinates, density={density}")
    print(f"{'='*70}")
    
    # Create sparse test volume in COO format
    vol_sparse = create_sparse_volume(vol_shape, density=density, dtype=dtype)
    # Also create a DOK version with identical non-zero pattern
    vol_dok = coo_to_dok(vol_sparse)
    # Dense version for scipy baseline (include conversion cost in SciPy timing)
    start = time.perf_counter()
    vol_dense = vol_sparse.todense()
    densify_time = time.perf_counter() - start
    dense_bytes = vol_dense.nbytes
    
    # Create random coordinates within bounds
    x_coords = np.random.uniform(0, vol_shape[0]-1, n_coords)
    y_coords = np.random.uniform(0, vol_shape[1]-1, n_coords)
    z_coords = np.random.uniform(0, vol_shape[2]-1, n_coords)
    
    # Prepare coordinates for scipy (needs shape (n_points, 3))
    coords_scipy = np.column_stack([x_coords, y_coords, z_coords])
    
    # Initialize interpolators
    print("Initializing interpolators...")
    # Sparse-optimized interpolators operate directly on sparse volumes
    lean_interp_coo = SparseLeanVolumeInterpolator(vol_sparse, extrap_val=np.nan, dtype=dtype)
    lean_interp_dok = SparseLeanVolumeInterpolator(vol_dok, extrap_val=np.nan, dtype=dtype)
    
    # Create grid points for scipy
    x_grid = np.arange(vol_shape[0])
    y_grid = np.arange(vol_shape[1])
    z_grid = np.arange(vol_shape[2])
    scipy_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), vol_dense,
                                          method='linear', bounds_error=False, 
                                          fill_value=np.nan)
    
    # Warm-up runs
    _ = lean_interp_coo((x_coords[:10], y_coords[:10], z_coords[:10]))
    _ = lean_interp_dok((x_coords[:10], y_coords[:10], z_coords[:10]))
    _ = scipy_interp(coords_scipy[:10])
    
    # Benchmark SparseLeanVolumeInterpolator (COO)
    print("Benchmarking SparseLeanVolumeInterpolator (COO)...")
    lean_coo_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        lean_result_coo = lean_interp_coo((x_coords, y_coords, z_coords))
        lean_coo_times.append(time.perf_counter() - start)
    lean_coo_avg_time = np.mean(lean_coo_times)
    lean_coo_std_time = np.std(lean_coo_times)
    lean_coo_min_time = np.min(lean_coo_times)

    # Benchmark SparseDokVolumeInterpolator (DOK)
    print("Benchmarking SparseDokVolumeInterpolator (DOK)...")
    lean_dok_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        lean_result_dok = lean_interp_dok((x_coords, y_coords, z_coords))
        lean_dok_times.append(time.perf_counter() - start)
    lean_dok_avg_time = np.mean(lean_dok_times)
    lean_dok_std_time = np.std(lean_dok_times)
    lean_dok_min_time = np.min(lean_dok_times)
    
    # Benchmark scipy RegularGridInterpolator
    print("Benchmarking scipy RegularGridInterpolator...")
    scipy_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        scipy_result = scipy_interp(coords_scipy)
        scipy_times.append(time.perf_counter() - start)
    scipy_avg_time = np.mean(scipy_times)
    scipy_std_time = np.std(scipy_times)
    scipy_min_time = np.min(scipy_times)
    
    # Verify results match
    print("Verifying results...")
    all_close_coo, max_diff_coo, mean_diff_coo, n_nan_coo = verify_results(lean_result_coo, scipy_result)
    all_close_dok, max_diff_dok, mean_diff_dok, n_nan_dok = verify_results(lean_result_dok, scipy_result)
    
    # Calculate speedup (SciPy densify + interp time) / Lean time
    effective_scipy_avg_time = densify_time + scipy_avg_time
    speedup_coo = effective_scipy_avg_time / lean_coo_avg_time
    speedup_dok = effective_scipy_avg_time / lean_dok_avg_time
    
    # Print results
    print(f"\nResults:")
    print(f"  SparseLeanVolumeInterpolator (COO):")
    print(f"    Average time: {lean_coo_avg_time*1000:.3f} ms ± {lean_coo_std_time*1000:.3f} ms")
    print(f"    Min time:     {lean_coo_min_time*1000:.3f} ms")
    print(f"  SparseDokVolumeInterpolator (DOK):")
    print(f"    Average time: {lean_dok_avg_time*1000:.3f} ms ± {lean_dok_std_time*1000:.3f} ms")
    print(f"    Min time:     {lean_dok_min_time*1000:.3f} ms")
    print(f"  scipy RegularGridInterpolator:")
    print(f"    Average time (interp only): {scipy_avg_time*1000:.3f} ms ± {scipy_std_time*1000:.3f} ms")
    print(f"    Min time (interp only):     {scipy_min_time*1000:.3f} ms")
    print(f"    Densify time (.todense):    {densify_time*1000:.3f} ms")
    print(f"    Total SciPy time:           {effective_scipy_avg_time*1000:.3f} ms")
    print(f"    Dense volume size:          {dense_bytes/1e6:.2f} MB")
    print(f"  Speedup (COO, incl. densify): {speedup_coo:.2f}x")
    print(f"  Speedup (DOK, incl. densify): {speedup_dok:.2f}x")
    print(f"  Results match COO: {all_close_coo}")
    if all_close_coo:
        print(f"    COO max diff:  {max_diff_coo:.2e}")
        print(f"    COO mean diff: {mean_diff_coo:.2e}")
        print(f"    COO NaN count: {n_nan_coo}")
    print(f"  Results match DOK: {all_close_dok}")
    if all_close_dok:
        print(f"    DOK max diff:  {max_diff_dok:.2e}")
        print(f"    DOK mean diff: {mean_diff_dok:.2e}")
        print(f"    DOK NaN count: {n_nan_dok}")
    
    return {
        'vol_shape': vol_shape,
        'n_coords': n_coords,
        'lean_coo_avg_time': lean_coo_avg_time,
        'lean_coo_std_time': lean_coo_std_time,
        'lean_coo_min_time': lean_coo_min_time,
        'lean_dok_avg_time': lean_dok_avg_time,
        'lean_dok_std_time': lean_dok_std_time,
        'lean_dok_min_time': lean_dok_min_time,
        'densify_time': densify_time,
        'dense_bytes': dense_bytes,
        'scipy_avg_time': scipy_avg_time,
        'scipy_std_time': scipy_std_time,
        'scipy_min_time': scipy_min_time,
        'scipy_effective_avg_time': effective_scipy_avg_time,
        'speedup_coo': speedup_coo,
        'speedup_dok': speedup_dok,
        'results_match_coo': all_close_coo,
        'results_match_dok': all_close_dok,
        'max_diff_coo': max_diff_coo,
        'mean_diff_coo': mean_diff_coo,
        'max_diff_dok': max_diff_dok,
        'mean_diff_dok': mean_diff_dok
    }


def main():
    """Run comprehensive benchmarks."""
    print("="*70)
    print("LeanVolumeInterpolator vs scipy RegularGridInterpolator Sparse Benchmark")
    print("="*70)
    
    # Test configurations: (volume_shape, n_coords)
    test_configs = [
        # Small volume, few coordinates
        ((50, 50, 50), 100),
        ((50, 50, 50), 1000),
        ((50, 50, 50), 10000),
        
        # Medium volume
        ((100, 100, 100), 100),
        ((100, 100, 100), 1000),
        ((100, 100, 100), 10000),
        ((100, 100, 100), 100000),
        
        # Large volume
        ((200, 200, 200), 1000),
        ((200, 200, 200), 10000),
        ((200, 200, 200), 100000),
        
        # Very large volume (like mentioned in README)
        ((500, 500, 500), 10000),
        ((500, 500, 500), 100000),

        ((1500, 1500, 1500), 100000),
    ]
    
    results = []
    
    for vol_shape, n_coords in test_configs:
        try:
            result = benchmark_interpolation(vol_shape, n_coords, n_runs=5, density=1e-2)
            results.append(result)
        except MemoryError:
            print(f"\n⚠️  Skipping {vol_shape} with {n_coords} coords: Out of memory")
            continue
        except Exception as e:
            print(f"\n⚠️  Skipping {vol_shape} with {n_coords} coords: {e}")
            continue
    
    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Volume Shape':<20} {'Format':<8} {'N Coords':<12} "
          f"{'Lean (ms)':<15} {'scipy interp (ms)':<18} {'Speedup*':<10} {'Match':<8}")
    print("-"*70)
    
    for r in results:
        # COO row
        print(f"{str(r['vol_shape']):<20} {'COO':<8} {r['n_coords']:<12} "
              f"{r['lean_coo_avg_time']*1000:<15.3f} {r['scipy_avg_time']*1000:<15.3f} "
              f"{r['speedup_coo']:<10.2f} {str(r['results_match_coo']):<8}")
        # DOK row
        print(f"{str(r['vol_shape']):<20} {'DOK':<8} {r['n_coords']:<12} "
              f"{r['lean_dok_avg_time']*1000:<15.3f} {r['scipy_avg_time']*1000:<15.3f} "
              f"{r['speedup_dok']:<10.2f} {str(r['results_match_dok']):<8}")
    
    # Overall statistics
    if results:
        all_speedups = [r['speedup_coo'] for r in results] + [r['speedup_dok'] for r in results]
        avg_speedup = np.mean(all_speedups)
        max_speedup = np.max(all_speedups)
        min_speedup = np.min(all_speedups)
        
        print(f"\n{'='*70}")
        print("Overall Statistics:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Maximum speedup: {max_speedup:.2f}x")
        print(f"  Minimum speedup: {min_speedup:.2f}x")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

