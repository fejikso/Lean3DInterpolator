#!/usr/bin/env python3
"""
Benchmark script comparing LeanVolumeInterpolator with scipy.interpolate.RegularGridInterpolator
using HDF5 datasets with sparse data (10% filled, 90% zeros).

This script creates a temporary HDF5 file with:
- Shape: (512, 512, 512)
- Chunking: (8, 8, 8)
- Compression: lzf
- Data: 10% random values, 90% zeros
"""

import numpy as np
import time
import tempfile
import os

try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py is required for this benchmark. Install it with: pip install h5py"
    )

from scipy.interpolate import RegularGridInterpolator
from LeanVolumeInterpolator import LeanVolumeInterpolator


def create_sparse_hdf5_dataset(filepath, shape=(512, 512, 512), fill_ratio=0.1, 
                               chunk_shape=(8, 8, 8), compression='lzf', dtype=np.float32):
    """
    Create an HDF5 dataset with sparse data (only fill_ratio percent filled).
    
    Parameters:
        filepath: Path to the HDF5 file
        shape: Shape of the dataset (nx, ny, nz)
        fill_ratio: Fraction of the volume to fill with random values (0.0 to 1.0)
        chunk_shape: Chunk size for HDF5 storage
        compression: Compression algorithm ('lzf', 'gzip', etc.)
        dtype: Data type for the dataset
    
    Returns:
        h5py.File object and h5py.Dataset object
    """
    print(f"Creating HDF5 file: {filepath}")
    print(f"  Shape: {shape}")
    print(f"  Fill ratio: {fill_ratio*100:.1f}%")
    print(f"  Chunk shape: {chunk_shape}")
    print(f"  Compression: {compression}")
    print(f"  Dtype: {dtype}")
    
    # Create HDF5 file
    h5_file = h5py.File(filepath, 'w')
    
    # Create dataset with chunking and compression
    dataset = h5_file.create_dataset(
        'volume',
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
        compression=compression,
        fillvalue=0.0
    )
    
    # Fill 10% of the volume with random values
    total_elements = np.prod(shape)
    n_filled = int(total_elements * fill_ratio)
    
    print(f"  Filling {n_filled:,} random positions ({fill_ratio*100:.1f}% of {total_elements:,} total)...")
    
    # Generate random indices and values
    np.random.seed(42)  # For reproducibility
    flat_indices = np.random.choice(total_elements, size=n_filled, replace=False)
    random_values = np.random.rand(n_filled).astype(dtype)
    
    # Create sparse array in memory (zeros everywhere, then fill random positions)
    # This is more efficient than writing element-by-element to HDF5
    print("  Creating sparse array in memory...")
    sparse_array = np.zeros(shape, dtype=dtype)
    
    # Convert flat indices to 3D indices and fill
    indices_3d = np.unravel_index(flat_indices, shape)
    sparse_array[indices_3d] = random_values
    
    # Write entire array to HDF5 at once (compression will handle zeros efficiently)
    print("  Writing to HDF5 dataset...")
    dataset[:] = sparse_array
    
    # Get some statistics
    dataset.flush()  # Ensure data is written
    file_size = os.path.getsize(filepath)
    print(f"  File size: {file_size / (1024**2):.2f} MB")
    print(f"  Non-zero elements: {np.count_nonzero(dataset[:]):,}")
    
    return h5_file, dataset


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


def benchmark_hdf5_interpolation(vol_shape=(512, 512, 512), n_coords=10000, 
                                 n_runs=5, dtype=np.float32):
    """
    Benchmark both interpolators using an HDF5 dataset.
    
    Parameters:
        vol_shape: Shape of the 3D volume (nx, ny, nz)
        n_coords: Number of coordinates to interpolate
        n_runs: Number of runs for averaging
        dtype: Data type for the volume
    
    Returns:
        Dictionary with timing results and verification info
    """
    print(f"\n{'='*70}")
    print(f"HDF5 Benchmark: Volume shape {vol_shape}, {n_coords:,} coordinates")
    print(f"{'='*70}")
    
    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        tmp_filepath = tmp_file.name
    
    try:
        # Create sparse HDF5 dataset
        h5_file, dataset = create_sparse_hdf5_dataset(
            tmp_filepath,
            shape=vol_shape,
            fill_ratio=0.1,
            chunk_shape=(8, 8, 8),
            compression='lzf',
            dtype=dtype
        )
        
        # Create random coordinates within bounds
        print(f"\nGenerating {n_coords:,} random coordinates...")
        np.random.seed(123)  # Different seed for coordinates
        x_coords = np.random.uniform(0, vol_shape[0]-1, n_coords)
        y_coords = np.random.uniform(0, vol_shape[1]-1, n_coords)
        z_coords = np.random.uniform(0, vol_shape[2]-1, n_coords)
        
        # Prepare coordinates for scipy (needs shape (n_points, 3))
        coords_scipy = np.column_stack([x_coords, y_coords, z_coords])
        
        # Initialize interpolators
        print("\nInitializing interpolators...")
        
        # Load HDF5 dataset into memory for both interpolators
        # Note: Current LeanVolumeInterpolator implementation uses fancy indexing
        # that HDF5 doesn't support, so we load into memory for the benchmark.
        # The HDF5 file still demonstrates compression benefits (114MB vs 512MB uncompressed).
        print("  Loading HDF5 dataset into memory...")
        vol_memory = dataset[:]  # Load entire dataset into memory
        
        # Initialize LeanVolumeInterpolator with in-memory dense array
        t0 = time.perf_counter()
        lean_interp = LeanVolumeInterpolator(vol_memory, extrap_val=np.nan, dtype=dtype)
        lean_init_time = time.perf_counter() - t0
        print(f"  ✓ LeanVolumeInterpolator initialized in {lean_init_time*1000:.3f} ms")

        x_grid = np.arange(vol_shape[0])
        y_grid = np.arange(vol_shape[1])
        z_grid = np.arange(vol_shape[2])

        t0 = time.perf_counter()
        scipy_interp = RegularGridInterpolator(
            (x_grid, y_grid, z_grid), 
            vol_memory,
            method='linear', 
            bounds_error=False, 
            fill_value=np.nan
        )
        scipy_init_time = time.perf_counter() - t0
        print(f"  ✓ scipy RegularGridInterpolator initialized in {scipy_init_time*1000:.3f} ms")
        
        # Warm-up runs
        print("\nWarming up (small subset, cached)...")
        _ = lean_interp((x_coords[:100], y_coords[:100], z_coords[:100]))
        _ = scipy_interp(coords_scipy[:100])
        
        # Optional pause to allow manual cache clearing before SciPy benchmark
        print(
            "\nIf you want cold-cache timings for scipy RegularGridInterpolator, clear the OS page cache now\n"
            "in another terminal, for example:\n"
            "  echo 3 | sudo tee /proc/sys/vm/drop_caches\n"
        )
        input("Press Enter to start scipy RegularGridInterpolator benchmark...")
        
        # Benchmark scipy RegularGridInterpolator
        print(f"\nBenchmarking scipy RegularGridInterpolator ({n_runs} runs)...")
        scipy_times = []
        for run in range(n_runs):
            start = time.perf_counter()
            scipy_result = scipy_interp(coords_scipy)
            elapsed = time.perf_counter() - start
            scipy_times.append(elapsed)
            print(f"  Run {run+1}/{n_runs}: {elapsed*1000:.3f} ms")
        scipy_avg_time = np.mean(scipy_times)
        scipy_std_time = np.std(scipy_times)
        scipy_min_time = np.min(scipy_times)
        
        # Optional pause to allow manual cache clearing before Lean benchmark
        print(
            "\nIf you want cold-cache timings for LeanVolumeInterpolator, clear the OS page cache now\n"
            "in another terminal, for example:\n"
            "  echo 3 | sudo tee /proc/sys/vm/drop_caches\n"
        )
        input("Press Enter to start LeanVolumeInterpolator benchmark...")
        
        # Benchmark LeanVolumeInterpolator
        print(f"\nBenchmarking LeanVolumeInterpolator ({n_runs} runs)...")
        lean_times = []
        for run in range(n_runs):
            start = time.perf_counter()
            lean_result = lean_interp((x_coords, y_coords, z_coords))
            elapsed = time.perf_counter() - start
            lean_times.append(elapsed)
            print(f"  Run {run+1}/{n_runs}: {elapsed*1000:.3f} ms")
        lean_avg_time = np.mean(lean_times)
        lean_std_time = np.std(lean_times)
        lean_min_time = np.min(lean_times)
        
        # Verify results match
        print("\nVerifying results...")
        all_close, max_diff, mean_diff, n_nan = verify_results(lean_result, scipy_result)
        
        # Calculate speedup
        speedup = scipy_avg_time / lean_avg_time
        
        # Print results
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"LeanVolumeInterpolator:")
        print(f"  Average time: {lean_avg_time*1000:.3f} ms ± {lean_std_time*1000:.3f} ms")
        print(f"  Min time:     {lean_min_time*1000:.3f} ms")
        print(f"  Init time:    {lean_init_time*1000:.3f} ms")
        print(f"  Throughput:   {n_coords/lean_avg_time:,.0f} points/sec")
        print(f"\nscipy RegularGridInterpolator:")
        print(f"  Average time: {scipy_avg_time*1000:.3f} ms ± {scipy_std_time*1000:.3f} ms")
        print(f"  Min time:     {scipy_min_time*1000:.3f} ms")
        print(f"  Init time:    {scipy_init_time*1000:.3f} ms")
        print(f"  Throughput:   {n_coords/scipy_avg_time:,.0f} points/sec")
        print(f"\nSpeedup:        {speedup:.2f}x")
        print(f"Results match:  {all_close}")
        if all_close:
            print(f"Max difference: {max_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")
        print(f"NaN count:      {n_nan:,} (out of {n_coords:,})")
        print(f"{'='*70}")
        
        # Memory usage info
        print(f"\nMemory Information:")
        print(f"  HDF5 file size: {os.path.getsize(tmp_filepath) / (1024**2):.2f} MB")
        print(f"  In-memory size (for scipy): {vol_memory.nbytes / (1024**2):.2f} MB")
        print(f"  Memory savings (HDF5 vs in-memory): {(1 - os.path.getsize(tmp_filepath) / vol_memory.nbytes) * 100:.1f}%")
        
        h5_file.close()
        
        return {
            'vol_shape': vol_shape,
            'n_coords': n_coords,
            'lean_avg_time': lean_avg_time,
            'lean_std_time': lean_std_time,
            'lean_min_time': lean_min_time,
            'lean_init_time': lean_init_time,
            'scipy_avg_time': scipy_avg_time,
            'scipy_std_time': scipy_std_time,
            'scipy_min_time': scipy_min_time,
            'scipy_init_time': scipy_init_time,
            'speedup': speedup,
            'results_match': all_close,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'hdf5_file_size_mb': os.path.getsize(tmp_filepath) / (1024**2),
            'memory_size_mb': vol_memory.nbytes / (1024**2)
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
            print(f"\nCleaned up temporary file: {tmp_filepath}")


def main():
    """Run HDF5 benchmarks with different coordinate counts."""
    print("="*70)
    print("HDF5 Benchmark: LeanVolumeInterpolator vs scipy RegularGridInterpolator")
    print("="*70)
    print("\nConfiguration:")
    print("  Volume shape: (512, 512, 512)")
    print("  Data sparsity: 10% filled, 90% zeros")
    print("  Chunking: (8, 8, 8)")
    print("  Compression: lzf")
    print("="*70)
    
    # Test with different numbers of coordinates
    test_configs = [
        1000,
        10000,
        100000,
        1000000,  # 1M points
    ]
    
    results = []
    
    for n_coords in test_configs:
        try:
            result = benchmark_hdf5_interpolation(
                vol_shape=(512, 512, 512),
                n_coords=n_coords,
                n_runs=5,
                dtype=np.float32
            )
            results.append(result)
        except MemoryError:
            print(f"\n⚠️  Skipping {n_coords:,} coordinates: Out of memory")
            continue
        except Exception as e:
            print(f"\n⚠️  Error with {n_coords:,} coordinates: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary table
    if results:
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(
            f"{'N Coords':<12} "
            f"{'Lean init (ms)':<15} {'scipy init (ms)':<17} "
            f"{'Lean (ms)':<12} {'scipy (ms)':<12} "
            f"{'Speedup':<10} {'Match':<8} {'HDF5 (MB)':<12} {'Memory (MB)':<12}"
        )
        print("-"*70)
        
        for r in results:
            print(
                f"{r['n_coords']:<12,} "
                f"{r['lean_init_time']*1000:<15.3f} {r['scipy_init_time']*1000:<17.3f} "
                f"{r['lean_avg_time']*1000:<12.3f} {r['scipy_avg_time']*1000:<12.3f} "
                f"{r['speedup']:<10.2f} {str(r['results_match']):<8} "
                f"{r['hdf5_file_size_mb']:<12.2f} {r['memory_size_mb']:<12.2f}"
            )
        
        # Overall statistics
        avg_speedup = np.mean([r['speedup'] for r in results])
        max_speedup = np.max([r['speedup'] for r in results])
        min_speedup = np.min([r['speedup'] for r in results])
        
        print(f"\n{'='*70}")
        print("Overall Statistics:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Maximum speedup: {max_speedup:.2f}x")
        print(f"  Minimum speedup: {min_speedup:.2f}x")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

