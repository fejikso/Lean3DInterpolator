# LeanVolumeInterpolator

## Description
LeanVolumeInterpolator is a Python class designed for efficient trilinear interpolation of 3D volumetric data. It offers a practical alternative to `scipy.interpolate.RegularGridInterpolator`, especially for handling large datasets stored as numpy memmaps, sparse 3D volumes, and compressed HDF5 datasets.

The motivation for developing LeanVolumeInterpolator arose from the need to perform arbitrary two-dimensional slices on a 3D volume, with a focus on efficiency and speed.

### Performance Comparison (Dense Volumes & Memmaps)
In a benchmark test using a large volume of size `(1300, 900, 2300)` stored as a NumPy memmap on disk, LeanVolumeInterpolator demonstrated significant performance improvements compared to SciPy:

- **Task**: Interpolating arbitrary 3D coordinates from the memmapped volume (e.g., a dense \((1000 \times 1000)\) grid).
- **Memmap size**: The on-disk and logical in-memory size of this volume is approximately **2.5–10 GB** depending on the chosen dtype.

Two aspects stand out:

- **Object initialization cost**
  - LeanVolumeInterpolator:
    - Initialization time: on the order of **tens of microseconds** (e.g. **~0.008 ms**).
  - SciPy RegularGridInterpolator:
    - Initialization time: on the order of **seconds** for this volume (e.g. **~2.2 s** in one memmap experiment), due to internally materializing and indexing the full volume.

- **Per-call interpolation performance**
  - In previous dense memmap benchmarks with \(10^6\) coordinates:
    - SciPy RegularGridInterpolator:
      - Average time: **~0.87 s**
      - Throughput: **~1.15M points/sec**
    - LeanVolumeInterpolator:
      - Average time: **~0.79 s**
      - Throughput: **~1.27M points/sec**
    - Speedup (Lean vs SciPy): **≈1.1×** on the interpolation itself.
  - In a smaller memmap benchmark with 1,000 coordinates on the same volume:
    - LeanVolumeInterpolator:
      - Average interpolation time: **~0.50 ms**
      - Initialization time: **~0.008 ms**
      - Throughput: **~2.0M points/sec**
    - SciPy RegularGridInterpolator:
      - Average interpolation time: **~1.0 ms**
      - Initialization time: **~2,214 ms**
      - Throughput: **~1.0M points/sec**
    - Speedup (Lean vs SciPy): **≈2.0×** per interpolation call, with **orders-of-magnitude lower initialization cost**.

These experiments showcase LeanVolumeInterpolator's capability to handle large-scale volumetric interpolation efficiently on memmapped volumes, both by reducing one-time initialization overhead and by improving per-call interpolation throughput compared to the standard SciPy tools.

### Performance & Memory with Sparse Volumes
When working with truly sparse 3D volumes (created with the [`sparse` PyData library](https://pypi.org/project/sparse/)), `SparseLeanVolumeInterpolator` can operate directly on the sparse structure, while SciPy must first densify the volume before interpolation:

- **SciPy (RegularGridInterpolator)**:
  - Cannot consume `sparse` arrays directly; attempts raise an error instructing the user to call `.todense()` manually.
  - Densification time and memory can be substantial:
    - For a `(100, 100, 100)` volume at density 0.01, densifying takes on the order of **1–2 ms** and creates a dense array of about **4 MB**.
    - For a `(500, 500, 500)` sparse volume at density 0.01, densifying takes on the order of **70 ms** and creates a dense array of about **500 MB**.
    - For a `(1500, 1500, 1500)` sparse volume at density 0.01, densifying takes on the order of **2 seconds** and creates a dense array of about **13.5 GB**.
  - In our benchmarks we treat SciPy’s effective cost as **densify + interpolation** when starting from a sparse volume.

- **SparseLeanVolumeInterpolator**:
  - Works directly on `sparse.COO` or `sparse.DOK` volumes without densification.
  - For moderate volumes where densification dominates (e.g. `(100, 100, 100)` or `(200, 200, 200)` with relatively few query points), including SciPy’s `.todense()` time yields **speedups from ~2× up to more than 10×** in favor of `SparseLeanVolumeInterpolator`.
  - For very large volumes where SciPy’s densification is extremely costly (e.g. `(500, 500, 500)` and `(1500, 1500, 1500)` at density 0.01 with modest coordinate counts), the sparse interpolator avoids hundreds of milliseconds to seconds of densification and **saves hundreds of megabytes to tens of gigabytes of RAM**, often resulting in **overall speedups of ~4–5× or more** compared to the densify+interpolate pipeline.

Overall, when the source data is sparse, `SparseLeanVolumeInterpolator` not only avoids the large memory overhead of densifying but can also outperform SciPy in wall-clock time once the densification step is included in the comparison.

### Performance with Compressed HDF5 Volumes
`LeanVolumeInterpolator` also works well with large 3D volumes stored in compressed HDF5 datasets. In these scenarios, SciPy’s `RegularGridInterpolator` typically requires loading the full dataset into memory first, whereas the HDF5 backend and chunking/compression settings are better aligned with LeanVolumeInterpolator’s access pattern.

In a benchmark using a `(512, 512, 512)` volume stored as an HDF5 dataset with:

- 10% non-zeros (90% zeros),
- Chunking `(8, 8, 8)`,
- `lzf` compression,
- `float32` data,

we observed:

- **Memory characteristics**
  - HDF5 file size on disk: **~115 MB**
  - Fully dense in-memory volume (for SciPy): **~512 MB**
  - Effective storage reduction: **≈77%** compared to the dense in-memory representation.

- **Interpolation throughput (Lean vs SciPy)**  
  (All timings are averages over 5 runs; speedup is SciPy time / Lean time.)
  - 1,000 random coordinates:
    - LeanVolumeInterpolator: **~0.53 ms**
    - SciPy RegularGridInterpolator: **~0.75 ms**  
    - Speedup: **≈1.4×**
  - 10,000 random coordinates:
    - LeanVolumeInterpolator: **~1.77 ms**
    - SciPy RegularGridInterpolator: **~3.69 ms**  
    - Speedup: **≈2.1×**
  - 100,000 random coordinates:
    - LeanVolumeInterpolator: **~40.9 ms**
    - SciPy RegularGridInterpolator: **~49.5 ms**  
    - Speedup: **≈1.2×**
  - 1,000,000 random coordinates:
    - LeanVolumeInterpolator: **~383 ms**
    - SciPy RegularGridInterpolator: **~534 ms**  
    - Speedup: **≈1.4×**

Across all HDF5 configurations tested, LeanVolumeInterpolator was between **~1.2× and ~2.1× faster** than SciPy for interpolation itself (average speedup ≈**1.5×**), while the HDF5 representation reduced storage from ~512 MB down to ~115 MB for the same logical volume.

### Background and Motivation
The development of LeanVolumeInterpolator was inspired by limitations encountered with `scipy.interpolate.RegularGridInterpolator` in handling certain 3D data structures. Specifically, I found the following problems:

- **Incompatibility with the `Sparse` Package**: RegularGridInterpolator cannot handle 3D volumes created using the `sparse` package, which is an extension of `scipy.sparse` (which only allows 2D sparse arrays) that allows sparse n-dimensional datasets. `LeanVolumeInterpolator` seems to work well with 3D sparse volumes created with the `sparse` package, without having to convert them to full (dense) arrays.
- **Inefficient use of NumPy Memmaps**: I observed that RegularGridInterpolator seems to pre-load the contents of NumPy memory maps (numpy.memmap), which defeats the benefits of using memory maps, especially with large volumes.
- **Inefficient use of compressed HDF5 Datasets**: As with NumPy memmaps, when 3D volumes are stored as hdf5 datasets with compression, RegularGridInterpolator seems to load (and therefore decompressing) the whole volume at initialization, which defeats the purpose of efficient storage as a compressed dataset. 

It should work with other 3D array-like structures that allow slicing. 

As I encountered in some applications, this makes LeanVolumeInterpolator a more suitable choice for applications dealing with large or complex 3D volumetric data, where memory efficiency and flexibility are key concerns.

## Features
- Efficient trilinear interpolation on 3D volumes. Calls to the underlying volume are vectorized, so calls are very efficient.
- Supports various data types including numpy arrays and memmaps. Sparse volumes are supported via the `SparseLeanVolumeInterpolator` class.
- Customizable constant extrapolation values (when input coordinates are outside the volume).
- Customizable data types for inner computations and output. For example, the volume data might be `np.uint8`, but the output will be `np.float32`.

## Caveats

- **Coordinate system is implicit and uniform**
  - LeanVolumeInterpolator assumes that your volume `vol` lives on a regular grid with integer indices:
    - X dimension: `0, 1, ..., vol.shape[0] - 1`
    - Y dimension: `0, 1, ..., vol.shape[1] - 1`
    - Z dimension: `0, 1, ..., vol.shape[2] - 1`
  - The interpolator does **not** take physical coordinates (e.g. millimeters) or arbitrary axis arrays.
  - If your data uses a different coordinate system (for example, voxel size of 0.5 mm, or non-zero origin), you must:
    - Convert your query points into this “canonical” index space before calling the interpolator, and/or
    - Apply an affine transform so that your coordinates align with the implicit `[0, ..., n_dim-1]` grid.

- **Dense vs sparse volumes**
  - `LeanVolumeInterpolator` is designed for **dense** NumPy-compatible volumes (arrays, memmaps, dense HDF5 datasets already loaded into memory).
  - For **sparse** 3D volumes created with the PyData `sparse` library, you should use `SparseLeanVolumeInterpolator` instead:
    - It accepts `sparse.COO` or `sparse.DOK` volumes directly.
    - It never densifies the full volume, and treats missing entries as zeros.

- **HDF5 volumes and fancy indexing**
  - The current implementation of `LeanVolumeInterpolator` uses advanced NumPy indexing internally.
  - Some storage backends (notably HDF5 datasets accessed via `h5py`) do not support these indexing patterns directly.
  - In those cases, you should first load the HDF5 dataset into a NumPy array (or memmap) and then pass that dense array into LeanVolumeInterpolator, as shown in the HDF5 benchmark script.

## Installation
Clone the repository or download the `LeanVolumeInterpolator.py` file directly into your project.
```
git clone https://github.com/fejikso/LeanVolumeInterpolator.git
```

It requires the NumPy package.
## Usage
Here are a couple of examples to show how to use the LeanVolumeInterpolator.

### With NumPy Arrays
```python
import numpy as np
from LeanVolumeInterpolator import LeanVolumeInterpolator

# Load a large numpy 3D volume
# test_vol = np.load("big_3d_volume.npy", mmap_mode="r")
# or create a random matrix for testing
test_vol = np.random.rand((100,100,100))

# Initialize the interpolator
# * if input coordinates are outside the volume, np.nan will be used for extrapolation.
# * the inner interpolation and output will be using np.float32
# * since we are not using sparse volumes, no need to convert to dense.
lean_interp = LeanVolumeInterpolator(test_vol, extrap_val=np.nan, dtype=np.float32)

# Single coordinate interpolation
single_val = lean_interp((51.5, 13.1, 10.5))
print(f"Interpolated value at (51.5, 13.1, 10.5): {single_val}")

# Slice the volume in an arbitrary plane
X, Y, Z = np.mgrid[0:100, 0:100, 50:51]
XYZ = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T; # Transpose so XYZ is shape (n_coords, 3)

from scipy.spatial.transform import Rotation
rot = Rotation.from_rotvec([30,-45, 5], degrees=True);
XYZrot = rot.apply(XYZ)

rot_slice = lean_interp((XYZrot[:,0], XYZrot[:,1], XYZrot[:,2]))
rot_slice = rot_slice.reshape(X.shape) # reshape to original coordinate matrix

```

### With Sparse Volumes
```python
import numpy as np
import sparse
from LeanVolumeInterpolator import SparseLeanVolumeInterpolator

# Create a sparse 3D volume
test_vol_sparse = sparse.random((100, 100, 100), density=0.1)

# Initialize the sparse interpolator
lean_interp = SparseLeanVolumeInterpolator(test_vol_sparse, extrap_val=np.nan, dtype=np.float32)

# Vector of coordinates
# Here we subsample a sub-volume at a finer resolution

x_coords = np.linspace(0, 50, 150)
y_coords = np.linspace(0, 50, 150)
z_coords = np.linspace(0, 50, 150)

vector_vals = lean_interp((x_coords, y_coords, z_coords))
print(f"Interpolated values at vectorized coordinates: {vector_vals}")
```

### Future?
This code was whipped-up for some applications that were three-dimensional. It can definitely be expanded for n-dimensional cases.
## Contributing
Contributions to LeanVolumeInterpolator are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.

## License
This project is licensed under the Apache License 2.0 - see https://www.apache.org/licenses/LICENSE-2.0.txt  for details.

## Contact
For any queries or feedback, please contact https://github.com/fejikso

## Support the author

If you find LeanVolumeInterpolator useful and would like to support further development, you can buy me a coffee here:

- https://buymeacoffee.com/fejikso
