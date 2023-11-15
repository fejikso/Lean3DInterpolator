# LeanVolumeInterpolator

## Description
LeanVolumeInterpolator is a Python class designed for efficient trilinear interpolation of 3D volumetric data. It offers a practical alternative to `scipy.interpolate.RegularGridInterpolator`, especially for handling large datasets stored as numpy memmaps, sparse 3D volumes, and compressed HDF5 datasets.

The motivation for developing LeanVolumeInterpolator arose from the need to perform arbitrary two-dimensional slices on a 3D volume, with a focus on efficiency and speed.

### Performance Comparison
In a benchmark test using a large volume of size (1300, 900, 2300) stored as a memmap on disk, LeanVolumeInterpolator demonstrated significant performance improvements:

- **Task**: Slicing with a two-dimensional _oblique_ (angled, non-orthogonal to volume) plane discretized as a (1000x1000) coordinate grid.
- **scipy.interpolate.interpn**: Took 2.49 seconds.
- **LeanVolumeInterpolator**: Completed the task in just 0.29 seconds.

This test showcases LeanVolumeInterpolator's capability to handle these interpolation tasks more efficiently than the standard solutions, particularly for large-scale volumetric data.

### Background and Motivation
The development of LeanVolumeInterpolator was inspired by limitations encountered with `scipy.interpolate.RegularGridInterpolator` in handling certain 3D data structures. Specifically, I found the following problems:

- **Incompatibility with the `Sparse` Package**: RegularGridInterpolator cannot handle 3D volumes created using the `sparse` package, which is an extension of `scipy.sparse` (which only allows 2D sparse arrays) that allows sparse n-dimensional datasets. `LeanVolumeInterpolator` seems to work well with 3D sparse volumes created with the `sparse` package, without having to convert them to full (dense) arrays.
- **Inefficient use of NumPy Memmaps**: I observed that RegularGridInterpolator seems to pre-load the contents of NumPy memory maps (numpy.memmap), which defeats the benefits of using memory maps, especially with large volumes.
- **Inefficient use of compressed HDF5 Datasets**: As with NumPy memmaps, when 3D volumes are stored as hdf5 datasets with compression, RegularGridInterpolator seems to load (and therefore decompressing) the whole volume at initialization, which defeats the purpose of efficient storage as a compressed dataset. 

It should work with other 3D array-like structures that allow slicing. 

As I encountered in some applications, this makes LeanVolumeInterpolator a more suitable choice for applications dealing with large or complex 3D volumetric data, where memory efficiency and flexibility are key concerns.

## Features
- Efficient trilinear interpolation on 3D volumes. Calls to the underlying volume are vectorized, so calls are very efficient.
- Supports various data types including numpy arrays, memmaps, and sparse volumes.
- Customizable constant extrapolation values (when input coordinates are outside the volume).
- Customizable data types for inner computations and output. For example, the volume data might be `np.uint8`, but the output will be `np.float32`.

## Caveats
* Given a 3D volume dataset `vol`, This lean interpolation approach assumes that the coordinates are `0, 1, ... , n_dim` for each dimension. If you want to use a different coordinate system, you will have to do some transformation with respect to this "canonical" coordinate system.
* When using `sparse` volumes, the argument `to_dense` must be set to `True`. Otherwise inner computations will throw an error. The output will always be dense (np.array)

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
lean_interp = LeanVolumeInterpolator(test_vol, extrap_val=np.nan, dtype=np.float32, to_dense=False)

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
from LeanVolumeInterpolator import LeanVolumeInterpolator

# Create a sparse 3D volume
test_vol_sparse = sparse.random((100, 100, 100), density=0.1)

# Initialize the interpolator
lean_interp = LeanVolumeInterpolator(test_vol_sparse, extrap_val=np.nan, dtype=np.float32, to_dense=True)

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
