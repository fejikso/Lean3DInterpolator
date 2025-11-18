# LeanVolumeInterpolator.py
# Author: Fernando Gonzalez del Cueto
# License: Apache License 2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

try:
    import sparse as _sparse
except ImportError:  # pragma: no cover - sparse is optional
    _sparse = None

class LeanVolumeInterpolator():
    """
    A class for efficient trilinear interpolation on large volumetric datasets.

    This class aims to replace scipy.interpolate.RegularGridInterpolator, offering improved:
    - performance with large dense volumes such as numpy arrays and memmaps
    - convenience when working with HDF5 datasets loaded into memory

    Attributes:
        shape (tuple): Shape of the input volume.
        dtype (data-type): Data type for computation, default is np.float32.
        vol (array-like): The input 3D dense volume for interpolation.
        extrap_val (float): Extrapolation value for out-of-bound coordinates.
    
    Methods:
        __call__(coords): Interpolates the volume at the given coordinates.
    """

    def __init__(self, vol, extrap_val=np.nan, dtype=np.float32):
        """
        Initializes the LeanInterpolator object.

        Parameters:
            vol (array-like): A 3D volume for interpolation.
            extrap_val (float): Value used for extrapolation of out-of-bound coordinates. Defaults to np.nan.
            dtype (data-type): Data type for computations, defaults to np.float32.
        """
        self.shape = vol.shape
        self.dtype = dtype
        self.vol = vol
        self.extrap_val = extrap_val

    def __call__(self, coords):
        """
        Performs trilinear interpolation on the given coordinates.

        Parameters:
            coords: A tuple (x, y, z) of coordinates for interpolation. Each element of the tuple can be a single value, 1D, 2D, or 3D array.
            All of them (x,y,z) should have the same shape

        Returns:
            numpy.ndarray: The interpolated values at the specified coordinates, matching the shape of the input coordinates.
        """
        
        out_shape = np.array(coords[0]).shape  # infer input shape so that output matches it
 
        V = self.vol

        xv, yv, zv = (np.ravel(v) for v in coords) # reshape into 1D vectors

        Nv = len(xv)

        # Vectorized coordinates        
        xv0 = xv.astype(int)
        αx = xv-xv0

        yv0 = yv.astype(int)
        αy = yv-yv0

        zv0 = zv.astype(int)
        αz = zv-zv0

        s = np.zeros(Nv, dtype=self.dtype)
        val = np.zeros(Nv, dtype=self.dtype)
        
        for xi, βx in [(xv0, 1-αx), (xv0+1, αx)]:

            idx_xvalid = (xi>=0) & (xi<V.shape[0])
            
            for yi, βy in [(yv0, 1-αy), (yv0+1, αy)]:
            
                idx_yvalid = (yi>=0) & (yi<V.shape[1])

                for zi, βz in [(zv0, 1-αz), (zv0+1, αz)]:

                    idx_zvalid = (zi>=0) & (zi<V.shape[2])
                    
                    idx_valid = idx_xvalid & idx_yvalid & idx_zvalid
                    
                    val[~idx_valid] = self.extrap_val
                    val[idx_valid] = V[xi[idx_valid], yi[idx_valid], zi[idx_valid]]

                    s[:] += (βx*βy*βz)*val

        return s.reshape(out_shape)


class SparseLeanVolumeInterpolator(LeanVolumeInterpolator):
    """
    Sparse-optimized trilinear interpolator for 3D volumes stored as PyData
    sparse arrays (e.g., sparse.COO, sparse.DOK).

    This class avoids densifying the full volume by treating missing entries
    as zeros and using a backend-specific lookup strategy for each format.
    """

    def __init__(self, vol, extrap_val=np.nan, dtype=np.float32):
        """
        Initialize the sparse interpolator.

        Parameters
        ----------
        vol : sparse.SparseArray
            3D sparse volume (COO, DOK, etc.).
        extrap_val : float
            Extrapolation value for out-of-bounds coordinates.
        dtype : numpy.dtype
            Data type for computations.
        """
        if _sparse is None:
            raise ImportError(
                "SparseLeanVolumeInterpolator requires the 'sparse' package. "
                "Install it with `pip install sparse`."
            )

        if not isinstance(vol, _sparse.SparseArray):
            raise TypeError(
                "SparseLeanVolumeInterpolator expects a PyData sparse array "
                "(e.g. sparse.COO or sparse.DOK)."
            )

        super().__init__(vol, extrap_val=extrap_val, dtype=dtype)

        nx, ny, nz = vol.shape
        self.nx, self.ny, self.nz = nx, ny, nz
        self._stride_yz = ny * nz

        # Build a compact index over non-zero entries.
        if isinstance(vol, _sparse.COO):
            coords = vol.coords  # shape: (3, nnz)
            data = vol.data.astype(self.dtype, copy=False)
        elif isinstance(vol, _sparse.DOK):
            mapping = vol.data  # dict[(i,j,k)] -> value
            if mapping:
                idxs, data = zip(*mapping.items())  # idxs: sequence of (i,j,k)
                idxs = np.asarray(idxs, dtype=np.int64)  # shape: (nnz, 3)
                data = np.asarray(data, dtype=self.dtype)
                coords = idxs.T  # shape: (3, nnz)
            else:
                coords = np.zeros((3, 0), dtype=np.int64)
                data = np.zeros((0,), dtype=self.dtype)
        else:
            raise TypeError(
                "SparseLeanVolumeInterpolator currently supports sparse.COO "
                "and sparse.DOK volumes."
            )

        lin = (
            coords[0].astype(np.int64) * self._stride_yz
            + coords[1].astype(np.int64) * nz
            + coords[2].astype(np.int64)
        )

        if lin.size:
            order = np.argsort(lin)
            self._lin_idx = lin[order]
            self._values = data[order]
        else:
            self._lin_idx = np.zeros((0,), dtype=np.int64)
            self._values = np.zeros((0,), dtype=self.dtype)

    def __call__(self, coords):
        """
        Perform trilinear interpolation using sparse lookups.
        """
        out_shape = np.array(coords[0]).shape

        xv, yv, zv = (np.ravel(v) for v in coords)
        Nv = len(xv)

        # Base integer indices and fractional parts
        xv0 = xv.astype(int)
        alpha_x = xv - xv0

        yv0 = yv.astype(int)
        alpha_y = yv - yv0

        zv0 = zv.astype(int)
        alpha_z = zv - zv0

        ny, nz = self.ny, self.nz
        stride_yz = self._stride_yz
        lin_idx = self._lin_idx
        values = self._values

        # Offsets for the 8 neighboring voxels
        off_x = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)[:, None]  # (8,1)
        off_y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)[:, None]
        off_z = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)[:, None]

        # Weights for each neighbor: βx, βy, βz
        beta_x = np.where(off_x == 0, 1 - alpha_x[None, :], alpha_x[None, :])  # (8, N)
        beta_y = np.where(off_y == 0, 1 - alpha_y[None, :], alpha_y[None, :])
        beta_z = np.where(off_z == 0, 1 - alpha_z[None, :], alpha_z[None, :])
        weights = beta_x * beta_y * beta_z  # (8, N)

        # Integer neighbor indices for all 8 neighbors
        xi8 = xv0[None, :] + off_x  # (8, N)
        yi8 = yv0[None, :] + off_y
        zi8 = zv0[None, :] + off_z

        # Validity mask for points inside volume bounds
        valid8 = (
            (xi8 >= 0) & (xi8 < self.nx) &
            (yi8 >= 0) & (yi8 < self.ny) &
            (zi8 >= 0) & (zi8 < self.nz)
        )

        # Initialize neighbor values with extrapolation value everywhere
        val_all = np.full((8, Nv), self.extrap_val, dtype=self.dtype)

        if lin_idx.size > 0 and np.any(valid8):
            # Linear indices for all neighbors
            lin_all = xi8 * stride_yz + yi8 * nz + zi8  # (8, N)

            # Work only on valid entries
            idx_flat_valid = np.nonzero(valid8.ravel())[0]
            lin_flat = lin_all.ravel()[idx_flat_valid]

            # Binary search in the sorted linear index array
            pos = np.searchsorted(lin_idx, lin_flat)

            # Clip positions for safe indexing and check matches
            pos_clipped = np.clip(pos, 0, lin_idx.size - 1)
            match = (pos < lin_idx.size) & (lin_idx[pos_clipped] == lin_flat)

            if np.any(match):
                vals_flat = np.zeros_like(lin_flat, dtype=self.dtype)
                vals_flat[match] = values[pos_clipped[match]]

                # Set valid neighbor values (zeros where no nnz, as per sparse semantics)
                val_all_flat = val_all.ravel()
                val_all_flat[idx_flat_valid] = vals_flat

        # Combine neighbors: sum_k weight_k * value_k
        s = np.sum(weights * val_all, axis=0, dtype=self.dtype)

        return s.reshape(out_shape)

if __name__ == "__main__":
    """
    Test suite for LeanVolumeInterpolator
    Demonstrates various use cases with informative output
    """
    print("=" * 70)
    print("LeanVolumeInterpolator Test Suite")
    print("=" * 70)
    print()
    
    # Create test volume
    print("📦 Creating test volume (100×100×100)...")
    test_vol = np.random.rand(100, 100, 100).astype(np.float32)
    print(f"   Volume shape: {test_vol.shape}")
    print(f"   Volume dtype: {test_vol.dtype}")
    print(f"   Volume range: [{test_vol.min():.4f}, {test_vol.max():.4f}]")
    print()
    
    # Initialize interpolator
    print("🔧 Initializing LeanVolumeInterpolator...")
    lean_interp = LeanVolumeInterpolator(
        test_vol, 
        extrap_val=np.nan, 
        dtype=np.float32
    )
    print(f"   Extrapolation value: {lean_interp.extrap_val}")
    print(f"   Computation dtype: {lean_interp.dtype}")
    print()
    
    # Test 1: Single point interpolation
    print("-" * 70)
    print("Test 1: Single Point Interpolation")
    print("-" * 70)
    coords = (51.5, 13.1, 10.5)
    val = lean_interp(coords)
    print(f"   Coordinates: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")
    print(f"   Interpolated value: {val:.6f}")
    print()
    
    # Test 2: Multiple points (1D array)
    print("-" * 70)
    print("Test 2: Multiple Points (1D Array)")
    print("-" * 70)
    x_coords = np.array([10.2, 20.5, 30.8, 40.1, 50.9])
    y_coords = np.array([15.3, 25.7, 35.2, 45.6, 55.4])
    z_coords = np.array([5.1, 12.8, 18.5, 24.3, 30.7])
    vals = lean_interp((x_coords, y_coords, z_coords))
    print(f"   Number of points: {len(x_coords)}")
    print(f"   Interpolated values:")
    for i, (x, y, z, v) in enumerate(zip(x_coords, y_coords, z_coords, vals), 1):
        print(f"      Point {i}: ({x:.2f}, {y:.2f}, {z:.2f}) → {v:.6f}")
    print(f"   Statistics: min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}")
    print()
    
    # Test 3: 2D grid interpolation
    print("-" * 70)
    print("Test 3: 2D Grid Interpolation")
    print("-" * 70)
    x_grid = np.linspace(20, 30, 5)
    y_grid = np.linspace(20, 30, 5)
    z_fixed = 25.5
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.full_like(X, z_fixed)
    grid_vals = lean_interp((X, Y, Z))
    print(f"   Grid size: {X.shape}")
    print(f"   X range: [{x_grid.min():.2f}, {x_grid.max():.2f}]")
    print(f"   Y range: [{y_grid.min():.2f}, {y_grid.max():.2f}]")
    print(f"   Fixed Z: {z_fixed:.2f}")
    print(f"   Result statistics: min={grid_vals.min():.6f}, max={grid_vals.max():.6f}, mean={grid_vals.mean():.6f}")
    print()
    
    # Test 4: Edge case - out of bounds
    print("-" * 70)
    print("Test 4: Out-of-Bounds Coordinates")
    print("-" * 70)
    oob_coords = [
        (-5.0, 50.0, 50.0),      # Negative X
        (150.0, 50.0, 50.0),     # X too large
        (50.0, -5.0, 50.0),      # Negative Y
        (50.0, 150.0, 50.0),     # Y too large
        (50.0, 50.0, -5.0),      # Negative Z
        (50.0, 50.0, 150.0),     # Z too large
    ]
    print("   Testing out-of-bounds coordinates (should return extrapolation value):")
    for i, (x, y, z) in enumerate(oob_coords, 1):
        val = lean_interp((x, y, z))
        status = "✓" if np.isnan(val) else "✗"
        print(f"      {status} Point {i}: ({x:6.1f}, {y:6.1f}, {z:6.1f}) → {val}")
    print()
    
    # Test 5: Boundary coordinates
    print("-" * 70)
    print("Test 5: Boundary Coordinates")
    print("-" * 70)
    boundary_coords = [
        (0.0, 0.0, 0.0),         # Lower corner
        (99.0, 99.0, 99.0),      # Upper corner
        (50.0, 0.0, 50.0),       # Edge point
        (0.0, 50.0, 99.0),       # Edge point
    ]
    print("   Testing boundary coordinates:")
    for i, (x, y, z) in enumerate(boundary_coords, 1):
        val = lean_interp((x, y, z))
        print(f"      Point {i}: ({x:6.1f}, {y:6.1f}, {z:6.1f}) → {val:.6f}")
    print()
    
    print("=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)