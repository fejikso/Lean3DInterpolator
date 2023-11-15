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

class LeanVolumeInterpolator():
    """
    A class for efficient trilinear interpolation on large volumetric datasets.

    This class aims to replace scipy.interpolate.RegularGridInterpolator, offering improved performance with large volumes, such as numpy memmaps, sparse volumes, or HDF5 datasets.

    Attributes:
        shape (tuple): Shape of the input volume.
        dtype (data-type): Data type for computation, default is np.float32.
        vol (array-like): The input 3D volume for interpolation.
        extrap_val (float): Extrapolation value for out-of-bound coordinates.
        to_dense (bool): Flag to convert sparse input volume to dense, default is False. Only needed if the volume is a 3D sparse volume.
    
    Methods:
        __call__(coords): Interpolates the volume at the given coordinates.
    """

    def __init__(self, vol, extrap_val=np.nan, dtype=np.float32, to_dense=False):
        """
        Initializes the LeanInterpolator object.

        Parameters:
            vol (array-like): A 3D volume for interpolation.
            extrap_val (float): Value used for extrapolation of out-of-bound coordinates. Defaults to np.nan.
            dtype (data-type): Data type for computations, defaults to np.float32.
            to_dense (bool): If True, converts a sparse volume to a dense numpy array. Defaults to False.
        """
        self.shape = vol.shape
        self.dtype = dtype
        self.vol = vol
        self.extrap_val = extrap_val
        self.to_dense = to_dense

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
                    
                    if self.to_dense:
                        val[idx_valid] = V[xi[idx_valid], yi[idx_valid], zi[idx_valid]].todense()
                    else:
                        val[idx_valid] = V[xi[idx_valid], yi[idx_valid], zi[idx_valid]]

                    s[:] += (βx*βy*βz)*val

        return s.reshape(out_shape)

if __name__ == "__main__":
    # Brief example usage of LeanInterpolator
    # code demonstrating how to use the class

    test_vol = np.random.rand((100,100,100))    

    lean_interp = LeanVolumeInterpolator(test_vol, extrap_val=np.nan, dtype=np.float32, to_dense=False)

    # will interpolate the value 
    val = lean_interp((51.5, 13.1, 10.5))
    print(val)