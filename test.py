import numpy as np
from LeanVolumeInterpolator import LeanVolumeInterpolator

if __name__ == "__main__":
    # Example usage of LeanVolumeInterpolator

    # Create a random 3D volume
    test_vol = np.random.rand(100, 100, 100)    

    # Initialize the LeanVolumeInterpolator
    lint = LeanVolumeInterpolator(test_vol, extrap_val=np.nan, dtype=np.float32, to_dense=False)

    # Single coordinate interpolation
    single_val = lint((51.5, 13.1, 10.5))
    print(f"Interpolated value at a single coordinate (51.5, 13.1, 10.5): {single_val}")

    # Vector of coordinates interpolation
    x_coords = np.linspace(0, 99, 50)  # 50 evenly spaced coordinates along x
    y_coords = np.linspace(0, 99, 50)  # 50 evenly spaced coordinates along y
    z_coords = np.linspace(0, 99, 50)  # 50 evenly spaced coordinates along z

    vector_vals = lint((x_coords, y_coords, z_coords))
    print(f"Interpolated values at 50 vectorized coordinates: {vector_vals}")

    # Demonstrating efficiency with a larger set of coordinates
    large_x_coords = np.random.uniform(0, 99, 1000)
    large_y_coords = np.random.uniform(0, 99, 1000)
    large_z_coords = np.random.uniform(0, 99, 1000)

    large_vector_vals = lint((large_x_coords, large_y_coords, large_z_coords))
    print(f"Interpolated values at 1000 random coordinates: {large_vector_vals}")
