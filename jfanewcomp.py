import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cucim.core.operations.morphology import distance_transform_edt

# Device setup
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize grid with seeds for 2D
def initialize_seed_grid_2D(shape, seed_positions, device):
    grid = torch.full(shape, float('inf'), device=device)
    for pos in seed_positions:
        grid[pos] = 0
    return grid

# Initialize grid with seeds for 3D
def initialize_seed_grid_3D(shape, seed_positions, device):
    grid = torch.full(shape, float('inf'), device=device)
    for pos in seed_positions:
        grid[pos] = 0
    return grid

import torch.nn.functional as F

# 2D Jump Flood Algorithm using max-pooling
def jump_flood_2D(grid, max_iter=100, device=device_cpu, debug=False):

    height, width = grid.shape
    device = grid.device
    grid = grid.to(device)

    # Calculate the initial jump size as the largest power of two less than max dimension
    jump = 1 << (max(height, width).bit_length() - 1)

    # Iterate until the jump size reduces to zero
    while jump > 0:
        for dy in [-jump, 0, jump]:
            for dx in [-jump, 0, jump]:
                if dy == 0 and dx == 0:
                    continue  # Skip the zero offset case

                # Roll the grid to get the neighbor's values
                rolled_grid = torch.roll(grid, shifts=(dy, dx), dims=(0, 1))

                # Compute new distances considering the Euclidean distance from the offset
                distance = (dy**2 + dx**2)**0.5
                updated_grid = rolled_grid + distance

                # Update the grid with the minimum distance encountered
                grid = torch.min(grid, updated_grid)

        # Halve the jump distance for the next iteration
        jump //= 2

    return grid
# 3D Jump Flood Algorithm using max-pooling
def jump_flood_3D(grid, max_iter=5, device=device_cpu, debug=False):
    depth, height, width = grid.shape
    device = grid.device
    grid = grid.to(device)

    # Calculate the initial jump size as the largest power of two less than max dimension
    jump = 1 << (max(depth, height, width).bit_length() - 1)

    # Iterate until the jump size reduces to zero
    while jump > 0:
        for dz in [-jump, 0, jump]:
            for dy in [-jump, 0, jump]:
                for dx in [-jump, 0, jump]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue  # Skip the zero offset case

                    # Roll the grid to get the neighbor's values
                    rolled_grid = torch.roll(grid, shifts=(dz, dy, dx), dims=(0, 1, 2))

                    # Compute new distances considering the Euclidean distance from the offset
                    distance = (dz**2 + dy**2 + dx**2)**0.5
                    updated_grid = rolled_grid + distance

                    # Update the grid with the minimum distance encountered
                    grid = torch.min(grid, updated_grid)

        # Halve the jump distance for the next iteration
        jump //= 2

    return grid
# Benchmarking 2D JFA and cuCIM
def benchmark_jfa_2d_with_cucim(shape, seed_positions, max_iter=10):
    # Initialize seed grids for CPU and GPU
    grid_cpu = initialize_seed_grid_2D(shape, seed_positions, device_cpu)
    grid_gpu = initialize_seed_grid_2D(shape, seed_positions, device_gpu)
    
    # JFA on CPU
    start = time.time()
    distance_map_cpu = jump_flood_2D(grid_cpu, max_iter=max_iter, device=device_cpu)
    cpu_time = time.time() - start
    
    # JFA on GPU
    start = time.time()
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    distance_map_gpu = jump_flood_2D(grid_gpu, max_iter=max_iter, device=device_gpu)
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # cuCIM Distance Transform on GPU with initial grid of cp.inf
    cucim_init_grid = cp.full(shape, cp.inf, dtype=cp.float64)
    for pos in seed_positions:
        cucim_init_grid[pos] = 0  # Set seed positions to 0
    
    start = time.time()
    cucim_distance_map = distance_transform_edt(cucim_init_grid, float64_distances=True)
    cucim_time = time.time() - start
    
    # Return the execution times
    return cpu_time, gpu_time, cucim_time

# Benchmarking 3D JFA and cuCIM
def benchmark_jfa_3d_with_cucim(shape, seed_positions, max_iter=5):
    # Initialize seed grids for CPU and GPU
    grid_cpu = initialize_seed_grid_3D(shape, seed_positions, device_cpu)
    grid_gpu = initialize_seed_grid_3D(shape, seed_positions, device_gpu)
    
    # JFA on CPU
    start = time.time()
    distance_map_cpu = jump_flood_3D(grid_cpu, max_iter=max_iter, device=device_cpu)
    cpu_time = time.time() - start
    
    # JFA on GPU
    start = time.time()
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    distance_map_gpu = jump_flood_3D(grid_gpu, max_iter=max_iter, device=device_gpu)
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # cuCIM Distance Transform on GPU with initial grid of cp.inf
    cucim_init_grid = cp.full(shape, cp.inf, dtype=cp.float64)
    for pos in seed_positions:
        cucim_init_grid[pos] = 0  # Set seed positions to 0
    
    start = time.time()
    cucim_distance_map = distance_transform_edt(cucim_init_grid, float64_distances=True)
    cucim_time = time.time() - start
    
    # Return the execution times
    return cpu_time, gpu_time, cucim_time

# Define spatial sizes and run benchmarks for 2D and 3D
spatial_sizes_2D = [64, 128, 256, 512, 1024]  # 2D spatial sizes
spatial_sizes_3D = [64, 128, 256, 512]        # 3D spatial sizes

# Example seed positions
seed_positions_2D = [(32, 32), (16, 48), (48, 16), (32, 48)]
seed_positions_3D = [(16, 16, 16), (8, 32, 24), (24, 8, 32), (16, 24, 8)]

# Store results for 2D and 3D
cpu_times_2D, gpu_times_2D, cucim_times_2D = [], [], []
cpu_times_3D, gpu_times_3D, cucim_times_3D = [], [], []

# Run benchmarks for 2D sizes
for size in spatial_sizes_2D:
    shape = (size, size)
    cpu_time, gpu_time, cucim_time = benchmark_jfa_2d_with_cucim(shape, seed_positions_2D, max_iter=10)
    cpu_times_2D.append(cpu_time)
    gpu_times_2D.append(gpu_time)
    cucim_times_2D.append(cucim_time)
    print(f"2D {size}x{size}: CPU = {cpu_time:.4f}s, GPU = {gpu_time:.4f}s, cuCIM = {cucim_time:.4f}s")

# Run benchmarks for 3D sizes
for size in spatial_sizes_3D:
    shape = (size, size, size)
    cpu_time, gpu_time, cucim_time = benchmark_jfa_3d_with_cucim(shape, seed_positions_3D, max_iter=5)
    cpu_times_3D.append(cpu_time)
    gpu_times_3D.append(gpu_time)
    cucim_times_3D.append(cucim_time)
    print(f"3D {size}x{size}x{size}: CPU = {cpu_time:.4f}s, GPU = {gpu_time:.4f}s, cuCIM = {cucim_time:.4f}s")

# Plotting the execution times for 2D and 3D
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Execution Time vs Spatial Size for Different Methods (2D and 3D)")

# Plot for 2D
axes[0].plot(spatial_sizes_2D, cpu_times_2D, 'o-', color='red', label="CPU JFA")
axes[0].plot(spatial_sizes_2D, gpu_times_2D, 'o-', color='purple', label="GPU JFA")
axes[0].plot(spatial_sizes_2D, cucim_times_2D, 'o-', color='green', label="cuCIM")
axes[0].set_xlabel("Spatial Size (2D)")
axes[0].set_ylabel("Execution Time (seconds)")
axes[0].set_title("2D images")
  # Set x-axis to log scale
axes[0].set_yscale("log") 
axes[0].legend()
axes[0].grid()

# Plot for 3D
axes[1].plot(spatial_sizes_3D, cpu_times_3D, 'o-', color='red', label="CPU JFA")
axes[1].plot(spatial_sizes_3D, gpu_times_3D, 'o-', color='purple', label="GPU JFA")
axes[1].plot(spatial_sizes_3D, cucim_times_3D, 'o-', color='green', label="cuCIM")
axes[1].set_xlabel("Spatial Size (3D)")
axes[1].set_ylabel("Execution Time (seconds)")
axes[1].set_title("3D volumes")
  # Set x-axis to log scale
axes[1].set_yscale("log") 
axes[1].legend()
axes[1].grid()

plt.show()
