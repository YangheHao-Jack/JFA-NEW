import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cucim.core.operations.morphology import distance_transform_edt

# Device setup
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize grid with seeds for 2D or 3D
def initialize_seed_grid(shape, seed_positions, device):
    grid = torch.full(shape, float('inf'), device=device)
    for pos in seed_positions:
        grid[pos] = 0
    return grid
def joint_histogram_comparison(map1, map2, title1, title2):
    plt.figure(figsize=(6, 6))
    plt.hist2d(map1.flatten().cpu().numpy(), map2.flatten().cpu().numpy(), bins=100, cmap='viridis')
    plt.xlabel(title1)
    plt.ylabel(title2)
    plt.title(f"Joint Histogram of {title1} vs {title2}")
    plt.colorbar()
    plt.show()

import torch.nn.functional as F

# 2D Jump Flood Algorithm using max-pooling
def jump_flood_2D(grid, max_iter=100, device=device_cpu, debug=False):
    height, width = grid.shape
    device = grid.device
    grid = grid.to(device)

    # Calculate the initial jump size as the largest power of two less than max dimension
    jump = 1 << (max(height, width).bit_length() - 1)

    # Jump Flooding Algorithm (JFA) stage
    while jump > 0:
        shifts = [
            (-jump, -jump), (-jump, 0), (-jump, jump),
            (0, -jump),             (0, jump),
            (jump, -jump), (jump, 0), (jump, jump),
        ]
        rolled_grids = []
        for dy, dx in shifts:
            rolled_grid = torch.roll(grid, shifts=(dy, dx), dims=(0, 1))
            rolled_grids.append(rolled_grid)

        rolled_grids = torch.stack(rolled_grids, dim=0)
        distances = torch.tensor(
            [torch.sqrt(torch.tensor(dx**2 + dy**2, dtype=torch.float32, device=device)) for dy, dx in shifts],
            device=device
        ).view(-1, 1, 1)
        updated_grids = rolled_grids + distances
        min_grid, _ = torch.min(updated_grids, dim=0)
        grid = torch.min(grid, min_grid)
        jump //= 2

    # Local Refinement Stage
    kernel_size = 5 # Size of the local window (can adjust for accuracy vs. performance)
    padding = kernel_size // 2
    refinement_passes=100
    for _ in range(refinement_passes):
        # Extract a local window around each pixel
        padded_grid = F.pad(grid.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='replicate')
        unfolded_grid = F.unfold(padded_grid, kernel_size=kernel_size)
        unfolded_grid = unfolded_grid.squeeze(0)  # Shape: (kernel_size^2, height * width)

        # Compute exact distances within the local window
        offsets = torch.tensor(
            [[dy, dx] for dy in range(-padding, padding + 1) for dx in range(-padding, padding + 1)],
            dtype=torch.float32, device=device
        )
        distances = torch.sqrt(offsets[:, 0]**2 + offsets[:, 1]**2).view(-1, 1)  # Shape: (kernel_size^2, 1)
        exact_distances = unfolded_grid + distances  # Add distances to unfolded values

        # Take the minimum distance for each pixel
        refined_grid = exact_distances.min(dim=0).values.view(height, width)
        grid = torch.min(grid, refined_grid)  # Update grid with refined distances

    return grid

# 3D Jump Flood Algorithm using max-pooling
def jump_flood_3D(grid, max_iter=5, device=device_cpu, debug=False):
    depth, height, width = grid.shape
    device = grid.device
    grid = grid.to(device)

    # Calculate the initial jump size
    jump = 1 << (max(depth, height, width).bit_length() - 1)

    # Precompute distances for each jump level
    distances = {}
    j = jump
    while j > 0:
        # Compute distances for all relevant shifts at this jump level
        shifts = [
            (dz, dy, dx)
            for dz in [-j, 0, j]
            for dy in [-j, 0, j]
            for dx in [-j, 0, j]
            if not (dz == 0 and dy == 0 and dx == 0)
        ]
        for dz, dy, dx in shifts:
            distance = torch.sqrt(torch.tensor(dz**2 + dy**2 + dx**2, dtype=torch.float32, device=device))
            distances[(dz, dy, dx)] = distance
        j //= 2

    # JFA propagation
    j = jump
    while j > 0:
        shifts = [
            (dz, dy, dx)
            for dz in [-j, 0, j]
            for dy in [-j, 0, j]
            for dx in [-j, 0, j]
            if not (dz == 0 and dy == 0 and dx == 0)
        ]
        for dz, dy, dx in shifts:
            rolled_grid = torch.roll(grid, shifts=(dz, dy, dx), dims=(0, 1, 2))
            distance = distances[(dz, dy, dx)]
            updated_grid = rolled_grid + distance
            grid = torch.min(grid, updated_grid)
        j //= 2

    return grid

# Exact Distance Transform Calculation
def compute_exact_distance_map(shape, seed_positions):
    coords = torch.stack(torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]), indexing='ij'
    ), dim=-1)
    exact_map = torch.full(shape, float('inf'))
    for pos in seed_positions:
        seed_coord = torch.tensor(pos)
        distances = torch.sqrt(((coords - seed_coord) ** 2).sum(dim=-1).float())
        exact_map = torch.min(exact_map, distances)
    return exact_map

# Benchmarking 2D or 3D JFA and cuCIM
def benchmark_jfa_with_cucim(shape, seed_positions, max_iter=500, dim=2):
    # Initialize seed grids for CPU and GPU
    grid_cpu = initialize_seed_grid(shape, seed_positions, device_cpu)
    grid_gpu = initialize_seed_grid(shape, seed_positions, device_gpu)
    
    # JFA on CPU
    start = time.time()
    if dim == 2:
        distance_map_cpu = jump_flood_2D(grid_cpu, max_iter=max_iter, device=device_cpu)
    else:
        distance_map_cpu = jump_flood_3D(grid_cpu, max_iter=max_iter, device=device_cpu)
    cpu_time = time.time() - start
    
    # JFA on GPU
    start = time.time()
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    if dim == 2:
        distance_map_gpu = jump_flood_2D(grid_gpu, max_iter=max_iter, device=device_gpu)
    else:
        distance_map_gpu = jump_flood_3D(grid_gpu, max_iter=max_iter, device=device_gpu)
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # Exact Distance Transform on CPU
    exact_distance_map = compute_exact_distance_map(shape, seed_positions)
    
    # cuCIM Distance Transform on GPU with initial grid of cp.inf
    cucim_init_grid = cp.full(shape, cp.inf, dtype=cp.float64)
    for pos in seed_positions:
        cucim_init_grid[pos] = 0  # Set seed positions to 0
    
    start = time.time()
    cucim_distance_map = distance_transform_edt(cucim_init_grid, float64_distances=False)
    cucim_time = time.time() - start
    cucim_distance_map = torch.tensor(cp.asnumpy(cucim_distance_map), device=device_gpu)
    
    # Replace remaining inf values in distance maps with a large finite value
    max_possible_distance = np.sqrt(sum(d**2 for d in shape))
    distance_map_cpu = torch.where(torch.isinf(distance_map_cpu), torch.tensor(max_possible_distance), distance_map_cpu)
    distance_map_gpu = torch.where(torch.isinf(distance_map_gpu), torch.tensor(max_possible_distance), distance_map_gpu)
    cucim_distance_map = torch.where(torch.isinf(cucim_distance_map), torch.tensor(max_possible_distance), cucim_distance_map)
    exact_distance_map = torch.where(torch.isinf(exact_distance_map), torch.tensor(max_possible_distance), exact_distance_map)
    
    # Accuracy Calculations
    mse_cpu = torch.mean((distance_map_cpu - exact_distance_map) ** 2).item()
    mse_gpu = torch.mean((distance_map_gpu.cpu() - exact_distance_map) ** 2).item()
    mse_cucim = torch.mean((cucim_distance_map.cpu() - exact_distance_map) ** 2).item()
    
    # Report results
    print(f"JFA CPU Execution Time: {cpu_time:.4f} seconds")
    print(f"JFA GPU Execution Time: {gpu_time:.4f} seconds")
    print(f"cuCIM Execution Time: {cucim_time:.4f} seconds")
    print(f"Mean Squared Error (CPU JFA vs Exact): {mse_cpu:.4f}")
    print(f"Mean Squared Error (GPU JFA vs Exact): {mse_gpu:.4f}")
    print(f"Mean Squared Error (cuCIM vs Exact): {mse_cucim:.4f}")
    
    return distance_map_cpu, distance_map_gpu, cucim_distance_map, exact_distance_map, mse_cpu, mse_gpu, mse_cucim

# Visualization for 2D slice comparisons
def visualize_comparisons(cpu_map, gpu_map, cucim_map, exact_map, seed_positions, slice_idx=32):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"2D Slices of 3D Distance Maps at Depth Slice {slice_idx}")

    # CPU JFA Slice
    axes[0].imshow(cpu_map[slice_idx].cpu().numpy(), cmap='viridis')
    axes[0].set_title("CPU JFA Approximation")
    for pos in seed_positions:
        if pos[0] == slice_idx:
            axes[0].plot(pos[2], pos[1], 'ro', markersize=5)
    
    # GPU JFA Slice
    axes[1].imshow(gpu_map[slice_idx].cpu().numpy(), cmap='viridis')
    axes[1].set_title("GPU JFA Approximation")
    for pos in seed_positions:
        if pos[0] == slice_idx:
            axes[1].plot(pos[2], pos[1], 'ro', markersize=5)
    
    # cuCIM Exact EDT Slice
    axes[2].imshow(cucim_map[slice_idx].cpu().numpy(), cmap='viridis')
    axes[2].set_title("cuCIM Exact Distance Transform")
    for pos in seed_positions:
        if pos[0] == slice_idx:
            axes[2].plot(pos[2], pos[1], 'ro', markersize=5)

    # Exact EDT Slice
    axes[3].imshow(exact_map[slice_idx].numpy(), cmap='viridis')
    axes[3].set_title("Exact Euclidean Distance Transform")
    for pos in seed_positions:
        if pos[0] == slice_idx:
            axes[3].plot(pos[2], pos[1], 'ro', markersize=5)
    
    plt.show()

# Visualization for seed positions in 3D
def visualize_seeds_in_3D(shape, seed_positions):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each seed as a scatter point in 3D
    x_coords, y_coords, z_coords = zip(*seed_positions)
    ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o', s=50, label="Seeds")
    
    # Set limits based on grid shape
    ax.set_xlim(0, shape[2] - 1)
    ax.set_ylim(0, shape[1] - 1)
    ax.set_zlim(0, shape[0] - 1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Seed Positions in Grid")
    ax.legend()
    plt.show()

# Generate random seed positions
def generate_random_seeds(shape, num_seeds):
    return [(np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2])) for _ in range(num_seeds)]

# Benchmark for complex cases
def benchmark_complex_cases_with_cucim(cases):
    for i, case in enumerate(cases):
        shape = case["shape"]
        max_iter = case["max_iter"]
        num_seeds = case["seeds"]
        seed_positions = generate_random_seeds(shape, num_seeds)
        
        # Print case details
        print(f"\n--- Running Case {i + 1} ---")
        print(f"Grid Shape: {shape}, Seeds: {num_seeds}, Max Iterations: {max_iter}")
        
        # Run the benchmark
        dim = len(shape)
        cpu_map, gpu_map, cucim_map, exact_map, mse_cpu, mse_gpu, mse_cucim = benchmark_jfa_with_cucim(shape, seed_positions, max_iter, dim=dim)
        
        # Visualize 2D slice comparisons and 3D seed positions for non-2D cases
        if dim == 3:
            visualize_comparisons(cpu_map, gpu_map, cucim_map, exact_map, seed_positions, slice_idx=shape[0] // 2)
            visualize_seeds_in_3D(shape, seed_positions)
            joint_histogram_comparison(cpu_map, exact_map, "3D CPU JFA", "Exact EDT")
            joint_histogram_comparison(gpu_map, exact_map, "3D GPU JFA", "Exact EDT")
            joint_histogram_comparison(cucim_map, exact_map, "3D cuCIM", "Exact EDT")
            joint_histogram_comparison(cpu_map, cucim_map, "3D CPU JFA", "cuCIM")
            joint_histogram_comparison(gpu_map, cucim_map, "3D GPU JFA", "cuCIM")

# Define configurations for complex cases
complex_cases = [
    #{"shape": (128, 128, 128), "seeds": 20, "max_iter": 7},
    #{"shape": (256, 256, 256), "seeds": 50, "max_iter": 8},
    {"shape": (128, 128, 128), "seeds": 20, "max_iter": 1000},  # Non-cubic case
]



# Exact 2D Distance Transform Calculation
def compute_exact_distance_map_2D(shape, seed_positions):
    coords = torch.stack(torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij'
    ), dim=-1)
    exact_map = torch.full(shape, float('inf'))
    for pos in seed_positions:
        seed_coord = torch.tensor(pos)
        distances = torch.sqrt(((coords - seed_coord) ** 2).sum(dim=-1).float())
        exact_map = torch.min(exact_map, distances)
    return exact_map

# Benchmarking 2D JFA and cuCIM
def benchmark_jfa_2d_with_cucim(shape, seed_positions, max_iter=500):
    # Initialize seed grids for CPU and GPU
    grid_cpu = initialize_seed_grid(shape, seed_positions, device_cpu)
    grid_gpu = initialize_seed_grid(shape, seed_positions, device_gpu)
    
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
    
    # Exact Distance Transform on CPU
    exact_distance_map = compute_exact_distance_map_2D(shape, seed_positions)
    
    # cuCIM Distance Transform on GPU with initial grid of cp.inf
    cucim_init_grid = cp.full(shape, cp.inf, dtype=cp.float64)
    for pos in seed_positions:
        cucim_init_grid[pos] = 0  # Set seed positions to 0
    
    start = time.time()
    cucim_distance_map = distance_transform_edt(cucim_init_grid, float64_distances=True)
    cucim_time = time.time() - start
    cucim_distance_map = torch.tensor(cp.asnumpy(cucim_distance_map), device=device_gpu)
    
    # Replace remaining inf values in distance maps with a large finite value
    max_possible_distance = np.sqrt(sum(d**2 for d in shape))
    distance_map_cpu = torch.where(torch.isinf(distance_map_cpu), torch.tensor(max_possible_distance), distance_map_cpu)
    distance_map_gpu = torch.where(torch.isinf(distance_map_gpu), torch.tensor(max_possible_distance), distance_map_gpu)
    cucim_distance_map = torch.where(torch.isinf(cucim_distance_map), torch.tensor(max_possible_distance), cucim_distance_map)
    exact_distance_map = torch.where(torch.isinf(exact_distance_map), torch.tensor(max_possible_distance), exact_distance_map)
    
    # Accuracy Calculations
    mse_cpu = torch.mean((distance_map_cpu - exact_distance_map) ** 2).item()
    mse_gpu = torch.mean((distance_map_gpu.cpu() - exact_distance_map) ** 2).item()
    mse_cucim = torch.mean((cucim_distance_map.cpu() - exact_distance_map) ** 2).item()
    
    # Report results
    print(f"2D JFA CPU Execution Time: {cpu_time:.4f} seconds")
    print(f"2D JFA GPU Execution Time: {gpu_time:.4f} seconds")
    print(f"2D cuCIM Execution Time: {cucim_time:.4f} seconds")
    print(f"Mean Squared Error (2D CPU JFA vs Exact): {mse_cpu:.4f}")
    print(f"Mean Squared Error (2D GPU JFA vs Exact): {mse_gpu:.4f}")
    print(f"Mean Squared Error (2D cuCIM vs Exact): {mse_cucim:.4f}")
    
    return distance_map_cpu, distance_map_gpu, cucim_distance_map, exact_distance_map, mse_cpu, mse_gpu, mse_cucim

# Visualization for 2D distance map comparisons
def visualize_2d_comparisons(cpu_map, gpu_map, cucim_map, exact_map, seed_positions):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("2D Distance Map Comparisons")

    # CPU JFA
    axes[0].imshow(cpu_map.cpu().numpy(), cmap='viridis')
    axes[0].set_title("2D CPU JFA Approximation")
    for pos in seed_positions:
        axes[0].plot(pos[1], pos[0], 'ro', markersize=5)
    
    # GPU JFA
    axes[1].imshow(gpu_map.cpu().numpy(), cmap='viridis')
    axes[1].set_title("2D GPU JFA Approximation")
    for pos in seed_positions:
        axes[1].plot(pos[1], pos[0], 'ro', markersize=5)
    
    # cuCIM Exact EDT
    axes[2].imshow(cucim_map.cpu().numpy(), cmap='viridis')
    axes[2].set_title("2D cuCIM Exact Distance Transform")
    for pos in seed_positions:
        axes[2].plot(pos[1], pos[0], 'ro', markersize=5)
    
    # Exact EDT
    axes[3].imshow(exact_map.numpy(), cmap='viridis')
    axes[3].set_title("2D Exact Euclidean Distance Transform")
    for pos in seed_positions:
        axes[3].plot(pos[1], pos[0], 'ro', markersize=5)
    
    plt.show()

# Define parameters for 2D example
shape_2D = (256, 256)  # Size of the 2D grid
seed_positions_2D = [(128, 128), (50, 50), (200, 200), (50, 200), (200, 50)]

# Run 2D benchmark and obtain maps
cpu_map_2d, gpu_map_2d, cucim_map_2d, exact_map_2d, mse_cpu_2d, mse_gpu_2d, mse_cucim_2d = benchmark_jfa_2d_with_cucim(shape_2D, seed_positions_2D)

# Visualize comparisons for 2D case
visualize_2d_comparisons(cpu_map_2d, gpu_map_2d, cucim_map_2d, exact_map_2d, seed_positions_2D)
# Run the benchmark for all complex cases with cuCIM



# Generate joint histograms for quantitative comparisons
joint_histogram_comparison(cpu_map_2d, exact_map_2d, "CPU JFA", "Exact EDT")
joint_histogram_comparison(gpu_map_2d, exact_map_2d, "GPU JFA", "Exact EDT")
joint_histogram_comparison(cucim_map_2d, exact_map_2d, "cuCIM", "Exact EDT")
joint_histogram_comparison(cpu_map_2d, cucim_map_2d, "CPU JFA", "cuCIM")
joint_histogram_comparison(gpu_map_2d, cucim_map_2d, "GPU JFA", "cuCIM")


benchmark_complex_cases_with_cucim(complex_cases) 


