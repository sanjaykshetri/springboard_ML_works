"""
Visualization utilities for distance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import pandas as pd


def plot_distance_comparison(points: np.ndarray, 
                           reference_point: np.ndarray,
                           title: str = "Distance Comparison") -> plt.Figure:
    """
    Create visualization comparing Euclidean and Manhattan distances.
    
    Args:
        points: Array of points to plot
        reference_point: Reference point for distance calculation
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    from .distance_metrics import euclidean_distance, manhattan_distance
    
    # Calculate distances
    euc_distances = [euclidean_distance(point, reference_point) for point in points]
    man_distances = [manhattan_distance(point, reference_point) for point in points]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scatter plot with Euclidean distances
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], 
                          c=euc_distances, cmap='viridis', s=50)
    ax1.scatter(reference_point[0], reference_point[1], 
               c='red', s=200, marker='*', label='Reference')
    ax1.set_title('Euclidean Distance')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    fig.colorbar(scatter1, ax=ax1, label='Distance')
    
    # Scatter plot with Manhattan distances
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], 
                          c=man_distances, cmap='viridis', s=50)
    ax2.scatter(reference_point[0], reference_point[1], 
               c='red', s=200, marker='*', label='Reference')
    ax2.set_title('Manhattan Distance')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    fig.colorbar(scatter2, ax=ax2, label='Distance')
    
    # Comparison scatter plot
    ax3.scatter(euc_distances, man_distances, alpha=0.6)
    ax3.plot([min(euc_distances), max(euc_distances)], 
             [min(euc_distances), max(euc_distances)], 
             'r--', label='y=x')
    ax3.set_xlabel('Euclidean Distance')
    ax3.set_ylabel('Manhattan Distance')
    ax3.set_title('Distance Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_distance_contours(center: Tuple[float, float] = (0, 0),
                          max_dist: float = 5,
                          resolution: int = 100) -> plt.Figure:
    """
    Plot distance contours for Euclidean and Manhattan metrics.
    
    Args:
        center: Center point for contours
        max_dist: Maximum distance to plot
        resolution: Grid resolution
        
    Returns:
        matplotlib Figure object
    """
    from .distance_metrics import euclidean_distance, manhattan_distance
    
    # Create grid
    x = np.linspace(center[0] - max_dist, center[0] + max_dist, resolution)
    y = np.linspace(center[1] - max_dist, center[1] + max_dist, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distances for each point in grid
    euclidean_grid = np.zeros((resolution, resolution))
    manhattan_grid = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            point = [X[i, j], Y[i, j]]
            euclidean_grid[i, j] = euclidean_distance(point, center)
            manhattan_grid[i, j] = manhattan_distance(point, center)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Euclidean contours
    contour1 = ax1.contour(X, Y, euclidean_grid, levels=10, colors='blue')
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.contourf(X, Y, euclidean_grid, levels=50, alpha=0.6, cmap='Blues')
    ax1.scatter(*center, c='red', s=100, marker='*')
    ax1.set_title('Euclidean Distance Contours')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Manhattan contours
    contour2 = ax2.contour(X, Y, manhattan_grid, levels=10, colors='green')
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.contourf(X, Y, manhattan_grid, levels=50, alpha=0.6, cmap='Greens')
    ax2.scatter(*center, c='red', s=100, marker='*')
    ax2.set_title('Manhattan Distance Contours')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_clustering_comparison(points: np.ndarray, 
                             labels_euclidean: np.ndarray,
                             labels_manhattan: np.ndarray) -> plt.Figure:
    """
    Compare clustering results using different distance metrics.
    
    Args:
        points: Data points
        labels_euclidean: Cluster labels using Euclidean distance
        labels_manhattan: Cluster labels using Manhattan distance
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Euclidean-based clustering
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], 
                          c=labels_euclidean, cmap='tab10', s=50)
    ax1.set_title('Clustering with Euclidean Distance')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Manhattan-based clustering
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], 
                          c=labels_manhattan, cmap='tab10', s=50)
    ax2.set_title('Clustering with Manhattan Distance')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_performance_comparison(distances_euc: List[float],
                              distances_man: List[float],
                              times_euc: List[float],
                              times_man: List[float]) -> plt.Figure:
    """
    Plot performance comparison between distance metrics.
    
    Args:
        distances_euc: Euclidean distances
        distances_man: Manhattan distances  
        times_euc: Euclidean computation times
        times_man: Manhattan computation times
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distance correlation
    ax1.scatter(distances_euc, distances_man, alpha=0.6)
    min_val = min(min(distances_euc), min(distances_man))
    max_val = max(max(distances_euc), max(distances_man))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax1.set_xlabel('Euclidean Distance')
    ax1.set_ylabel('Manhattan Distance')
    ax1.set_title('Distance Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Computation time comparison
    metrics = ['Euclidean', 'Manhattan']
    times = [np.mean(times_euc), np.mean(times_man)]
    errors = [np.std(times_euc), np.std(times_man)]
    
    ax2.bar(metrics, times, yerr=errors, capsize=5)
    ax2.set_ylabel('Average Computation Time (seconds)')
    ax2.set_title('Performance Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig