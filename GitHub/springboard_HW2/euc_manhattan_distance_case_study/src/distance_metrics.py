"""
Distance Metrics Implementation

This module contains implementations of various distance metrics
for use in clustering and similarity analysis.
"""

import numpy as np
from typing import Union, List, Tuple
import pandas as pd


def euclidean_distance(point1: Union[List, np.ndarray], 
                      point2: Union[List, np.ndarray]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point coordinates
        point2: Second point coordinates
        
    Returns:
        Euclidean distance as float
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    return np.sqrt(np.sum((point1 - point2) ** 2))


def manhattan_distance(point1: Union[List, np.ndarray], 
                      point2: Union[List, np.ndarray]) -> float:
    """
    Calculate Manhattan distance between two points.
    
    Args:
        point1: First point coordinates  
        point2: Second point coordinates
        
    Returns:
        Manhattan distance as float
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    return np.sum(np.abs(point1 - point2))


def minkowski_distance(point1: Union[List, np.ndarray], 
                      point2: Union[List, np.ndarray], 
                      p: float = 2) -> float:
    """
    Calculate Minkowski distance (generalized metric).
    
    Args:
        point1: First point coordinates
        point2: Second point coordinates  
        p: Order parameter (p=1: Manhattan, p=2: Euclidean)
        
    Returns:
        Minkowski distance as float
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)


def distance_matrix_euclidean(points: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise Euclidean distance matrix.
    
    Args:
        points: Array of points (n_points, n_features)
        
    Returns:
        Distance matrix (n_points, n_points)
    """
    n_points = len(points)
    distances = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = euclidean_distance(points[i], points[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
    return distances


def distance_matrix_manhattan(points: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise Manhattan distance matrix.
    
    Args:
        points: Array of points (n_points, n_features)
        
    Returns:
        Distance matrix (n_points, n_points)
    """
    n_points = len(points)
    distances = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = manhattan_distance(points[i], points[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
    return distances


class DistanceCalculator:
    """
    A class for calculating various distance metrics.
    """
    
    def __init__(self, metric: str = 'euclidean'):
        """
        Initialize distance calculator.
        
        Args:
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
        """
        self.metric = metric.lower()
        self.valid_metrics = ['euclidean', 'manhattan', 'minkowski']
        
        if self.metric not in self.valid_metrics:
            raise ValueError(f"Metric must be one of {self.valid_metrics}")
    
    def calculate(self, point1: Union[List, np.ndarray], 
                  point2: Union[List, np.ndarray], 
                  **kwargs) -> float:
        """
        Calculate distance using specified metric.
        
        Args:
            point1: First point coordinates
            point2: Second point coordinates
            **kwargs: Additional parameters (e.g., p for Minkowski)
            
        Returns:
            Distance as float
        """
        if self.metric == 'euclidean':
            return euclidean_distance(point1, point2)
        elif self.metric == 'manhattan':
            return manhattan_distance(point1, point2)
        elif self.metric == 'minkowski':
            p = kwargs.get('p', 2)
            return minkowski_distance(point1, point2, p)
    
    def distance_matrix(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate pairwise distance matrix.
        
        Args:
            points: Array of points (n_points, n_features)
            **kwargs: Additional parameters
            
        Returns:
            Distance matrix (n_points, n_points)
        """
        if self.metric == 'euclidean':
            return distance_matrix_euclidean(points)
        elif self.metric == 'manhattan':
            return distance_matrix_manhattan(points)
        elif self.metric == 'minkowski':
            p = kwargs.get('p', 2)
            n_points = len(points)
            distances = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = minkowski_distance(points[i], points[j], p)
                    distances[i, j] = dist
                    distances[j, i] = dist
                    
            return distances
    
    def find_nearest_neighbors(self, query_point: Union[List, np.ndarray], 
                              points: np.ndarray, 
                              k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to query point.
        
        Args:
            query_point: Query point coordinates
            points: Array of candidate points
            k: Number of neighbors to find
            
        Returns:
            Tuple of (distances, indices) of k nearest neighbors
        """
        distances = []
        
        for point in points:
            dist = self.calculate(query_point, point)
            distances.append(dist)
        
        distances = np.array(distances)
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        
        return nearest_distances, nearest_indices