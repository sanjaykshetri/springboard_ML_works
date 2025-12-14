# Euclidean and Manhattan Distance Case Study

**Author:** Sanjay Kumar Chhetri  
**Date:** December 14, 2025  
**Project:** Springboard Data Science Course  

A comprehensive analysis and comparison of Euclidean and Manhattan distance metrics for clustering and similarity analysis.

## Project Structure

```
euc_manhattan_distance_case_study/
├── data/                               # Dataset files
├── notebooks/                          # Jupyter notebooks for analysis  
├── src/                               # Source code modules
├── plots/                             # Generated visualizations
├── results/                           # Output files and analysis results
├── tests/                             # Unit tests
├── docs/                              # Documentation
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── Euclidean_and_Manhattan_Distances_Case_Study.ipynb  # Main analysis notebook
```

## Overview

This case study explores two fundamental distance metrics in data science:

### **Euclidean Distance**
- **Definition**: Straight-line distance between two points in Euclidean space
- **Formula**: √[(x₁-x₂)² + (y₁-y₂)² + ... + (zₙ-zₙ)²]
- **Use Cases**: K-means clustering, nearest neighbor analysis, similarity measurement

### **Manhattan Distance (L1 Distance)**  
- **Definition**: Sum of absolute differences between coordinates
- **Formula**: |x₁-x₂| + |y₁-y₂| + ... + |zₙ-zₙ|
- **Use Cases**: Grid-based pathfinding, feature selection, robust clustering

## Key Analyses

1. **Distance Calculations**: Implementation of both metrics from scratch
2. **Comparative Analysis**: Side-by-side comparison on sample datasets
3. **Clustering Applications**: K-means vs K-medoids clustering comparison
4. **Visualization**: 2D and 3D plots showing distance contours
5. **Performance Metrics**: Computational efficiency and accuracy analysis

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

## Key Findings

- **Euclidean Distance**: Better for continuous, normally distributed data
- **Manhattan Distance**: More robust to outliers and high-dimensional sparse data
- **Clustering Performance**: Context-dependent based on data characteristics
- **Computational Cost**: Manhattan distance generally faster for high dimensions

## Applications

- **Machine Learning**: Distance-based algorithms (KNN, K-means)
- **Computer Vision**: Image similarity and pattern matching
- **Recommendation Systems**: User/item similarity calculations
- **Geographic Analysis**: Route planning and location services

## Dependencies

- Python 3.8+
- NumPy
- Pandas  
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter

## License

This project is part of the Springboard Data Science coursework.

## Contact

**Sanjay Kumar Chhetri**  
Springboard Data Science Student