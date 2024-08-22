## Stationary Distribution of the ELO Rating System for two players

### Description
This repository contains the implementation of the stationary distribution of the ELO rating system. The project focuses on the numerical approximation of the stationary cumulative distribution function (CDF) associated with the ELO rating system, utilizing various mathematical and computational methods.

### Features
- **Meshing Functions**: Tools for generating meshes and constructing the system matrix.
- **Newton-Raphson Method**: Implementation of the Newton-Raphson method to compute the inverse of the function `g`.
- **System Solver**: Methods to build and solve the linear system representing the ELO rating system to obtain the stationary distribution.
- **Grid Convergence Study**: Analysis of grid convergence to assess the accuracy of numerical approximations.
- **Statistical Analysis**: Computation of expected values, second moments, and variance.
- **Symmetry Tests**: Procedures for testing the symmetry of the stationary distribution.
- **Elo Conjecture Test**: Evaluation of Elo's conjecture regarding the relationship between expected match outcomes and rating differences.
- **ELO vs. Normal Distribution**: Comparison between the stationary distribution of the ELO system and the normal distribution.
- **Dependency Studies**: Investigation of the relationship between the mean, variance, and the adjustment factor \(K\).

### Dependencies
- NumPy
- Matplotlib
- SciPy
- tqdm

### Usage
The main script provides various analyses and visualizations related to the stationary distribution of the ELO rating system. To execute the code, ensure that all dependencies are installed and run the script in a Python environment.

### Author
David Man
