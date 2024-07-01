## Invariant Cumulative Function of the ELO Rating System

### Description
This repository contains the implementation of the invariant cumulative function of the ELO rating system, developed by David Man. The project involves the numerical solution of the ELO rating system's invariant cumulative function using various mathematical and computational techniques.

### Features
- **Meshing Functions**: Functions for meshing and construction of the system matrix.
- **Newton-Raphson Method**: Implementation of the Newton-Raphson method to find the inverse of function `g`.
- **System Solver**: Methods to construct and solve the linear system representing the ELO rating system in order to obtain the Elo CDF.
- **Statistical Analysis**: Calculation of expected values, second moments, and variance.
- **Symmetry Tests**: Tests for the symmetry of the ELO CDF.
- **Grid Convergence Study**: Analysis of grid convergence.
- **ELO vs. Normal Distribution**: Comparison of ELO CDF with the normal distribution CDF.
- **Residual Analysis**: Analysis of residuals between the ELO CDF and normal distribution.
- **Dependency Studies**: Examination of the independence of mean and variance in relation to the constant `k`.

### Dependencies
- NumPy
- Matplotlib
- SciPy
- tqdm

### Usage
The main script demonstrates various analyses and visualizations related to the ELO rating system. To run the code, ensure that all dependencies are installed and execute the script in a Python environment.

### Author
David Man
