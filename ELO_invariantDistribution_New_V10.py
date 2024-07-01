
'''
The invariant cumulative fonction 
of the ELO rating system

by David Man 
'''

#%% Meshing Functions and A Construction

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from tqdm import tqdm


# Function definitions for meshing
def b_bm(x):
    """
    Bonus-malus function for Elo rating system.
    
    Parameters:
    x (float): Rating difference.
    
    Returns:
    float: Calculated bonus-malus value.
    """
    return 1 / (1 + 10**(-x / 400))

def g(x, k):
    """
    Function g used in the Elo rating system update.
    
    Parameters:
    x (float): Rating difference.
    k (float): Constant k used in the update formula.
    
    Returns:
    float: Updated rating difference.
    """
    return x - 2 * k * b_bm(x)

def b_bm_prime(x):
    """
    Derivative of the bonus-malus function.
    
    Parameters:
    x (float): Rating difference.
    
    Returns:
    float: Derivative of the bonus-malus function.
    """
    return (np.log(10) * 10**(-x / 400)) / (400 * (1 + 10**(-x / 400))**2)

def g_prime(x, k):
    """
    Derivative of function g.
    
    Parameters:
    x (float): Rating difference.
    k (float): Constant k used in the update formula.
    
    Returns:
    float: Derivative of the g function.
    """
    return 1 - 2 * k * b_bm_prime(x)

def h(x, k, y_0, tolerance=1e-8, max_iterations=100):
    """
    Newton-Raphson method to find the inverse of function g.
    
    Parameters:
    x (float): Rating difference.
    k (float): Constant k used in the update formula.
    y_0 (float): Initial guess for the inverse.
    tolerance (float): Convergence tolerance.
    max_iterations (int): Maximum number of iterations.
    
    Returns:
    float: Inverse of g at x.
    
    Raises:
    ValueError: If the method does not converge.
    """
    y_n = y_0
    
    for i in range(max_iterations):
        g_n = g(y_n, k)
        g_n_prime = g_prime(y_n, k)
        
        # Newton-Raphson step
        y_nPlus1 = y_n - (g_n - x) / g_n_prime
        
        # Check for convergence
        if abs(y_nPlus1 - y_n) < tolerance:
            return y_nPlus1
        
        y_n = y_nPlus1
    
    raise ValueError("Newton-Raphson method did not converge")

def construct_A_with_source_term(x_mesh, k, p, q):
    """
    Construct the matrix A and source term b for the linear system.
    
    Parameters:
    x_mesh (ndarray): Discretized mesh of rating differences.
    k (float): Constant k used in the update formula.
    p (float): Probability of winning.
    q (float): Probability of losing.
    
    Returns:
    tuple: Matrix A and source term b.
    """
    mesh_size = len(x_mesh)
    A = np.zeros((mesh_size, mesh_size), dtype=np.float32)
    b = np.zeros(mesh_size, dtype=np.float32)
    
    F_boundary_lower = 0
    F_boundary_upper = 1

    for i in range(mesh_size):
        
        x_i = x_mesh[i]
        
        # Calculating h()
        h_x_i = h(x_i, k, x_i + k)
        h_x_i_minus_2k = h(x_i - 2*k, k, x_i - k)
        
        
        # Formule: F_x = p * F(h_x_i_minus_2k) + q * F(h_x_i)
        
        
        # Handle h_x_i_minus_2k within the mesh range
        if h_x_i_minus_2k < x_mesh[0]:
            b[i] += p * F_boundary_lower # Set lower boundary conditions for source term
            
        else:
            # Find x_a_prime, x_b_prime
            x_a_prime = np.max(x_mesh[x_mesh <= h_x_i_minus_2k])
            x_b_prime = np.min(x_mesh[x_mesh >= h_x_i_minus_2k])
            
            # Calculating the weight
            w_i_prime = (h_x_i_minus_2k - x_a_prime) / (x_b_prime - x_a_prime) if x_a_prime != x_b_prime else 0

            # Find the indices for x_a_prime, x_b_prime
            a_prime_idx = np.where(x_mesh == x_a_prime)[0][0]
            b_prime_idx = np.where(x_mesh == x_b_prime)[0][0]

            # Construct the matrix A
            A[i, a_prime_idx] += p * (1 - w_i_prime)
            A[i, b_prime_idx] += p * w_i_prime
            
            

        # Handle h_x_i within the mesh range
        if h_x_i > x_mesh[-1]:
            b[i] += q * F_boundary_upper # Set upper boundary conditions for source term
            
        else:
            # Find x_a, x_b
            x_a = np.max(x_mesh[x_mesh <= h_x_i])
            x_b = np.min(x_mesh[x_mesh >= h_x_i])
            
            # Calculating the weight
            w_i = (h_x_i - x_a) / (x_b - x_a) if x_a != x_b else 0
            
            # Find the indices for x_a, x_b
            a_idx = np.where(x_mesh == x_a)[0][0]
            b_idx = np.where(x_mesh == x_b)[0][0]
            
            # Construct the matrix A
            A[i, a_idx] += q * (1 - w_i)
            A[i, b_idx] += q * w_i



    return csr_matrix(A), b # A is sparse compressed


def solve_system(m, k, p, q, L):
    """
    Solve the linear system for a given number of grid points m dividing 2K.
    
    Parameters:
    m (int): Number of grid points of 2K
    k (float): Constant K used in the update formula.
    p (float): Probability of winning.
    q (float): Probability of losing.
    L (float): Limit for the range of rating differences.
    
    Returns:
    tuple: Discretized mesh and the solution F.
    """
    x_negative_limit = -L
    x_positive_limit = L
    delta_x = 2 * k / m
    
    x_mesh = np.arange(x_negative_limit, x_positive_limit, delta_x)
    
    A, b = construct_A_with_source_term(x_mesh, k, p, q)
    
    
    I = csr_matrix(np.eye(A.shape[0])) # converting to Compressed Sparse Row
    
    I_minus_A = I - A 
    
    F = spsolve(I_minus_A, b) # Using faster algorithm for solving the linear system that is sparse
    

    
    return x_mesh, F


# Calculate the expected value E[X]

def calculate_expected_value(x_mesh, F):
    # Exclude zero point from both sides
    negative_part = - np.trapz(F[x_mesh < 0], x_mesh[x_mesh < 0])
    positive_part = np.trapz(1 - F[x_mesh > 0], x_mesh[x_mesh > 0])
    
    # Check if 0 is part of x_mesh and handle its contribution equally
    zero_part = 0
    if 0 in x_mesh:
        zero_index = np.where(x_mesh == 0)[0][0]  # Find the index where x_mesh is 0
        dx = x_mesh[1] - x_mesh[0]  # Assuming uniform spacing
        zero_part = - 0.5 * F[zero_index] * dx + 0.5 * (1 - F[zero_index]) * dx   # Equally share the zero contribution
    
    E_X = negative_part + zero_part + positive_part
    return E_X

# Calculate second moment E[X**2]

def calculate_second_moment(x_mesh, F):
    # Exclude zero point from both sides
    negative_part = - np.trapz(2 * x_mesh[x_mesh < 0] *    F[x_mesh < 0],    x_mesh[x_mesh < 0])
    positive_part =   np.trapz(2 * x_mesh[x_mesh > 0] * (1 - F[x_mesh > 0]), x_mesh[x_mesh > 0])
    
    # Check if 0 is part of x_mesh and handle its contribution equally
    zero_part = 0
    if 0 in x_mesh:
        zero_index = np.where(x_mesh == 0)[0][0]  # Find the index where x_mesh is 0
        dx = x_mesh[1] - x_mesh[0]  # Assuming uniform spacing
        
        zero_part = - 0.5 * 2 * x_mesh[zero_index] *    F[zero_index]    * dx + \
                      0.5 * 2 * x_mesh[zero_index] * (1 - F[zero_index]) * dx
                    # Equally share the zero contribution
    
    E_X2 = negative_part + zero_part + positive_part
    return E_X2


def calculate_variance(E_X, E_X2):
    return E_X2 - E_X**2


# Function to compute E[b(X)] using the CDF F and x_mesh
def compute_E_b_X(x_mesh, F):
    # Exclude zero point from both sides
    negative_part = - np.trapz(F[x_mesh < 0] * b_bm_prime(x_mesh[x_mesh < 0]), x_mesh[x_mesh < 0])
    positive_part =   np.trapz((1 - F[x_mesh > 0]) * b_bm_prime(x_mesh[x_mesh > 0]), x_mesh[x_mesh > 0])
    
    # Check if 0 is part of x_mesh and handle its contribution equally
    zero_part = 0
    if 0 in x_mesh:
        zero_index = np.where(x_mesh == 0)[0][0]  # Find the index where x_mesh is 0
        dx = x_mesh[1] - x_mesh[0]  # Assuming uniform spacing
        zero_part = - 0.5 * F[zero_index] * b_bm_prime(x_mesh[zero_index]) * dx + \
                      0.5 * (1 - F[zero_index]) * b_bm_prime(x_mesh[zero_index]) * dx 
                    # Equally share the zero contribution
    
    E_b_X = 0.5 + negative_part + zero_part + positive_part
    
    return E_b_X






#%% Elo CDF computation


m = 40 # grid point dividing 2K
L = 1200 # Limit of the box size [-L,L]
k = 20

# interest limit factor for ploting
limit_factor = 4

configurations = [
    (m , L, 0.1, 0.9),
    (m , L, 0.3, 0.7),
    (m , L, 0.5, 0.5),
    (m , L, 0.8, 0.2)
    ]

fig, axs = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

# Using tqdm to display progress
for idx, (m, L, p, q) in enumerate(tqdm(configurations, desc="Processing")):
    
    x_mesh, F = solve_system(m, k, p, q, L)
    
    
    mu = calculate_expected_value(x_mesh, F)
    E_X2 = calculate_second_moment(x_mesh, F)
    variance = calculate_variance(mu, E_X2)
    sigma = np.sqrt(variance)
    
    interest_limits = (mu - limit_factor * sigma, mu + limit_factor * sigma)

    ax = axs[idx // 2, idx % 2]
    
    ax.plot(x_mesh, F, label= fr'$(E[\Delta_{{ij}}] = {mu:.2f}, V[\Delta_{{ij}}] = {variance:.2f})$')
    
    ax.axvline(x= mu, color='r', linestyle='--')
    ax.text(mu + 0.1 * sigma, 0.3, fr'$E[\Delta_{{ij}}] = {mu:.2f}$', color='r', ha='left')
    
    ax.set_xlim(*interest_limits)

    ax.set_title(f'F(x) for m={m}, L={L}, p={p}, q={q}')
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.legend()
    ax.grid(True)

plt.suptitle('Elo CDF for several configurations')
plt.show()


#%% Grid convergence study
# Parameters
k = 20
p = 0.5
q = 0.5
L = 1200

# Range of m values for grid convergence study
m_values = [10, 20, 60, 80, 100, 120, 140]

errors = []
grid_spacings = []

# Grid convergence study
for m in tqdm(m_values,  desc="Processing"):
    x_m, F_m = solve_system(m, k, p, q, L)
    x_2m, F_2m = solve_system(2*m, k, p, q, L)
    
    # Direct comparison: F_m(i) vs. F_2m(2i)
    F_2m_reduced = F_2m[::2]  # Take every second element from F_2m
    
    # Compute the L_inf norm (maximum absolute error)
    error = np.max(np.abs(F_m - F_2m_reduced))
    
    errors.append(error)
    grid_spacings.append(2 * k / m)



# Log-log plot of error vs. grid spacing
log_grid_spacings = np.log(grid_spacings)
log_errors = np.log(errors)

plt.figure(figsize=(8, 6))
plt.plot(log_grid_spacings, log_errors, 'o-', label='Error')

# Linear regression to determine the order of convergence
coefficients = np.polyfit(log_grid_spacings, log_errors, 1)
slope, intercept = coefficients

plt.plot(log_grid_spacings, slope * log_grid_spacings + intercept, 'r--', label=f'Fit: slope = {slope:.2f}')

plt.xlabel('log(Grid Spacing)')
plt.ylabel('log(Error)')
plt.title(f'Grid Convergence Study p = {p}, q = {q}, L = {L}')
plt.legend()
plt.grid(True)
plt.show()

# Display the order of convergence
print(f"Order of convergence: {slope:.2f}")

#%% Symetry of Elo CDF

# Configurations for p and q
configurations = [(0.3, 0.7), 
                  (0.5, 0.5), 
                  (0.9, 0.1)]
k = 20
L = 1200
m = 100

# Function to perform the symmetry test around the mean
def symmetry_test(x_mesh, F_Elo):
    symmetries = []
    mu = calculate_expected_value(x_mesh, F_Elo)
    
    for y in x_mesh:
        if mu - y >= x_mesh[0] and mu + y <= x_mesh[-1]:
            F_mu_minus_y = np.interp(mu - y, x_mesh, F_Elo)
            F_mu_plus_y = np.interp(mu + y, x_mesh, F_Elo)
            symmetry_value = F_mu_minus_y + F_mu_plus_y
            
            symmetries.append((y, symmetry_value))
    
    return symmetries

# Store the results_symetry
results_symetry = {}

# Perform the symmetry test for different configurations of p and q
for p, q in tqdm(configurations, desc="Processing"):
    
    x_mesh, F_Elo = solve_system(m, k, p, q, L)
    
    # symmetry test values = F_X(mu - y) + F_X(mu + y)
    # symmetries = (y, symmetry_value) for all x_mesh 
    symmetries = symmetry_test(x_mesh, F_Elo)
    
    results_symetry[(p, q)] = (x_mesh, F_Elo, symmetries)

#%% Plotting Section 1: Symmetry Test

limit_factor = 5

fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for idx, (p, q) in enumerate(configurations):
    x_mesh, F_Elo, symmetries = results_symetry[(p, q)]
    ys, symmetry_values = zip(*symmetries)
    
    mu = calculate_expected_value(x_mesh, F_Elo)
    E_X2 = calculate_second_moment(x_mesh, F_Elo)
    variance = calculate_variance(mu, E_X2)
    sigma = np.sqrt(variance)
    
    interest_limits = ( - limit_factor * sigma,  limit_factor * sigma)
    
    ax = axs[idx]
    ax.plot(ys, symmetry_values, label=f'p={p}, q={q}')
    
    ax.set_xlim(*interest_limits)
    ax.set_xlabel('y')
    ax.set_ylabel(r'$F_X(\mu - y) + F_X(\mu + y)$')
    ax.set_title(f'Symmetry Test for p={p}, q={q}')
    ax.axhline(1, color='red', linestyle='--', label='Expected Value for Symmetric Distribution')
    ax.legend()
    ax.grid(True)

plt.show()


#%% symetry L_inf


# Configurations for p and q
configurations = [(0.3, 0.7), 
                  (0.5, 0.5), 
                  (0.9, 0.1)]
k = 20
L = 1200
m_values = [10, 20, 40, 80, 160]  # Different values of m to test

# Store the results_symetry_2
results_symetry_2 = {}


# Perform the symmetry test for different configurations of p and q
for p, q in tqdm(configurations, desc="Processing 1"):
    results_symetry_2[(p, q)] = []
    for m in m_values:
        x_mesh, F_Elo = solve_system(m, k, p, q, L)
        symmetries = symmetry_test(x_mesh, F_Elo)
        delta_x = 2 * k / m
        results_symetry_2[(p, q)].append((delta_x, symmetries))

# Compute L_inf norm of difference and store results
L_inf_differences = {config: [] for config in configurations}

for (p, q), res in results_symetry_2.items():
    for (delta_x, symmetries) in res:
        ys, symmetry_values = zip(*symmetries)
        symmetry_values = np.array(symmetry_values)
        L_inf_difference = np.max(np.abs(symmetry_values - 1))  # Compute L_inf norm of difference
        L_inf_differences[(p, q)].append((delta_x, L_inf_difference))

#%% Plotting section 2: the L_inf of symetries differences

fig, ax = plt.subplots(figsize=(10, 6))

for (p, q), differences in L_inf_differences.items():
    delta_xs, L_inf_vals = zip(*differences)
    ax.plot(delta_xs, L_inf_vals, 'o-', label=f'p={p}, q={q}')
    
ax.set_xlabel(r'Grid Spacing $\delta{x}$')
ax.set_ylabel(r'$L_\infty$ Norm of Symmetry Difference')
ax.set_title(r'$L_\infty$ Norm of Symmetry Difference vs $\delta{x}$')
ax.legend()
ax.grid(True)

plt.show()



#%% Elo Conjecture


# Define the parameters
k = 20
L = 1200
p_values = [0.3, 0.4, 0.5, 0.8, 0.9]
m_values = [10, 20, 60, 80, 100]



# Prepare for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Compute and plot normalized errors for different parameter values
for p in p_values:
    q = round(1 - p, 1)
    errors = []
    grid_spacings = []
    for m in m_values:
        # Solve the system to get x_mesh and F
        x_mesh, F = solve_system(m, k, p, q, L)
        
        delta_x = 2 * k / m
        
        # Compute E[b(X)] and b(E[X)]
        E_b_X = compute_E_b_X(x_mesh, F)
        E_X = calculate_expected_value(x_mesh, F)
        b_E_X = b_bm(E_X)
        
        # Compute the error with norm L_1
        error = abs(E_b_X - b_E_X) 
        
        # Store the results
        grid_spacings.append(delta_x)
        errors.append(error)
        
        # Print the results
        print(f"p={p}, q={q}, m={m}, E[b(X)]={E_b_X}, b(E[X])={b_E_X}, error={error}, E_X = {E_X}")
    
    # Plot the normalized errors for this combination of p and q
    ax.plot(grid_spacings, errors, label=f"p={p}, q={q}")


# Plot formatting Elo Conjecture

#ax.axhline(tolerance, color='red', linestyle='--', label='Tolerance')
ax.set_xlabel(r'grid spacing $\delta{{x}}$')
ax.set_ylabel('Difference')
ax.set_title('Difference between E[b(X)] and b(E[X]) for Different p and q')
ax.legend()
ax.grid(True)
plt.show()



#%% Elo CDF vs Normal CDF comparisson


m = 40 # grid point dividing 2K
L = 1200 # Limit of the box size [-L,L]
k = 20

# interest limit factor for ploting
limit_factor = 4

configurations = [
    (m , L, 0.1, 0.9),
    (m , L, 0.3, 0.7),
    (m , L, 0.5, 0.5),
    (m , L, 0.8, 0.2)
    ]

fig, axs = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

# Using tqdm to display progress
for idx, (m, L, p, q) in enumerate(tqdm(configurations, desc="Processing")):
    
    x_mesh, F = solve_system(m, k, p, q, L)
    
    
    mu = calculate_expected_value(x_mesh, F)
    E_X2 = calculate_second_moment(x_mesh, F)
    variance = calculate_variance(mu, E_X2)
    sigma = np.sqrt(variance)
    
    normal_cdf = norm.cdf(x_mesh, loc=mu, scale=sigma)
    
    interest_limits = (mu - limit_factor * sigma, mu + limit_factor * sigma)

    ax = axs[idx // 2, idx % 2]
    
    ax.plot(x_mesh, F, label= fr'Numerical Elo CDF - $(E[\Delta_{{ij}}] = {mu:.2f}, V[\Delta_{{ij}}] = {variance:.2f})$')
    ax.plot(x_mesh, normal_cdf, label= fr'Normal CDF $\mathcal{{N}}(\mu = {mu:.2f}, \sigma^2 = {variance:.2f})$', linestyle='--')
    
    ax.axvline(x= mu, color='r', linestyle='--')
    ax.text(mu + 0.1 * sigma, 0.3, fr'$E[\Delta_{{ij}}] = {mu:.2f}$', color='r', ha='left')
    
    ax.set_xlim(*interest_limits)

    ax.set_title(f'F(x) for m={m}, L={L}, p={p}, q={q}')
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.legend()
    ax.grid(True)

plt.suptitle(r'Elo CDF vs $\mathcal{{N}}(\mu, \sigma^2)$ CDF for several configurations')
plt.show()



#%% Difference Elo CDF vs Normal CDF


# Perform the residual analysis for different mesh sizes and configurations
def difference_L_inf_analysis(m_values, k, p, q, L):
    errors = []
    grid_spacings = []
    
    for m in m_values:
        
        delta_x = 2 * k / m
        
        # Solve for Elo distribution to get mu and sigma
        x_mesh, F_Elo = solve_system(m, k, p, q, L)
        mu = calculate_expected_value(x_mesh, F_Elo)
        E_X2 = calculate_second_moment(x_mesh, F_Elo)
        variance = calculate_variance(mu, E_X2)
        sigma = np.sqrt(variance)
        
        F_Norm = norm.cdf(x_mesh, loc=mu, scale=sigma)
        
        difference = F_Elo - F_Norm
        
        # Compute the L_inf norm (maximum absolute error)
        sup_abs_difference = np.max(np.abs(difference))
        
        errors.append(sup_abs_difference)
        grid_spacings.append(delta_x)
    
    return grid_spacings, errors


# Configurations for p and q
configurations = [(0.3, 0.7), 
                  (0.5, 0.5), 
                  (0.9, 0.1)]

# Perform the analysis
m_values = [5, 10, 20, 60, 80]
k = 20
L = 1200

# Store the results_error
results_differences = {}

for p, q in tqdm(configurations, desc="Processing"):
    grid_spacings, errors = difference_L_inf_analysis(m_values, k, p, q, L)
    results_differences[(p, q)] = (grid_spacings, errors)

#%% Plot the results_error
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for idx, (p, q) in enumerate(configurations):
    grid_spacings, errors = results_differences[(p, q)]
    
    ax = axs[idx]
    ax.plot(grid_spacings, errors, 'o-', label=f'p={p}, q={q}')
    ax.set_xlabel(r'Grid Spacing ($\delta{{x}}$)')
    ax.set_ylabel(r'$\|F_{Elo}-F_{\mathcal{{N}}(\mu, \sigma^2)}\|_{{\infty}}$')
    ax.set_title(fr'Difference $F_{{Elo}}$ vs $F_{{\mathcal{{N}}(\mu, \sigma^2)}}$ Analysis for p={p}, q={q}')
    ax.legend()
    ax.grid(True)

plt.show()


#%% Normal residual Analysis


# Function to compute residuals using normal distribution with Elo-derived mean and std
def compute_residual(x_mesh, k, p, q, mu, sigma):
    residuals = np.zeros_like(x_mesh)
    for i, x in enumerate(x_mesh):
        h_x_minus_2k  = h(x - 2 * k, k, x - k)
        h_x = h(x, k, x + k)
        
        F_h_x_minus_2k = norm.cdf(h_x_minus_2k, loc=mu, scale=sigma)
        F_h_x = norm.cdf(h_x, loc=mu, scale=sigma)
        F_x = norm.cdf(x, loc=mu, scale=sigma)
        
        residuals[i] = p * F_h_x_minus_2k + q * F_h_x - F_x
    return residuals

# Perform the residual analysis for different mesh sizes and configurations
def residual_error_analysis(m_values, k, p, q, L):
    errors = []
    grid_spacings = []
    
    for m in m_values:
        
        delta_x = 2 * k / m
        
        # Solve for Elo distribution to get mu and sigma
        x_mesh, F_Elo = solve_system(m, k, p, q, L)
        mu = calculate_expected_value(x_mesh, F_Elo)
        E_X2 = calculate_second_moment(x_mesh, F_Elo)
        variance = calculate_variance(mu, E_X2)
        sigma = np.sqrt(variance)
        
        
        residuals = compute_residual(x_mesh, k, p, q, mu, sigma)
        
        # Compute the L_inf norm (maximum absolute error)
        sup_abs_residual = np.max(np.abs(residuals)) 
        
        
        errors.append(sup_abs_residual)
        grid_spacings.append(delta_x)
    
    return grid_spacings, errors


# Configurations for p and q
configurations = [(0.3, 0.7), 
                  (0.5, 0.5), 
                  (0.9, 0.1)]

# Perform the analysis
m_values = [5, 10, 20, 60, 80]
k = 20
L = 1200

# Store the results_error
results_residual_error = {}

for p, q in tqdm(configurations, desc="Processing"):
    grid_spacings, errors = residual_error_analysis(m_values, k, p, q, L)
    results_residual_error[(p, q)] = (grid_spacings, errors)

#%% Plot the results_error
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for idx, (p, q) in enumerate(configurations):
    grid_spacings, errors = results_residual_error[(p, q)]
    
    ax = axs[idx]
    ax.plot(grid_spacings, errors, 'o-', label=f'p={p}, q={q}')
    ax.set_xlabel('Grid Spacing (Delta x)')
    ax.set_ylabel('Residual Error')
    ax.set_title(f'Normal Residual Error Analysis for p={p}, q={q}')
    ax.legend()
    ax.grid(True)

plt.show()



#%% mu - k independence and sigma**2 - k relation

m = 40 # grid point dividing 2K
L = 1200 # Limit of the box size [-L,L]

# interest limit factor for ploting
limit_factor = 4

configurations = [
    (m , L, 0.1, 0.9),
    (m , L, 0.3, 0.7),
    (m , L, 0.5, 0.5),
    (m , L, 0.8, 0.2)
    ]


k_values = np.array([10, 15, 20, 25, 30, 40])

# Storage for results
mu_results = {config: [] for config in configurations}

# Storage for variance results
variance_results = {config: [] for config in configurations}

# Compute variance for each configuration and k value
for k in  tqdm(k_values, desc="Processing"):
    for config in  configurations:
        m, L, p, q = config
        x_mesh, F = solve_system(m, k, p, q, L)
        mu = calculate_expected_value(x_mesh, F)
        E_X2 = calculate_second_moment(x_mesh, F)
        variance = calculate_variance(mu, E_X2)
        
        mu_results[config].append(mu) # mu - k independence
        variance_results[config].append(variance) # sigma**2 - k relation


#%% Plot the results

# Function to calculate R-squared value
def calculate_r_squared(x, y, slope, intercept):
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared



# Plot for mu vs k
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

for idx, config in enumerate(configurations):
    m, L, p, q = config
    mu_values = mu_results[config]
    
    variance_values = variance_results[config]
    sigma_values = np.sqrt(variance_values)
    sigma_min = np.min(sigma_values)
    
    mu_max_diff = np.max(mu_values) - np.min(mu_values)
    
    mu_diff_sigma_rate = mu_max_diff/sigma_min
    
    # Perform linear regression using np.polyfit
    coeffs = np.polyfit(k_values, mu_values, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    r_squared = calculate_r_squared(k_values, mu_values, slope, intercept)
    
    ax = axs[idx // 2, idx % 2]
    ax.plot(k_values, mu_values, 'o-', label=fr'p={p}, q={q}, slope={slope:.2f}, $R^2$={r_squared:.2f}, $ \frac{{\max \Delta E[X]}}{{\sqrt{{\min V[X]}}}}$ = { mu_diff_sigma_rate:.2f}')
    ax.plot(k_values, slope * k_values + intercept, '--')
    ax.set_xlabel('k')
    ax.set_ylabel(r'$E[\Delta_{{ij}}]$')
    ax.set_title(r'$E[\Delta_{{ij}}]$ vs k')
    ax.legend()
    ax.grid(True)

plt.tight_layout()


    


#%% # Plot for variance vs k


fig, axs = plt.subplots(2, 2, figsize=(14, 12))


for idx, config in enumerate(configurations):
    m, L, p, q = config
    variance_values = variance_results[config]
    # Perform linear regression using np.polyfit
    coeffs = np.polyfit(k_values, variance_values, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    r_squared = calculate_r_squared(k_values, variance_values, slope, intercept)
    
    ax = axs[idx // 2, idx % 2]
    ax.plot(k_values, variance_values, 'o-', label=fr'p={p}, q={q}, slope={slope:.2f}, $R^2$={r_squared:.3f}')
    ax.plot(k_values, slope * k_values + intercept, '--')
    ax.set_xlabel('k')
    ax.set_ylabel(r'$V[\Delta_{{ij}}]$')
    ax.set_title(r'$V[\Delta_{{ij}}]$ vs k')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
