import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel

# Load and sample data
data = np.load("output/author_topic_1.npy")
data = data[np.random.choice(data.shape[0], 5000, replace=False)]

# Prepare data for Stan
N = data.shape[0]  # number of observations
K = data.shape[1]  # number of categories

# Convert to format expected by Stan
# Assuming data contains counts for each category
y = data.astype(int)
n = np.sum(y, axis=1)  # total trials per observation

# Prepare Stan data dictionary
stan_data = {
    'N': N,
    'K': K,
    'y': y.tolist(),  # Convert to list for Stan
    'n': n.tolist()
}

# Compile the Stan model
model = CmdStanModel(stan_file='code/correlation.stan')


# Create proper initialization for Cholesky factor
def init_function():
    """Initialize parameters to avoid Cholesky constraint violations"""
    init_dict = {}

    # Initialize mean parameters
    init_dict['mu'] = np.random.normal(0, 0.1, K - 1)

    # Initialize scale parameters (must be positive)
    init_dict['tau'] = np.random.exponential(0.5, K - 1)

    # Initialize raw beta parameters
    init_dict['beta_raw'] = np.random.normal(0, 0.1, (K - 1, N))

    # Initialize Cholesky factor properly
    # Start with identity correlation matrix
    L_Omega = np.eye(K - 1)
    # Add small random perturbations to off-diagonal elements
    for i in range(K - 1):
        for j in range(i):
            L_Omega[i, j] = np.random.normal(0, 0.05)

    # Ensure diagonal elements are positive (they should be 1 for correlation matrix)
    for i in range(K - 1):
        L_Omega[i, i] = max(0.5, abs(L_Omega[i, i]))

    init_dict['L_Omega'] = L_Omega

    return init_dict


# Optimize the model (find MAP estimate)
try:
    fit = model.optimize(
        data=stan_data,
        inits=init_function(),  # Use custom initialization
        algorithm='lbfgs',  # L-BFGS optimization algorithm
        iter=10000,  # Maximum iterations
        init_alpha=0.001,  # Initial step size
        tol_obj=1e-8,  # Relaxed objective tolerance
        tol_rel_obj=1e-6,  # Relaxed relative objective tolerance
        tol_grad=1e-6,  # Relaxed gradient tolerance
        tol_rel_grad=1e-6,  # Relaxed relative gradient tolerance
        history_size=5,  # L-BFGS history size
        seed=42  # Random seed for reproducibility
    )

    print("Optimization successful!")
    print(f"Log probability: {fit.optimized_params_np[0]}")

    # Extract optimized parameters
    params = fit.optimized_params_dict

    # Print key parameters
    print("\nOptimized parameters:")
    print(f"mu: {params['mu']}")
    print(f"tau: {params['tau']}")

    # Print some beta parameters (first few observations)
    beta = np.array(params['beta'])
    print(f"\nBeta parameters shape: {beta.shape}")
    print(f"First 5 beta parameters:\n{beta[:5]}")

    # Print correlation matrix
    if 'Omega' in params:
        Omega = np.array(params['Omega'])
        print(f"\nCorrelation matrix:\n{Omega}")

    # Save results
    np.save('optimized_params.npy', params)
    print("\nResults saved to 'optimized_params.npy'")

except Exception as e:
    print(f"Optimization failed: {e}")
    print("This might be due to:")
    print("1. Model specification issues")
    print("2. Data format problems")
    print("3. Initialization problems")
    print("4. Convergence issues")

    # Try with different initialization
    print("\nAttempting optimization with different settings...")
    try:
        # Alternative initialization strategy
        def simple_init():
            return {
                'mu': np.zeros(K - 1),
                'tau': np.ones(K - 1),
                'beta_raw': np.random.normal(0, 0.01, (K - 1, N)),
                'L_Omega': np.eye(K - 1)  # Identity matrix for correlation
            }


        fit = model.optimize(
            data=stan_data,
            inits=simple_init,
            algorithm='lbfgs',
            iter=5000,
            init_alpha=0.01,
            seed=123
        )
        print("Second optimization attempt successful!")
        params = fit.optimized_params_dict
        np.save('optimized_params_backup.npy', params)

    except Exception as e2:
        print(f"Second optimization also failed: {e2}")

        # Diagnostic information
        print(f"\nData diagnostics:")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Data range: [{np.min(data)}, {np.max(data)}]")
        print(f"Any negative values: {np.any(data < 0)}")
        print(f"Any NaN values: {np.any(np.isnan(data))}")
        print(f"Sample data:\n{data[:5]}")
        print(f"Sample n (totals): {n[:5]}")