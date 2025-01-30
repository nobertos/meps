import numpy as np
import math


def sigma_equation(sigma, mu, laplace_fn, *args):
    """
    Equation to solve: σ = A*(μ(1-σ))
    where A* is the Laplace transform of the arrival time distribution
    
    Args:
        sigma: Variable to solve for
        mu: Service rate
        laplace_fn: Laplace transform function
        *args: Arguments for Laplace function
    """
    return sigma - laplace_fn(mu * (1 - sigma), *args)

def newton_raphson(f, x0, args=(), tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for root finding
    
    Args:
        f: Function to find root of
        x0: Initial guess
        args: Additional arguments for f
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        float: Root of the equation
        
    Raises:
        RuntimeError: If method fails to converge
    """
    x0 = float(x0)  # Ensure x0 is float for numerical stability
    
    for _ in range(max_iter):
        fx = f(x0, *args)
        # Compute numerical derivative with central difference
        h = max(tol, abs(x0 * 1e-8))  # Adaptive step size
        fpx = (f(x0 + h, *args) - f(x0 - h, *args)) / (2 * h)
        
        if abs(fpx) < tol:
            raise RuntimeError("Derivative too close to zero")
            
        x1 = x0 - fx / fpx
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
        
    raise RuntimeError(f"Failed to converge after {max_iter} iterations")

def calculate_performance(sigma, mu, lamda_):
    """
    Calculate queue performance metrics using σ
    
    Args:
        sigma: Solution to the functional equation
        mu: Service rate
        lamda_: Effective arrival rate
        
    Returns:
        tuple: (Q, R, W) where:
            Q: Average number of customers in system
            R: Average response time
            W: Average waiting time
    """
    R = 1/mu * (1/(1 - sigma))  # Average response time
    Q = lamda_ * R                # Average number in system
    W = R - 1/mu               # Average waiting time
    return Q, R, W

def get_lambda(probs, lambdas):
    """
    Calculate effective arrival rate from hyper-exponential parameters
    
    Args:
        probs: List of probabilities
        lambdas: List of rate parameters
        
    Returns:
        float: Effective arrival rate
    """
    # Mean arrival time is weighted sum of individual mean times
    mean_time = sum(p * (1/l) for p, l in zip(probs, lambdas))
    return 1/mean_time

def hyperexp_laplace(s, probs, lambdas):
    """
    Laplace transform of hyper-exponential (Hk) distribution
    
    Args:
        s: Laplace transform variable
        probs: List of probabilities for each branch
        lambdas: List of rate parameters for each branch
        
    Returns:
        float: Value of Laplace transform at s
        
    Raises:
        ValueError: If probabilities don't sum to 1 or if lengths don't match
    """
    if len(probs) != len(lambdas):
        raise ValueError("Number of probabilities must match number of rates")
    if not np.isclose(sum(probs), 1.0, rtol=1e-5):
        raise ValueError("Probabilities must sum to 1")
    if any(p < 0 or p > 1 for p in probs):
        raise ValueError("All probabilities must be between 0 and 1")
    if any(l <= 0 for l in lambdas):
        raise ValueError("All rates must be positive")
        
    return sum(p * lambda_i / (lambda_i + s) 
              for p, lambda_i in zip(probs, lambdas))


# Added Laplace transforms for common distributions
def exp_laplace(s, lambda_):
    """Laplace transform of exponential distribution"""
    if lambda_ <= 0:
        raise ValueError("Rate must be positive")
    return lambda_ / (lambda_ + s)

def erlang_laplace(s, k, mu):
    """Laplace transform of Erlang-k distribution"""
    if k <= 0 or not isinstance(k, int):
        raise ValueError("k must be positive integer")
    if mu <= 0:
        raise ValueError("mu must be positive")
    return (mu / (mu + s)) ** k

def deterministic_laplace(s, D):
    """Laplace transform of deterministic distribution"""
    if D <= 0:
        raise ValueError("D must be positive")
    return np.exp(-s * D)

def gamma_laplace(s, shape, rate):
    """Laplace transform of Gamma distribution"""
    if shape <= 0 or rate <= 0:
        raise ValueError("Shape and rate must be positive")
    return (rate / (rate + s)) ** shape

def weibull_laplace(s, shape, scale):
    """Laplace transform of Weibull distribution (series approximation)"""
    if shape <= 0 or scale <= 0:
        raise ValueError("Shape and scale must be positive")
    
    # Numerical approximation using Taylor series
    return np.sum([(-1)**n * (scale**n * s**n) * math.gamma(1 + n/shape)
                  / math.factorial(n) for n in range(0, 20)])

def pareto_laplace(s, alpha, xm):
    """Laplace transform of Pareto distribution (series approximation)"""
    if alpha <= 0 or xm <= 0:
        raise ValueError("Alpha and xm must be positive")
    
    # Numerical approximation using integration
    from scipy.integrate import quad
    integrand = lambda x: np.exp(-s*x) * alpha * xm**alpha / x**(alpha+1)
    result, _ = quad(integrand, xm, np.inf)
    return result


def general_gm1_queue(mu, distribution='hyperexp', laplace_fn=None, 
                     probs=None, lambdas=None, **dist_args):
    """
    Unified G/M/1 queue analyzer supporting multiple distributions
    
    Args:
        mu: Service rate
        distribution: Distribution type ('hyperexp', 'erlang', 'custom')
        laplace_fn: Custom Laplace transform function (for 'custom' distribution)
        probs: List of probabilities (for hyperexponential)
        lambdas: List of rate parameters (for hyperexponential)
        dist_args: Additional distribution-specific arguments
        
    Returns:
        dict: Queue performance metrics or error message
    """
    try:
        # Handle hyper-exponential case
        if distribution == 'hyperexp':
            if probs is None or lambdas is None:
                raise ValueError("probs and lambdas required for hyperexp")
                
            # Validate hyper-exponential parameters
            if len(probs) != len(lambdas):
                raise ValueError("probs and lambdas must have same length")
            if not np.isclose(sum(probs), 1.0, rtol=1e-5):
                raise ValueError("probs must sum to 1")
            if any(p < 0 or p > 1 for p in probs):
                raise ValueError("probs must be in [0,1]")
            if any(l <= 0 for l in lambdas):
                raise ValueError("lambdas must be positive")
                
            laplace_fn = hyperexp_laplace
            laplace_args = (probs, lambdas)
            
        elif distribution == 'erlang':
            k = dist_args.get('k', 1)
            laplace_fn = erlang_laplace
            laplace_args = (k, dist_args['mu'])
            
        elif distribution == 'custom':
            if laplace_fn is None:
                raise ValueError("laplace_fn required for custom distribution")
            laplace_args = dist_args.get('args', ())
            
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Calculate effective arrival rate
        h = 1e-6  # For numerical differentiation
        A0 = laplace_fn(0, *laplace_args)
        Ah = laplace_fn(h, *laplace_args)
        E_T = (A0 - Ah)/h  # E[T] = -dA/ds at s=0
        lamda_ = 1/E_T
        
        # Check stability
        rho = lamda_/mu
        if rho >= 1:
            return {"error": f"Unstable (ρ = {rho:.2f} ≥ 1)"}

        # Solve for σ
        sigma = newton_raphson(
            sigma_equation, 
            0.5,  # Default guess
            args=(mu, laplace_fn) + laplace_args  # Fix here
        )

        # Calculate performance metrics
        Q, R, W = calculate_performance(sigma, mu, lamda_)
        
        return {
            "distribution": distribution,
            "sigma": sigma,
            "traffic_intensity": rho,
            "average_customers": Q,
            "average_response_time": R,
            "average_waiting_time": W,
            "effective_arrival_rate": lamda_,
            "probs": probs
        }
        
    except Exception as e:
        return {"error": str(e)}

def print_results(results, indent=2):
    """
    Print queue analysis results in a readable format
    
    Args:
        results: Dictionary returned by general_gm1_queue
        indent: Number of spaces for indentation
    """
    space = ' ' * indent
    
    if "error" in results:
        print(f"{space}Error: {results['error']}")
        return
    
    # Header with distribution type
    dist_name = results.get('distribution', 'unknown').capitalize()
    print(f"{space}{dist_name} Distribution Results:")
    
    # Main metrics
    print(f"{space}- Effective Arrival Rate (λ): {results['effective_arrival_rate']:.4f}")
    print(f"{space}- Traffic Intensity (ρ): {results['traffic_intensity']:.4f}")
    print(f"{space}- Sigma (σ): {results['sigma']:.4f}")
    print(f"{space}- Average Customers in System: {results['average_customers']:.4f}")
    print(f"{space}- Average Response Time: {results['average_response_time']:.4f}")
    print(f"{space}- Average Waiting Time: {results['average_waiting_time']:.4f}")
    
    # Additional distribution-specific parameters
    if results['distribution'] == 'hyperexp':
        print(f"{space}- Phases: {len(results.get('probs', []))}")
    elif results['distribution'] == 'erlang':
        print(f"{space}- Erlang Stages: {results.get('k', 'N/A')}")
    
    print("-" * 50)# Example usage patterns
if __name__ == "__main__":
    # Hyper-exponential (2 phases)
    print("Hyper-exponential Example:")
    res = general_gm1_queue(
        mu=3.0,
        distribution='hyperexp',
        probs=[1/3, 2/3],
        lambdas=[1, 2]
    )
    print_results(res)
    # 
    # # Erlang-3 arrivals
    # print("\nErlang Example:")
    # res = general_gm1_queue(
    #     mu=4.0,
    #     distribution='erlang',
    #     k=3,
    # )
    # print_results(res)
    # 
    # # Custom distribution (e.g., deterministic)
    # print("\nCustom Distribution Example:")
    # res = general_gm1_queue(
    #     mu=0.8,
    #     distribution='custom',
    #     laplace_fn=deterministic_laplace,
    #     D=1.2
    # )
    # print_results(res)
