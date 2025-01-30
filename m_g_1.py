import numpy as np
from scipy import stats

class MG1Queue:
    """
    Analysis of M/G/1 queue using the Pollaczek-Khinchin formula.
    
    Notation:
    - m₁: First moment of service time (E[X])
    - m₂: Second moment of service time (E[X²])
    - CV²: Squared coefficient of variation = (m₂ - m₁²)/m₁²
    """
    
    def __init__(self, arrival_rate):
        if arrival_rate <= 0:
            raise ValueError("Arrival rate must be positive")
        self.lambda_ = arrival_rate
    
    def calculate_cv_squared(self, m1, m2):
        """
        Calculate squared coefficient of variation using moments.
        
        Args:
            m1: First moment (mean)
            m2: Second moment
            
        Returns:
            float: Squared coefficient of variation
        """
        return (m2 - m1**2) / (m1**2)
    
    def analyze_with_moments(self, m1, m2):
        """
        Analyze queue using first two moments of service time distribution.
        
        Args:
            m1: First moment E[X]
            m2: Second moment E[X²]
        """
        # Traffic intensity
        rho = self.lambda_ * m1
        
        if rho >= 1:
            return {
                "error": f"System unstable: traffic intensity ρ = {rho:.4f} ≥ 1",
                "traffic_intensity": rho
            }
        
        # Calculate CV² using moments
        cv_squared = self.calculate_cv_squared(m1, m2)
        
        # Calculate performance metrics using P-K formula
        EWq = (self.lambda_ * m2) / (2 * (1 - rho))
        ER = EWq + m1
        EQq = self.lambda_ * EWq
        EQ = self.lambda_ * ER
        
        return {
            "traffic_intensity": rho,
            "E[R]": ER,           # Expected response time
            "E[Wq]": EWq,         # Expected waiting time
            "E[Q]": EQ,           # Expected number in system
            "E[Qq]": EQq,         # Expected queue length
            "E[X]": m1,           # First moment (mean service time)
            "E[X²]": m2,          # Second moment
            "CV²": cv_squared,    # Squared coefficient of variation
            "CV": np.sqrt(cv_squared),  # Coefficient of variation
            "utilization": rho
        }
    
    def analyze_erlang_service(self, EX, k):
        """
        Analyze queue with Erlang-k distributed service times.
        
        For Erlang-k distribution:
        - m₁ = EX
        - m₂ = EX² * (k+1)/k
        - CV² = 1/k
        
        Args:
            EX: Expected service time (m₁)
            k: Shape parameter (number of phases)
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        
        # Calculate moments for Erlang-k
        m1 = EX
        m2 = (EX**2) * ((k + 1) / k)  # This gives us E[X²] directly
        
        results = self.analyze_with_moments(m1, m2)
        
        if "error" not in results:
            results["distribution"] = f"Erlang-{k}"
            results["shape_parameter"] = k
            results["phase_rate"] = k/EX
            results["interpretation"] = {
                "phases": f"Service process consists of {k} exponential phases",
                "variability": "Low" if k > 5 else "Medium" if k > 1 else "High",
                "theoretical_cv_squared": 1/k  # Should match our calculated CV²
            }
        
        return results

# Rest of the code remains the same...    
    def analyze_exponential_service(self, mean_service):
        """
        Special case: M/M/1 queue (exponential service times).
        This is equivalent to Erlang-1 service times.
        """
        return self.analyze_erlang_service(mean_service, k=1)
    
    def analyze_constant_service(self, service_time):
        """
        Special case: M/D/1 queue (deterministic service times).
        This is equivalent to Erlang-∞ service times.
        """
        return self.analyze_with_moments(service_time, 0)
    
    def analyze_lognormal_service(self, mean, sigma):
        """Analyze queue with lognormal service times."""
        mu = np.log(mean) - sigma**2/2
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        return self.analyze_with_moments(mean, variance)
    
    def analyze_gamma_service(self, mean, shape):
        """Analyze queue with gamma-distributed service times."""
        scale = mean / shape
        variance = shape * scale**2
        return self.analyze_with_moments(mean, variance)

def print_results(results):
    """
    Helper function to print queue analysis results in a clear, educational format.
    
    This function presents the queueing metrics using formal notation (E[X], E[R], etc.)
    while providing descriptive labels to help users understand each measure's meaning.
    """
    if "error" in results:
        print("\nError:", results["error"])
        return
        
    print("\nQueue Analysis Results")
    print("=" * 50)
    
    # System stability metrics
    print("\nStability Metrics:")
    print(f"Traffic Intensity (ρ) = λE[X]: {results['traffic_intensity']:.4f}")
    print(f"Server Utilization: {results['utilization']:.2%}")
    
    # Time-based metrics
    print("\nTime Metrics:")
    print(f"E[X]  (Mean Service Time): {results['E[X]']:.4f}")
    print(f"E[Wq] (Mean Waiting Time): {results['E[Wq]']:.4f}")
    print(f"E[R]  (Mean Response Time): {results['E[R]']:.4f}")
    
    # Queue length metrics
    print("\nQueue Length Metrics:")
    print(f"E[Qq] (Mean Queue Length): {results['E[Qq]']:.4f}")
    print(f"E[Q]  (Mean System Size): {results['E[Q]']:.4f}")
    
    # Variability metrics
    print("\nVariability Metrics:")
    print(f"E[X²] (Second Moment of Service): {results['E[X²]']:.4f}")
    print(f"CV²   (Squared Coefficient of Variation): {results['CV²']:.4f}")
    print(f"CV    (Coefficient of Variation): {results['CV']:.4f}")
    
    # Distribution-specific information for Erlang
    if "distribution" in results and results["distribution"].startswith("Erlang"):
        print(f"\nErlang Distribution Details:")
        print(f"Shape Parameter (k): {results['shape_parameter']}")
        interp = results["interpretation"]
        print(f"Process Characteristics:")
        print(f"- {interp['phases']}")
        print(f"- Variability Level: {interp['variability']}")
        print(f"- Phase Rate (μ): {results['phase_rate']:.4f}")
if __name__ == "__main__":
    # Example usage with different Erlang-k service times
    arrival_rate = 1.0  # Average of 1 customer per unit time
    mean_service = 1/6  # Mean service time of 0.5 time units
    queue = MG1Queue(arrival_rate)
    
    # Compare different Erlang-k services
    k_values = [2]
    
    for k in k_values:
        print(f"\nM/G/1 Queue with Erlang-{k} Service Times")
        print("=" * 50)
        results = queue.analyze_erlang_service(mean_service, k)
        print_results(results)
