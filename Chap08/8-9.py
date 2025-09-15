import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def sample_distances(n_samples, n_dims):
    """
    Sample from standard normal distribution in n_dims dimensions
    and calculate distances from the origin.
    """
    # Sample from standard normal distribution
    samples = np.random.normal(0, 1, (n_dims, n_samples))
    
    # Calculate Euclidean distance from origin for each sample
    distances = np.sqrt(np.sum(samples**2, axis=0))
    
    return distances

def theoretical_chi_distribution(n_dims, x):
    """
    Theoretical probability density function for distances from origin
    of standard normal distribution in n_dims dimensions.
    This follows a chi distribution with n_dims degrees of freedom.
    """
    return stats.chi.pdf(x, df=n_dims)

def plot_distance_distributions():
    """
    Plot histograms of distances for different dimensions and compare
    with theoretical chi distribution.
    """
    # Parameters
    n_samples = 10000
    dimensions = [25, 100, 500]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, n_dims in enumerate(dimensions):
        # Sample distances
        distances = sample_distances(n_samples, n_dims)
        
        # Plot histogram
        axes[i].hist(distances, bins=50, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', label='Empirical')
        
        # Plot theoretical chi distribution
        x_range = np.linspace(0, np.max(distances), 1000)
        theoretical_pdf = theoretical_chi_distribution(n_dims, x_range)
        axes[i].plot(x_range, theoretical_pdf, 'r-', linewidth=2, 
                    label=f'Chi({n_dims})')
        
        # Formatting
        axes[i].set_xlabel('Distance from Origin')
        axes[i].set_ylabel('Probability Density')
        axes[i].set_title(f'{n_dims} Dimensions')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        axes[i].axvline(mean_dist, color='green', linestyle='--', 
                       label=f'Mean: {mean_dist:.2f}')
        
        print(f"\n{n_dims} Dimensions:")
        print(f"  Mean distance: {mean_dist:.4f}")
        print(f"  Std distance: {std_dist:.4f}")
        print(f"  Theoretical mean: {np.sqrt(2) * np.sqrt(n_dims/2) * np.sqrt(np.pi/2):.4f}")
        print(f"  Theoretical std: {np.sqrt(n_dims - (np.sqrt(2) * np.sqrt(n_dims/2) * np.sqrt(np.pi/2))**2):.4f}")
    
    plt.tight_layout()
    plt.suptitle('Distribution of Distances from Origin in High-Dimensional Spaces', 
                 y=1.02, fontsize=14)
    plt.show()

def analyze_chi_distribution_properties():
    """
    Analyze the properties of the chi distribution for different dimensions.
    """
    dimensions = [25, 100, 500]
    
    print("Chi Distribution Properties:")
    print("=" * 50)
    
    for n_dims in dimensions:
        # Theoretical properties of chi distribution
        mean_theoretical = np.sqrt(2) * np.sqrt(n_dims/2) * np.sqrt(np.pi/2)
        var_theoretical = n_dims - mean_theoretical**2
        std_theoretical = np.sqrt(var_theoretical)
        
        print(f"\n{n_dims} dimensions:")
        print(f"  Mean: {mean_theoretical:.4f}")
        print(f"  Standard deviation: {std_theoretical:.4f}")
        print(f"  Coefficient of variation: {std_theoretical/mean_theoretical:.4f}")

def verify_concentration_measure():
    """
    Verify that distances concentrate around sqrt(n) as dimension increases.
    """
    dimensions = [25, 100, 500, 1000]
    n_samples = 5000
    
    print("\nConcentration of Distances:")
    print("=" * 40)
    print("Dimension | Mean Distance | sqrt(n) | Ratio")
    print("-" * 40)
    
    for n_dims in dimensions:
        distances = sample_distances(n_samples, n_dims)
        mean_dist = np.mean(distances)
        sqrt_n = np.sqrt(n_dims)
        ratio = mean_dist / sqrt_n
        
        print(f"{n_dims:9d} | {mean_dist:12.4f} | {sqrt_n:7.2f} | {ratio:5.3f}")

if __name__ == "__main__":
    print("Empirical Verification of Distance Distributions in High-Dimensional Spaces")
    print("=" * 80)
    
    # Plot the distributions
    plot_distance_distributions()
    
    # Analyze theoretical properties
    analyze_chi_distribution_properties()
    
    # Verify concentration measure
    verify_concentration_measure()
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The distances from the origin of samples from standard normal")
    print("distributions follow a CHI DISTRIBUTION with degrees of freedom")
    print("equal to the number of dimensions.")
    print("\nKey properties:")
    print("- Mean distance ≈ sqrt(n) for large n")
    print("- The distribution becomes more concentrated around sqrt(n) as n increases")
    print("- This is the theoretical foundation for the 'curse of dimensionality'")
