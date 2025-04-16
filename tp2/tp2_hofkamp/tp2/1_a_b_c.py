import numpy as np
import timeit

np.random.seed(42)

def suma_uniformes(mu, sigma2, size=1):
    """Genera muestras de una distribución normal sumando 12 distribuciones uniformes."""
    uniforms = np.random.uniform(0, 1, (size, 12))
    standard_normal_samples = np.sum(uniforms, axis=1) - 6  
    return mu + np.sqrt(sigma2) * standard_normal_samples

def metodo_rechazo(mu, sigma2, size=1):
    """Genera muestras de una distribución normal usando el método de rechazo."""
    samples = []
    while len(samples) < size:
        u1, u2 = np.random.uniform(0, 1, 2)
        x = -np.log(u1)
        if u2 <= np.exp(-0.5 * (x - 1) ** 2):
            samples.append(x if np.random.rand() < 0.5 else -x)
    return mu + np.sqrt(sigma2) * np.array(samples)

def box_muller(mu, sigma2, size=1):
    """Genera muestras de una distribución normal usando el método de Box-Muller."""
    u1, u2 = np.random.uniform(0, 1, (2, size))
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mu + np.sqrt(sigma2) * z0

def main():
    mu, sigma2, size = 0, 1, 10000
    time_summing = timeit.timeit(lambda: suma_uniformes(mu, sigma2, size), number=10)
    time_rejection = timeit.timeit(lambda: metodo_rechazo(mu, sigma2, size), number=10)
    time_box_muller = timeit.timeit(lambda: box_muller(mu, sigma2, size), number=10)
    time_numpy = timeit.timeit(lambda: np.random.normal(mu, np.sqrt(sigma2), size), number=10)
    
    print(f"suma de uniformes: {time_summing:.3f} s")
    print(f"rechazo: {time_rejection:.3f} s")
    print(f"Box-Muller: {time_box_muller:.3f} s")
    print(f"numpy.random.normal: {time_numpy:.3f} s")


if __name__ == "__main__":
    main()