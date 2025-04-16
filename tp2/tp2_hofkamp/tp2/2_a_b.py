import numpy as np
import matplotlib.pyplot as plt

def one_sample(mu, sigma2, size=1):
    return np.random.normal(mu, sigma2, size).item()

def odometry_motion_model(xt, ut, alpha):
    """Modelo de movimiento basado en odometría con ruido."""
    x, y, theta = xt 
    delta_r1,delta_r2,delta_t = ut 

    # Ruido
    delta_r1_n = delta_r1 + one_sample(0, alpha[0] * abs(delta_r1) + alpha[1] * abs(delta_t))
    delta_t_n = delta_t + one_sample(0, alpha[2] * abs(delta_t) + alpha[3] * (abs(delta_r1) + abs(delta_r2)))
    delta_r2_n = delta_r2 + one_sample(0, alpha[0] * abs(delta_r2) + alpha[1] * abs(delta_t))

    x_new = x + delta_t_n * np.cos(theta + delta_r1_n)
    y_new = y + delta_t_n * np.sin(theta + delta_r1_n)
    theta_new = theta + delta_r1_n + delta_r2_n
    
    return np.array([x_new, y_new, theta_new])

def evaluate_model():
    xt = np.array([2.0, 4.0, 0.0])  
    ut = np.array([np.pi/2, 0, 1]) 
    alpha = np.array([0.1, 0.1, 0.01, 0.01]) 

    samples = []
    for i in range(5000):
        samples.append(odometry_motion_model(xt, ut, alpha)) 

    samples = np.array(samples) 
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Distribución de la nueva posición del robot")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
