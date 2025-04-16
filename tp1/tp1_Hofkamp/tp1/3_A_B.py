import numpy as np
import matplotlib.pyplot as plt

def diffdrive(x, y, theta, v_l, v_r, t, l):
    """
    Calcula la nueva pose (x_n, y_n, theta_n) de un robot diferencial.
    """
    if v_l == v_r:
        # Movimiento en línea recta
        x_n = x + v_l * t * np.cos(theta)
        y_n = y + v_l * t * np.sin(theta)
        theta_n = theta
    else:
        R = (l / 2) * (v_l + v_r) / (v_r - v_l)
        omega = (v_r - v_l) / l
        ICC_x = x - R * np.sin(theta)
        ICC_y = y + R * np.cos(theta)
        
        delta_theta = omega * t
        cos_dt = np.cos(delta_theta)
        sin_dt = np.sin(delta_theta)
        
        x_n = cos_dt * (x - ICC_x) - sin_dt * (y - ICC_y) + ICC_x
        y_n = sin_dt * (x - ICC_x) + cos_dt * (y - ICC_y) + ICC_y
        theta_n = theta + delta_theta
        
    return x_n, y_n, theta_n

def get_positions ():
    vl = [0.1,0.5,0.2,1,0.4,0.2,0.5]
    vr = [0.5,0.1,0.2,0,0.4,-0.2,0.5]
    t = [2,2,2,4,2,2,2]

    X = [0]
    Y = [0]
    theta = [np.pi/4]

    for i in range(len(t)):
        x_n, y_n, theta_n = diffdrive(X[-1], Y[-1], theta[-1], vl[i], vr[i], t[i], 0.5)
        X.append(x_n)
        Y.append(y_n)
        theta.append(theta_n)
    return X,Y

def main():
    X,Y = get_positions()
    # Graficar la trayectoria
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, marker='o', linestyle=':', color='g')
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid()
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()