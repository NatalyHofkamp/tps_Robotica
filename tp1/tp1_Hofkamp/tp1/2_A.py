import numpy as np
import matplotlib.pyplot as plt

scan = np.loadtxt('/home/nataly/Escritorio/robotica/tp1/laserscan.dat')
angle = np.linspace(-np.pi/2, np.pi/2, np.shape(scan)[0], endpoint=True)

x_lidar = scan * np.cos(angle)
y_lidar = scan * np.sin(angle)

# Graficar 
plt.figure(figsize=(6,6))
plt.scatter(x_lidar, y_lidar, s=5, color='g', label="Mediciones LiDAR")
plt.gca().set_aspect('equal')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.show()
