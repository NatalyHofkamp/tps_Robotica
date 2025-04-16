import numpy as np
import matplotlib.pyplot as plt

# Pose del robot en la terna global
x_robot, y_robot, theta_robot = 5, -7, -np.pi/4

# Pose del LIDAR en la terna del robot
x_lidar_robot, y_lidar_robot, theta_lidar = 0.2, 0, np.pi

# Transformación homogénea del robot en la terna global
T_robot_global = np.array([
    [np.cos(theta_robot), -np.sin(theta_robot), x_robot],
    [np.sin(theta_robot), np.cos(theta_robot), y_robot],
    [0, 0, 1]
])

# Transformación homogénea del LIDAR en la terna del robot
T_lidar_robot = np.array([
    [np.cos(theta_lidar), -np.sin(theta_lidar), x_lidar_robot],
    [np.sin(theta_lidar), np.cos(theta_lidar), y_lidar_robot],
    [0, 0, 1]
])

# Transformación homogénea del LIDAR en la terna global
T_lidar_global = T_robot_global @ T_lidar_robot
x_lidar_global, y_lidar_global = T_lidar_global[:2, 2]

# Cargar datos del LiDAR
scan = np.loadtxt('/home/nataly/Escritorio/robotica/tp1/laserscan.dat')

# Definir ángulos de medición en la terna del LIDAR
angle = np.linspace(-np.pi/2, np.pi/2, np.shape(scan)[0], endpoint=True)


# Transformar las mediciones del LIDAR a coordenadas globales
x_measurements = []
y_measurements = []

for r, alpha in zip(scan,angle):
    p_lidar = np.array([r * np.cos(alpha), r * np.sin(alpha), 1])
    p_global = T_lidar_global @ p_lidar
    x_measurements.append(p_global[0])
    y_measurements.append(p_global[1])

# Graficar
plt.figure(figsize=(8, 6))
plt.scatter(x_robot, y_robot, c='blue', marker='s', label='Robot')
plt.scatter(x_lidar_global, y_lidar_global, c='red', marker='x', label='LIDAR')
plt.scatter(x_measurements, y_measurements, c='green', s=10)
plt.legend()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid()
plt.axis('equal')
plt.show()
