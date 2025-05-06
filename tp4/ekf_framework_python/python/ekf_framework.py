# -*- coding: utf-8 -*-
"""
Extended Kalman Filter Framework

"""


from math import *
import numpy as np
import numpy.linalg as lng
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def read_world_data(filename):
    '''
    read_world_data reads the world data from the file and returns a dictionary
    with the key as the id of the landmark and the value as the x and y coordinates
    of the landmark
    '''
    world_dict = defaultdict()
    f = open(filename)
    for line in f:
        line_s  = line.split('\n')
        line_spl  = line_s[0].split(' ')
        world_dict[float(line_spl[0])] = [float(line_spl[1]),float(line_spl[2])]
    return world_dict


def read_sensor_data(filename):
    '''
    read_sensor_data reads the sensor data from the file and returns a dictionary
    with tuples as keys. The keys are either (timestamp,'odom') or (timestamp,'sensor')
    where timestamp is the timestamp of the data, and "odom" and "sensor" diferenciate
    between odometry and sensor data. The values are dictionaries with the odometry data
    or the sensor data. The odometry data has keys 'r1','t' and 'r2' for the rotation 1,
    translation and rotation 2 values respectively. The sensor data has keys 'id','range'
    and 'bearing' for the id of the landmark, the range and the bearing of the landmark
    respectively. The sensor data is stored in a list for each timestamp where the length
    depends on the number of landmarks observed at that timestamp.
    '''
    data_dict = defaultdict()
    timestamp=-1
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line_s = line.strip()
        line_spl = line_s.split(' ') # split the line
        if (line_spl[0] =='ODOMETRY'):
            data_dict[timestamp,'odom'] = {'r1':float(line_spl[1]),'t':float(line_spl[2]),'r2':float(line_spl[3])}   
            timestamp = timestamp+1                       
            
        if(line_spl[0]=='SENSOR'):
            if (timestamp,'sensor') not in data_dict:
                data_dict[timestamp,'sensor'] = {'id':[],'range':[],'bearing':[]}
            data_dict[timestamp,'sensor']['id'].append(float(line_spl[1]))    
            data_dict[timestamp,'sensor']['range'].append(float(line_spl[2]))
            data_dict[timestamp,'sensor']['bearing'].append(float(line_spl[3]))
                                        
    return data_dict


def prediction_step(mu, sigma, odometry):
    #Read in the state from the mu vector
    x,y,theta = mu 
    #Read in the odometry i.e r1, t , r2
    r1 = odometry ['r1']
    t = odometry ['t']
    r2 = odometry ['r2']
    #Compute the noise free motion.
    mu = np.array([
        x + t * np.cos(theta + r1),
        y + t * np.sin(theta + r1),
        theta + r1 + r2
    ])
    #Computing the Jacobian of G with respect to the state
    Gt = np.array([
        [1, 0, -t * np.sin(theta + r1)],
        [0, 1,  t * np.cos(theta + r1)],
        [0, 0, 1]
    ])
    #Use the Motion Noise as given in the exercise 
    Q = [[0.2, 0,   0],
        [0,   0.2, 0],
        [0,   0,   0.02]]
    # Predict the covariance
    sigma = Gt @ sigma @ Gt.T + Q
    return mu, sigma
    
    
def correction_step(mu, sigma, measurements, world_dict):
    x, y, theta = mu
    ids = measurements['id']
    ranges = measurements['range']
    
    n = len(ids)
    H = np.zeros((n, 3))
    expected_ranges = np.zeros(n)

    for i, lm_id in enumerate(ids):
        lx, ly = world_dict[lm_id]
        dx = x - lx
        dy = y - ly
        q = np.sqrt(dx**2 + dy**2)

        if q == 0:
            continue

        # Jacobiana Hi
        H[i, 0] = dx / q
        H[i, 1] = dy / q
        H[i, 2] = 0

        # h(mu): medición esperada
        expected_ranges[i] = q

    # Matriz de ruido de sensor (asumimos independiente)
    Rt = np.eye(n) * 0.5

    # Ganancia de Kalman
    S = H @ sigma @ H.T + Rt
    K = sigma @ H.T @ np.linalg.inv(S)

    # Diferencia entre medición real y esperada
    z = np.array(ranges)
    z_hat = expected_ranges
    innovation = z - z_hat

    # Corrección de la media y covarianza
    mu = mu + K @ innovation
    sigma = (np.eye(3) - K @ H) @ sigma

    return mu, sigma



def plot_ellipse(ax, mu, sigma, color="k"):
    """
    Draws ellipse from xy of covariance matrix
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(xy=mu, width=w, height=h, angle=theta, color=color)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)


  
        
## Main loop Starts here
parser = argparse.ArgumentParser()
parser.add_argument('sensor_data', type=str, help='Sensor Data')
parser.add_argument('world_data', type=str, help='World Data')

args= parser.parse_args()

data_dict = read_sensor_data(args.sensor_data)
world_data = read_world_data(args.world_data)


#Initial Belief
mu = np.array([0.0,0.0, 0.0]).T
sigma =np.array([[1.0,0.0 , 0.0],[0.0, 1.0 , 0.0],[0.0, 0.0 , 1.0]])


# Landmark Positions
lx=[]
ly=[]

for i in range (len(world_data)):
    lx.append(world_data[i+1][0])
    ly.append(world_data[i+1][1])



for t in range(len(data_dict)//2):
    # Perform the prediction step of the EKF
    [mu, sigma] = prediction_step(mu, sigma, data_dict[t,'odom'])
   
    # Perform the correction step of the EKF
    [mu, sigma] = correction_step(mu, sigma, data_dict[t,'sensor'], world_data)
    
    x_pos = mu[0]
    y_pos = mu[1]
    
    
    ''' Plotting  the state Estimate '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(x_pos,y_pos,'ro',markersize=10)
    quiver_len = 3.0
    theta_n = mu[2]
    plt.quiver(x_pos, y_pos , quiver_len * np.cos(theta_n), quiver_len * np.sin(theta_n),angles='xy',scale_units='xy')

    plot_ellipse(ax, mu[0:2], sigma[0:2,0:2], color="g")

    plt.plot(lx,ly,'bo',markersize=10)
    plt.axis([-2, 12, -2, 12])
    plt.draw()
    file = "plots/ekf_plot_" + str(t).zfill(3)
    plt.savefig(file)
    plt.close()

    # Note: for creating animation, in plots folder do:
    # ffmpeg -framerate 25 -i ekf_plot_%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ekf_movie.mp4

    
    