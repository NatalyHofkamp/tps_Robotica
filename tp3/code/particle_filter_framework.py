from math import *
import numpy as np
import random
import argparse
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


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
    
        
class robot():

    def __init__(self):
        '''
        __init__ : creates the robot and initializes the location/orientation estimates
        '''
        self.x = random.random()  # initial x position
        self.y = random.random() # initial y position
        self.orientation = random.uniform(-math.pi,math.pi) # initial orientation
        self.weights = 1.0
       

    def set(self, new_x, new_y, new_orientation):
        '''
        set: sets a robot coordinate, including x, y and orientation
        '''
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
       
       
    def measurement_prob_range(self, ids, ranges, wrld_dict):
        '''
        measurement_prob_range:
            computes the probability of a measurement. This function takes in the ranges
            and ids of the sensor data for each measurement update. World data is passed
            as the dictionary. The positions of the landmarks in the world can be accessed
            as wrld_dict[ids[i]][0],wrld_dict[ids[i]][1] where i is looped over the number
            of measurements for the current timestamp

            Complete this Stub for the measurement model which is based on range only
            Range only implies that the error has to be calculated based only on the range values
        '''
        error = 1
            for landmark in ids:
                    lx,ly = wrld_dict[landmark]
                    exp_dist = np.sqrt((self.x - lx)**2 + (self.y - ly)**2 )
                    real_dist = ranges[landmark]
                    error *= norm.pdf(real_dist, loc=exp_dist, scale=0.2)
        return error


    def mov_odom(self,odom,noise):
        '''
        move_odom: Takes in Odometry data and moves the robot based on the odometry data
        '''
    
        # Calculate the distance and Guassian noise
        dist  = odom['t']
        # calculate delta rotation 1 and delta rotation 1

        delta_rot1  = odom['r1']
        delta_rot2 = odom['r2']

        # noise sigma for delta_rot1 
        sigma_delta_rot1 = noise[0]*abs(delta_rot1)  + noise[1]*abs(dist)
        delta_rot1_noisy = delta_rot1 + random.gauss(0,sigma_delta_rot1)

        # noise sigma for translation
        sigma_translation = noise[2]*abs(dist)  + noise[3]*abs(delta_rot1+delta_rot2)
        translation_noisy = dist + random.gauss(0,sigma_translation)

        # noise sigma for delta_rot2
        sigma_delta_rot2 = noise[0]*abs(delta_rot2)  + noise[1]*abs(dist)
        delta_rot2_noisy = delta_rot2 + random.gauss(0,sigma_delta_rot2)

        # Estimate of the new position of the robot
        x_new = self.x  + translation_noisy * cos(self.orientation+delta_rot1_noisy)
        y_new = self.y  + translation_noisy * sin(self.orientation+delta_rot1_noisy)
        theta_new = self.orientation + delta_rot1_noisy + delta_rot2_noisy    

        result = robot()
        result.set(x_new, y_new,theta_new )
        
        return result


    def set_weights(self, weight):
        '''
        set_weights: sets the weight of the particles
        '''
        #noise parameters
        self.weights  = float(weight)



# def get_mean_position(p,t):
#     '''
#     get_mean_position: extract position from particle set
#     '''
#     x = 0.0
#     y = 0.0
#     orientation = 0.0
   
#     # TODO: Write the code here to calculate the mean position and orientation
#     # of all the particles

#     # Particles are plotted here 
#     # Lists x_pos and y_pos contains the x and y positions of all the particles which
#     # can be accessed from p[i].x, p[i].y. avg_oreint is the average orientation
#     # of all the particles
        
#     plt.clf()
#     plt.plot(x_pos,y_pos,'r.')
#     quiver_len = 3.0
#     plt.quiver(x, y, quiver_len * np.cos(orientation), quiver_len * np.sin(orientation),angles='xy',scale_units='xy')

#     plt.plot(lx,ly,'bo',markersize=10)
#     plt.axis([-2, 12, -2, 12])
#     plt.draw()
#     ##
#     # file = "./plots/pf_plot_" + str(t).zfill(3)
#     # plt.savefig(file)
#     # Note: for creating animation, in plots folder do:
#     # ffmpeg -framerate 25 -i pf_plot_%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p pf_movie.mp4
#     plt.pause(0.1)
#     return [x, y, orientation]


def resample_particles(weights, particles):
    '''
    resample_particles: takes the current particles and weights and resamples them based on that.
    '''
    # TODO: Complete this Stub to resample the weights of the particles

    #Sum the weights 
    
    
    
    
    
    #normalize the weights
    
    

    
    
    
    # calculate the PDF of the weights
    pdf=[]        
    for k in range(len(p)):
        

    

    # Calculate the step for random sampling , it depends on number of 
    #particles

    
    step=        
    
    
    # Sample a value in between [0,step) uniformly
    
    seed = 
    #print 'Seed is %0.15s and step is %0.15s' %(seed, step)


    # resample the particles based on the seed , step and cacluated pdf
    for h in range(len(p)):
        # Write the code here                 
            
            
        
    return p_sampled


def particle_filter(data_dict,world_dict, N):
    '''
    particle_filter: The main particle filter function. Creates the particles and updates them
    ''' 

    # Make particles
    p = []
    for i in range(N):
        r = robot()
        p.append(r)

    # Update particles
    for t in range((len(data_dict)-1)//2):
        # motion update (prediction)
        p2 = []
        for i in range(N):
            p2.append(p[i].mov_odom(data_dict[t,'odom'],noise_param))
        p = p2

        # measurement update
        w = []
        for i in range(N):
            w.append(p[i].measurement_prob_range(data_dict[t,'sensor']['id'],data_dict[t,'sensor']['range'],world_dict))
           
        
        # TODO: Complete this Stub to resample the weights of the particles


        get_mean_position(p,t)
        p = resample_particles(w,p)
        
    return get_mean_position(p,t)

## Main loop Starts here
parser = argparse.ArgumentParser()
parser.add_argument('sensor_data', type=str, help='Sensor Data')
parser.add_argument('world_data', type=str, help='World Data')
parser.add_argument('N', type=int, help='Number of particles')


args = parser.parse_args()
N = args.N


noise_param = [0.1, 0.1 ,0.05 ,0.05]

plt.ion()
plt.axis([0, 15, 0, 15])

data_dict = read_sensor_data(args.sensor_data)
world_data  =read_world_data(args.world_data)

lx=[]
ly=[]

for i in range (len(world_data)):
    lx.append(world_data[i+1][0])
    ly.append(world_data[i+1][1])

estimated_position = particle_filter(data_dict,world_data,N)

plt.ioff()
plt.show()