from random import random
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation


def function(x,y):
    """Function which to find local mins or maxes

    Args:
        x (float): x-value
        y (float):y-value

    Returns:
        float: function value
    """     
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Constants
C1 = 0.4
C2 = 1.5
max_velocity = 5
start_range = [-5, 5]
weight = 1.4
min_weight = 0.3
beta = 0.9
number_of_particles = 40
max_iterations = 40
log_constant = 0.01

        
x = np.linspace(start_range[0], start_range[1], 500)
y = np.linspace(start_range[0], start_range[1], 500)
X,Y = np.meshgrid(x,y)
Z = function(X,Y)
logfunc = np.log(Z + log_constant)

best_particle = np.array([0, 0, np.infty])
# swarm- current x, current y, pb x, pb y, pb val
swarm = np.full((number_of_particles, 5), np.infty)
velocities = np.zeros((number_of_particles, 2))

for i in range(number_of_particles):
    swarm[i,0] = start_range[0] + random() * (start_range[1] - start_range[0])
    swarm[i,1] = start_range[0] + random() * (start_range[1] - start_range[0])

    velocities[i,0] = -(start_range[1] - start_range[0])/2 + random() * (start_range[1] - start_range[0])
    velocities[i,1] = -(start_range[1] - start_range[0])/2 + random() * (start_range[1] - start_range[0])

for iterations in range(max_iterations):
    
    for ind, particle in enumerate(swarm):
        current_value = function(particle[0], particle[1])
        if current_value < particle[4]:
            swarm[ind, 2], swarm[ind, 3] = particle[0], particle[1]
            swarm[ind, 4] = current_value
        
        if current_value < best_particle[2]:
            best_particle[0], best_particle[1] = particle[0], particle[1]
            best_particle[2] = current_value
        
        r = random()
        q = random()
        
        for i in range(2):
            velocities[ind, i] = weight * velocities[ind, i] + C1 * q * (
            particle[i+2] - particle[i])+ C2 * r * (best_particle[i] - particle[i])
        
            if velocities[ind, i] > max_velocity:
                velocities[ind, i] = max_velocity
            elif velocities[ind, i] < -max_velocity:
                velocities[ind, i] = -max_velocity
                
            swarm[ind, i] += velocities[ind, i]
    
    weight *= beta
    if weight < min_weight:
        weight = min_weight
    scatter = plt.scatter(swarm[:,0], swarm[:,1], color="black", zorder=2)
    plot = plt.contour(X,Y, logfunc)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.pause(0.05)
    plt.clf()
    

plt.show()


