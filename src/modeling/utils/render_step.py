
import math


def fibonacci_sphere(samples=12, distance=5):
    """
    :param samples: number of views
    :param distance: distance from the center of the object
    :return: # samples points of view around the model
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = (1 - (i / float(samples - 1)) * 2) * distance   # y goes from 1 to -1
        radius = math.sqrt(distance * distance - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
        
        ass = math.sqrt(math.pow(x,2)+math.pow(y,2)+math.pow(z,2) )
        
    return points


