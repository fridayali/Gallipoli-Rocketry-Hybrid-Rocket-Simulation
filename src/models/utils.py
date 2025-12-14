#This function is a basic sampler. It gives all function data based on time.
import numpy as np
def sampler(dt, total_time, function):
    y_values=[]
    x_values=np.arange(0, total_time + dt, dt)
    for t in x_values: 
        y= function(t)
        y_values.append(y)
    
    return x_values, np.array(y_values)
