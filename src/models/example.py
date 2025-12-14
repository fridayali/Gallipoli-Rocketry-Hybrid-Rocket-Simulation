#This file indicates regulations for the python functions.
#Such as how a function inputs and outputs variables.
#This regulations firstly for the user API layer between GUI and models,
#and also this is for having more readable codes.



#!!!! ADD LIBRARIES ARE USED TO THE requirements.txt

import numpy
import matplotlib
#etc...

#if there is a function that is used many times and from different .py files,
#it can be added utils.py file
#Then you can use function as importing utils  

import utils
import matplotlib.pyplot as plt
#using of utils:
dt=0.01
total_time=10
def flow_time_function(): pass
function= flow_time_function
time_data, y_data = utils.sampler(dt, total_time, function)
plt.plot(time_data,y_data)
plt.xlabel("Time")
plt.ylabel("Flow rate")
