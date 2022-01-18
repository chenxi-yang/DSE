import random
import numpy as np
from keras.models import Sequential
from keras import models
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from six.moves import cPickle as pickle

def optPlan(x, maxD):
    return np.concatenate((maxD*np.sign(x[0:3]), [0, 0, 0]), axis=0)

def normalize(s, env):    
    high = env.observation_space.high
    low = env.observation_space.low

    mean = (high + low) / 2.0
    spread = abs(high - low) / 2.0

    return (s - mean) / spread

def F(ALL_x, opt_a, opt_b, g):
    col1 = ALL_x[:,3,None] - opt_b[:,0,None]
    col2 = ALL_x[:,4,None] - opt_b[:,1,None]
    col3 = ALL_x[:,5,None] - opt_b[:,2,None]
    col4 = g*opt_a[:,0,None]
    col5 = -g*opt_a[:,1,None]
    col6 = opt_a[:,2,None] - g
       
    return np.concatenate((col1,col2,col3,col4,col5,col6),axis=1)

#these dynamics are for the absolute states, not the relative ones
def F_nr(ALL_x, opt_a, g):    
    col1 = ALL_x[3]
    col2 = ALL_x[4]
    col3 = ALL_x[5]
    col4 = g*opt_a[0]
    col5 = -g*opt_a[1]
    col6 = opt_a[2] - g

    return np.array([col1,col2,col3,col4,col5,col6])

#This is the Runge Kutta method for the planner
def step_plan(planner_state, next_dir, dtt):

    der = np.array([next_dir[0], next_dir[1], next_dir[2], 0, 0, 0])
    
    k1 = der

    k2 = der
        
    k3 = der
        
    k4 = der

    Snx = planner_state + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4))
    return Snx;

#This is the Runge Kutta method for absolute states, not the relative ones
def step_nr(ALL_x,dtt,opt_a,g):

    k1 = F_nr(ALL_x,opt_a,g)
    ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1)

    k2 = F_nr(ALL_tmp,opt_a,g)
    ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2)
        
    k3 = F_nr(ALL_tmp,opt_a,g)
    ALL_tmp = ALL_x + np.multiply(dtt,k3)
        
    k4 = F_nr(ALL_tmp,opt_a,g)

    Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4))
    return Snx;

#This is the Runge Kutta method
def step(ALL_x,dtt,opt_a,opt_b,g):

    k1 = F(ALL_x,opt_a,opt_b,g)
    ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1)

    k2 = F(ALL_tmp,opt_a,opt_b,g)
    ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2)
        
    k3 = F(ALL_tmp,opt_a,opt_b,g)
    ALL_tmp = ALL_x + np.multiply(dtt,k3)
        
    k4 = F(ALL_tmp,opt_a,opt_b,g)

    Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4))
    return Snx;

def getAction(index):
    if index == 0:
        return [np.tan(-0.1), np.tan(-0.1), 7.81]
    elif index == 1:
        return [np.tan(-0.1), np.tan(-0.1), 11.81]
    elif index == 2:
        return [np.tan(-0.1), np.tan(0.1), 7.81]
    elif index == 3:
        return [np.tan(-0.1), np.tan(0.1), 11.81]
    elif index == 4:
        return [np.tan(0.1), np.tan(-0.1), 7.81]
    elif index == 5:
        return [np.tan(0.1), np.tan(-0.1), 11.81]
    elif index == 6:
        return [np.tan(0.1), np.tan(0.1), 7.81]
    return [np.tan(0.1), np.tan(0.1), 11.81]

def Conv(ALL_x):
    pos = ALL_x[0:3]/5.0
    vel = ALL_x[3:6]/10.0
    ret_val = np.concatenate((pos,vel),axis=0)
    return ret_val

g = 9.81
maxD = 0.25
dtt = 0.1
episodes = 100

states = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0])
planner = np.array([0.2, 0.1, 0.1, 0, 0, 0])
norm_states = Conv(states)

#model = models.load_model('./networks/tanh32x32_-2.0.h5')
#model = models.load_model('./networks/tanh32x32_-1.5.h5')
#model = models.load_model('./networks/tanh32x32_-1.0.h5')

model = models.load_model('./networks/tanh20x20_-0.5.h5')

maxDist = np.linalg.norm(states[0:3] - planner[0:3])

for e in range(episodes):

    observation = Conv(states - planner)

    # print(observation, states, planner)
    print(f"planner: {planner}")

    actions = model.predict(observation.reshape(1,len(observation)))
    action_index = np.argmax(actions[0])
    # print(f"action index: {action_index}")
    action = getAction(action_index)

    cur_plan = optPlan(states, maxD)
    planner = step_plan(planner, cur_plan, dtt)    
    
    states = step_nr(states, dtt, action, g)    

    newDist = np.linalg.norm(states[0:3] - planner[0:3])
    if newDist > maxDist:
        maxDist = newDist

print(planner)
print(states)
print(maxDist)
