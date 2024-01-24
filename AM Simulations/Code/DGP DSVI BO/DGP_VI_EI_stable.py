import numpy as np
from scipy.io import savemat
import sys

#from DSVI_EI_analytical_5May import DSVI_EI_v1
from DSVI_EI_analytical_stable import DSVI_EI_v1

from eval_functions_28Apr2022 import *
import time
import warnings
warnings.filterwarnings("ignore")

def main():
    args = sys.argv[1:]
    label = args[0]
    
    if(label == 'art1'):
        dispatcher={label:art1}
        d = 1
        actual_optimum = -1.0578 
        folder = './art1/'
    elif(label == 'perm3'):
        dispatcher={label:perm3}
        d = 3
        actual_optimum = 0
        folder = './perm3/'
        max_steps = 200
    elif(label == 'crossit'):
        dispatcher={label:cross_in_tray}
        d = 2
        actual_optimum = -2.06261
        folder = './crossit/'
        max_steps = 150
    elif(label == 'camel6h'):
        dispatcher={label:camel_6h}
        d = 2
        actual_optimum = -1.0316
        folder = './camel6h/'
        max_steps = 150
    elif(label == 'camel3h'):
        dispatcher={label:camel_3h}
        d = 2
        actual_optimum = 0
        folder = './camel3h/'
        max_steps = 150
    elif(label == 'levy'):
        dispatcher={label:levy}
        d = 3
        actual_optimum = 0
        folder = './levy/'
        max_steps = 200
    elif(label == 'hart4'):
        dispatcher={label:Hartmann4}
        d = 4
        actual_optimum = -3.135474
        folder = './hart4/'
        max_steps = 250
    elif(label == 'hart3'):
        dispatcher={label:Hartmann3}
        d = 3
        actual_optimum = -3.86278
        folder = './hart3/'
        max_steps = 200
    elif(label == 'hart6'):
        dispatcher={label:Hartmann6}
        d = 6
        actual_optimum = -3.32236801 
        folder = './hart6/'
        max_steps = 200
    elif(label == 'mich10d'):
        dispatcher={label:mich10d}
        d = 10
        actual_optimum = -9.66015
        folder = './mich10d/'
        max_steps = 400
        max_iters = 5
    elif(label == 'shekel4'):
        dispatcher={label:shekel4}
        d = 4
        actual_optimum = -10.5364
        folder = './shekel4/'
        max_steps = 200
    elif(label == 'netfabb'):
        dispatcher={label:netfabb}
        d = 3
        actual_optimum = -178.948476
        folder = './netfabb/'
        max_steps = 150
    else:
        dispatcher = 'xxx'
        d=0
        actual_optimum = 0
        folder = './xxx/'
    
    try:
        f=dispatcher[label]
    except KeyError:
        raise ValueError('invalid function name')
    
    n_layers = 2

    
    n = 5*d 
    new_n = max_steps-n
    
    max_iters = 5
    m = 100
        
    x_lb = np.zeros((d,1))
    x_ub = np.ones((d,1))

    xlimits = np.hstack((x_lb,x_ub))
    
    current_opt_EI,regret_EI,run_time_EI = DSVI_EI_v1(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,n_layers,label = label,folder = folder)
    mdic = {"current_opt_DGPBO_EI": current_opt_EI,"regret_DGPBO_EI": regret_EI,"runtime_EI":run_time_EI}
    savemat(label+"_DGPVIEI_full"+str(time.asctime(time.localtime(time.time())))+".mat", mdic)

if __name__ == "__main__":
    main()