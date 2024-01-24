import numpy as np
from scipy.io import savemat

from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

from lcb_gp import lcb_gp
from GP_UCB_tri_stable import GP_BO

from eval_functions_28Apr2022 import *
import sys
import time

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
    elif(label == 'mich10d'):
        dispatcher={label:mich10d}
        d = 10
        actual_optimum = -9.66015
        folder = './mich10d/'
        max_steps = 400
        max_iters = 5
    elif(label == 'langer'):
        dispatcher={label:langer}
        d = 4
        actual_optimum = -1.4
        folder = './langer/'
        max_steps = 200
    elif(label == 'shekel4'):
        dispatcher={label:shekel4}
        d = 4
        actual_optimum = -10.5364
        folder = './shekel4/'
        max_steps = 200
    elif(label == 'netfabb'):
        dispatcher={label:netfabb}
        d = 3
        # actual_optimum = -178.94847600602935
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
    

    #n_layers = 2
    max_iters = 5
    n = 5*d 
    new_n = max_steps-n
    m = 100

    x_lb = np.zeros((d,1))
    x_ub = np.ones((d,1))

    xlimits = np.hstack((x_lb,x_ub))
    
    rbf_kernel = ConstantKernel(1.0) * RBF() + ConstantKernel(1.0)
    current_opt,regret,run_time_GP = GP_BO(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,rbf_kernel,lcb_gp,label,folder)

    mdic = {"current_opt_GPUCB": current_opt,"regret_GPUCB": regret, "runtime_GPUCB":run_time_GP}
    savemat(label+"_GPUCB_full"+str(time.asctime(time.localtime(time.time())))+".mat", mdic)

if __name__ == "__main__":
    main()