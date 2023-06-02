import numpy as np
from scipy.io import savemat
import sys

#from dgpsi import dgp, kernel, combine, lgp, path, emulator
import dgp
from kernel_class import kernel
from kernel_class import combine
from likelihood_class import Hetero

from SI_UCB_stable import DGPSI_LCB_stable

from eval_functions_28Apr2022 import *
import time

def main():
    args = sys.argv[1:]
    label = args[0]
    
    if(label == 'art1'):
        dispatcher={label:art1}
        d = 1
        actual_optimum = -1.0578 
        folder = './art1/'
        max_steps = 100
    elif(label == 'perm3'):
        dispatcher={label:perm3}
        d = 3
        actual_optimum = 0
        folder = './perm3/'
        max_steps = 200
    elif(label == 'schwef3'):
        dispatcher={label:schwef3}
        d = 3
        actual_optimum = 0
        folder = './schwef3/'
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
        max_steps = 200
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
    else:
        dispatcher = 'xxx'
        d=0
        actual_optimum = 0
        folder = './xxx/'
    
    try:
        f=dispatcher[label]
    except KeyError:
        raise ValueError('invalid function name')
    

    if (d==1):
        layer1=[kernel(length=np.array([1]),name='sexp')]
        layer2=[kernel(length=np.array([1]),name='sexp',connect=np.arange(1))]
        all_layer=combine(layer1,layer2) 
    else: 
        layer1 = []   
        for _ in range(d):
            layer1.append(kernel(length=np.array([1]),name='sexp',nugget_est=True))
        layer2 = [kernel(length=np.array([1]),name='sexp',nugget_est=True,connect = np.arange(d))]
        all_layer=combine(layer1,layer2)

    #n_layers = 2

    
    n = 5*d 
    new_n = max_steps-n
    
    if (d<4):
        N_predict = 50
        bag_size = 50
    else:
        N_predict = 25
        bag_size = 25
    
    if (d>7):
        max_iters = 5
        m = 100
        N_train = 150
    else:
        max_iters = 10
        m = 100
        N_train = 150
    
    # N_predict = 1000
    # bag_size = 1000
        
    if(d==1):    
        n= 3
    else:
        n = 5*d
        
    x_lb = np.zeros((d,1))
    x_ub = np.ones((d,1))

    xlimits = np.hstack((x_lb,x_ub))

    current_opt,regret,run_time_DGP = DGPSI_LCB_stable(max_iters=max_iters,xlimits=xlimits,n=n,d=d,new_n=new_n,m = m,f=f,actual_optimum=actual_optimum,all_layer_init=all_layer,N_train=N_train,N_predict=N_predict,label = label,folder = folder,bag_size = bag_size)
    mdic = {"current_opt_DGPUCB": current_opt,"regret_DGPUCB": regret, "runtime_DGPUCB":run_time_DGP}
    savemat(label+"_DGPUCB_full"+str(time.asctime(time.localtime(time.time())))+".mat", mdic)

if __name__ == "__main__":
    main()