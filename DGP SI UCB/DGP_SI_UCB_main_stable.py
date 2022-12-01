import numpy as np
from scipy.io import savemat
import sys

#from dgpsi import dgp, kernel, combine, lgp, path, emulator
import dgp
from kernel_class import kernel
from kernel_class import combine

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
    elif(label == 'perm3'):
        dispatcher={label:perm3}
        d = 3
        actual_optimum = 0
        folder = './perm3/'
    elif(label == 'crossit'):
        dispatcher={label:cross_in_tray}
        d = 2
        actual_optimum = -2.06261
        folder = './crossit/'
    elif(label == 'camel6h'):
        dispatcher={label:camel_6h}
        d = 2
        actual_optimum = -1.0316
        folder = './camel6h/'
    elif(label == 'camel3h'):
        dispatcher={label:camel_3h}
        d = 2
        actual_optimum = 0
        folder = './camel3h/'
    elif(label == 'levy'):
        dispatcher={label:levy}
        d = 3
        actual_optimum = 0
        folder = './levy/'
    elif(label == 'hart4'):
        dispatcher={label:Hartmann4}
        d = 4
        actual_optimum = -3.135474
        folder = './hart4/'
    elif(label == 'hart3'):
        dispatcher={label:Hartmann3}
        d = 3
        actual_optimum = -3.86278
        folder = './hart3/'
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
        layer1=[kernel(length=np.array([1]),name='sexp'),kernel(length=np.array([1]),name='sexp')]
        layer2=[kernel(length=np.array([1]),name='sexp',connect=np.arange(2))]
        all_layer=combine(layer1,layer2)

    #n_layers = 2

    max_iters = 10
    n = 10*d
    new_n = 200-n
    m = 100

    x_lb = np.zeros((d,1))
    x_ub = np.ones((d,1))

    xlimits = np.hstack((x_lb,x_ub))

    

    current_opt,regret,run_time_DGP = DGPSI_LCB_stable(max_iters=max_iters,xlimits=xlimits,n=n,d=d,new_n=new_n,m = m,f=f,actual_optimum=actual_optimum,all_layer_init=all_layer,N_train=150,N_predict=50,label = label,folder = folder)
    mdic = {"current_opt_DGPUCB": current_opt,"regret_DGPUCB": regret, "runtime_DGPUCB":run_time_DGP}
    savemat(label+"_DGPUCB_full"+str(time.asctime(time.localtime(time.time())))+".mat", mdic)

if __name__ == "__main__":
    main()