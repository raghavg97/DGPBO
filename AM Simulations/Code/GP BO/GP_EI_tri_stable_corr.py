from smt.sampling_methods import LHS
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
import numpy as np
from ei_gp import ei_gp
import time
from scipy.io import savemat
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import triangulation

def GP_BO_EI(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,kernel_init,lcb_gp,label,folder):
    current_opt = np.empty((max_iters,n+new_n))
    regret = np.empty((max_iters,n+new_n))
    time_elapsed = np.empty((max_iters,n+new_n))
    current_opt[:] = np.NaN
    regret[:] = np.NaN
    time_elapsed[:] = np.NaN 
    
    init_t = time.time()
    
    log_file_name = label+str(int(time.time()))+'.txt'
    log_file = open(log_file_name, 'w')
    
    x_shift = np.array([233.25,772.17,0.132]).reshape(-1,3)
    x_scale = np.array([250.0,800.0,0.150]).reshape(-1,3)
        
    y_shift = -80.0
    y_scale = 100.0
    
    y_pred= np.zeros((n,))
    for loocv in range(n):
        
    # for iters in range(max_iters):
    # for iters in range(4,max_iters):
        seed = 0
        # all_layer = all_layer_init
        kernel = kernel_init
        np.random.seed(seed) 
        dat = np.array(pd.read_csv('dataBO.csv', header = None))
        
        
        # dat = np.array(pd.read_csv('init'+'%.2d'%(iters+1)+'_BO.csv', header = None))
        dat = np.array(pd.read_csv('init'+'%.2d'%(0+1)+'_BO.csv', header = None))
        
        dat_1 = np.delete(dat,loocv,axis= 0)
        x = dat_1[:,:3]
        y = -1.0*dat_1[:,3].reshape(-1,1)
        
        x = (x-x_shift)/x_scale
        y = (y-y_shift)/y_scale
        
        xx_dat = dat[loocv,:3]
        y_dat = -1.0*dat[loocv,3].reshape(-1,1)
    
        xx = (xx_dat - x_shift)/x_scale
        yy_dat = (y_dat-y_shift)/y_scale
      
      
    
#         current_opt[iters,n-1] = np.amin(y)*y_scale + y_shift
#         regret[iters,n-1] = current_opt[iters,n-1] - actual_optimum
        
        # print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1])     
        
        i=n
        
        start_t = time.time()

        np.random.seed(1000*loocv + 2*i)
        gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,random_state=i)
        gpr.fit(x, y)

        y_pred[loocv] = gpr.predict(xx)*y_scale + y_shift
        # kernel = gpr.kernel_
    
            
        print("LOOCV: ",loocv,"y_mean: ",y_pred[loocv])
        del x
        del y
        del gpr

        print("\n")
        
    mdic = {"y_pred_GP": y_pred}
    savemat("y_pred_GP"+ str(time.asctime(time.localtime(time.time())))+".mat", mdic)
        
    return None