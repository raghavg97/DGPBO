from smt.sampling_methods import LHS
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
import numpy as np
from lcb_gp import lcb_gp
import time
from scipy.io import savemat
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import triangulation

def GP_BO(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,kernel_init,lcb_gp,label,folder):
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
    
    for iters in range(max_iters):
        
        seed = 0
        kernel = kernel_init
        
        dat = np.array(pd.read_csv('dataBO.csv', header = None))
        xx_dat = dat[:,:3]
        y_dat = -1.0*dat[:,3].reshape(-1,1)
    
        xx = (xx_dat - x_shift)/x_scale
        yy_dat = (y_dat-y_shift)/y_scale
        
        # dat = np.array(pd.read_csv('init'+'%.2d'%(iters+1)+'_BO.csv', header = None))
        dat = np.array(pd.read_csv('init'+'%.2d'%(0+1)+'_BO.csv', header = None))
        
        x = dat[:,:3]
        y = -1.0*dat[:,3].reshape(-1,1)
        
        x = (x-x_shift)/x_scale
        y = (y-y_shift)/y_scale
    
        current_opt[iters,n-1] = np.amin(y)*y_scale + y_shift
        regret[iters,n-1] = current_opt[iters,n-1] - actual_optimum
        
        print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1])     
        
        for i in range(n,n+new_n):
            start_t = time.time()
            
            np.random.seed(1000*iters + 2*i)
            gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,random_state=1000*iters + 2*i)
            gpr.fit(x, y)
           
           
            kernel = gpr.kernel_
    
            yy = lcb_gp(xx,i,d,gpr)
            minlcb_idx = np.argmin(yy)
            
            x0 = xx[minlcb_idx,:]
            new_x = np.reshape(x0,(1,d))
            y_new = yy_dat[minlcb_idx].reshape(1,1)
            
            x = np.append(x,new_x,axis=0)
            y = np.append(y,y_new,axis=0)
            
            # xx = np.delete(xx,minlcb_idx,axis = 0)
            # yy_dat = np.delete(yy_dat,minlcb_idx,axis = 0)
      
            if (y_new < (current_opt[iters,i-1] - y_shift)/y_scale):
                current_opt[iters,i] = y_new*y_scale + y_shift
                regret[iters,i] = current_opt[iters,i] - actual_optimum
            else:
                current_opt[iters,i] = current_opt[iters,i-1]
                regret[iters,i] = regret[iters,i-1]
            end_t = time.time()
            time_elapsed[iters,i] = end_t - start_t
            
            if(i%20==0):
                log_file = open(log_file_name, 'a')
                log_file.write('\n')
                log_file.write("Iters "+str(iters)+" i=" + str(i) + " Time "+str(time.asctime(time.localtime(time.time()))))
                log_file.close()

            print (minlcb_idx,' i = ',i ,' Regret Current = ',regret[iters,i],end="\r")
            
            if(regret[iters,i]<0.5):
                regret[iters,i+1:] = regret[iters,i]
                break
            
        del x
        del y
        del gpr
        
        print("\n")
        
        mdic = {"current_opt_GPUCB": current_opt,"regret_GPUCB": regret, "runtime_GPUCB":time_elapsed}
        savemat(folder+label+"_GPUCB_"+ str(time.asctime(time.localtime(time.time())))+"_iter"+str(iters)+".mat", mdic)
        
    return current_opt,regret,time_elapsed