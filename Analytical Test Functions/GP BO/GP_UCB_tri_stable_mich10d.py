from smt.sampling_methods import LHS
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
import numpy as np
# from lcb_gp import lcb_gp
from ei_gp import ei_gp
import time
from scipy.io import savemat
import warnings
warnings.filterwarnings("ignore")
import random

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
    
    np.random.seed(1234) 
    xx = np.random.uniform(low = xlimits[:,0],high = xlimits[:,1],size = (1000,d))
    
    for iters in range(max_iters):
    # for iters in range(2,max_iters):
        seed = iters
        np.random.seed(seed) 
        # kernel = kernel_init
        kernel = ConstantKernel(np.random.uniform()) * RBF() + ConstantKernel(np.random.uniform())
        
        
        lhs_sampling = LHS(xlimits = xlimits,random_state = seed, criterion = 'maximin')
        x = lhs_sampling(n)
        y = f(x).reshape(-1,1)
    
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        y = (y-y_mean)/y_std
        
        yy_dat = f(xx)
        actual_optimum = np.min(yy_dat)
        
        yy_dat = (yy_dat-y_mean)/y_std
    
        current_opt[iters,n-1] = np.amin(y)*y_std + y_mean
        regret[iters,n-1] = current_opt[iters,n-1] - actual_optimum
        
        print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1])     
        
         
        for i in range(n,n+new_n):
            start_t = time.time()
            
            np.random.seed(1000*iters + 2*i)
            gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,random_state=1000*iters + 2*i)
            gpr.fit(x, y)
           
           
            kernel = gpr.kernel_

            # yy = lcb_gp(xx,i,d,gpr)
            # minlcb_idx = np.argmin(yy)
            
            yy = ei_gp(xx,i,d,gpr,(current_opt[iters,i-1]-y_mean)/y_std)
            
            # print(np.sum(np.array(yy==0,dtype = 'int')))
            # if(np.std(yy)<1e-10):
            #     np.random.seed(4*iters + 3*i)
            #     minlcb_idx = np.random.randint(0,1000)
            # else:
            minlcb_idx = np.argmax(yy)
            # print(yy[minlcb_idx-2:minlcb_idx+2])
            
            if np.count_nonzero(yy== yy[minlcb_idx])>1:
                max_indices = np.where(yy == yy[minlcb_idx])[0]
                random.seed(40*iters + 3*i)
                minlcb_idx = random.choice(max_indices)

            
            x0 = xx[minlcb_idx,:]
            
            new_x = np.reshape(x0,(1,d))
            y_new = yy_dat[minlcb_idx].reshape(1,1)
            
            x = np.append(x,new_x,axis=0)
            y = np.append(y,y_new,axis=0)
      
            if (y_new < (current_opt[iters,i-1]-y_mean)/y_std):
                current_opt[iters,i] = y_new*y_std + y_mean
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

            print (minlcb_idx,'i = ',i ,' Regret Current = ',regret[iters,i],end="\r")
            
            if(regret[iters,i]<0.01):
                regret[iters,i+1:] = regret[iters,i]
                break
        
        print("\n")
        del x
        del y
        del gpr
        
        mdic = {"current_opt_GPEI": current_opt,"regret_GPEI": regret, "runtime_GPEI":time_elapsed}
        savemat(folder+label+"_GPEI_"+ str(time.asctime(time.localtime(time.time())))+"_iter"+str(iters)+".mat", mdic)
        
    return current_opt,regret,time_elapsed