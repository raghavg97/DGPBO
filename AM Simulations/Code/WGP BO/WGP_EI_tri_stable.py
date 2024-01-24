import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import scipy
import torch
from botorch.distributions import Kumaraswamy
from botorch import fit_gpytorch_model
from EI_WGP import ei_wgp
# from wgp_helper import wgp_helper
from botorch.utils.transforms import standardize
from botorch.models.transforms.input import Warp
from botorch.models import SingleTaskGP
import pandas as pd

import gpytorch

from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior



import triangulation
from scipy.io import savemat

import time



def wgp_ei_v1(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,dtype,device,label,folder):
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
        
        x_tensor = torch.from_numpy(x).float().to(device)
        y_obj_tensor = torch.from_numpy(y).float().to(device)
        
        print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1])     
    
        # torch.manual_seed(iters)
        # c1 = torch.rand(d, dtype=dtype, device=device) * 3 + 0.1
        # c0 = torch.rand(d, dtype=dtype, device=device) * 3 + 0.1
        # k = Kumaraswamy(concentration1=c1, concentration0=c0)
        
        for i in range(n,n+new_n):
            start_t = time.time()
            np.random.seed(1000*iters + 2*i)
            #tf.random.set_seed(1000*iters + 2*i)
            torch.manual_seed(1000*iters + 2*i)
       
            #warp_tf = Warp(indices=list(range(x_tensor.shape[-1])),concentration1_prior=LogNormalPrior(0.0, 0.75 ** 0.5),concentration0_prior=LogNormalPrior(0.0, 0.75 ** 0.5))
            gpytorch.settings.cholesky_jitter(float=None, double=None, half=None)
            warp_tf = Warp(indices=list(range(x_tensor.shape[-1])),concentration1_prior=LogNormalPrior(0.0, 0.75 ** 0.5),concentration0_prior=LogNormalPrior(0.0, 0.75 ** 0.5))
    
            #model = SingleTaskGP(x_tensor,standardize(y_obj_tensor),input_transform=warp_tf).to(x_tensor)
            model = SingleTaskGP(x_tensor,standardize(y_obj_tensor),input_transform=warp_tf).to(x_tensor)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
            
            fit_gpytorch_model(mll,max_retries=10)

            # print(xx.shape)
            yy = ei_wgp(xx,model, (current_opt[iters,i-1]- y_shift)/y_scale,device,dtype,d)
            
            maxei_idx = np.argmax(yy)
            # print(maxei_idx)
            new_x = xx[maxei_idx,:].reshape(1,d)
            y_new = yy_dat[maxei_idx].reshape(1,1)
            
            #print(new_x)

            # if (np.sum(new_x>1) > 0):
            #     new_x[new_x>1] = 1
            #     print("A value atleast crossed 1 in new_x")
            # elif(np.sum(new_x<0) > 0):
            #     new_x[new_x<0] = 0
            #     print("A value atleast went below in new_x")
                 
            x = np.append(x,new_x,axis=0)
            x_tensor = torch.cat((x_tensor,torch.from_numpy(new_x).float().to(device)))
            #y_new_tensor = wgp_helper(torch.from_numpy(new_x),f,xlimits,k,dtype,device)
            y_new_tensor = torch.from_numpy(y_new).float().to(device)
            #print(y_new)
            
            y_obj_tensor = torch.cat((y_obj_tensor,y_new_tensor))
            
            # x = np.append(x,new_x)
            # y = np.append(y,y_new_tensor.detach().numpy())
            
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

            print (maxei_idx,' i = ',i ,' Regret Current = ',regret[iters,i],end="\r")
            
            if(regret[iters,i]<0.5):
                regret[iters,i+1:] = regret[iters,i]
                break
            
        del x
        del y
        print("\n")
        
        mdic = {"current_opt_WGPEI": current_opt,"regret_WGPEI": regret, "runtime_WGPEI":time_elapsed}
        savemat(folder+label+"_WGPEI_"+ str(time.asctime(time.localtime(time.time())))+"_iter"+str(iters)+".mat", mdic)

    return current_opt,regret,time_elapsed