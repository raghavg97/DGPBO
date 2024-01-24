#THIS IS FOR POST PRELIMINARY UPDATE
from dgp import dgp
import kernel_class
import linkgp
from emulation import emulator
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import scipy
import triangulation
from scipy import stats as st

from scipy.io import savemat
from LCB_DGPSI_stable import lcb_stable
import time
import pandas as pd

def DGPSI_LCB_stable(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,all_layer_init,N_train,N_predict,label,folder,bag_size):
    current_opt = np.empty((max_iters,n+new_n))
    regret = np.empty((max_iters,n+new_n))
    time_elapsed = np.empty((max_iters,n+new_n))
    current_opt[:] = np.NaN
    regret[:] = np.NaN
    time_elapsed[:] = np.NaN 
    
    init_t = time.time()
    
    log_file_name = label+str(int(time.time()))+'.txt'
    log_file = open(log_file_name, 'w')
    
    if(d>2):
        total_bags = 400
    else:
        total_bags = 100
        
    x_shift = np.array([233.25,772.17,0.132]).reshape(-1,3)
    x_scale = np.array([250.0,800.0,0.150]).reshape(-1,3)
        
    y_shift = -80.0
    y_scale = 100.0
#     
    # for iters in range(max_iters):
    for iters in range(4,max_iters):
        seed = 0
        all_layer = all_layer_init
        
        np.random.seed(seed) 
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
        
        print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1] )
    
        for i in range(n,n+new_n):
            start_t = time.time()
            np.random.seed(1000*iters + 2*i)  
          
            model=dgp(x,[y],1000*iters + 2*i,all_layer = all_layer,check_rep = False)
            
            if(i==n):
                model.initialize(seed = 1000*iters + 2*i)
                model.train(N=500,disable = True)
                all_layer=model.estimate()
            else:
                model.train(N=N_train,disable = True)
                all_layer=model.estimate(burnin = int(N_train/2))
            
            emu=emulator(all_layer,N=N_predict,nb_parallel = True)
        
            np.random.seed(1000*iters + 2*i) 
          
            #res = emu.ppredict(x=np.reshape(xx,(-1,d)), sample_size=1, method='mean_var',chunk_num = 28,core_num = 28)
            res = emu.ppredict(x=np.reshape(xx,(-1,d)), sample_size=1, method='mean_var',chunk_num = 28,core_num = 28)
            

            mu_all = np.transpose(np.array(res[2]))
            var_all = np.transpose(np.array(res[3]))
             
            #print(mu_all.shape)
            #print(var_all.shape)
            

            minlcb_idx_bag  = np.zeros((total_bags,),dtype = int)
            
            for bag in range(total_bags):
                np.random.seed(900*iters + 8*i + 5*bag)
                bag_idx = np.random.randint(0,N_predict,size = bag_size)
                mu_all_bag = mu_all[bag_idx]
                var_all_bag = var_all[bag_idx]
          
                yy = lcb_stable(mu_all = mu_all_bag,var_all = var_all_bag,n=i,d=d)
               
                minlcb_idx_bag[bag] = np.argmin(yy)
            
            # minlcb_idx = np.array(st.mode(minlcb_idx_bag)[0],dtype = int)
            # minlcb_idx = np.array(np.median(minlcb_idx_bag),dtype = int)
            # x0 = xx[minlcb_idx,:]
            
            x_bag_mean = np.mean(xx[minlcb_idx_bag,:],axis = 0).reshape(1,d)
            dist_bag = np.linalg.norm(xx-x_bag_mean,axis=1)
            # print(dist_bag.shape)
            minlcb_idx = np.argmin(dist_bag)
             
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

            print(minlcb_idx,' i = ',i ,' Regret Current = ',regret[iters,i],' Time = ',time_elapsed[iters,i],end="\r")
        #print('Time Lapsed = ',time.time() - init_t)
        
            if(regret[iters,i]<0.5):
                regret[iters,i+1:] = regret[iters,i]
                break
        
      
        del x
        del y
        del model
        
        mdic = {"current_opt_DGPUCB": current_opt,"regret_DGPUCB": regret, "runtime_DGPUCB":time_elapsed}
        savemat(folder+label+"_DGPUCB_"+ str(time.asctime(time.localtime(time.time())))+"_iter"+str(iters)+".mat", mdic)
    
    
    return current_opt,regret,time_elapsed