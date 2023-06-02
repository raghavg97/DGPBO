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

from scipy.io import savemat
from LCB_DGPSI_stable import lcb_stable
import time

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
    
    for iters in range(max_iters):
    # for iters in range(5,max_iters):
        seed = iters
        all_layer = all_layer_init
        
        np.random.seed(seed) 
        lhs_sampling = LHS(xlimits = xlimits,random_state = seed, criterion = 'maximin')
        x = lhs_sampling(n)
        y = f(x).reshape(-1,1)
        
        y_std = np.std(y)
        
        y = y/y_std
    
        current_opt[iters,n-1] = np.amin(y)*y_std
        regret[iters,n-1] = current_opt[iters,n-1] - actual_optimum
        
        print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1] )
    
        for i in range(n,n+new_n):
            start_t = time.time()
            np.random.seed(1000*iters + 2*i)  
          
            model=dgp(x,[y],1000*iters + 2*i,all_layer = all_layer,check_rep = False)
            
            if(i==n):
                if(d==10):
                    model.initialize(seed = 1000*iters + 2*i)
                    model.train(N=100,disable = True)
                    all_layer=model.estimate()
                else:
                    model.initialize(seed = 1000*iters + 2*i)
                    model.train(N=500,disable = True)
                    all_layer=model.estimate()
            else:
                model.train(N=N_train,disable = True)
                all_layer=model.estimate(burnin = int(N_train/2))
            
            emu=emulator(all_layer,N=N_predict,nb_parallel = True)
        
            np.random.seed(1000*iters + 2*i) 
            #xx = np.random.uniform(low = xlimits[:,0],high = xlimits[:,1],size = (m,d))
            if (d>1 and d<5):
                xx = triangulation.tricands(x)
            elif (d==10):
                xx = np.random.uniform(low = xlimits[:,0],high = xlimits[:,1],size = (10000,d))
            else:
                xx = np.random.uniform(low = xlimits[:,0],high = xlimits[:,1],size = (m,d))
            
            print(xx.shape)
            #res = emu.ppredict(x=np.reshape(xx,(-1,d)), sample_size=1, method='mean_var',chunk_num = 28,core_num = 28)
            res = emu.ppredict(x=np.reshape(xx,(-1,d)), sample_size=1, method='mean_var',chunk_num = 28,core_num = 28)
            
            # end_t = time.time()
            mu_all = np.transpose(np.array(res[2]))
            var_all = np.transpose(np.array(res[3]))
             
            #print(mu_all.shape)
            #print(var_all.shape)
            

            x0_bag  = np.zeros((total_bags,d))
            
            for bag in range(total_bags):
                np.random.seed(900*iters + 8*i + 5*bag)
                bag_idx = np.random.randint(0,N_predict,size = bag_size)
                mu_all_bag = mu_all[bag_idx]
                var_all_bag = var_all[bag_idx]
          
                yy = lcb_stable(mu_all = mu_all_bag,var_all = var_all_bag,n=i,d=d)
                minlcb_idx = np.argmin(yy)
                x0_bag[bag] = xx[minlcb_idx,:]
            
            print(x0_bag)
            mdic = {"x0_bag": x0_bag}
            savemat("xb.mat", mdic)
            x0 = np.mean(x0_bag,axis=0)
             
            new_x = np.reshape(x0,(1,d))
            x = np.append(x,new_x,axis=0)
            y_new = (f(new_x)/y_std).reshape(-1,1)
            y = np.append(y,y_new,axis=0)
      
            if (y_new < (current_opt[iters,i-1])/y_std):
                current_opt[iters,i] = y_new*y_std
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

            print ('i = ',i ,' Regret Current = ',regret[iters,i],' Time:',time_elapsed[iters,i],end="\r")
        #print('Time Lapsed = ',time.time() - init_t)
        
            if(regret[iters,i]<(-2e-5)):
                regret[iters,i+1:] = regret[iters,i]
                break
        
        print("\n")
        del x
        del y
        del model
        
        mdic = {"current_opt_DGPUCB": current_opt,"regret_DGPUCB": regret, "runtime_DGPUCB":time_elapsed}
        savemat(folder+label+"_DGPUCB_"+ str(time.asctime(time.localtime(time.time())))+"_iter"+str(iters)+".mat", mdic)
    
    
    return current_opt,regret,time_elapsed