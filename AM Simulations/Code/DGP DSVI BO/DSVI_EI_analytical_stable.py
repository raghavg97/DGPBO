import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import scipy

from scipy.io import savemat
from ei_dsvi import ei_dsvi
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
import gpflux

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables, construct_gp_layer

import scipy
import triangulation
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



tf.keras.backend.set_floatx("float64")  # we want to carry out GP calculations in 64 bit
tf.get_logger().setLevel("INFO")

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

import time

def DSVI_EI_v1(max_iters,xlimits,n,d,new_n,m,f,actual_optimum,n_layers,label,folder):
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
        
        kernel1 = gpflow.kernels.SquaredExponential() + gpflow.kernels.Constant()
        kernel2 = gpflow.kernels.SquaredExponential() + gpflow.kernels.Constant()
    
        likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
        
        print('Iteration ', iters,'\nRegret Start = ',regret[iters,n-1] )
         
        for i in range(n,n+new_n):
            start_t = time.time()
            np.random.seed(1000*iters + 2*i)
            tf.random.set_seed(1000*iters + 2*i)
            
            #print(xlimits.shape)
            Z = np.linspace(xlimits[:,0], xlimits[:,1], n)
            
            inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z.copy())
            gp_layer1 = gpflux.layers.GPLayer(kernel1, inducing_variable1, num_data=x.shape[0],num_latent_gps = x.shape[1])
            
            
            # Layer 2
            inducing_variable2 = gpflow.inducing_variables.InducingPoints(Z.copy())
            gp_layer2 = gpflux.layers.GPLayer(kernel2,inducing_variable2,num_data=x.shape[0],num_latent_gps = 1,mean_function=gpflow.mean_functions.Zero(),)
            # gp_layer2 = construct_gp_layer(num_data = x.shape[0],num_inducing = i,input_dim = 2,output_dim = 1,kernel_class = kernel2,z_init=Z.copy())

            # Initialise likelihood and build model
    
            two_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)
            # two_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2])
            # two_layer_dgp = gpflux.models.DeepGP([gp_layer1], likelihood_layer,input_dim = d,target_dim = 1)

            # Compile and fit
            model = two_layer_dgp.as_training_model()
            
            model.compile(tf.optimizers.Adam(0.001))
            if (i==n):
                history = model.fit({"inputs": x, "targets": y}, epochs=500, verbose=0)
            else:
                history = model.fit({"inputs": x, "targets": y}, epochs=150, verbose=0)                     
            
            
            prediction_model = two_layer_dgp.as_prediction_model()
            
            yy = ei_dsvi(xx = xx,n=i,d=d,pred_model = prediction_model,min_y = (current_opt[iters,i-1] - y_shift)/y_scale)
            
            maxei_idx = np.argmax(yy)
          
            new_x = xx[maxei_idx,:].reshape(1,d)
            y_new = yy_dat[maxei_idx].reshape(1,1)
            
            kernel1 = two_layer_dgp.f_layers[0].kernel
            kernel2 = two_layer_dgp.f_layers[1].kernel
            likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
 
            x = np.append(x,new_x,axis=0)
            y = np.append(y,y_new,axis=0)
            
            # xx = np.delete(xx,maxei_idx,axis = 0)
            # yy_dat = np.delete(yy_dat,maxei_idx,axis = 0)
       
      
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

            print (maxei_idx,'i = ',i ,' Regret Current = ',regret[iters,i],end="\r")
            
            if(regret[iters,i]<0.5):
                regret[iters,i+1:] = regret[iters,i]
                break
        
      
        del x
        del y
        del two_layer_dgp
        print("\n")
        mdic = {"current_opt_DGPVIEI": current_opt,"regret_DGPVIEI": regret, "runtime_DGPVIEI":time_elapsed}
        savemat(folder+label+"_DGPVIEI_"+ str(time.asctime(time.localtime(time.time())))+"_iter"+str(iters)+".mat", mdic)  
    
    return current_opt,regret,time_elapsed