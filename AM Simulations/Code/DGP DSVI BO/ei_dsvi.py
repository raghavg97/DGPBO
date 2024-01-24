import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm

import tensorflow as tf
from tqdm import tqdm


tf.keras.backend.set_floatx("float64")

def ei_dsvi(xx,n,d,pred_model,min_y):
    xx = np.reshape(xx,(-1,d))
    
    out = pred_model(xx)
    
    mu = out.f_mean.numpy().squeeze()
    var = out.f_var.numpy().squeeze()
    
    s = np.sqrt(var)

    gam = np.divide((min_y - mu),s)
    

    ei = np.multiply(s,(np.multiply(gam,norm.cdf(gam)) + norm.pdf(gam)))

    
    return ei.reshape(-1,)