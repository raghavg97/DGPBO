import numpy as np
def lcb_gp(xx,n,d,gpr_model):
    xx = np.reshape(xx,(-1,d))
    
    mu, sigma = gpr_model.predict(xx, return_std=True)
    sigma = sigma.reshape(-1, 1)
    
    beta_t = 0.2 *np.log(2*n*d)
 
    lcb = mu - np.sqrt(beta_t)*sigma
    
    return lcb.reshape(-1,)