import numpy as np
from scipy.stats import norm

def ei_gp(xx,n,d,gpr_model,current_min):
    xx = np.reshape(xx,(-1,d))
    
    mu, sigma = gpr_model.predict(xx, return_std=True)
    sigma = sigma.reshape(-1, 1)
    
    f_term = (current_min - mu)/sigma
    ei = (current_min - mu)*norm.cdf(f_term) + sigma*norm.pdf(f_term)
    
    return ei.reshape(-1,)