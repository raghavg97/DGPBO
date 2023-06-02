import torch
import numpy as np
from scipy.stats import norm

def ei_wgp(xx,wgp_model,min_y,device,dtype,d):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize# observe new values
    xx = torch.from_numpy(xx).float().to(device)
    xx = xx.view(-1,d)
    
    postr = wgp_model.posterior(X = xx)
    mu = postr.mean
    s = torch.sqrt(postr.variance)
    # print(mu.shape)
    # print(s.shape)
    
    gam = torch.divide(min_y - mu,s).cpu().detach().numpy()
    
    s =  s.cpu().detach().numpy()
    
    ei = np.multiply(s,(np.multiply(gam,norm.cdf(gam)) + norm.pdf(gam)))
    #print(ei.shape)
    return ei.reshape(-1,)