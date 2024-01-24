import torch
import numpy as np

def wgp_helper(x_tensor,f,xlimits,k_icdf,dtype,device):
    xlimits_tensor = torch.from_numpy(xlimits)
 
    d = x_tensor.shape[1]    
    x_scaled = x_tensor
    #print(x_scaled)
    #x_scaled = torch.zeros(list(x_tensor.shape)) 
    
    #for i in range(d):
    #    limits_diff = (xlimits_tensor[i,1]-xlimits_tensor[i,0])
    #    limits_min = xlimits_tensor[i,0]
    #    x_scaled[:,i] = (x_tensor[:,i]-limits_min)/limits_diff
        
    x_warp = k_icdf.icdf(x_tensor)
    x_warp_rescaled = torch.zeros(list(x_warp.shape),dtype = dtype,device = device)
    
    #print(x_warp)
    
    for i in range(d):
        limits_diff = (xlimits_tensor[i,1]-xlimits_tensor[i,0])
        limits_min = xlimits_tensor[i,0]
        x_warp_rescaled[:,i] = x_warp[:,i]*limits_diff+ limits_min
    
    #print(x_warp_rescaled.detach().numpy())
    y =  f(x_warp_rescaled.detach().numpy()).reshape(-1,1)
   
        
    return torch.from_numpy(y)    
   
    
        
        
    
    
    