import numpy as np

def lcb_mixt_bag(mu_all,var_all,n,d):
    s_all = np.sqrt(var_all)

    beta_t = 0.2 *np.log(2*n*d)

    lcb = np.mean(mu_all,axis=0) - np.mean(np.sqrt(beta_t)*s_all,axis=0)

    return lcb.reshape(-1,)