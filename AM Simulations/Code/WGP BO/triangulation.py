import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import numpy as np

def tricands_interior(X):
    m = X.shape[1]
    n = X.shape[0]
    if (n < m+1):
        raise Exception("must have nrow(X) >= ncol(X) + 1")

    ## possible to further vectorize?
    ## find the middle of triangles
    # tri = Delaunay(X, qhull_options="Q12").vertices
    tri = Delaunay(X).vertices
    Xcand = np.zeros([tri.shape[0], m])
    for i in range(tri.shape[0]):
        Xcand[i,:] = np.mean(X[tri[i,],], axis = 0)

    return {'cand':Xcand, 'tri':tri}

def tricands_fringe(X):
    ## extract dimsions and do sanity checks
    m = X.shape[1]
    n = X.shape[0]
    if (n < m+1):
        raise Exception("must have nrow(X) >= ncol(X) + 1")

    ## get midpoints of external (convex hull) facets and normal vectors
    qhull = ConvexHull(X, qhull_options="n")
    qhull = ConvexHull(X)
    norms = np.zeros((qhull.simplices.shape[0],m))
    Xbound = np.zeros((qhull.simplices.shape[0],m))
    for i in range(qhull.simplices.shape[0]): 
        Xbound[i,] = np.mean(X[qhull.simplices[i,],:], axis = 0)
        norms[i,] = qhull.equations[i,0:m] 

    ## norms off of the boundary points to get fringe candidates
    ## half-way from the facet midpoints to the boundary
    eps = np.sqrt(np.finfo(float).eps)
    alpha = np.zeros(Xbound.shape[0])
    ai = np.zeros([Xbound.shape[0], m])
    pos = norms > 0
    ai[pos] = (1-Xbound[pos])/norms[pos]
    ai[np.logical_not(pos)] = -Xbound[np.logical_not(pos)]/norms[np.logical_not(pos)]
    ai[np.abs(norms) < eps] = np.Inf
    alpha = np.min(ai, axis = 1)

    ## half way to the edige
    Xfringe = Xbound + norms*alpha[:,np.newaxis]/2

    return {'XF':Xfringe, 'XB':Xbound, 'qhull':qhull}

def tricands(X, fringe=True, nmax=None, best=None, vis=False, imgname = 'tricands.pdf'):
    """
    Triangulation Candidates for Bayesian Optimization

    X - A numpy.array of size "NxD" with each row giving a design point and each column a feature.
    fring - A boolean variable giving whether to include fringe points to allow exploration outside the convex hull of our points.
    best - A (zero-indexed) integer giving the row which corresponds to the best currently observed point; only matters if there is subsampling involved.
    vis - A boolen telling us if you want a visualization of your triangulation; only applicable to 2D designs.

    Returns a numpy array which is like X.
    """
    ## extract dimsions and do sanity checks
    m = X.shape[1]
    n = X.shape[0]
    if nmax is None:
        nmax = 100*n
    if vis and m != 2:
        raise Exception("visuals only possible when ncol(X)=2")
    if n < m+1:
        raise Exception("must have nrow(X) >= ncol(X) + 1")

    ## possible visual
    if vis:
        fig = plt.figure()
        plt.scatter(X[:,0], X[:,1])
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot(X, xlim=c(0,1), ylim=c(0,1))

    ## interior candidates
    ic = tricands_interior(X)
    Xcand = ic['cand']
    if vis:
        for i in range(ic['tri'].shape[0]): 
            X[np.append(ic['tri'][i,:], ic['tri'][i,0]),].T
            for j in range(ic['tri'].shape[1]+1):
                if j < ic['tri'].shape[1]:
                    xpoints = X[ic['tri'][i,j:(j+2)],0]
                    ypoints = X[ic['tri'][i,j:(j+2)],1]
                else:
                    xpoints = X[ic['tri'][i,[-1,0]],0]
                    ypoints = X[ic['tri'][i,[-1,0]],1]
                plt.plot(xpoints, ypoints, color = 'black')

        plt.scatter(Xcand[:,0], Xcand[:,1])
        
    ## calculate midpoints of convex hull vectors
    if fringe:
        fr = tricands_fringe(X)
        Xcand = np.concatenate([Xcand, fr['XF']], axis = 0)
        ## possibly visualize fringe candidates
        if vis: 
            for i in range(fr['XB'].shape[0]):
                plt.arrow(fr['XB'][i,0], fr['XB'][i,1], fr['XF'][i,0]-fr['XB'][i,0], fr['XF'][i,1]-fr['XB'][i,1], width = 0.005, color = 'red')

    ## throw some away?
    if(nmax < Xcand.shape[0]):

        ## check to see if we are guaranteeing some
        if best is not None:
            ## find candidates adjacent to best
            adj = np.where(np.apply_along_axis(lambda x: np.any(x==best), 1, ic['tri']))[0]
            if len(adj) > nmax/10:
                sample = np.random.choice(adj, round(nmax/10), replace=False)
            if vis:
                plt.scatter(X[best:(best+1),0],X[best:(best+1),1], color = 'green')
        else:
            adj = np.array([])
            
        if len(adj) >= nmax:
            raise Exception("adjacent to best >= nmax")

        ## get the rest randomly
        remain = np.array(list(range(Xcand.shape[0])))
        if len(adj > 0):
            remain = np.delete(remain, adj, 0)
        rest = np.random.choice(remain, (nmax - len(adj)), replace=False)
        sel = np.concatenate([adj, rest], axis = 0).astype(int)
        Xcand = Xcand[sel,:]

        ## possibly visualize
        if(vis):
            plt.scatter(Xcand[:,0], Xcand[:,1], color="green")

    if vis:
        plt.savefig(imgname)
        plt.close()
    return Xcand