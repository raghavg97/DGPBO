#Maximization
##Currently implements EI and TS-10D with 2 layers
### Load libraries
library(geometry)

## tricands.interior:
## 
## interior, Delaunay triangulated candidates; subroutine
## used by tricands wrapper below

tricands.interior <- function(X)
{
  ## extract dimsions and do sanity checks
  m <- ncol(X)
  n <- nrow(X)
  if(n < m+1) stop("must have nrow(X) >= ncol(X) + 1")
  
  ## possible to further vectorize?
  ## find the middle of triangles
  suppressWarnings( tri <- delaunayn(X, options="Q12") )
  Xcand <- matrix(NA, nrow=nrow(tri), ncol=m)
  for(i in 1:nrow(tri)) {
    Xcand[i,] <- colMeans(X[tri[i,],])
  }
  
  return(list(cand=Xcand, tri=tri))
}



## tricands.fringe:
## 
## fringe, outside convex hull candidates; subroutine
## used by tricands wrapper below

tricands.fringe <- function(X)
{
  ## extract dimsions and do sanity checks
  m <- ncol(X)
  n <- nrow(X)
  if(n < m+1) stop("must have nrow(X) >= ncol(X) + 1")
  
  ## get midpoints of external (convex hull) facets and normal vectors
  ## qhull <- convhulln(X, output.options="n")
  suppressWarnings( qhull <- convhulln(X, output.options="n", options="Q12") )
  norms <- Xbound <- matrix(NA, nrow=nrow(qhull$hull), ncol=m)
  for(i in 1:nrow(qhull$hull)) {
    Xbound[i,] <- colMeans(X[qhull$hull[i,],])
    norms[i,] <- qhull$normals[i,1:m] 
  }
  
  ## norms off of the boundary points to get fringe candidates
  ## half-way from the facet midpoints to the boundary
  eps <- sqrt(.Machine$double.eps)
  alpha <- rep(NA, nrow(Xbound))
  ai <- matrix(NA, nrow(Xbound), m)
  pos <- norms > 0
  ai[pos] <- (1-Xbound[pos])/norms[pos]
  ai[!pos] <- -Xbound[!pos]/norms[!pos]
  ai[abs(norms) < eps] <- Inf
  alpha <- do.call(pmin, as.data.frame(ai))
  
  ## half way to the edige
  Xfringe <- Xbound + norms*alpha/2
  
  ## done
  return(list(XF=Xfringe, XB=Xbound, qhull=qhull))
}



## tricands:
## 
## new gap-filling candidates based on triangulation and 
## exploration beyond the convex hull; vis only appropriate in 2d
## uses above two subroutines for interior and fringe candidates

tricands <- function(X, fringe=TRUE, max=100*ncol(X), best=NULL, vis=FALSE)
{
  ## extract dimsions and do sanity checks
  m <- ncol(X)
  n <- nrow(X)
  if(vis && m != 2) stop("visuals only possible when ncol(X)=2")
  if(n < m+1) stop("must have nrow(X) >= ncol(X) + 1")
  
  ## possible visual
  if(vis) plot(X, xlab="x1", ylab="x2", xlim=c(0,1), ylim=c(0,1))
  
  ## interior candidates
  int <- tricands.interior(X)
  Xcand <- int$cand
  if(vis) {
    for(i in 1:nrow(int$tri)) {
      lines(X[c(int$tri[i,], int$tri[i,1]),])
    }
    points(Xcand, col=2, pch=18)
  }
  
  ## calculate midpoints of convex hull vectors
  if(fringe) {
    fr <- tricands.fringe(X)
    ## possibly visualize fringe candidates
    if(vis) arrows(fr$XB[,1], fr$XB[,2], fr$XF[,1], fr$XF[,2], lty=2)
    Xcand <- rbind(Xcand, fr$XF)
  } 
  
  ## throw some away?
  if(max < nrow(Xcand)) {
    
    ## check to see if we are guaranteeing some
    if(!is.null(best)) {
      ## find candidates adjacent to best
      adj <- which(apply(int$tri, 1, function(x) { any(x == best) }))
      if(length(adj) > max/10) adj <- sample(adj, round(max/10), replace=FALSE)
      if(vis) {
        points(X[best,,drop=FALSE], col="green", pch=17)
      }
    } else adj <- c()
    if(length(adj) >= max) stop("adjacent to best >= max")
    
    ## get the rest randomly
    remain <- (1:nrow(Xcand))
    if(length(adj > 0)) remain <- remain[-adj]
    rest <- sample(remain, (max - length(adj)), replace=FALSE)
    Xcand <- Xcand[c(adj, rest),]
    
    ## possibly visualize
    if(vis) points(Xcand, col="green")
    
    ## otherwise, maybe fill with random candidates
  } 
  
  ## done
  return(Xcand)
}


camel3h <- function(xx,dim_X)
{ #Optimum -1.0316
  x1 <- xx[,1]*10-5
  x2 <- xx[,2]*10-5
  
  term1 <- 2*x1^2
  term2 <- -1.05*x1^4
  term3 <- x1^6 / 6
  term4 <- x1*x2
  term5 <- x2^2
  
  y <- term1 + term2 + term3 + term4 + term5
  return(y)
}



library(lhs)
library(MASS)
library(mvtnorm)
library(foreach)
library(doParallel)
library(R.matlab)
library(deepgp)

init_python <- readMat("Func2D_X.mat")
x_python <- init_python$x
xx_python <- init_python$xx

### Set command line variables
layers <- 2 # 1, 2, or 3

### Read command line arguments

args <- commandArgs(TRUE)
if(length(args) > 0)
  for(i in 1:length(args))
    eval(parse(text = args[[i]]))


cat("layers is ", layers, "\n")


dim_X <- 2
dim_W <- 2 #Always \leq dim_X
dim_Z <-1 #Always \leq dim_X
actual_optimum_y <- 0 #For calculating regret
max_iters <- 10 # number of times to repeat the optimization for average performance
n <- 5*dim_X# number of points in starting design - General rule 5*dimension
#n <- 20
new_n <- 150-n# number of points to add sequentially
m <- 100 # number of candidate and testing points - General rule 100*d
f <- camel3h

###Initialize for storing the required values
current_opt <- matrix(data=NA,nrow=max_iters,ncol=n+new_n)
instant_regret <- matrix(data=NA,nrow=max_iters,ncol=n+new_n)
comp_time <- matrix(data=NA,nrow=max_iters,ncol=n+new_n)
acq_time <- matrix(data=NA,nrow=max_iters,ncol=n+new_n)

init_time <- Sys.time()
for (iters in 1:max_iters)
{ 
  cat('Iteration number',iters,'\n')
  
  seed <- iters
  cat("seed is ", seed, "\n")
  set.seed(seed)
  
  rm(x) #To remove the x and y from previous iterations
  rm(y)
  x <- x_python[iters,,]
  #print(x)
  #y <- apply(x, 1, f) #+ rnorm(n, 0, noise)
  y <- c(f(x,dim_X))
  cat('Regret start ',min(y)-actual_optimum_y,'\n')
  ### Set initial values for MCMC (function defaults)
  g_0 <- 0.01
  if (layers == 1) 
  {
    theta_0 <- 0.5
  } 
  else if (layers == 2) 
  {
    theta_y_0 <- 0.5
    theta_w_0 <- rep(1,dim_W)
    w_0 <- x[,1:dim_W]
    #w_0 <- matrix(x[,1:dim_W])
  } 
  else if (layers == 3) 
  {
    theta_y_0 <- 0.5
    theta_w_0 <- rep(1,dim_W)
    theta_z_0 <- rep(1,dim_Z)
    w_0 <- matrix(x[,1:dim_W])
    z_0 <- matrix(x[,1:dim_Z])
  }
  
  current_opt[iters,n] <- min(y)
  instant_regret[iters,n] <- current_opt[iters,n] - actual_optimum_y
  
  for (t in (n+1):(n + new_n)) {
    # Select new random set of candidate/testing points
    start_time <- proc.time()[[3]]
    set.seed(iters+1000*t)
    xx <- tricands(x)
    #dim(xx) <- c(length(xx),dim_X)
    #yy <- apply(xx, 1, f)
    #yy <- f(xx,dim_X)
    
    
    if (t == n+1) { # run more iterations the first time
      nmcmc <- 10000
      burn <- 8000
      thin <- 2
      
      #for code debugging
      #nmcmc <- 1000
      #burn <- 800
      #thin <- 2
    } 
    else {
      nmcmc <- 3000
      burn <- 1000
      thin <- 2
    }
    
    # Fit Model
    if (layers == 1) {
      fit <- fit_one_layer(x, y, nmcmc = nmcmc,verb = FALSE, g_0 = g_0, theta_0 = theta_0)
    }else if (layers == 2) {
      fit <- fit_two_layer(x, y, D = dim_W, nmcmc = nmcmc,verb = FALSE, g_0 = g_0, theta_y_0 = theta_y_0,
                           theta_w_0 = theta_w_0, w_0 = w_0)
    } else if (layers == 3) {
      fit <- fit_three_layer(x, y, D = dim_W, nmcmc = nmcmc, verb = FALSE,g_0 = g_0, theta_y_0 = theta_y_0,
                             theta_w_0 = theta_w_0, theta_z_0 = theta_z_0)
    }
    
    cores=detectCores()
    
    # Trim, predict, and calculate ALC
    fit <- trim(fit, burn = burn, thin = thin)
    fit <- predict(fit, xx, lite = TRUE, EI = TRUE ,cores=cores)
    
    #acq_obj <- acq_fun(fit,cores=cores)
    #acq_vals <- acq_obj$value
    
    acq_vals <- fit$EI
    
    
    
    
    # Select next design point
    x_new <- matrix(xx[which.max(acq_vals),], ncol = dim_X)
    
    x <- rbind(x, x_new)
    y <- c(y, f(x_new,dim_X)) #+ rnorm(1, 0, noise))
    
    
    # Adjust starting locations for the next iteration
    g_0 <- fit$g[fit$nmcmc]
    if (layers == 1) {
      theta_0 <- fit$theta[fit$nmcmc]
    } else if (layers == 2 | layers == 3) {
      theta_y_0 <- fit$theta_y[fit$nmcmc]
      theta_w_0 <- fit$theta_w[fit$nmcmc]
      w_0 <- fit$w[[fit$nmcmc]]
    }
    if (layers == 3){
      theta_z_0 <- fit$theta_z[fit$nmcmc]
      z_0 <- fit$z[[fit$nmcmc]]
    }
    
    comp_time[iters,t] <- proc.time()[[3]] - start_time
    #acq_time[iters,t] <- acq_obj$time
    #cat('Time since start ',Sys.time() - init_time,'\n')
    #(acq_samples,'Samples \n')
    cat('i = ', t, 'Regret',min(y)-actual_optimum_y,'Time ',comp_time[iters,t],'\r')
    #cat('Regret',min(y)-actual_optimum_y,'\r')
    current_opt[iters,t] <- min(y)
    instant_regret[iters,t] <- current_opt[iters,t] - actual_optimum_y
    
    if (t == n + new_n) break
    
  } } 
filename_base <- paste("DGP-EI","Camel3h2D10iters",layers,".rds",sep="_")
filename1 <- paste("current_opt",filename_base)
filename2 <- paste("regret",filename_base)
filename3 <- paste("time",filename_base)
filename4 <- paste("Acq_time",filename_base)
saveRDS(current_opt,file=filename1)
saveRDS(instant_regret,file=filename2)
saveRDS(comp_time,file=filename3)
saveRDS(acq_time,file=filename4)
