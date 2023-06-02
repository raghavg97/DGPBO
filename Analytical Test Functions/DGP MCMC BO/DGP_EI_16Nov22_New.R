#Maximization
##Currently implements EI and TS-10D with 2 layers
### Load libraries
art1 <- function(x,d=1)
{
  s <- length(x)
  y <- x
  
  for (i in 1:s)
  { 
    if (x[i]<0.3 || x[i]>0.6)
    {
      y[i] = sin(pi*x[i] + pi) + x[i] + cos(2*pi*x[i]+pi)
    }
    else
    {
      y[i] = -0.9 + 0.1*sin(50*pi*x[i])
    }
  }
  
  return(y)
}


library(deepgp)
library(lhs)
library(MASS)
library(mvtnorm)
library(foreach)
library(doParallel)
library(R.matlab)

init_python <- readMat("Art1_X_May2023.mat")
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


dim_X <- 1
dim_W <- 1 #Always \leq dim_X
dim_Z <-1 #Always \leq dim_X
actual_optimum_y <- -1.0578 #For calculating regret
max_iters <- 10 # number of times to repeat the optimization for average performance
n <- 3# number of points in starting design - General rule 5*dimension
#n <- 20
new_n <- 97# number of points to add sequentially
m <- 100 # number of candidate and testing points - General rule 100*d
f <- art1

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
  x <- x_python[,iters,drop = FALSE]
  #y <- apply(x, 1, f) #+ rnorm(n, 0, noise)
  y <- c(f(x,dim_X))
  cat('Regret start ',min(y)-actual_optimum_y,'\n')
  y_std = sd(y)
  y <- y/y_std
  ### Set initial values for MCMC (function defaults)
  g_0 <- 0.01
  if (layers == 1) 
  {
    theta_0 <- 0.5
  } 
  else if (layers == 2) 
  {
    theta_y_0 <- runif(1)
    theta_w_0 <- rnorm(dim_W,1,0.1)
    w_0 <- matrix(rnorm(dim(x)[1]*dim_W), nrow = dim(x)[1], ncol = dim_W)
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
  
  current_opt[iters,n] <- min(y)*y_std
  instant_regret[iters,n] <- current_opt[iters,n] - actual_optimum_y
  
  for (t in (n+1):(n + new_n)) {

    # Select new random set of candidate/testing points
    start_time <- proc.time()[[3]]
    set.seed(iters+1000*t)
    xx <- xx_python[,t-n,iters]
    dim(xx) <- c(length(xx),1)
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
      fit <- fit_three_layer(x, y, D = dim_W, nmcmc = nmcmc,verb = FALSE, g_0 = g_0, theta_y_0 = theta_y_0,
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
    y <- c(y, f(x_new,dim_X)/y_std) #+ rnorm(1, 0, noise))
    
    
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
    
    #cat('Regret',min(y)-actual_optimum_y,'\r')
    current_opt[iters,t] <- min(y)*y_std
    instant_regret[iters,t] <- current_opt[iters,t] - actual_optimum_y
    cat('i = ', t, 'Regret',instant_regret[iters,t],'Time ',comp_time[iters,t],'\r')
      
    if (instant_regret[iters, t] < -2e-5) {
        instant_regret[iters, (t + 1):ncol(instant_regret)] <- instant_regret[iters, t]
        break
    }
    
    if (t == n + new_n) break
    
  } } 
filename_base <- paste("DGP-EI","Art1D",layers,".rds",sep="_")
filename1 <- paste("current_opt",filename_base)
filename2 <- paste("regret",filename_base)
filename3 <- paste("time",filename_base)
filename4 <- paste("Acq_time",filename_base)
saveRDS(current_opt,file=filename1)
saveRDS(instant_regret,file=filename2)
saveRDS(comp_time,file=filename3)
saveRDS(acq_time,file=filename4)
