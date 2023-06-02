import numpy as np

def Hartmann3(x):
    #3D function evaluated in [0,1]
    #minimum -3.86278 at x* = (0.114614,0.555649,0.852547)
    alpha = np.array([1.0,1.2,3.0,3.2])
    A = np.array([[3.0,10,30],[0.1,10,35],[3.0,10,30],[0.1,10,35]])
    P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])
    P = 0.0001*P
    
    outer = np.zeros((np.shape(x)[0],1))
    
    for i in range(4):
        inner = np.zeros((np.shape(x)[0],1))
        for j in range(3):
            inner = inner + A[i,j]*np.square((x[:,j] - P[i,j])).reshape(-1,1)
            #print(np.shape(inner))
        outer = outer + alpha[i]*np.exp(-1*inner)
    
    return (-1*outer)

def Hartmann4(x):
    #[0,1]^4
    #Actual Optimum -3.135474
    alpha = np.array([1.0,1.2,3.0,3.2])
    A = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    P = np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],[2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
    P = 0.0001*P
    
    outer = np.zeros((np.shape(x)[0],1))
    for i in range(4):
        inner = np.zeros((np.shape(x)[0],1))
        
        for j in range(4):
            inner = inner + A[i,j]*np.square(x[:,j] - P[i,j]).reshape(-1,1)
        outer = outer + alpha[i]*np.exp(-1*inner)
  
    
    y =(1.1 - outer)/0.839
    return y

def cross_in_tray(x):
    #2D [-10,10]
    #optimum -2.06261
    
    x1 = x[:,0]*20 - 10
    x2 = x[:,1]*20 - 10
    
    term1 = np.sqrt(np.square(x1)+np.square(x2))/np.pi
    term2 = np.exp(np.abs(100-term1))
    
    term3 = np.abs(np.sin(x1)*np.sin(x2)*term2)  
    
    term4 = -0.0001*np.power((term3 + 1),0.1)    
    
    return term4

def levy(x):
    #any D [-10,10]
    #f(x*) = 0 at x*=(1,1,...,1)
    d = np.shape(x)[1]
    x = x*20 - 10
    
    w = 1+ (x-1)/4
    
    term3 = np.zeros((np.shape(x)[0],1))
    for i in range(d-1):
        wi = w[:,i].reshape(-1,1)
        term1 = np.square(wi-1)
        term2 = 1 + 10*np.square(np.sin(np.pi*wi + 1))
        term3 = term3 + term1*term2
    
    wd = w[:,d-1].reshape(-1,1)
    w1 = w[:,0].reshape(-1,1)
    term3 = np.square(np.sin(np.pi*w1)) + term3 + np.square(wd-1)*(1 + np.square(np.sin(2*np.pi*wd))) 
    
    return term3

def camel_3h(x):
    #2D [-5,5]
    #optimum 0 at 0
    
    x1 = (x[:,0]*10 - 5).reshape(-1,1)
    x2 = (x[:,1]*10 - 5).reshape(-1,1)
    
    y = 2*np.square(x1) -1.05*np.power(x1,4) + np.power(x1,6)/6 + x1*x2 + np.square(x2)
    
    return y

def camel_6h(x):
    #2D [-3,3],[-2,2]
    #optimum -1.0316 at multiple points [0.0898,-0.7126],[-0.0898,0.7126] WITHOUT SCALING
    
    x1 = (x[:,0]*6 - 3).reshape(-1,1)
    x2 = (x[:,1]*4 - 2).reshape(-1,1)
    
    y = (4 - 2.1*np.square(x1) + np.power(x1,4)/3)*np.square(x1) + x1*x2 + (-4+4*np.square(x2))*np.square(x2)
    
    return y

def perm3(x):
    #any D [-d,d]
    # Optimum 0 at [1,2,...d]
    
    d = np.shape(x)[1]
    
    x = x*2*d - d
    
    beta = 0.5
    
    outer = np.zeros((np.shape(x)[0],1))
    for i in range(d):
        inner = np.zeros((np.shape(x)[0],1))
        for j in range(d):
            inner = inner + (np.power((j+1),i+1)+beta)*(np.power(x[:,j]/(j+1),(i+1)).reshape(-1,1) - 1)
        outer = outer + np.square(inner).reshape(-1,1)
           
    return outer  

def art1(x):
    #1D Min -1.0578 at 0.0541
    y = np.zeros((np.shape(x)))
    
    for i in range(np.shape(x)[0]):
        if(x[i] < 0.3 or x[i] > 0.6):
            y[i] = np.sin(np.pi*x[i] + np.pi) + x[i] + np.cos(2*np.pi*x[i] + np.pi)
        else:
            y[i] = -0.9 + 0.1*np.sin(50*np.pi*x[i])
      
    return y 

def mich10d(x):
    m = 10

    term12 = np.zeros((np.shape(x)[0],1))
    x = x*np.pi
    
    for k in range(10):
        i = k+1
        term1 = np.sin(x[:,k]).reshape(-1,1)
        term2 = np.power(np.sin(i*np.square(x[:,k])/np.pi),2*m).reshape(-1,1)
        
        term12 = term12 + term1*term2

    
    return -1*term12 

def Hartmann6(x):
    #Optimum -3.32236801  WITHOUT SCALING [0,1]^6
    #6D, At [0.20169,0.150011,0.476874,0.275332,0.311652,0.6573] 
    alpha = np.array([1.0,1.2,3.0,3.2])
    A = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    P = np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],[2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
    P = 0.0001*P
    
    outer = np.zeros((np.shape(x)[0],1))
    
    for i in range(4):
        inner = np.zeros((np.shape(x)[0],1))
        
        for j in range(6):
            inner = inner + A[i,j]*np.square(x[:,j] - P[i,j]).reshape(-1,1)
        outer = outer + alpha[i]*np.exp(-1*inner)
  
    
    return -1*outer

def shekel4(x):
    #Global optimum m=10, -10.5364
    x = x*10
    d= 4
    m=10
    c = np.array([0.1,1.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    A = np.array([[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,5,1,2,3.6],[4,1,8,6,3,2,3,8,6,7],[4,1,8,6,7,9,3,1,2,3.6]]).reshape(10,4)
    
    term2 = np.zeros((x.shape[0],1))
    for i in range(m):
        term1 = np.zeros((x.shape[0],1))
        for j in range(d):
            term1 = term1+np.square(x[:,j] - A[i,j]).reshape(-1,1)
        
        term2 = term2 + 1/(term1+c[i])
            
    return -1*term2