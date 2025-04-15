import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import inv
from scipy import differentiate
from timeit import default_timer as timer

def f(x): #currently this is just Rosenbrock written by a fellow better at code then I
    #FUNCTION TO BE OPTIMISED
    d = len(x)
    sum = 0
    for i in range(len(x)):
        sum = sum + 10*(x[i])**2 + 10*x[i] 
    return sum

def grad(f,x): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    d = len(x)
    sum = np.zeros(d)
    for i in range(len(x)):
        sum[i] = 20*x[i]+10 
    return sum

def line_search(f,x,p,nabla): #;-;
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x)
    x_new = x + a * p 
    nabla_new = grad(f,x_new)
    while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
        a *= 0.5
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
    return a


def BFGS(f,x0,max_it,plot=False):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.


    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.

    OUTPUTS: 
    x:      the optimal solution of the function f 

    '''
    d = len(x0) # dimension of problem 
    nabla = grad(f,x0) # initial gradient 
    H = np.eye(d) # initial hessian
    x = x0[:]
    rn = np.array([x])
    it = 2 
    if plot == True: 
        if d == 2: 
            x_store =  np.zeros((1,2)) # storing x values 
            x_store[0,:] = x 
        else: 
            print('Too many dimensions to produce trajectory plot!')
            plot = False

    while np.linalg.norm(nabla) > 1e-5: # while gradient is positive
        if it > max_it: 
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = line_search(f,x,p,nabla) # line search 
        s = a * p 
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:] 
        x = x_new[:]
        if plot == True:
            x_store = np.append(x_store,[x],axis=0) # storing x
        rn = np.append(rn, np.array([x]), axis=0)
    
    rN=x
    rnN = rn
    
    numN = rnN.shape[0];
    print(rnN[0:(numN-1)])
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

    return x




guess2 = 100*np.ones(2)
guess5 = 100*np.ones(5)
guess10 = 100*np.ones(10)
guess50 = 100*np.ones(50)
guess100 = 100*np.ones(100)

start = timer()
x_opt = BFGS(f,guess2,100,plot=True)
print(x_opt)
end = timer()
t2 = end - start

start = timer()
x_opt = BFGS(f,guess5,100,plot=True)
print(x_opt)
end = timer()
t5 = end - start

start = timer()
x_opt = BFGS(f,guess10,100,plot=True)
print(x_opt)
end = timer()
t10 = end - start

start = timer()
x_opt = BFGS(f,guess50,100,plot=True)
print(x_opt)
end = timer()
t50 = end - start

start = timer()
x_opt = BFGS(f,guess100,100,plot=True)
print(x_opt)
end = timer()
t100 = end - start

print('time 2d:')
print(t2)
print('time 5d:')
print(t5)
print('time 10d:')
print(t10)
print('time 50d:')
print(t50)
print('time 100d:')
print(t100)

