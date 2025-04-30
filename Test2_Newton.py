import numpy as np
import numpy.linalg as la
from scipy.linalg import inv
from matplotlib import pyplot as plt
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

def Hessian(f,x_guess):
    return differentiate.hessian(f,x_guess)

def Newton(guess,tol):
    count = 0 # this variable is a safegaurd against infinite loops
    x = guess # this is our initial guess, and later our x_n
    rn = np.array([x])
    x1 = 2 * x # after this, we use x1 to save our x_n-1
    print('hi:')
    print(x1)
    print(f(x1))
    print(np.abs(f(x)-f(x1)))
    while (count < 20 and (np.abs(f(x)-f(x1))) > tol).all():
        #While we're done less then 100 iters #While |f(x_n)-f(x_n-1)| is less then tol
        count = count + 1
        hessian = Hessian(f, x) # for Newton direction
        x1 = x # save x_n-1

        grad_x = grad(f,x)
        print('grad')
        print(grad_x)
        x = x - 1*(inv(hessian.ddf)@grad_x)

        rn = np.append(rn, np.array([x]), axis=0)
    print('num iters: ')
    print(count)
    print('xmin: ')
    print(x)
    
    rN=x
    rnN = rn
    
    numN = rnN.shape[0];
    print(rnN[0:(numN-1)])
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    #plt.show();

guess2 = 10*np.ones(2)
guess5 = 10*np.ones(5)
guess10 = 10*np.ones(10)
guess50 = 10*np.ones(50)
guess100 = 10*np.ones(100)
guess500 = 10*np.ones(500)
guess1000 = 10*np.ones(1000)

start = timer()
#Newton(guess2,1e-5)
end = timer()
t2 = end - start

start = timer()
#Newton(guess5,1e-5)
end = timer()
t5 = end - start

start = timer()
#Newton(guess10,1e-5)
end = timer()
t10 = end - start

start = timer()
#Newton(guess50,1e-5)
end = timer()
t50 = end - start

start = timer()
#Newton(guess100,1e-5)
end = timer()
t100 = end - start

start = timer()
Newton(guess500,1e-5)
end = timer()
t500 = end - start

start = timer()
#Newton(guess1000,1e-5)
end = timer()
t1000 = end - start

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
print('time 500d:')
print(t500)
print('time 1000d:')
print(t1000)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))

