import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import inv
from scipy import differentiate
from scipy.optimize import minimize
from scipy.optimize import newton
from timeit import default_timer as timer

#Citation: https://github.com/trsav/bfgs

def f(x): #currently this is just Rosenbrock written by a fellow better at code then I
    #FUNCTION TO BE OPTIMISED
    d = len(x)
    sum = 0
    for i in range(len(x)):
        sum = sum + 10*(x[i])**2 + 10*x[i] 
    return sum

def grad(x): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    d = len(x)
    sum = np.zeros(d)
    for i in range(len(x)):
        sum[i] = 20*x[i]+10 
    return sum

def hessian_fd(x):
    h=1e-5
    fun = f
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        # diagonal second derivative using central difference
        H[i, i] = (fun(x + h*e_i) - 2*fun(x) + fun(x - h*e_i)) / (h**2)
        for j in range(i+1, n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            # partial derivatives using central difference
            H[i, j] = (fun(x + h*e_i + h*e_j) - fun(x + h*e_i - h*e_j) 
                        - fun(x - h*e_i + h*e_j) + fun(x - h*e_i - h*e_j)) / (4*h**2)
            H[j, i] = H[i, j]
    return H

guess2 = 10*np.ones(2)
guess5 = 10*np.ones(5)
guess10 = 10*np.ones(10)
guess50 = 10*np.ones(50)
guess100 = 10*np.ones(100)
guess200 = 10*np.ones(200)
guess300 = 10*np.ones(300)
guess500 = 10*np.ones(500)
guess1000 = 10*np.ones(1000)
guess1500 = 10*np.ones(1500)
guess2000 = 10*np.ones(2000)
arr = [guess2, guess5, guess10, guess50, guess100, guess200, guess300, guess500, guess1000, guess1500, guess2000]

'''
for i in arr:
    print(len(i))

    guess = i
    start = timer()
    x_opt = minimize(f, guess, method='BFGS', tol = 1e-5)
    end = timer()
    t = end - start
    print('BFGS time:')
    print(t)
'''

for i in arr:
    print(len(i))
    guess = i
    start = timer()
    x_opt = minimize(f, guess, method='Newton-CG', jac = grad, hess = hessian_fd, tol = 1e-5)
    end = timer()
    t = end - start
    print('NEWTON time:')
    print(t)
