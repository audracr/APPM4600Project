import numpy as np
import numpy.linalg as la
from scipy.linalg import inv
from matplotlib import pyplot as plt
from scipy import differentiate
from timeit import default_timer as timer
from scipy.linalg import solve  

def f(x):
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


def hessian_fd(fun, x, h=1e-5):
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


def Lazy_Newton_Rosenbrock_d2(xguess, tol=1e-5, max_iter=10000):
    count = 0
    # same initial guess
    x = xguess
    iterates = [x.copy()]
    
    while count < max_iter:
        g = grad(f,x)
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break

        H = hessian_fd(f, x, h=1e-5)
        
        p = solve(H, g)
        alpha = 1.0
        x_new = x - alpha * p
        ls_iter = 0
        max_ls_iter = 10
        while f(x_new) >= f(x) and ls_iter < max_ls_iter:
            alpha *= 0.5
            x_new = x - alpha * p
            ls_iter += 1
        
        if f(x_new) >= f(x):
            print("Line search failed")
            break
        
        x = x_new.copy()
        iterates.append(x.copy())
        count += 1
        
        print(f"Iter: {count:4d} | x = {x} | f(x) = {f(x):.6e} | ||grad|| = {grad_norm:.3e}")
    
    print("\nFinal x (d2):", x)
    print("Number of iterations (d2):", count)
    iterates = np.array(iterates)
    final_x = x
    errors = np.linalg.norm(iterates - final_x, axis=1)
    
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(errors)), np.log10(errors + 1e-18), 'r-o', label='Newton analytic')
    plt.title('Newton Iteration: log₁₀|x - x*| (Analytic Hessian)')
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Error)')
    plt.legend()
    plt.grid(True)

guess2 = 10*np.ones(2)
guess5 = 10*np.ones(5)
guess10 = 10*np.ones(10)
guess50 = 10*np.ones(50)
guess100 = 10*np.ones(100)
guess500 = 10*np.ones(200)
guess1000 = 10*np.ones(1000)

start = timer()
Lazy_Newton_Rosenbrock_d2(guess2, 1e-5, 10000)
end = timer()
t2 = end - start

start = timer()
Lazy_Newton_Rosenbrock_d2(guess5, 1e-5, 10000)
end = timer()
t5 = end - start

start = timer()
Lazy_Newton_Rosenbrock_d2(guess10, 1e-5, 10000)
end = timer()
t10 = end - start

start = timer()
Lazy_Newton_Rosenbrock_d2(guess50, 1e-5, 10000)
end = timer()
t50 = end - start

start = timer()
Lazy_Newton_Rosenbrock_d2(guess100, 1e-5, 10000)
end = timer()
t100 = end - start

start = timer()
Lazy_Newton_Rosenbrock_d2(guess500, 1e-5, 10000)
end = timer()
t500 = end - start

start = timer()
Lazy_Newton_Rosenbrock_d2(guess1000, 1e-5, 10000)
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
#print('time 1000d:')
#print(t1000)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))

