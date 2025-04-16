import numpy as np
from scipy.linalg import solve  
from matplotlib import pyplot as plt

# d1 version
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

def Rosen_fun_d1(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (x[0] - 1.0)**2

def grad(x):
    dfdx0 = -400 * x[0] * (x[1] - x[0]**2) + 2*(x[0] - 1.0)
    dfdx1 = 200 * (x[1] - x[0]**2)
    return np.array([dfdx0, dfdx1])

def Lazy_Newton_Rosenbrock_d1(tol=1e-12, max_iter=10000):
    count = 0
    # diff initial guess
    x = np.array([1.2, 1.2])
    iterates = [x.copy()]
    
    while count < max_iter:
        g = grad(x)
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break
        
        # compute Hessian using finite differences
        H = hessian_fd(Rosen_fun_d1, x)

        p = solve(H, g)
        
        # kine search logic
        alpha = 1.0
        x_new = x - alpha * p
        ls_iter = 0
        max_ls_iter = 10  
        
        while Rosen_fun_d1(x_new) >= Rosen_fun_d1(x) and ls_iter < max_ls_iter:
            alpha *= 0.5
            x_new = x - alpha * p
            ls_iter += 1
        
        if Rosen_fun_d1(x_new) >= Rosen_fun_d1(x):
            print("Line search failed")
            break
        
        x = x_new.copy()
        iterates.append(x.copy())
        count += 1
        
        print(f"Iter: {count:4d} | x = {x} | f(x) = {Rosen_fun_d1(x):.6e} | ||grad|| = {grad_norm:.3e}")
    
    print("\nFinal x (d1):", x)
    print("Number of iterations (d1):", count)
    
    iterates = np.array(iterates)
    final_x = x
    errors = np.linalg.norm(iterates - final_x, axis=1)
    
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(errors)), np.log10(errors + 1e-18), 'b-o', label='Newton finite-diff')
    plt.title('Newton Iteration: log₁₀|x - x*| (Finite Differences)')
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Error)')
    plt.legend()
    plt.grid(True)
    plt.show()

# d2 

def Rosen_fun_d2(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (x[0] - 1.0)**2

def grad_d2(x):
    #analytic gradient for the Rosenbrock function
    dfdx = -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1.0)
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])

def hessian_analytic_d2(x):
    # analytic hessian for the rosenbrock function:
    H = np.zeros((2,2))
    H[0,0] = 1200 * x[0]**2 - 400 * x[1] + 2
    H[0,1] = -400 * x[0]
    H[1,0] = -400 * x[0]
    H[1,1] = 200
    return H

def Lazy_Newton_Rosenbrock_d2(tol=1e-12, max_iter=10000):
    count = 0
    # same initial guess
    x = np.array([1.2, 1.2])
    iterates = [x.copy()]
    
    while count < max_iter:
        g = grad_d2(x)
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break

        H = hessian_analytic_d2(x)
        
        p = solve(H, g)
        alpha = 1.0
        x_new = x - alpha * p
        ls_iter = 0
        max_ls_iter = 10
        while Rosen_fun_d2(x_new) >= Rosen_fun_d2(x) and ls_iter < max_ls_iter:
            alpha *= 0.5
            x_new = x - alpha * p
            ls_iter += 1
        
        if Rosen_fun_d2(x_new) >= Rosen_fun_d2(x):
            print("Line search failed")
            break
        
        x = x_new.copy()
        iterates.append(x.copy())
        count += 1
        
        print(f"Iter: {count:4d} | x = {x} | f(x) = {Rosen_fun_d2(x):.6e} | ||grad|| = {grad_norm:.3e}")
    
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
    plt.show()

print("Running d1 version with finite differences for Hessian:")
Lazy_Newton_Rosenbrock_d1()

print("\nRunning d2 version with analytic Hessian:")
Lazy_Newton_Rosenbrock_d2()


