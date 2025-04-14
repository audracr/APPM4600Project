import numpy as np
import numpy.linalg as la
from scipy.linalg import inv
from matplotlib import pyplot as plt
from scipy import differentiate


def Rosen_fun_d1(x_guess):
    term_1 = 100*(x_guess[1]-x_guess[0]**2)**2+(x_guess[0]-1)**2
    return term_1

def grad(x_guess):
    df_dx0 = 200*(x_guess[1]-x_guess[0]**2)*(-2*x_guess[0])+2*(x_guess[0]-1)
    df_dx1 = 200*(x_guess[1]-x_guess[0]**2)
    return np.array([df_dx0, df_dx1])

def Hessian(x_guess):
    H = differentiate.hessian(Rosen_fun_d1,x_guess)
    return H

def Lazy_Newton_Rosenbrack_d1(tol):
    count = 0 # this variable is a safegaurd against infinite loops
    x = np.array([15,15]) # this is our initial guess, and later our x_n
    rn = np.array([x])
    x1 = 2 * x # after this, we use x1 to save our x_n-1
    p_k = x # in general p_k = x_k+1 - x_k
    while (count < 10000 and (np.abs(Rosen_fun_d1(x)-Rosen_fun_d1(x1))) > tol):
        #While we're done less then 100 iters #While |f(x_n)-f(x_n-1)| is less then tol
        count = count + 1
        x1 = x # save x_n-1

        g_k = grad(x)
        if (count != 0): # if we should update the hessian
            hessian = Hessian(x).ddf
            B_k = hessian
         # for Newton direction
        print(g_k)
        print(-g_k)
        p_k = (inv(B_k))@(g_k)
        alpha = 10
        for i in range(1,51):
            alpha = alpha / i
            x = x - alpha*p_k
            print(Rosen_fun_d1(x))
            print(Rosen_fun_d1(x1))
            if (Rosen_fun_d1(x)<Rosen_fun_d1(x1)):
                break
            else:
                x = x1
        #print('x is:')
        #print(x)
        #print(Rosen_fun_d1(x))
        rn = np.append(rn, np.array([x]), axis=0)
        #print('rn')
        #print(rn)
        #break
    print('num iters: ')
    print(count)
    print('xmin: ')
    print(x)
    print(rn)
    
    rN=x
    rnN = rn
    
    numN = rnN.shape[0];
    print(rnN[0:(numN-1)])
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

Lazy_Newton_Rosenbrack_d1(1e-16)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))


def Rosen_fun_d2(x_guess):
    term_1 = 10*(x_guess[1]-x_guess[0]**2)**2+(x_guess[0]-1)**2
    term_2 = 10*(x_guess[2]-x_guess[1]**2)**2+(x_guess[1]-1)**2
    return term_1 + term_2

def grad_d2(x_guess):
    df_dx0 = 20*(x_guess[1]-x_guess[0]**2)*(-2*x_guess[0])+2*(x_guess[0]-1)
    df_dx1 = 20*(x_guess[1]-x_guess[0]**2)+20*(x_guess[2]-x_guess[1]**2)*(-2*x_guess[1])+2*(x_guess[1]-1)
    df_dx2 = 20*(x_guess[2]-x_guess[1]**2)
    return np.array([df_dx0, df_dx1, df_dx2])

def Hessian_d2(x_guess):
    H = differentiate.hessian(Rosen_fun_d2,x_guess)
    return H

def Lazy_Newton_Rosenbrack_d2(tol):
    count = 0 # this variable is a safegaurd against infinite loops
    x = np.array([1.2,1.2,1.2]) # this is our initial guess, and later our x_n
    rn = np.array([x])
    x1 = 2 * x # after this, we use x1 to save our x_n-1
    p_k = x # in general p_k = x_k+1 - x_k
    while (count < 100 and (np.abs(Rosen_fun_d2(x)-Rosen_fun_d2(x1))) > tol):
        #While we're done less then 100 iters #While |f(x_n)-f(x_n-1)| is less then tol
        count = count + 1
        x1 = x # save x_n-1

        g_k = grad_d2(x)
        if (count != 0): # if we should update the hessian
            hessian = Hessian_d2(x).ddf
            B_k = hessian
         # for Newton direction
        print(g_k)

        p_k = (inv(B_k))@(g_k)
        print((inv(B_k)))
        alpha = 10
        for i in range(1,101):
            alpha = alpha / i
            x = x - alpha*p_k
            print(Rosen_fun_d2(x))
            print(Rosen_fun_d2(x1))
            if (Rosen_fun_d2(x)<Rosen_fun_d2(x1)):
                break
            else:
                x = x1
        #print('x is:')
        #print(x)
        #print(Rosen_fun_d1(x))
        rn = np.append(rn, np.array([x]), axis=0)
        #print('rn')
        #print(rn)
        #break
    print('num iters: ')
    print(count)
    print('xmin: ')
    print(x)
    print(rn)
    
    rN=x
    rnN = rn
    
    numN = rnN.shape[0];
    print(rnN[0:(numN-1)])
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

Lazy_Newton_Rosenbrack_d2(1e-16)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))


