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

def grad_d2(x_guess):
    df_dx0 = 200*(x_guess[1]-x_guess[0]**2)*(-2*x_guess[0])+2*(x_guess[0]-1)
    df_dx1 = 200*(x_guess[1]-x_guess[0]**2)+200*(x_guess[2]-x_guess[1]**2)*(-2*x_guess[1])+2*(x_guess[1]-1)
    df_dx2 = 200*(x_guess[2]-x_guess[1]**2)
    return np.array([df_dx0, df_dx1, df_dx2])

def Hessian_d2(x_guess):
    H = differentiate.hessian(Rosen_fun_d2,x_guess)
    return H

def Rosen_fun_d2(x_guess):
    term_1 = 100*(x_guess[1]-x_guess[0]**2)**2+(x_guess[0]-1)**2
    term_2 = 100*(x_guess[2]-x_guess[1]**2)**2+(x_guess[1]-1)**2
    return term_1 + term_2

def grad_d3(x_guess):
    df_dx0 = 200*(x_guess[1]-x_guess[0]**2)*(-2*x_guess[0])+2*(x_guess[0]-1)
    df_dx1 = 200*(x_guess[1]-x_guess[0]**2)+200*(x_guess[2]-x_guess[1]**2)*(-2*x_guess[1])+2*(x_guess[1]-1)
    df_dx2 = 200*(x_guess[2]-x_guess[1]**2)+200*(x_guess[3]-x_guess[2]**2)*(-2*x_guess[2])+2*(x_guess[2]-1)
    df_dx3 = 200*(x_guess[3]-x_guess[2]**2)
    return np.array([df_dx0, df_dx1, df_dx2, df_dx3])

def Hessian_d3(x_guess):
    H = differentiate.hessian(Rosen_fun_d3,x_guess)
    return H

def Rosen_fun_d3(x_guess):
    term_1 = 100*(x_guess[1]-x_guess[0]**2)**2+(x_guess[0]-1)**2
    term_2 = 100*(x_guess[2]-x_guess[1]**2)**2+(x_guess[1]-1)**2
    term_3 = 100*(x_guess[3]-x_guess[2]**2)**2+(x_guess[2]-1)**2
    return term_1 + term_2 + term_3



def Newton_Rosenbrack_d1(tol):
    count = 0 # this variable is a safegaurd against infinite loops
    x = np.array([15,15]) # this is our initial guess, and later our x_n
    rn = np.array([x])
    x1 = 2 * x # after this, we use x1 to save our x_n-1
    while (count < 100 and (np.abs(Rosen_fun_d1(x)-Rosen_fun_d1(x1))) > tol):
        #While we're done less then 100 iters #While |f(x_n)-f(x_n-1)| is less then tol
        count = count + 1
        hessian = Hessian(x) # for Newton direction
        x1 = x # save x_n-1
        
        grad_x = grad(x)
        x = x - 1*(inv(hessian.ddf)@grad_x)

        print('x is:')
        print(x)
        print(Rosen_fun_d1(x))
        rn = np.append(rn, np.array([x]), axis=0)
        print('rn')
        print(rn)
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

Newton_Rosenbrack_d1(1e-16)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))




def Newton_Rosenbrack_d2(tol):
    count = 0 # this variable is a safegaurd against infinite loops
    x = np.array([15,15,15]) # this is our initial guess, and later our x_n
    rn = np.array([x])
    x1 = 2 * x # after this, we use x1 to save our x_n-1
    while (count < 100 and (np.abs(Rosen_fun_d2(x)-Rosen_fun_d2(x1))) > tol):
        #While we're done less then 100 iters #While |f(x_n)-f(x_n-1)| is less then tol
        count = count + 1
        hessian = Hessian_d2(x) # for Newton direction
        x1 = x # save x_n-1
        
        grad_x = grad_d2(x)
        x = x - 1*(inv(hessian.ddf)@grad_x)

        print('x is:')
        print(x)
        print(Rosen_fun_d2(x))
        rn = np.append(rn, np.array([x]), axis=0)
        print('rn')
        print(rn)
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

Newton_Rosenbrack_d2(1e-16)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))



def Newton_Rosenbrack_d3(tol):
    count = 0 # this variable is a safegaurd against infinite loops
    x = np.array([15,15,15,15]) # this is our initial guess, and later our x_n
    rn = np.array([x])
    x1 = 2 * x # after this, we use x1 to save our x_n-1
    while (count < 100 and (np.abs(Rosen_fun_d3(x)-Rosen_fun_d3(x1))) > tol):
        #While we're done less then 100 iters #While |f(x_n)-f(x_n-1)| is less then tol
        count = count + 1
        hessian = Hessian_d3(x) # for Newton direction
        x1 = x # save x_n-1
        
        grad_x = grad_d3(x)
        x = x - 1*(inv(hessian.ddf)@grad_x)

        print('x is:')
        print(x)
        print(Rosen_fun_d3(x))
        rn = np.append(rn, np.array([x]), axis=0)
        print('rn')
        print(rn)
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

Newton_Rosenbrack_d3(1e-16)

#print(Rosen_fun_d1(np.array([guess,guess])))
#print(Hessian(np.array([guess,guess])))


