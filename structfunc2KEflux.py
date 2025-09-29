from scipy.special import jv # Imports Bessel function 
import math
import numpy as np
import xarray as xr
import scipy as sp
from scipy.integrate import simpson as simps

def RLS(Y, W, P, X):
    
    ''' Regularized Least-Squares
    # Input:
    # Y = data
    # W = weights (either 2D or 1D array)
    # P = uncertainty in parameters (either 2D or 1D)
    # X = model to be fit
    #
    # Output:
    # x0 = coefficinets
    # n0 = residuals
    # Cxx = uncertainity
    '''
    if len(W.shape) == 2:
        if W.shape[0] != W.shape[1]:
            raise SyntaxError('W is not square')
        elif np.diag(W).shape[0] != X.shape[0]:
            raise SyntaxError('diag(W) not the same length as rows of X')
    elif len(W.shape) == 1:
        W = np.diag(W)
        if W.shape[0] != X.shape[0]:
            raise SyntaxError('diag(W) not the same length as rows of X')
    
    if len(P.shape) == 1:
        if len(P) == 1:
            P = P*np.identity(X.shape[1])
        elif len(P) != 1:
            if len(P) == X.shape[1]:
                P = np.diag(P)
            elif len(P) != X.shape[1]:
                raise SyntaxError('diag(P) not the same length as columns of X')
    elif len(P.shape) == 2:
        if P.shape[0] != P.shape[1]:
            raise SyntaxError('P is not a square array')
        elif P.shape[0] == P.shape[1]:
            if len(np.diag(P)) != X.shape[1]:
                raise SyntaxError('diag(P) not the same length as columns of X')
    
    if Y.shape[0] != X.shape[0]:
        raise SyntaxError('len(Y) not the same as rows of X')
        
    
    if np.mean(P, axis=(0, 1)) == 0:
        Pinv = P
    else:
        Pinv = np.linalg.inv(P) # P^{-1}

    Winv = np.linalg.inv(W) # W^{-1}
    
    M = np.dot(np.dot(X.T, Winv), X) # M = (A' x W^{-1} x A)
    Minv = np.linalg.inv(M + Pinv) # Minv = (M + P^{-1})^{-1}
    N = np.dot(np.dot(Minv, X.T), Winv) # N = Minv x A' x W^{-1}
    
    
    # Solution
    x0 = np.dot(N, Y) # x0 = N x Y
    
    # Residuals
    y0 = np.dot(X, x0)
    n0 = Y - y0 # n0 = Y - A x x0
    
    # Uncertainity
#     B = np.dot(np.dot(np.dot(Minv, X.T), Winv), W)
#     B = np.dot(np.dot(B, Winv), X)
    Cxx = Minv
    
    return x0, y0, n0, Cxx

def errorsFlux(Cxx, H):
    ''' Converts from errors in KE injection rates to erros in Energy Flux
        
        Input:
        Cxx: square matrix containing errors in KE injection rates
        H: Transformation matrix'''
    
    return np.dot(np.dot(H, Cxx), H.T)

def defH(k, dk):
    ''' Defines matrix that transforms from errors in KE injection rates to errors in KE flux
        
        Input:
        k: wavenumber array
        dk: delta k array'''
    
    H = np.zeros((len(k), len(k)+1))
    Hlog = np.zeros((len(k), len(k)))
    
    for i in range(len(k)):
        Hlog[:, i] = np.heaviside(k - k[i], 1)
    
    # Iterates for all 
    for jj in range(len(k)):
        for ii in range(len(k)):
            H[jj, ii+1] = Hlog[jj, ii]*dk[jj]
    H[:, 0] = -np.ones((len(k)))
    return H

def defA(r, k, dk):

    '''Defines model matrix
    
       Input:
       r: distance bins
       k: wavenumber bins
       dk: delta k'''
    
    A = np.zeros((len(r), len(k)+1))
    
    # Iterates for all 
    for jj in range(len(k)):
        for ii in range(len(r)):
            A[ii, jj+1] = -4*jv(1, k[jj]*r[ii])/k[jj]*dk[jj]
    
    A[:, 0] = 2*r
    
    return A

def tick_function(X):
    V = 1/X
    return ["%3.0f" % z for z in V]

def calcFk(eps, k, dk):
    '''Calculates KE flux:
        
        Input:
        eps: KE injection rates; first element is eps_u and next ones are eps_j increasing in wavenumber k
        k: wavenumber bins
        dk: delta k'''

    # Calculat KE flux
    Fk = np.zeros((len(k),))
    Fk[0] = -eps[0]
        
    for jj in range(len(k)-1):
        Fk[jj+1] = Fk[jj] + eps[jj+1]*dk[jj]
        
    return Fk

# def Fk2SF3(Flux, k, dk, r):
#     '''Converts the KE flux to structure function using a Hankel Transform (Xie and Buhler, 2018)
       
#        Input:
#        Flux: Spectral Flux
#        k: wavenumber array
#        dk: wavenumber resolution array
#        r: distance bins
       
#        Output:
#        SF3: third-order structure function'''
    
#     flux_spec = Flux
#     k_spec = k

#     rspec, kkspec = np.meshgrid(r, k_spec)
#     J2 = jv(2, rspec*kkspec)
#     intl = J2.T*flux_spec*dk
#     intl = intl*1/k_spec
#     inlf = np.sum(intl, axis=1)
    
#     return -4*r*inlf



def Fk2SF3(Flux, k, dk, r):
    """
    Converts the KE flux to the third-order structure function using a Hankel Transform
    (Xie and Bühler, 2018) with Simpson's rule for higher-accuracy integration.

    Parameters
    ----------
    Flux : ndarray
        Spectral flux F(k).
    k : ndarray
        Wavenumber array (1D).
    dk : ndarray
        Wavenumber resolution (1D, same size as k).
    r : ndarray
        Distance bins.

    Returns
    -------
    SF3 : ndarray
        Third-order structure function D3(r).
    """
    k = np.asarray(k)
    r = np.asarray(r)
    Flux = np.asarray(Flux)

    # Mesh for J2(k*r)
    rspec, kspec = np.meshgrid(r, k, indexing='ij')
    J2 = jv(2, kspec * rspec)

    # Integrand: F(k)/k * J2(k*r)
    integrand = (Flux / k)[:, None] * J2.T  # shape: (k, r)

    # Simpson's rule along k dimension
    D3 = -4 * r * simps(integrand, k, axis=0)

    return D3


def PnoiseW(X, P, W):
    # Equation 10 from Bruce's notes
    if len(W.shape) == 2:
        if W.shape[0] != W.shape[1]:
            raise SyntaxError('W is not square')
        elif np.diag(W).shape[0] != X.shape[0]:
            raise SyntaxError('diag(W) not the same length as rows of X')
    elif len(W.shape) == 1:
        W = np.diag(W)
        if W.shape[0] != X.shape[0]:
            raise SyntaxError('diag(W) not the same length as rows of X')
    
    if len(P.shape) == 1:
        if len(P) == 1:
            P = P*np.identity(X.shape[1])
        elif len(P) != 1:
            if len(P) == X.shape[1]:
                P = np.diag(P)
            elif len(P) != X.shape[1]:
                raise SyntaxError('diag(P) not the same length as columns of X')
    elif len(P.shape) == 2:
        if P.shape[0] != P.shape[1]:
            raise SyntaxError('P is not a square array')
        elif P.shape[0] == P.shape[1]:
            if len(np.diag(P)) != X.shape[1]:
                raise SyntaxError('diag(P) not the same length as columns of X')
    
    Pnoise = np.dot(np.dot(X, P), X.T) + W
    return Pnoise

def Pomm(X, Phat):
    if len(Phat.shape) == 1:
        if len(Phat) == 1:
            Phat = Phat*np.identity(X.shape[1])
        elif len(Phat) != 1:
            if len(Phat) == X.shape[1]:
                Phat = np.diag(Phat)
            elif len(Phat) != X.shape[1]:
                raise SyntaxError('diag(Phat) not the same length as columns of X')
    elif len(Phat.shape) == 2:
        if Phat.shape[0] != Phat.shape[1]:
            raise SyntaxError('Phat is not a square array')
        elif Phat.shape[0] == Phat.shape[1]:
            if len(np.diag(Phat)) != X.shape[1]:
                raise SyntaxError('diag(Phat) not the same length as columns of X')
                
    Pomm = np.dot(np.dot(np.linalg.inv(X), Phat), np.linalg.inv(X).T)
    return Pomm

def confidence_interval(x, ci=0.95):
    '''Calculates confidence intervals using 95% t-student distribution'''
    
    x_mean = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(x.shape[0])
    h = se * sp.stats.t._ppf((1 + ci)/2. , x.shape[0])
    return x_mean - h, x_mean + h