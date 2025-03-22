import scipy.signal as sig
import numpy as np
import math
from scipy.special import jv # Imports Bessel function 


def spectralFlux(un, vn, x, y, wind, detr):
    ''' CALCULATES SPECTRAL FLUXES USING CONVENTIONAL METHOD (Ayaji et al. 2019)
        Input:
        un, vn: horizontal velocity components (gridded)
        x, y: distance vectors in meters
        wind: if == 1, windowing is performed
        detr: if == 1, trend and mean are removed '''
    
    Nx = len(x)
    Ny = len(y)
    
    # Removes trends and mean
    if detr == 1:
        un = un - np.mean(un)
        vn = vn - np.mean(vn)
        un = sig.detrend(un, axis=0, type='linear')
        un = sig.detrend(un, axis=1, type='linear')
        vn = sig.detrend(vn, axis=0, type='linear')
        vn = sig.detrend(vn, axis=1, type='linear')

    # Windowing
    if wind == 1:
        Nx, Ny = un.shape
        hannx = sig.hann(Nx, sym=False)
        hanny = sig.hann(Ny, sym=False)
        hann = hannx[np.newaxis, ...]*hanny[..., np.newaxis]
        un *= hann
        vn *= hann

    dx = np.mean(np.diff(x, axis=0))
    dy = np.mean(np.diff(y, axis=0))
    
    # Defines wavenumbers
    kx = np.fft.fftfreq(Nx, dx)
    ky = np.fft.fftfreq(Ny, dy)
    Kmax = max(kx.max(), ky.max())
    kk, ll = np.meshgrid(kx, ky)
    K2D = np.sqrt(kk**2 + ll**2)
    ddk = 1./(dx*Nx)
    ddl = 1./(dy*Ny)
    dK = max(ddk, ddl)
    K1D = dK*np.arange(1, int(Kmax/dK))


    # Calculates divergences in fourier space \int^{k_max}_0 \nabla u e^{-ikx}dx = k*\int^{k_max}_0 u e^{-ikx}dx
    dudx = np.real(np.fft.ifft2(np.fft.fft2(un)*(1j*kx*2*np.pi)[None, :]));
    dudy = np.real(np.fft.ifft2(np.fft.fft2(un)*(1j*ky*2*np.pi)[:, None]));
    dvdx = np.real(np.fft.ifft2(np.fft.fft2(vn)*(1j*kx*2*np.pi)[None, :]));
    dvdy = np.real(np.fft.ifft2(np.fft.fft2(vn)*(1j*ky*2*np.pi)[:, None]));

    # Multiplies by complex conjugate of velocity
    t1 = un*dudx + vn*dudy
    t2 = un*dvdx + vn*dvdy

    # Fourier transform and complex conjugate
    KEtrans = (np.fft.fft2(un).conj())*np.fft.fft2(t1) + (np.fft.fft2(vn).conj())*np.fft.fft2(t2)
    KEtransf2D = -np.real(KEtrans)/np.square(Nx*Ny)
#     specflux = np.zeros(len(K1D))
    
#     for i in range(K1D.size):
#         kfilt =  ((K1D[i]  <= K2D))
#         specflux[i] = (KEdiv2D[kfilt]).sum()

    transfer = np.zeros(len(K1D))
    for i in range(K1D.size):
        kfilt =  ((K1D[i] - K1D[0]) <= K2D) & (K2D <= K1D[i])
        transfer[i] = (KEtransf2D[kfilt]).sum()
    
    specflux = np.cumsum(transfer[::-1])[::-1] # - Get flux
    
    return K1D, specflux

def E2SF2(Ek, k, dk, r):
    '''Converts the isotropic KE spectra to second-order structure function (Xie and Buhler, 2018)
       
       Input:
       Ek: Isotropic spectrum
       k: wavenumber array
       dk: wavenumber resolution
       r: distance bins
       
       Output:
       SF2: third-order structure function'''
    

    rspec, kkspec = np.meshgrid(r, k)
    J2 = 1 - jv(0, rspec*kkspec)
    intl = Ek[:, None]*J2*dk
    sf2 = 2*np.sum(intl, axis=0)
    
    return sf2

def specK(x, y):
    '''Calculates 1D wavenumber vector 
       
       Input:
       x, y: distance arrays in meters'''
    
    dx = np.mean(np.diff(x, axis=0))
    dy = np.mean(np.diff(y, axis=0))
    
    Nx = len(x)
    Ny = len(y)
    
    # Defines wavenumbers
    kx = np.fft.fftfreq(Nx, dx)
    ky = np.fft.fftfreq(Ny, dy)
    Kmax = max(kx.max(), ky.max())
    kk, ll = np.meshgrid(kx, ky)
    K2D = np.sqrt(kk**2 + ll**2)
    ddk = 1./(dx*Nx)
    ddl = 1./(dy*Ny)
    dK = max(ddk, ddl)
    return dK*np.arange(1, int(Kmax/dK))

def spec_est2(A,d1,d2,win=True):

    """    Computes 2D spectral estimate of A

           Input: 
           obs: the returned array is fftshifted
           and consistent with the f1,f2 arrays
           d1,d2 are the sampling rates in rows,columns   """
    
    import numpy as np

    l1,l2= A.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)

    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    
    if win == True:
        wx = np.matrix(np.hanning(l1))
        wy =  np.matrix(np.hanning(l2))
        window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2)
    else:
        window_s = np.ones((l1,l2))

    an = np.fft.fft2(A*window_s,axes=(0,1))
    E = (an*an.conjugate()) / (df1*df2) / ((l1*l2)**2)
    E = np.fft.fftshift(E)
#     E = E.mean(axis=2)
    
    return np.real(E),f1,f2,df1,df2,f1Ny,f2Ny


def Ek2DEkiso(E, k, l, dk, dl):
    '''Calculates Isotropic Wavenumber spectra from 2D spectra
       
       Input:
       E: 2D wavenumber spectra
       k,l: zonal and meridional wavenumbers
       dk, dl: zonal and meridional wavenumber resolution'''

    ## try to go POLAR
    ki,li = np.meshgrid(k,l)
    K = np.sqrt(ki**2+li**2)
    # K = np.ma.masked_array(K,K<1.e-10)

    phi = np.math.atan2(dl,dk)
    dK = dk*np.cos(phi)
    Ki = np.arange(K.min(),K.max(), dK*2/3)
    Ki2  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK*(2/3)/2.

    Eiso = np.zeros(Ki2.size)

    for i in range(Ki2.size):
        f =  (K>= Ki2[i]-dK2)&(K<=Ki2[i]+dK2)
        dtheta = (2*np.pi)/(f.sum())
#     Eiso[i] = ((E[f].sum()))*Ki2[i]*dtheta
        Eiso[i] = np.mean(E.mean(axis=2)[f])*Ki2[i]

    Ki2 = Ki2[~np.isnan(Eiso)]    
    Eiso = Eiso[~np.isnan(Eiso)]
    return Eiso, Ki2