import numpy as np
from scipy.fft import fftfreq
from utilities.additionalfunctions import constants as cts

#Create a spatial and momentum grid
def creategrid(fundparameters):
    dx = cts.FTYPE(fundparameters[0]/fundparameters[1])
    xlin = np.linspace(-fundparameters[0]/2.0, fundparameters[0]/2.0, fundparameters[1],endpoint=False, dtype=cts.FTYPE)
    klin = 2*np.pi*fftfreq(fundparameters[1], dx).astype(cts.FTYPE)
    
    xx, yy, zz = np.meshgrid(xlin,xlin,xlin, indexing='ij', copy=False)
    kx, ky, kz = np.meshgrid(klin,klin,klin, indexing='ij', copy=False)
    ksq=(kx**2+ky**2+kz**2).astype(cts.FTYPE)
    zerokmask = np.abs(ksq) < 1e-8
    invksq=np.divide(1.0,ksq,where=~zerokmask,out=np.zeros_like(ksq,dtype=cts.FTYPE))
    return ksq, invksq, xx, kx, yy, ky, zz, kz