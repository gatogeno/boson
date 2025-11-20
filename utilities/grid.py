import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift

#Create a spatial and momentum grid
def creategrid(parameters):
    dx = np.float32(parameters[1]/parameters[2])
    xlin = np.linspace(-parameters[1]/2.0, parameters[1]/2.0, parameters[2],endpoint=False, dtype=np.float32)
    klin = 2*np.pi*fftfreq(parameters[2], dx).astype(np.float32)
    
    if parameters[0]==1:
        ksq=(klin**2).astype(np.float32)
        return dx, ksq, xlin, klin
    elif parameters[0]==2:
        xx, yy = np.meshgrid(xlin,xlin, indexing='ij', copy=False)
        kx, ky = np.meshgrid(klin,klin, indexing='ij', copy=False)
        ksq=(kx**2+ky**2).astype(np.float32)
        return dx, ksq, xx, kx, yy, ky
    elif parameters[0]==3:
        xx, yy, zz = np.meshgrid(xlin,xlin,xlin, indexing='ij', copy=False)
        kx, ky, kz = np.meshgrid(klin,klin,klin, indexing='ij', copy=False)
        ksq=(kx**2+ky**2+kz**2).astype(np.float32)
        return dx, ksq, xx, kx, yy, ky, zz, kz