import numpy as np
from utilities.additionalfunctions import constants as cts

def gaussianpackets(grids):
    alpha=0.25
    x0=np.zeros(3)
    k0=np.zeros(3)
    x0[:] = [2.5, 2.5, 0.0][:3]
    k0[:] = [0.1, 0.1, 0.0][:3]

    r2=(grids[2]-x0[0])**2+(grids[4]-x0[1])**2+(grids[6]-x0[2])**2
    r2b=(grids[2]+x0[0])**2+(grids[4]+x0[1])**2+(grids[6]+x0[2])**2
    kx=k0[0]*grids[2]+k0[1]*grids[4]+k0[2]*grids[6]
    field=np.exp(-0.5*r2/alpha+1.0j*kx).astype(cts.CTYPE)+np.exp(-0.5*r2b/alpha-1.0j*kx).astype(cts.CTYPE)

    return field