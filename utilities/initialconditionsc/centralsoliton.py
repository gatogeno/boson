import numpy as np
from utilities.additionalfunctions import constants as cts

def centralsoliton(fundparameters,fieldparameters,grids,grid_shape):
    r2=grids[2]**2+grids[4]**2+grids[6]**2
    n0=fundparameters[3]*fundparameters[2]/(11.59*fieldparameters[0][1]**3)
    field=(np.sqrt(n0/(1+0.091*(r2/fieldparameters[0][1])**2)**8)*np.exp(-1.0j*np.zeros(grid_shape))).astype(cts.CTYPE)

    return field