import numpy as np
from numba import njit, prange

#Function that corrects the position of a particle if the value goes outside the box according to periodical boundary conditions
@njit(parallel=True, fastmath=True)
def poscorrectornumba(x, Lmin, Lmax, grid_spacing):
    L = (Lmax - Lmin) + grid_spacing
    Nparticles = x.shape[0]
    for i in prange(Nparticles):
        x[i, 0] = Lmin + ((x[i, 0] - Lmin) % L)
        x[i, 1] = Lmin + ((x[i, 1] - Lmin) % L)
        x[i, 2] = Lmin + ((x[i, 2] - Lmin) % L)
    return x

#wrapper
def poscorrector(x,grid):
    dx = grid[1] - grid[0]
    return poscorrectornumba(x, np.amin(grid), np.amax(grid), dx)