import numpy as np
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions import interpolations as inter
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def gradient3d(f, dx):
    nx, ny, nz = f.shape
    inv2dx = cts.FTYPE(1.0 / (2.0 * dx))
    gx = np.empty((nx, ny, nz), dtype=cts.FTYPE)
    gy = np.empty((nx, ny, nz), dtype=cts.FTYPE)
    gz = np.empty((nx, ny, nz), dtype=cts.FTYPE)

    for i in prange(nx):
        inext = (i + 1) % nx
        iprev = (i - 1) % nx
        for j in range(ny):
            jnext = (j + 1) % ny
            jprev = (j - 1) % ny
            f_ij  = f[i,  j,  :]
            f_inextj = f[inext, j,  :]
            f_iprevj = f[iprev, j,  :]
            f_ijnext = f[i,  jnext, :]
            f_ijprev = f[i,  jprev, :]

            gx[i, j, 0] = (f_inextj[0]  - f_iprevj[0] ) * inv2dx
            gy[i, j, 0] = (f_ijnext[0]  - f_ijprev[0] ) * inv2dx
            gz[i, j, 0] = (f_ij [1]  - f_ij [nz-1]) * inv2dx

            for k in range(1, nz - 1):
                gx[i, j, k] = (f_inextj[k]   - f_iprevj[k]  ) * inv2dx
                gy[i, j, k] = (f_ijnext[k]   - f_ijprev[k]  ) * inv2dx
                gz[i, j, k] = (f_ij [k+1] - f_ij [k-1]) * inv2dx

            gx[i, j, nz-1] = (f_inextj[nz-1]  - f_iprevj[nz-1]) * inv2dx
            gy[i, j, nz-1] = (f_ijnext[nz-1]  - f_ijprev[nz-1]) * inv2dx
            gz[i, j, nz-1] = (f_ij [0]     - f_ij [nz-2]) * inv2dx

    return gx, gy, gz

#Cache functions for the gradient computation
gradcache = {}
def gradientcached(arr, dx, cache_key):
    entry = gradcache.get(cache_key)
    if entry is not None:
        return entry
    grad = gradient3d(arr, dx)
    gradcache[cache_key] = grad
    if len(gradcache) > 4:
        oldest = next(iter(gradcache))
        del gradcache[oldest]
    return grad

def cleargradientcache():
    gradcache.clear()

#Function that computes the acceleration for particles
def acceleration(grids, fieldY, pot):
    dx         = fieldY.getdx()
    xmin       = np.amin(fieldY.getunigrid())
    Ngrid      = len(fieldY.getunigrid())
    pos        = fieldY.fieldnc[0]

    key_q  = ('q',  id(pot.potentialq), pot.potversion)
    key_mf = ('mf', id(pot.mfpotential), pot.potversion)
    dU_q  = gradientcached(np.ascontiguousarray(pot.potentialq,  cts.FTYPE), dx, key_q)
    dU_mf = gradientcached(np.ascontiguousarray(pot.mfpotential, cts.FTYPE), dx, key_mf)
    acel_q  = -inter.interpolatevector(pos, dU_q,  Ngrid, dx, xmin)
    acel_mf = -inter.interpolatevector(pos, dU_mf, Ngrid, dx, xmin)
    return acel_q, acel_mf

