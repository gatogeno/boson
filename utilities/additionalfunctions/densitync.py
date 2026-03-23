import numpy as np
from numba import njit, prange
from numba.np.ufunc import parallel as numba_parallel
from numba import get_num_threads
from utilities.additionalfunctions import constants as cts

@njit(parallel=True, fastmath=True)
def computedensity(pos, w1, w2, Ngrid, dx, xmin):
    num_threads = get_num_threads()
    totalcells = Ngrid**3
    Nparticles = pos.shape[0]

    dens1 = np.zeros((num_threads, totalcells), dtype=cts.FTYPE)
    dens2 = np.zeros((num_threads, totalcells), dtype=cts.FTYPE)

    sy = Ngrid
    sx = Ngrid**2

    for i in prange(Nparticles):
        thread_id = numba_parallel.get_thread_id()
        rx = (pos[i, 0] - xmin) / dx
        ry = (pos[i, 1] - xmin) / dx
        rz = (pos[i, 2] - xmin) / dx

        p1 = w1[i]
        p2 = w2[i]

        bx = cts.ITYPE(np.floor(rx))
        by = cts.ITYPE(np.floor(ry))
        bz = cts.ITYPE(np.floor(rz))
        
        fx = rx - bx
        fy = ry - by
        fz = rz - bz
        
        for ix in range(2):
            wx = fx if ix else (1.0 - fx)
            idx_x = ((bx + ix) % Ngrid) * sx
            
            for iy in range(2):
                wy = wx * (fy if iy else (1.0 - fy))
                idx_y = ((by + iy) % Ngrid) * sy
                
                for iz in range(2):
                    wz = wy * (fz if iz else (1.0 - fz))
                    idx_z = (bz + iz) % Ngrid
                    
                    flat_idx = idx_x + idx_y + idx_z
                    dens1[thread_id, flat_idx] += wz * p1
                    dens2[thread_id, flat_idx] += wz * p2

    shape3d=(Ngrid,Ngrid,Ngrid)
    d1 = np.sum(dens1, axis=0).reshape(shape3d)
    d2 = np.sum(dens2, axis=0).reshape(shape3d)
    return d1, d2
