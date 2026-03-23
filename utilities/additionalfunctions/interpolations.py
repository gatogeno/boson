import numpy as np
from numba import njit, prange
from utilities.additionalfunctions import constants as cts

@njit(parallel=True, fastmath=True)
def interpolatescalar(pos, scalararray, Ngrid, dx, xmin):
    Nparticles = pos.shape[0]
    result = np.zeros(Nparticles, dtype=cts.FTYPE)

    for i in prange(Nparticles):
        rx = (pos[i, 0] - xmin) / dx
        ry = (pos[i, 1] - xmin) / dx
        rz = (pos[i, 2] - xmin) / dx

        bx, by, bz = cts.ITYPE(np.floor(rx)), cts.ITYPE(np.floor(ry)), cts.ITYPE(np.floor(rz))
        fx, fy, fz = rx - bx, ry - by, rz - bz
            
        accumula = cts.FTYPE(0.0)
        for ix in range(2):
            wx = fx if ix else (1.0 - fx)
            idx_x = (bx + ix) % Ngrid
            
            for iy in range(2):
                wy = wx * (fy if iy else (1.0 - fy))
                idx_y = (by + iy) % Ngrid
                
                for iz in range(2):
                    wz = wy * (fz if iz else (1.0 - fz))
                    idx_z = (bz + iz) % Ngrid
                    
                    accumula += wz * scalararray[idx_x, idx_y, idx_z]
            
        result[i] = accumula
            
    return result
    
@njit(parallel=True, fastmath=True)
def interpolatevector(pos, vecarray, Ngrid, dx, xmin):
    Nparticles = pos.shape[0]
    result = np.zeros(pos.shape, dtype=cts.FTYPE)
    
    for i in prange(Nparticles):
        rx = (pos[i, 0] - xmin) / dx
        ry = (pos[i, 1] - xmin) / dx
        rz = (pos[i, 2] - xmin) / dx

        bx, by, bz = cts.ITYPE(np.floor(rx)), cts.ITYPE(np.floor(ry)), cts.ITYPE(np.floor(rz))
        fx, fy, fz = rx - bx, ry - by, rz - bz
            
        for l in range(3): 
            veccomponent = vecarray[l]
            accumula = cts.FTYPE(0.0)     
            for ix in range(2):
                wx = fx if ix else (1.0 - fx)
                idx_x = (bx + ix) % Ngrid
                for iy in range(2):
                    wy = wx * (fy if iy else (1.0 - fy))
                    idx_y = (by + iy) % Ngrid
                    for iz in range(2):
                        wz = wy * (fz if iz else (1.0 - fz))
                        idx_z = (bz + iz) % Ngrid
                        
                        accumula += wz * veccomponent[idx_x, idx_y, idx_z]
            
            result[i, l] = accumula
              
    return result


#Function to interpolate values of the density in a grid to points inside of the grid
def interpolatordens(fieldY):
    dx = fieldY.getdx()
    pos = fieldY.fieldnc[0]
    unigrid = fieldY.getunigrid()
    xmin = np.amin(unigrid)
    Ngrid = len(unigrid)
    dens = np.ascontiguousarray(fieldY.densityc, dtype=cts.FTYPE)

    return interpolatescalar(pos, dens, Ngrid, dx, xmin)

def interpolatorntilde(fieldY):
    dx = fieldY.getdx()
    pos = fieldY.fieldnc[0]
    unigrid = fieldY.getunigrid()
    xmin = np.amin(unigrid)
    Ngrid = len(unigrid)
    dens = np.ascontiguousarray(fieldY.densityq, dtype=cts.FTYPE)

    return interpolatescalar(pos, dens, Ngrid, dx, xmin)

#Function to interpolate values of the potential in a grid to points inside of the grid
def interpolatorpot(fieldY,pot):
    dx = fieldY.getdx()
    pos = fieldY.fieldnc[0]
    unigrid = fieldY.getunigrid()
    xmin = np.amin(unigrid)
    Ngrid = len(unigrid)
    pote = np.ascontiguousarray(pot.mfpotential, dtype=cts.FTYPE)

    return interpolatescalar(pos, pote, Ngrid, dx, xmin)
    