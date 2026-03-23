import numpy as np
import math
from numba import njit
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions import auxiliarycollisions as acol
from utilities.additionalfunctions.weights import W1

def launchercolI2(fieldY, gncinterp, ntilde, totalomega, dt, subcells, subcellcoord, subcelllength, randarr):
    positions  = fieldY.fieldnc[0]
    velocities = fieldY.fieldnc[1]
    g = fieldY.g
    Nq = (1.0 - fieldY.getcohfrac()) * fieldY.gettotalnumber()
    Ntp = len(positions)
    gamma = cts.FTYPE(Nq / Ntp)
    Ngrid = len(fieldY.getunigrid())
    dx = fieldY.getdx()
    xmin = np.amin(fieldY.getunigrid())
    prefactor = cts.FTYPE(2.0 * g * g * Ntp / (math.pi * totalomega))
    
    collisionrate = np.zeros((Ngrid, Ngrid, Ngrid), dtype=cts.FTYPE)
    I2cells(subcells, subcellcoord, subcelllength, positions, velocities, gncinterp, ntilde, dt, prefactor, gamma, randarr, collisionrate, xmin, dx, Ngrid)
 
    return collisionrate


@njit(cache=True)
def I2cells(subcells,subcellcoord,subcelllength,positions,velocities,gncinterp,ntilde,dt,prefactor,gamma,randarr,collisionrate,xmin,dx,Ngrid):
    for i in range(len(subcells)):
        I2subcell(subcells[i],subcellcoord[i],subcelllength[i],positions,velocities,gncinterp,ntilde,dt,prefactor,gamma,randarr,collisionrate, xmin,dx,Ngrid)


@njit(cache=True)
def I2subcell(subcells,subcellcoord,subcelllength,positions,velocities,gncinterp,ntilde,dt,prefactor,gamma,randarr,collisionrate,xmin,dx,Ngrid):
    #No collisions if there is no two particles in the cell
    Numinsc = len(subcells)
    if Numinsc < 2:
        return
 
    dV = subcelllength * subcelllength * subcelllength
 
    velstorer = np.empty((Numinsc, 3), dtype=velocities.dtype)
    sumgnc = 0.0
    sumw1 = 0.0
    sumntilde = 0.0
 
    for k in range(Numinsc):
        indcel = subcells[k]
        v0 = velocities[indcel, 0]
        v1 = velocities[indcel, 1]
        v2 = velocities[indcel, 2]
        velstorer[k, 0] = v0
        velstorer[k, 1] = v1
        velstorer[k, 2] = v2
        vsq = v0*v0 + v1*v1 + v2*v2
 
        sumgnc += gncinterp[indcel]
        sumntilde += ntilde[indcel]
        sumw1 += W1(vsq, gncinterp[indcel])
 
    gncmean = sumgnc / cts.FTYPE(Numinsc)
    ntildemean = sumntilde / cts.FTYPE(Numinsc)
 
    granprefactor = prefactor * ntildemean * sumw1 / cts.FTYPE(Numinsc)
    invdx3 = 1.0 / (dx * dx * dx)

    #Center of subcell for collision rate
    midx = subcellcoord[0] + 0.5 * subcelllength
    midy = subcellcoord[1] + 0.5 * subcelllength
    midz = subcellcoord[2] + 0.5 * subcelllength
    fx = (midx - xmin) / dx
    fy = (midy - xmin) / dx
    fz = (midz - xmin) / dx
    ixl = cts.ITYPE(math.floor(fx)) % Ngrid
    iyl = cts.ITYPE(math.floor(fy)) % Ngrid
    izl = cts.ITYPE(math.floor(fz)) % Ngrid
    wxl = fx - math.floor(fx)
    wyl = fy - math.floor(fy)
    wzl = fz - math.floor(fz)
    ixr = (ixl + 1) % Ngrid
    iyr = (iyl + 1) % Ngrid
    izr = (izl + 1) % Ngrid
    wxr  = 1.0 - wxl
    wyr = 1.0 - wyl
    wzr = 1.0 - wzl

    #Collision process
    for a in range(0, Numinsc - 1, 2):
        i = subcells[a]
        j = subcells[a + 1]
        vi0 = velocities[i, 0]
        vi1 = velocities[i, 1]
        vi2 = velocities[i, 2]
        visq = vi0*vi0 + vi1*vi1 + vi2*vi2
        vj0 = velocities[j, 0]
        vj1 = velocities[j, 1]
        vj2 = velocities[j, 2]
        vjsq = vj0*vj0 + vj1*vj1 + vj2*vj2
 
        E12 = acol.bogoliubovenergy(visq, gncmean) + acol.bogoliubovenergy(vjsq, gncmean)
        v12x = vi0 + vj0
        v12y = vi1 + vj1
        v12z = vi2 + vj2

        #Marsaglia method to assign output direction
        ox = oy = oz = 0.0
        for attempt in range(30):
            x1 = 2.0 * acol.randomizer(randarr) - 1.0
            x2 = 2.0 * acol.randomizer(randarr) - 1.0
            r2 = x1*x1 + x2*x2
            if r2 < 1.0:
                sq = math.sqrt(1.0 - r2)
                ox = 2.0 * x1 * sq
                oy = 2.0 * x2 * sq
                oz = 1.0 - 2.0 * r2
                break
 
        s = acol.collenergycons(E12, v12x, v12y, v12z, ox, oy, oz, gncmean)
        if s == 0.0:
            continue
 
        v3x = s * ox
        v3y = s * oy
        v3z = s * oz
        v4x = v12x - v3x
        v4y = v12y - v3y
        v4z = v12z - v3z
        v3sq = s * s
        v4sq = v4x*v4x + v4y*v4y + v4z*v4z

        #Energy check
        Echeck = acol.bogoliubovenergy(v3sq, gncmean) + acol.bogoliubovenergy(v4sq, gncmean)
        tol = cts.FTYPE(1.0e3) * cts.FEPS
        if E12 > cts.MINVAL:
            if abs(Echeck - E12) / E12 > tol:
                continue
        else:
            if abs(Echeck - E12) > tol:
                continue

        #Weights
        W1i = W1(visq, gncmean)
        W1j = W1(vjsq, gncmean)
        W3 = W1(v3sq, gncmean)
        modv4 = math.sqrt(v4sq) if v4sq > 0.0 else 0.0
        W4 = W1(v4sq, gncmean)
        denom = W1i * W1j * abs(W3 * s + W4 * modv4)
        w = v3sq / denom

        #Bose enhancement factors
        f3   = acol.ffunction(v3x, v3y, v3z, velstorer, gamma, dV)
        f4   = acol.ffunction(v4x, v4y, v4z, velstorer, gamma, dV)
        probfactor = granprefactor * w * (1.0 + f3) * (1.0 + f4)
 
        P = probfactor * dt
        if P > 1.0:
            P = 1.0

        if acol.randomizer(randarr) < P:
            velocities[i, 0] = v3x
            velocities[i, 1] = v3y
            velocities[i, 2] = v3z
            velocities[j, 0] = v4x
            velocities[j, 1] = v4y
            velocities[j, 2] = v4z

            val = probfactor * invdx3
            collisionrate[ixl, iyl, izl] += wxr * wyr * wzr * val
            collisionrate[ixr, iyl, izl] += wxl * wyr * wzr * val
            collisionrate[ixl, iyr, izl] += wxr * wyl * wzr * val
            collisionrate[ixl, iyl, izr] += wxr * wyr * wzl * val
            collisionrate[ixr, iyr, izl] += wxl * wyl * wzr * val
            collisionrate[ixr, iyl, izr] += wxl * wyr * wzl * val
            collisionrate[ixl, iyr, izr] += wxr * wyl * wzl * val
            collisionrate[ixr, iyr, izr] += wxl * wyl * wzl * val
 

 
        