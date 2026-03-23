import numpy as np
import math
from numba import njit
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions.weights import W1, W2

@njit(cache=True)
def randomizer(rs):
    x = rs[0]
    x ^= x << cts.UITYPE(13)
    x ^= x >> cts.UITYPE(7)
    x ^= x << cts.UITYPE(17)
    rs[0] = x
    return cts.FTYPE(x) / cts.FTYPE(cts.UIMAX)

@njit(cache=True)
def bogoliubovenergy(vsq, gnc):
    arg = vsq * (vsq + 4.0 * gnc)
    return math.sqrt(arg)

@njit(cache=True)
def ffunction(vx, vy, vz, velstorer, gamma, dV, Nv=8):
    vmax = 0.0
    for k in range(velstorer.shape[0]):
        for d in range(3):
            av = abs(velstorer[k, d])
            if av > vmax:
                vmax = av
    av = abs(vx) if abs(vx) > abs(vy) else abs(vy)
    av = av if av > abs(vz) else abs(vz)
    if av > vmax:
        vmax = av
 
    dv = 2.0 * vmax / cts.FTYPE(Nv)
    dVphase = dV * dv * dv * dv
    invdv   = 1.0 / dv
 
    bx = min(max(cts.ITYPE((vx + vmax) * invdv), 0), Nv - 1)
    by = min(max(cts.ITYPE((vy + vmax) * invdv), 0), Nv - 1)
    bz = min(max(cts.ITYPE((vz + vmax) * invdv), 0), Nv - 1)
 
    count = 0
    for k in range(velstorer.shape[0]):
        kx = min(max(cts.ITYPE((velstorer[k, 0] + vmax) * invdv), 0), Nv - 1)
        ky = min(max(cts.ITYPE((velstorer[k, 1] + vmax) * invdv), 0), Nv - 1)
        kz = min(max(cts.ITYPE((velstorer[k, 2] + vmax) * invdv), 0), Nv - 1)
        if kx == bx and ky == by and kz == bz:
            count += 1
 
    return cts.FTYPE(count) * gamma / dVphase


@njit(cache=True)
def collenergycons(E12, v12x, v12y, v12z, ox, oy, oz, gnc):
    maxiterations=20
    tol=1.0e-8
    v12sq  = v12x*v12x + v12y*v12y + v12z*v12z
    v12dot = v12x*ox + v12y*oy + v12z*oz

    def solvereq(s):
        s2   = s * s
        v4sq = v12sq - 2.0 * s * v12dot + s2
        if v4sq < 0.0:
            v4sq = 0.0
        arg3 = s2 * (s2 + 4.0 * gnc)
        arg4 = v4sq * (v4sq + 4.0 * gnc)
        if arg3 <= 0.0 or arg4 <= 0.0:
            return -E12, 0.0
        ene3 = math.sqrt(arg3)
        ene4 = math.sqrt(arg4)
        F = ene3 + ene4 - E12
        dene3ds = ((2.0*s2   + 4.0 * gnc) / ene3) * s
        dv4sq = 2.0 * s - 2.0 * v12dot
        dene4ds = ((v4sq + 2.0 * gnc) / ene4)  * dv4sq
        return F, dene3ds + dene4ds

    #We analyse a possible classical solution before bisection or Newton-Raphson
    arg0 = v12sq * (v12sq + 4.0 * gnc)
    F0   = (math.sqrt(arg0) if arg0 > 0.0 else 0.0) - E12
    if F0 >= 0.0:
        disc = v12dot*v12dot - 2.0*(v12sq - E12)
        if disc >= 0.0:
            s = 0.5 * (v12dot + math.sqrt(disc))
            return s

    #If there is no classical solution, it starts with bisection method
    shigh = math.sqrt(E12) + math.sqrt(v12sq) + 1.0
    for _ in range(15):
        Fhigh, _ = solvereq(shigh)
        if Fhigh > 0.0:
            break
        shigh *= 2.0

    slow  = 0.0
    smid = 0.5 * (slow + shigh)
    for _ in range(15):
        Fmid, _ = solvereq(smid)
        if Fmid < 0.0:
            slow = smid
        else:
            shigh = smid
        smid = 0.5 * (slow + shigh)
        if shigh - slow < tol * (1.0 + smid):
            return max(smid, 0.0)
    s = smid

    #Newton-Raphson to reach the desired tolerance if has not been reached
    for _ in range(maxiterations):
        Fs, dFs = solvereq(s)
        if abs(dFs) < 1.0e-30:
            break
        snew = s - Fs / dFs
        if snew < slow:
            snew = 0.5 * (slow + s)
        if snew > shigh:
            snew = 0.5 * (s + shigh)
        if abs(snew - s) < tol * (1.0 + s):
            return max(snew, 0.0)
        s = snew

    return max(s, 0.0)