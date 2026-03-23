import numpy as np
import math
from numba import njit, prange
from numba.typed import List
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions import auxiliarycollisions as acol

def constructadaptivecells(fieldY, randarr, Nth=200):
    #Define the number of master cells
    Nm = max(1, len(fieldY.getunigrid()) // 4)

    positions = fieldY.fieldnc[0]
    dx   = fieldY.getdx()
    xmin   = np.amin(fieldY.getunigrid())
    xmax   = np.amax(fieldY.getunigrid()) + dx

    subcells, subcellcoord, subcelllength = buildcollisioncells(positions, xmin, xmax, Nm, Nth, randarr)
    return subcells, subcellcoord, subcelllength

@njit(cache=True, parallel=True)
def cellidsparallel(positions, indices, ox, oy, oz, nx, ny, nz, dx, dy, dz):
    N      = indices.shape[0]
    cellid = np.empty(N, dtype=cts.ITYPE)
    for k in prange(N):
        i = indices[k]
        lx = cts.ITYPE((positions[i, 0] - ox) / dx)
        ly = cts.ITYPE((positions[i, 1] - oy) / dy)
        lz = cts.ITYPE((positions[i, 2] - oz) / dz)
        lx = min(max(lx, cts.ITYPE(0)), nx - cts.ITYPE(1))
        ly = min(max(ly, cts.ITYPE(0)), ny - cts.ITYPE(1))
        lz = min(max(lz, cts.ITYPE(0)), nz - cts.ITYPE(1))
        cellid[k] = lx * ny * nz + ly * nz + lz
    return cellid
 
 
@njit(cache=True)
def cellidsserie(positions, indices, ox, oy, oz, nx, ny, nz, dx, dy, dz):
    N      = indices.shape[0]
    cellid = np.empty(N, dtype=cts.ITYPE)
    for k in range(N):
        i = indices[k]
        lx = cts.ITYPE((positions[i, 0] - ox) / dx)
        ly = cts.ITYPE((positions[i, 1] - oy) / dy)
        lz = cts.ITYPE((positions[i, 2] - oz) / dz)
        lx = min(max(lx, cts.ITYPE(0)), nx - cts.ITYPE(1))
        ly = min(max(ly, cts.ITYPE(0)), ny - cts.ITYPE(1))
        lz = min(max(lz, cts.ITYPE(0)), nz - cts.ITYPE(1))
        cellid[k] = lx * ny * nz + ly * nz + lz
    return cellid
 
 
@njit(cache=True)
def computecellids(positions, indices, ox, oy, oz, nx, ny, nz, dx, dy, dz):
    if indices.shape[0] >= cts.ITYPE(5000):
        return cellidsparallel(positions, indices, ox, oy, oz, nx, ny, nz, dx, dy, dz)
    return cellidsserie(positions, indices, ox, oy, oz, nx, ny, nz, dx, dy, dz)


@njit(cache=True)
def buildcollisioncells(positions, xmin, xmax, Nm, Nth, randarr):
    Ntp      = positions.shape[0]
    L        = xmax - xmin
    masterdx = L / cts.FTYPE(Nm)
    allindices = np.arange(Ntp, dtype=cts.ITYPE)
 
    # create the array with the information of number of master cell for each particle and initialize the array for the particles in the same cell
    cellid = computecellids(positions, allindices, xmin, xmin, xmin, Nm, Nm, Nm, masterdx, masterdx, masterdx)
    arrayparttogether = np.empty(Ntp, dtype=cts.ITYPE)
 
    Nm3 = Nm * Nm * Nm
    segmenter = np.zeros(Nm3 + 1, dtype=cts.ITYPE)
    for i in range(Ntp):
        segmenter[cellid[i] + 1] += 1
    for j in range(Nm3):
        segmenter[j + 1] += segmenter[j]
  
    occupationcounter = np.zeros(Nm3, dtype=cts.ITYPE)
    for i in range(Ntp):
        c = cellid[i]
        arrayparttogether[segmenter[c] + occupationcounter[c]] = i
        occupationcounter[c] += cts.ITYPE(1)
 
    subcells      = List()
    subcellcoord  = List()
    subcelllength = List()
 
    for indmc in range(Nm3):
        Nmc = segmenter[indmc + 1] - segmenter[indmc]
        if Nmc == 0:
            continue
 
        mz = indmc % Nm
        my = (indmc // Nm) % Nm
        mx = indmc // (Nm * Nm)
        cx = xmin + mx * masterdx
        cy = xmin + my * masterdx
        cz = xmin + mz * masterdx
 
        if Nmc <= Nth:
            l = cts.ITYPE(0)
        else:
            l = cts.ITYPE(math.floor(math.log2(cts.FTYPE(Nmc) / cts.FTYPE(Nth))))
 
        sx = cts.ITYPE(1) << cts.ITYPE((l + 2) // 3)
        sy = cts.ITYPE(1) << cts.ITYPE((l + 1) // 3)
        sz = cts.ITYPE(1) << cts.ITYPE(l       // 3)
        subdx   = masterdx / cts.FTYPE(sx)
        subdy   = masterdx / cts.FTYPE(sy)
        subdz   = masterdx / cts.FTYPE(sz)
        subside = min(subdx, min(subdy, subdz))
 
        subcellindices = arrayparttogether[segmenter[indmc]:segmenter[indmc] + Nmc]
        subcellid = computecellids(positions, subcellindices, cx, cy, cz, sx, sy, sz, subdx, subdy, subdz)
 
        numbersubcells = sx * sy * sz
        subsegmenter = np.zeros(numbersubcells + 1, dtype=cts.ITYPE)
        for k in range(Nmc):
            subsegmenter[subcellid[k] + 1] += 1
        for s in range(numbersubcells):
            subsegmenter[s + 1] += subsegmenter[s]
 
        suboccupationcounter = np.zeros(numbersubcells, dtype=cts.ITYPE)
        miniarrayparts = np.empty(Nmc, dtype=cts.ITYPE)
        for k in range(Nmc):
            sid = subcellid[k]
            miniarrayparts[subsegmenter[sid] + suboccupationcounter[sid]] = subcellindices[k]
            suboccupationcounter[sid] += cts.ITYPE(1)
 
        for s in range(numbersubcells):
            Ns = subsegmenter[s + 1] - subsegmenter[s]
            if Ns == 0:
                continue
            subcellarr  = miniarrayparts[subsegmenter[s]:subsegmenter[s] + Ns].copy()
            szcoord = s % sz
            sycoord = (s // sz) % sy
            sxcoord = s // (sy * sz)
            subcellcorner = np.array([cx + sxcoord * subdx, cy + sycoord * subdy, cz + szcoord * subdz], dtype=positions.dtype)

            #Shuffle subcellarr before TASC subdivision for random pairing
            for i in range(Ns - 1, 0, -1):
                j = cts.ITYPE(acol.randomizer(randarr) * cts.FTYPE(i + 1))
                tmp = subcellarr[i]
                subcellarr[i] = subcellarr[j]
                subcellarr[j] = tmp
            
            #TASC subdivision within each cell
            nt      = max(cts.ITYPE(1), cts.ITYPE(math.floor((Ns / 3.0) ** (1.0 / 3.0))))
            tascdx  = subside / cts.FTYPE(nt)
            tascid = cellidsserie(positions, subcellarr, subcellcorner[0], subcellcorner[1], subcellcorner[2], nt, nt, nt, tascdx, tascdx, tascdx)
 
            counteroftasc = cts.ITYPE(1)
            for i in range(Ns):
                if tascid[i] + cts.ITYPE(1) > counteroftasc:
                    counteroftasc = tascid[i] + cts.ITYPE(1)
 
            tasccounts  = np.zeros(counteroftasc, dtype=cts.ITYPE)
            for i in range(Ns):
                tasccounts[tascid[i]] += cts.ITYPE(1)
 
            tascsegmenter = np.zeros(counteroftasc + cts.ITYPE(1), dtype=cts.ITYPE)
            for i in range(counteroftasc):
                tascsegmenter[i + cts.ITYPE(1)] = tascsegmenter[i] + tasccounts[i]
 
            tascoccupationcounter = np.zeros(counteroftasc, dtype=cts.ITYPE)
            tascarray             = np.empty(Ns, dtype=cts.ITYPE)
            for i in range(Ns):
                tid = tascid[i]
                tascarray[tascsegmenter[tid] + tascoccupationcounter[tid]] = subcellarr[i]
                tascoccupationcounter[tid] += cts.ITYPE(1)
 
            for t in range(counteroftasc):
                Nt = tascsegmenter[t + cts.ITYPE(1)] - tascsegmenter[t]
                if Nt < cts.ITYPE(2):
                    continue
                tascarr = tascarray[tascsegmenter[t]:tascsegmenter[t] + Nt].copy()
                
                subcells.append(tascarr)
                subcellcoord.append(subcellcorner)
                subcelllength.append(subside)
 
    return subcells, subcellcoord, subcelllength
