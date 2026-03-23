import numpy as np
import math
from numba import njit, prange

@njit(cache=True)
def W1(vsq, gnc):
    denom =  math.sqrt(vsq * (vsq + 4.0 * gnc))
    return (vsq + 2.0 * gnc) / denom

@njit(cache=True)
def W2(vsq, gnc):
    denom = vsq + 4.0 * gnc
    return math.sqrt(vsq / denom)

@njit(cache=True, parallel=True, fastmath=True)
def weight1numba(modvelsq, densterm):
    N   = modvelsq.shape[0]
    out = np.empty(N, dtype=modvelsq.dtype)
    for i in prange(N):
        vsq    = modvelsq[i]
        gnc    = densterm[i]
        out[i] = (vsq + 2.0 * gnc) / math.sqrt(vsq * (vsq + 4.0 * gnc))
    return out
 
 
@njit(cache=True, parallel=True, fastmath=True)
def weight2numba(modvelsq, densterm):
    N   = modvelsq.shape[0]
    out = np.empty(N, dtype=modvelsq.dtype)
    for i in prange(N):
        vsq    = modvelsq[i]
        gnc    = densterm[i]
        out[i] = math.sqrt(vsq / (vsq + 4.0 * gnc))
    return out