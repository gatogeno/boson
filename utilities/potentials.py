import numpy as np
from scipy.ndimage import gaussian_filter
from utilities.additionalfunctions import constants as cts
#from scipy.fft import fftn, ifftn

class potential:
    def __init__(self, fundparameters, potparameters, grids, fieldY):
        #set constants
        self.geff=potparameters[1][1]

        #Initialize the harmonic potential
        self.harmpot = self.buildharmonic(potparameters, grids)
        
        #Initialize the potentials
        self.potentialc=self.buildpotentialc(fieldY)
        self.mfpotential=self.buildmfpotential(fieldY)
        self.potentialq=self.buildpotentialq(fieldY)

        #set a flag to be useful in the acceleration computation
        self.potversion = 0

    def buildharmonic(self,potparameters, grids):
        coord_indices = (2, 4, 6)[:3]
        coords = [grids[i] for i in coord_indices if len(grids) > i]
        if len(coords) < 3:
            coords += [np.zeros_like(coords[0])] * (3 - len(coords))
        
        omegas = np.array(potparameters[0][1:4], dtype=cts.FTYPE)
        shifts = np.array(potparameters[0][4:7], dtype=cts.FTYPE)
        vharm=0.5 * sum((omegas[i]**2) * (coords[i]-shifts[i])**2 for i in range(3))
        return vharm.astype(cts.FTYPE)

    def contactc(self, fieldY):
        return (self.geff*fieldY.densityc).astype(cts.FTYPE)

    def contactq(self, fieldY):
        return (self.geff*fieldY.densityq).astype(cts.FTYPE)

    def contactdep(self, fieldY):
        return (self.geff*fieldY.densitydep).astype(cts.FTYPE)

    def LHY(self, fieldY):
        coeff = 4.0 / (3.0 * np.pi**2)
        return (coeff*(self.geff**2.5)*fieldY.densityc**1.5).astype(cts.FTYPE)

    def thermpot(self,fieldY):
        return (self.geff*fieldY.thermalcontribution).astype(cts.FTYPE)

    def buildpotentialc(self, fieldY):
        if fieldY.getcohfrac() == 0:
            return np.zeros_like(fieldY.densityc, dtype=cts.FTYPE)
        g = self.geff
        nc, nq, ndep = fieldY.densityc, fieldY.densityq, fieldY.densitydep
        lhy = cts.FTYPE(4.0/(3.0*np.pi**2)) * (g**2.5) * nc**1.5
        return (self.harmpot + g*nc + g*nq + g*ndep + lhy + g*fieldY.thermalcontribution).astype(cts.FTYPE)

    def buildmfpotential(self, fieldY):
        g = self.geff
        nc, nq, ndep = fieldY.densityc, fieldY.densityq, fieldY.densitydep
        return (self.harmpot + g*nc + 2*g*nq + 2*g*ndep).astype(cts.FTYPE)

    def buildpotentialq(self, fieldY):
        if fieldY.getcohfrac() == 1:
            return np.zeros_like(fieldY.densityc, dtype=cts.FTYPE)
        return (self.geff * fieldY.densityc).astype(cts.FTYPE)

    def updatepotentials(self, fieldY, grids):
        self.potversion += 1
        g   = self.geff
        Vh  = self.harmpot
        nc  = fieldY.densityc
        nq  = fieldY.densityq
        nth = fieldY.thermalcontribution
        nc15 = nc ** 1.5

        if fieldY._cache_densitydep is None: 
            depl = (g**1.5 * nc15) / (3.0 * np.pi**2)
            if fieldY.smooth is not None:
                depl = gaussian_filter(depl, sigma=fieldY.smooth)
            fieldY._cache_densitydep = depl.astype(cts.FTYPE)
        ndep = fieldY._cache_densitydep
        lhy   = cts.FTYPE(4.0 / (3.0 * np.pi**2)) * (g**2.5) * (nc15)

        g_nc  = g * nc
        g_nq  = g * nq
        g_ndep  = g * ndep
        g_nth = g * nth

        if fieldY.getcohfrac() > 0:
            self.potentialc = (Vh + g_nc + g_nq + g_ndep + lhy + g_nth).astype(cts.FTYPE)
        else:
            self.potentialc = np.zeros_like(nc, dtype=cts.FTYPE)

        self.mfpotential = (Vh + g_nc + 2.0*g_nq + 2.0*g_ndep).astype(cts.FTYPE)

        if fieldY.getcohfrac() < 1:
            self.potentialq = g_nc.astype(cts.FTYPE)
        else:
            self.potentialq = np.zeros_like(nc, dtype=cts.FTYPE)
            
    def addextracontribution(self,cont):
        self.potentialc+=cont
        return None
        
