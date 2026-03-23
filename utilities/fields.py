import numpy as np
from utilities.initialconditionsc import *
from utilities.initialconditionsnc import *
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions import interpolations as interp
from utilities.additionalfunctions.densitync import computedensity
from utilities.additionalfunctions import weights as wg
from scipy.ndimage import gaussian_filter

class boson:
    def __init__(self, fundparameters, fieldparameters, potparameters, optparameters, grids):
        #Store quantities to be used in the implementation
        self.dx=cts.FTYPE(fundparameters[0]/fundparameters[1])
        self.totalnumber=fundparameters[2]
        self.cohfrac=fundparameters[3]
        self.tpnumber=fieldparameters[2][0]
        self.smooth=optparameters[0][1] if optparameters[0][0] else None
        self.unigrid=grids[6][0][0]
        self.g=potparameters[1][1]

        # Cache
        self._cache_densityc   = None
        self._cache_densitydep  = None
        self._cache_interp_dens = None
        self._cache_densityq   = None
        self._cache_thermal    = None
        
        #Initialize the coherent part
        grid_shape=(fundparameters[1],)*3
        if fieldparameters[0][0]=='None':
            self.fieldc=np.zeros(grid_shape, dtype=cts.CTYPE)
        elif fieldparameters[0][0]=='testing':
            self.fieldc=(np.ones(grid_shape, dtype=cts.CTYPE)*np.exp(-1.0j*np.random.random(grid_shape))).astype(cts.CTYPE)
        elif fieldparameters[0][0]=='random':
            self.fieldc=randomc(grid_shape)
        elif fieldparameters[0][0]=='centralsoliton':
            self.fieldc=centralsoliton(fundparameters,fieldparameters,grids,grid_shape)
        elif fieldparameters[0][0]=='gaussianpackets':
            self.fieldc=gaussianpackets(grids)       
        elif fieldparameters[0][0]=='from-file':
            self.fieldc=np.load(fieldparameters[0][1]) 
  
        #defining the attributes real and imaginary part for the coherent part
        self.re=self.fieldc.real
        self.imag=self.fieldc.imag
        
        #normalizing the coherent part
        self.normalizer()

        #Initialize the incoherent part
        self.fieldnc=np.zeros((2,fieldparameters[2][0],3), dtype=cts.FTYPE) #0: position, 1: velocity
        #Assign the values for incoherent part according the initial conditions
        if fieldparameters[1][0]!='None':
            if fieldparameters[1][0]=='testing':
                pass             
            elif fieldparameters[1][0]=='random':
                self.fieldnc=randomnc(fieldparameters,self.fieldnc,self.unigrid)    
            elif fieldparameters[1][0]=='normal':
                self.fieldnc=normal(fieldparameters,self.fieldnc)
            elif fieldparameters[1][0]=='thermaltest':
                self.fieldnc=thermaltest(fieldparameters,self.fieldnc,self.unigrid)
            elif fieldparameters[1][0]=='thermalharmonic':
                self.fieldnc=thermalharmonic(fieldparameters,potparameters,self.fieldnc)
            elif fieldparameters[1][0]=='nbodytest':
                self.fieldnc=nbodytest(fieldparameters,grids,self.fieldc,self.fieldnc,self.dx)
            elif fieldparameters[1][0]=='nfw':
                self.fieldnc=nfw(fundparamters,fieldparameters,grids,self.fieldnc)
            elif fieldparameters[1][0]=='from-file':
                self.fieldnc=np.load(fieldparameters[1][1])

        #Array to be a copy of self.fieldnc[0] to be used in the evolver for predictor-corrector scheme
        self.x0buffer = np.zeros_like(self.fieldnc[0], dtype=cts.FTYPE)

    #Invalidating cache
    def invalidate_fieldc_cache(self):
        self._cache_densityc    = None
        self._cache_densitydep  = None
        self._cache_interp_dens = None
        self._cache_densityq    = None
        self._cache_thermal     = None
    def _invalidate_phase_only(self):
        pass
    def invalidate_fieldnc_cache(self):
        self._cache_interp_dens = None
        self._cache_densityq    = None
        self._cache_thermal     = None
    def invalidate_positions_only(self):
        self._cache_interp_dens = None
    def invalidate_for_potential(self):
        self._cache_densityq    = None
        self._cache_thermal     = None
     
    #Function that normalize the coherent field
    def normalizer(self):
        Ntarget = self.cohfrac * self.totalnumber
        diff=Ntarget
        if Ntarget!=0:
            while abs(diff) / Ntarget > 1e-6:
                Nc0   = np.sum(self.densityc)   * self.dx**3
                Ndep0 = np.sum(self.densitydep) * self.dx**3
          
                diff = Ntarget - (Nc0 + Ndep0)
                denom = 2.0 * Nc0 + 3.0 * Ndep0
    
                if denom != 0:
                    delta = diff / denom
                    self.fieldc *= (1.0 + delta)
                    self.invalidate_fieldc_cache()
   
    #Functions to get the stored elements for the field
    def getdx(self):
        return self.dx

    def getunigrid(self):
        return self.unigrid
    
    def gettotalnumber(self):
        return self.totalnumber
    
    def getcohfrac(self):
        return self.cohfrac

    def gettpnumber(self):
        return self.tpnumber
        
    #Density and phase of the coherent part
    @property
    def densityc(self): 
        if self._cache_densityc is None:
            dens = np.abs(self.fieldc)**2
            if self.smooth is not None:
                dens = gaussian_filter(dens, sigma=self.smooth)
            self._cache_densityc = dens.astype(cts.FTYPE)
        return self._cache_densityc

    @property
    def phase(self):
        return np.angle(self.fieldc)

    #Density of the depletion part
    @property
    def densitydep(self):
        if self._cache_densitydep is None:
            dep = ((self.g**1.5) * (self.densityc**1.5)) / (3*np.pi**2)
            if self.smooth is not None:
                dep = gaussian_filter(dep, sigma=self.smooth)
            self._cache_densitydep = dep.astype(cts.FTYPE)
        return self._cache_densitydep

    #Interpolated coherent density
    def getinterpdens(self):
        if self._cache_interp_dens is None:
            self._cache_interp_dens = interp.interpolatordens(self)
        return self._cache_interp_dens

    #Weights
    def weight1(self):
        densterm = (self.g * self.getinterpdens()).astype(cts.FTYPE)
        modvelsq = np.sum(self.fieldnc[1] ** 2, axis=1).astype(cts.FTYPE)
        return wg.weight1numba(modvelsq, densterm)
 
    def weight2(self):
        densterm = (self.g * self.getinterpdens()).astype(cts.FTYPE)
        modvelsq = np.sum(self.fieldnc[1] ** 2, axis=1).astype(cts.FTYPE)
        return wg.weight2numba(modvelsq, densterm)

    #Density of the incoherent part
    def computenqandthermal(self):
        Ngrid = len(self.unigrid)
        Nq    = (1 - self.getcohfrac()) * self.gettotalnumber()
        dx    = self.dx
        densterm = self.g * self.getinterpdens()
        modvelsq = np.sum(self.fieldnc[1]**2, axis=1)

        w1raw = wg.weight1numba(modvelsq, densterm)
        w2raw = wg.weight2numba(modvelsq, densterm)
 
        w1sum = cts.FTYPE(np.sum(w1raw))
        w1     = (w1raw / w1sum).astype(cts.FTYPE)
        w2     = (w2raw / w1sum).astype(cts.FTYPE)

        #computation of the two objects
        d1, d2 = computedensity(self.fieldnc[0], w1, w2, Ngrid, dx, np.amin(self.unigrid))
        
        # Normalize the objects
        s1, s2 = cts.FTYPE(np.sum(d1)), cts.FTYPE(np.sum(d2))
        if s1 > 0:
            d1 *= Nq / (s1 * dx**3)
        if s2 > 0:
            d2 *= Nq / (s2 * dx**3)
        if self.smooth is not None:
            d1 = gaussian_filter(d1, sigma=self.smooth)
            d2 = gaussian_filter(d2, sigma=self.smooth)
        self._cache_densityq = d1.astype(cts.FTYPE)
        self._cache_thermal  = d2.astype(cts.FTYPE)

    #Quasiparticle density
    @property
    def densityq(self):
        if self._cache_densityq is None:
            if self.getcohfrac() == 1:
                self._cache_densityq=np.zeros((len(self.unigrid),) * 3, dtype=cts.FTYPE)
            else:
                self.computenqandthermal()
        return self._cache_densityq

    #Thermal contribution
    @property
    def thermalcontribution(self):
        if self._cache_thermal is None:
            if self.getcohfrac() == 1:
                self._cache_thermal=np.zeros((len(self.unigrid),) * 3, dtype=cts.FTYPE)
            else:
                self.computenqandthermal()
        return self._cache_thermal


    #Function to change the array of positions and velocities
    def replacefieldnc(self,newpositions,newvelocities):
        assert newpositions.shape==newvelocities.shape
        self.fieldnc=np.stack([newpositions,newvelocities],axis=0)
        return None

    def updatecohfrac(self):
        Nc  = cts.FTYPE(np.sum(self.densityc)   * self.dx**3)
        Ndep = cts.FTYPE(np.sum(self.densitydep) * self.dx**3)
        self.cohfrac = (Nc + Ndep) / self.totalnumber
        return self.cohfrac