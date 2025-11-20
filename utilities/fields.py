import numpy as np
from utilities import additionalfunctions as adf
from itertools import product
from scipy.ndimage import gaussian_filter

class boson:
    def __init__(self, parameters, grids):
        #Store quantities to be used in the implementation
        self.__dim=parameters[0]
        self.__dx=grids[0]
        self.__dv=2*parameters[6][2]/32
        self.__totalnumber=parameters[3]
        self.__cohfrac=parameters[4]
        self.__tpnumber=parameters[6][0]
        self.densassign=parameters[6][1]
        if parameters[9][0]==True:
            self.__smooth=parameters[9][1]
        else:
            self.__smooth=None
        
        if parameters[0]==1:
            self.__unigrid=grids[2]
        elif parameters[0]==2:
            self.__unigrid=grids[4][0]
        else:
            self.__unigrid=grids[6][0][0]
        
        #Initialize the coherent part
        grid_shape=(parameters[2],)*self.__dim
        if parameters[5][0][0]=='None':
            self.fieldc=np.zeros(grid_shape, dtype=np.complex64)
        elif parameters[5][0][0]=='testing':
            if self.__dim==1:
                wr2=parameters[7][0][0]*grids[2]**2
            if self.__dim==2:
                wr2=parameters[7][0][0]*grids[2]**2+parameters[7][0][1]*grids[4]**2
            else:
                wr2=parameters[7][0][0]*grids[2]**2+parameters[7][0][1]*grids[4]**2+parameters[7][0][2]*grids[6]**2
            self.fieldc=np.exp(-0.5*wr2).astype(np.complex64)*np.exp(-1.0j*np.random.random(grid_shape).astype(np.float32))
        elif parameters[5][0][0]=='soliton1D':
            if self.__dim==1:
                self.fieldc=0.5*np.abs(parameters[7][2][0])*(1/np.cosh(0.5*np.abs(parameters[7][2][0])*grids[2])).astype(np.complex64)*np.exp(1.0j*parameters[5][0][1]*grids[2]).astype(np.complex64)
            else:
                self.fieldc=np.zeros(grid_shape, dtype=np.complex64)
        elif parameters[5][0][0]=='gaussianpackets':
            alpha=0.25
            x0=np.zeros(self.__dim)
            k0=np.zeros(self.__dim)
            x0[:] = [2.5, 2.5, 0.0][:self.__dim]
            k0[:] = [0.1, 0.1, 0.0][:self.__dim]
            if self.__dim==1:
                r2=(grids[2]+x0[0])**2
                r2b=(grids[2]-x0[0])**2
                kx=k0[0]*grids[2]
            elif self.__dim==2:
                r2=(grids[2]+x0[0])**2+(grids[4]+x0[1])**2
                r2b=(grids[2]-x0[0])**2+(grids[4]-x0[1])**2
                kx=k0[0]*grids[2]+k0[1]*grids[4]
            else:
                r2=(grids[2]+x0[0])**2+(grids[4]+x0[1])**2+(grids[6]+x0[2])**2
                r2b=(grids[2]-x0[0])**2+(grids[4]-x0[1])**2+(grids[6]-x0[2])**2
                kx=k0[0]**grids[2]+k0[1]*grids[4]+k0[2]*grids[6]
            self.fieldc=np.exp(-0.5*r2/alpha-1.0j*kx).astype(np.complex64)+np.exp(-0.5*r2b/alpha+1.0j*kx).astype(np.complex64)       
        elif parameters[5][0][0]=='random':
            self.fieldc=(np.random.random(grid_shape)+1.0j*np.random.random(grid_shape)).astype(np.complex64)
        elif parameters[5][0][0]=='centralsoliton':
            if self.__dim==1:
                r2=grids[2]**2
            elif self.__dim==2:
                r2=grids[2]**2+grids[4]**2
            else:
                r2=grids[2]**2+grids[4]**2+grids[6]**2
            self.fieldc=(np.sqrt(((self.__totalnumber*1024*0.091**1.5)/(33*np.pi**2*parameters[5][0][1]**3))/pow(1+0.091*r2/(parameters[5][0][1]**2),8))*np.exp(-1.0j*np.random.random(grid_shape))).astype(np.complex64)
        elif parameters[5][0][0]=='from-file':
            self.fieldc=np.load(parameters[5][0][1]) 
  
        #defining the attributes real and imaginary part for the coherent part
        self.re=self.fieldc.real
        self.imag=self.fieldc.imag
        
        #normalizing the coherent part
        if self.__cohfrac>0:
            integnorm=np.sum(np.abs(self.fieldc)**2)*(self.__dx**self.__dim)
            self.fieldc*=np.sqrt(self.__cohfrac*self.__totalnumber/integnorm)

        
        #Initialize the incoherent part
        self.fieldnc=np.zeros((2,parameters[6][0],self.__dim), dtype=np.float32) #0: position, 1: velocity
        #Assign the values for incoherent part according the initial conditions
        if parameters[5][1][0]!='None':
            if parameters[5][1][0]=='testing':
                pass
                
            elif parameters[5][1][0]=='random':
                self.fieldnc[0]=np.random.uniform(np.amin(self.__unigrid),np.amax(self.__unigrid),size=(self.__tpnumber,self.__dim)).astype(np.float32)
                module=np.random.uniform(0,parameters[6][2],size=(self.__tpnumber,)).astype(np.float32)
                if self.__dim==1:
                    self.fieldnc[1][:,0]=module*(2*np.random.randint(0,2,size=self.__tpnumber)-1).astype(np.float32)
                elif self.__dim==2:
                    angleph=np.random.uniform(0,2*np.pi,self.__tpnumber).astype(np.float32)
                    self.fieldnc[1][:,0]=module*np.cos(angleph)
                    self.fieldnc[1][:,1]=module*np.sin(angleph)
                else:
                    angleth=np.random.uniform(0,np.pi,self.__tpnumber).astype(np.float32)
                    angleph=np.random.uniform(0,2*np.pi,self.__tpnumber).astype(np.float32)
                    self.fieldnc[1][:,0]=module*np.sin(angleth)*np.cos(angleph)
                    self.fieldnc[1][:,1]=module*np.sin(angleth)*np.sin(angleph)
                    self.fieldnc[1][:,2]=module*np.cos(angleth)
                    
            elif parameters[5][1][0]=='normal':
                self.fieldnc[0]=np.random.normal(parameters[5][1][1],parameters[5][1][2],size=(self.__tpnumber,self.__dim)).astype(np.float32)
                self.fieldnc[1]=np.random.normal(parameters[5][1][3],parameters[5][1][4],size=(self.__tpnumber,self.__dim)).astype(np.float32)

            elif parameters[5][1][0]=='thermaltest':
                self.fieldnc[0]=np.random.uniform(np.amin(self.__unigrid)/32.0,np.amax(self.__unigrid)/32.0,size=(self.__tpnumber,self.__dim)).astype(np.float32)
                module=np.random.uniform(0,parameters[6][2],size=(self.__tpnumber,)).astype(np.float32)
                if self.__dim==1:
                    self.fieldnc[1][:,0]=module*(2*np.random.randint(0,2,size=self.__tpnumber)-1).astype(np.float32)
                elif self.__dim==2:
                    angleph=np.random.uniform(0,2*np.pi,self.__tpnumber).astype(np.float32)
                    self.fieldnc[1][:,0]=module*np.cos(angleph)
                    self.fieldnc[1][:,1]=module*np.sin(angleph)
                else:
                    angleth=np.random.uniform(0,np.pi,self.__tpnumber).astype(np.float32)
                    angleph=np.random.uniform(0,2*np.pi,self.__tpnumber).astype(np.float32)
                    self.fieldnc[1][:,0]=module*np.sin(angleth)*np.cos(angleph)
                    self.fieldnc[1][:,1]=module*np.sin(angleth)*np.sin(angleph)
                    self.fieldnc[1][:,2]=module*np.cos(angleth)

            elif parameters[5][1][0]=='thermalharmonic':
                self.fieldnc[0]=np.random.normal(0,1/(np.sqrt(parameters[5][1][1])*parameters[7][0][0]),size=(self.__tpnumber,self.__dim)).astype(np.float32)
                self.fieldnc[1]=np.random.normal(0,1/np.sqrt(parameters[5][1][1]),size=(self.__tpnumber,self.__dim)).astype(np.float32)

            elif parameters[5][1][0]=='zng':
                if self.__dim == 1:
                    x = grids[2]
                    V_eff = 0.5 * parameters[7][0][0]**2 * x**2 + 2.0 * parameters[7][2][0] * np.abs(self.fieldc)**2
                    cell_centers = x.reshape(-1, 1)
                elif self.__dim == 2:
                    xx, yy = grids[2], grids[4]
                    V_eff = 0.5 * (parameters[7][0][0]**2 * xx**2 + parameters[7][0][1]**2 * yy**2) + 2.0 * parameters[7][2][0] * np.abs(self.fieldc)**2
                    cell_centers = np.stack([xx.flatten(), yy.flatten()], axis=-1)
                elif self.__dim == 3:
                    xx, yy, zz = grids[2], grids[4], grids[6]
                    V_eff = 0.5 * (parameters[7][0][0]**2 * xx**2 + parameters[7][0][1]**2 * yy**2 + parameters[7][0][2]**2 * zz**2) + 2.0 * parameters[7][2][0] * np.abs(self.fieldc)**2
                    cell_centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
                    
                V_flat=V_eff.flatten()
                z_local = np.exp(parameters[5][1][1] * (parameters[5][1][2] - V_flat))
                z_local = np.clip(z_local, 0, 0.999999)

                ksuma = np.arange(1, 30 + 1)
                n_cell = np.sum(z_local[:, None] ** ksuma[None, :] / ksuma[None, :]**(self.__dim/2.0), axis=1)
                P_cell = n_cell / np.sum(n_cell)
                cdf = np.cumsum(P_cell)
                rand_vals = np.random.rand(self.__tpnumber)
                cell_indices = np.searchsorted(cdf, rand_vals)
                cell_coords = cell_centers[cell_indices]

                offsets = np.random.rand(self.__tpnumber, self.__dim) * self.__dx
                self.fieldnc[0][:self.__tpnumber, :self.__dim] = cell_coords + offsets

                N_target = self.__tpnumber
                V_chosen = V_flat[cell_indices]
                
                factor_pool = 5
                N_pool = max(N_target * factor_pool, 1000)
                v_mag = np.random.uniform(0, parameters[6][2], N_pool)
                cell_idx = np.random.randint(0, N_target, N_pool)
                f_cand = 1.0 / (np.exp(parameters[5][1][1] * (0.5 * v_mag**2 + V_chosen[cell_idx] - parameters[5][1][2])) - 1.0)
                f_max = 1.0 / (np.exp(parameters[5][1][1] * (V_chosen[cell_idx] - parameters[5][1][2])) - 1.0)
                r = np.random.rand(N_pool)
                mask = r < f_cand / f_max
                v_accepted = v_mag[mask]
                cell_accepted = cell_idx[mask]
                n_accepted = len(v_accepted)

                while n_accepted < N_target:
                    N_missing = N_target - n_accepted
                    v_mag2 = np.random.uniform(0, parameters[6][2], N_missing*2)
                    cell_idx2 = np.random.randint(0, N_target, N_missing*2)
                    f_cand2 = 1.0 / (np.exp(parameters[5][1][1] * (0.5 * v_mag2**2 + V_chosen[cell_idx2] - parameters[5][1][2])) - 1.0)
                    f_max2 = 1.0 / (np.exp(parameters[5][1][1] * (V_chosen[cell_idx2] - parameters[5][1][2])) - 1.0)
                    r2 = np.random.rand(len(v_mag2))
                    mask2 = r2 < f_cand2 / f_max2
                    v_accepted2 = v_mag2[mask2]
                    cell_accepted2 = cell_idx2[mask2]

                    v_accepted = np.concatenate([v_accepted, v_accepted2])
                    cell_accepted = np.concatenate([cell_accepted, cell_accepted2])
                    n_accepted = len(v_accepted)

                v_accepted = v_accepted[:N_target]
                if self.__dim == 1:
                    v_vec = v_accepted[:, None] * np.sign(np.random.rand(N_target, 1) - 0.5)
                elif self.__dim == 2:
                    phi = np.random.uniform(0, 2*np.pi, N_target)
                    v_vec = np.stack([v_accepted * np.cos(phi), v_accepted * np.sin(phi)], axis=-1)
                else:
                    phi = np.random.uniform(0, 2*np.pi, N_target)
                    cos_theta = np.random.uniform(-1, 1, N_target)
                    sin_theta = np.sqrt(1 - cos_theta**2)
                    v_vec = np.stack([v_accepted * sin_theta * np.cos(phi),v_accepted * sin_theta * np.sin(phi),v_accepted * cos_theta], axis=-1)

                self.fieldnc[1][:N_target, :self.__dim] = v_vec

            elif parameters[5][1][0]=='from-file':
                self.fieldnc=np.load(parameters[5][1][1])
    
    #Properties of the class boson
    #Functions to get the stored elements for the field
    def getdim(self):
        return self.__dim
    
    def getdx(self):
        return self.__dx

    def getdv(self):
        return self.__dv
    
    def getunigrid(self):
        return self.__unigrid
    
    def gettotalnumber(self):
        return self.__totalnumber
    
    def getcohfrac(self):
        return self.__cohfrac

    def gettpnumber(self):
        return self.__tpnumber

    
    #Function to change the array of positions and velocities
    def replacefieldnc(self,newpositions,newvelocities):
        assert newpositions.shape==newvelocities.shape
        self.fieldnc=np.stack([newpositions,newvelocities],axis=0)
        return None
        
    #Density and phase of the coherent part
    @property
    def densityc(self): 
        dens=np.abs(self.fieldc)**2
        if self.__smooth is not None:
            dens=gaussian_filter(dens, sigma=self.__smooth)
        return dens

    @property
    def phase(self):
        return np.angle(self.fieldc)
    
    #Function that gives the weights for the elements of the grid for the particle density assignation
    @property
    def weightassign(self):  
        pos=self.fieldnc[0]
        Ntp=len(pos)
        dim=self.getdim()
        dx=self.getdx()
        unigrid=self.getunigrid()
        Ngrid=len(unigrid)
        xmin=np.amin(unigrid)
        
        if self.densassign=='nearestgridpoint':
            indice=np.round((pos-xmin)/dx).astype(int)%Ngrid
            weighter=np.ones(Ntp)
            
            indice = indice.reshape(1, Ntp, dim)
            weighter = weighter.reshape(1, Ntp)
        
        elif self.densassign=='cloudincell':
            indicelow=np.floor((pos-xmin)/dx).astype(int)
            indicehigh=(indicelow+1)
            indicelow=indicelow%Ngrid
            indicehigh=indicehigh%Ngrid
            fhigh = (pos-(xmin+indicelow*dx))/dx
            flow = 1.0 - fhigh   
            
            neighbors = np.array(list(product([0,1], repeat=dim)))
            indice = indicelow[None, :, :] * (1 - neighbors[:, None, :]) + indicehigh[None, :, :] * neighbors[:, None, :]
            weighter = np.prod(flow[None, :, :] * (1 - neighbors[:, None, :]) + fhigh[None, :, :] * neighbors[:, None, :], axis=2)

            indice=indice.astype(int)

        return indice, weighter

    #Density of the incoherent part
    @property
    def densitync(self):
        dim=self.getdim()
        dx=self.getdx()
        unigrid=self.getunigrid()
        Ngrid=len(unigrid)
        xmin=np.amin(unigrid)

        dens=np.zeros((Ngrid,)*dim, dtype=float)
        if self.getcohfrac()==1:
            return dens

        indice, weighter = self.weightassign
        
        if self.densassign=='nearestgridpoint':
            linearidx = np.ravel_multi_index(tuple(indice[0].T), (Ngrid,)*dim)
            np.add.at(dens.ravel(), linearidx, weighter[0])
        
        elif self.densassign=='cloudincell':
            didx = []
            for d in range(self.getdim()):
                didx.append(indice[:, :, d].ravel())
            didx = tuple(didx)
            weighterflat = weighter.ravel()
            np.add.at(dens, didx, weighterflat)

        
        Nnc=self.gettotalnumber()-np.sum(np.abs(self.fieldc)**2)*(dx**dim)
        dens*=Nnc/(np.sum(dens)*dx**dim)
        if self.__smooth is not None:
            dens=gaussian_filter(dens, sigma=self.__smooth)

        return dens