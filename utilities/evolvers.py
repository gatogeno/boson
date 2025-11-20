import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift
from utilities import potentials
from utilities import additionalfunctions as adf
from utilities import collisions as col

#Function that computes the acceleration for particles
def acceleration(parameters, grids, fieldY, potnc):
    dim = parameters[0]
    dU = np.gradient(potnc, grids[0], edge_order=2)
    dU = tuple(d.astype(np.float32) for d in dU)
    if dim == 1:
        dU = (dU[0].astype(np.float32),)

    indice, weighter = fieldY.weightassign
    Ntp = indice.shape[1]
    acel = np.zeros((Ntp, dim), dtype=np.float32)

    if parameters[6][1] == 'nearestgridpoint':
        for d in range(dim):
            acel[:, d] = -dU[d][tuple(indice[0, :, :].T)] * weighter[0]

    elif parameters[6][1] == 'cloudincell':
        for l in range(indice.shape[0]):
            for d in range(dim):
                acel[:, d] -= dU[d][tuple(indice[l, :, :].T)] * weighter[l]

    return acel

#Function that updates the positions in the kickdrift evolver
def drift(chi, dt, parameters, grids, fieldY, harmpot, potc, potnc):
    #Drift for non-coherent
    if fieldY.getcohfrac() < 1 and len(fieldY.fieldnc[0]) > 0:
        fieldY.fieldnc[0] += fieldY.fieldnc[1] * dt
        fieldY.fieldnc[0] = adf.poscorrector(fieldY.fieldnc[0], fieldY.getunigrid()) #It ensures periodic boundary conditions
        potnc = potentials.potential(parameters, grids, harmpot, fieldY, coherent=False)

    #Drift for coherent
    if fieldY.getcohfrac() > 0:
        fieldYk=fftn(fieldY.fieldc)
        fieldYk *= np.exp(-chi*0.5*grids[1]*dt).astype(np.complex64)
        fieldY.fieldc = ifftn(fieldYk).astype(np.complex64)
        potc = potentials.potential(parameters, grids, harmpot, fieldY, coherent=True)
        
    return potc, potnc

#Function that updates the velocities in the kickdrift evolver
def kick(chi, dt, parameters, grids, fieldY, potc, potnc):
    # Kick non-coherent
    if fieldY.getcohfrac() < 1 and len(fieldY.fieldnc[0]) > 0:
        acel = acceleration(parameters, grids, fieldY, potnc)
        fieldY.fieldnc[1] += acel * dt

    # Kick coherente
    if fieldY.getcohfrac() > 0:
        fieldY.fieldc *= np.exp(-chi * dt * potc)

    return None

#Function to save energy, total number and timestep in arrays
def storequantities(step, t, parameters, grids, fieldY, potc, potnc, timeline, totalnumber, energy, total_collisions_in_bin):
    #Kinetic energy non-coherent and coherent
    energy[0,1,step]=adf.kineticenergy(parameters, grids, fieldY, coherent=False)
    energy[1,1,step]=adf.kineticenergy(parameters, grids, fieldY, coherent=True)
    energy[2,1,step]=energy[0,1,step]+energy[1,1,step]
    #Potential energy non-coherent and coherent
    energy[0,2,step]=adf.potentialenergy(parameters, grids, fieldY, potnc, coherent=False)
    energy[1,2,step]=adf.potentialenergy(parameters, grids, fieldY, potc, coherent=True)
    energy[2,2,step]=energy[0,2,step]+energy[1,2,step]
    #Total energy non-coherent and coherent
    energy[0,0,step]=energy[0,1,step]+energy[0,2,step]
    energy[1,0,step]=energy[1,1,step]+energy[1,2,step]
    energy[2,0,step]=energy[0,0,step]+energy[1,0,step]
    #Total number non-coherent and coherent
    totalnumber[0,step]=np.sum(fieldY.densitync)*(grids[0]**parameters[0])
    totalnumber[1,step]=np.sum(fieldY.densityc)*(grids[0]**parameters[0])
    totalnumber[2,step]=totalnumber[0,step]+totalnumber[1,step]
    
    timeline[step]=t
    #Saving the data
    adf.savingstep(step, parameters, grids, fieldY, potc, potnc, timeline, totalnumber, energy, total_collisions_in_bin)
    
    return None

#Function related with the additive noise
def additivenoise(parameters,fieldY,D):
    sigma=np.sqrt(D/(fieldY.getdx()**fieldY.getdim())).astype(np.float32)
    eta1=np.random.normal(0.0,sigma).astype(np.float32)
    eta2=np.random.normal(0.0,sigma).astype(np.float32)
    return ((eta1+1.0j*eta2)/np.sqrt(2)).astype(np.complex64)
    

#Evolver kickdrift
def kickdrift(parameters, grids, fieldY, harmpot, potc, potnc):
    #Choose the 1 or i according if it is imaginary time propagation or not
    if parameters[8][3]==True:
        chi=1.0
    else:
        chi=1.0j

    #define elements to store physical quantities during the evolution
    num_store = parameters[8][1] // parameters[8][2] + 1
    energy=np.zeros((3,3,num_store), dtype=np.float32)
    totalnumber=np.zeros((3,num_store), dtype=np.float32)
    timeline=np.zeros(num_store, dtype=np.float32)
    rbins, tcolinbin=adf.containerscoll(parameters,300)
    t=np.float32(0.0) #set initial time
    
    #store the initial configurations for fields, energy, number and time
    store_index = 0
    storequantities(store_index, t, parameters, grids, fieldY, potc, potnc, timeline, totalnumber, energy, tcolinbin)
    
    #This part saves the initial potentials
    potc_old = np.copy(potc)
    potnc_old = np.copy(potnc)
    
    #The process of evolution    
    for h in range(1,parameters[8][1]+1):
        #We adapt the timesteps
        maxpot=np.float32(np.maximum(np.amax(np.abs(potc)),np.amax(np.abs(potnc))))
        if maxpot==0 or 1/maxpot==0:
            dt=np.float32(10**(-parameters[8][4])*(grids[0]**2)/6.0)
        else:
            dt=np.float32(10**(-parameters[8][4])*np.minimum((grids[0]**2)/6.0,1/maxpot))

        #To save the montecarlo rates of collision
        allradii = np.linalg.norm(fieldY.fieldnc[0], axis=1)
        particle_counts, _ = np.histogram(allradii, bins=rbins)
        tcolinbin[0] += particle_counts
        
        #Split step evolution of the deterministic part
        kick(chi,0.5*dt,parameters,grids,fieldY,potc,potnc)
        potc, potnc = drift(chi,dt,parameters,grids,fieldY,harmpot,potc,potnc)

        potc = 0.5 * (3 * potc - potc_old)
        potnc = 0.5 * (3 * potnc - potnc_old)
        #Here we observe if there are collisions.
        if parameters[4]<1 and (parameters[7][2][0]-parameters[7][3][0]/3.0)!=0:
            hasitcol = np.zeros(len(fieldY.fieldnc[1]))
            #Binning of all particles
            sortedidx,idxflat,vidxflat,uncells,startidx,endidx,Npercell=col.celldata(parameters,fieldY)
            #Coherent-Incoherent collision
            if parameters[10][0]==True:
                hasitcol,Rdt,D,toeliminate,v3out,v4out,v3in,v4in = col.collisionscnc(parameters,dt,fieldY,hasitcol,h,idxflat,vidxflat)
                fieldY.fieldc*=np.exp(-Rdt)
                #Addition of additive noise
                #zeta=additivenoise(parameters,fieldY,D)
                #fieldY.fieldc+=-1.0j*np.sqrt(dt)*zeta
            else:
                toeliminate,v3out,v4out,v3in,v4in=np.full(len(fieldY.fieldnc[1]), False), np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1]))
            #Incoherent-Incoherent collision
            if parameters[10][1]==True:
                hasitcol,v3,v4=col.collisionsnc(parameters,dt,fieldY,hasitcol,h,sortedidx,idxflat, vidxflat,uncells,startidx,endidx,Npercell)
            else:
                v3,v4=np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1]))
            #Change of velocities and creation or elimination of particles after collisions
            col.collisionresults(parameters,fieldY,tcolinbin,rbins,hasitcol,toeliminate,v3out,v4out,v3in,v4in,v3,v4)
        kick(chi,0.5*dt,parameters,grids,fieldY,potc,potnc)

        potc_old = np.copy(potc)
        potnc_old = np.copy(potnc)

        t += dt

        #Normalization only in imaginary time
        if parameters[8][3]==True and parameters[4] > 0:
            fieldY.fieldc *= np.sqrt(parameters[4] * parameters[3] / (np.sum(np.abs(fieldY.fieldc)**2) * grids[0]**parameters[0]))

        if h%(parameters[8][2])==0:
            store_index += 1
            storequantities(store_index,t,parameters,grids,fieldY,potc,potnc,timeline,totalnumber,energy,tcolinbin)
        
        #adf.progbar(h+1,parameters[8][1]+1)
    
    return energy, totalnumber, timeline, tcolinbin
    
    
    
        
        
    
   