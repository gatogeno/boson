import numpy as np
from scipy.fft import fftn, ifftn
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions import poscorrector as pcr
from utilities.additionalfunctions.acceleration import acceleration, cleargradientcache
from utilities.additionalfunctions import storing as sto
from utilities import potentials
from utilities import collisions as col
from utilities.additionalfunctions.adaptivecells import constructadaptivecells
from utilities.additionalfunctions import interpolations as interp
#from utilities.c12 import run_c12


#Evolver
def evolver(fundparameters,fieldparameters,evolverparameters,optparameters, grids, fieldY, pot):
    #Choose the 1 or i according if it is imaginary time propagation or not
    chi=1.0 if evolverparameters[2] else 1.0j

    #define elements to store physical quantities during the evolution
    num_store = evolverparameters[0] // evolverparameters[1] + 1
    timeline=np.zeros(num_store, dtype=cts.FTYPE)
    gammacol2 = np.zeros((len(fieldY.getunigrid()),) * 3, dtype=cts.FTYPE)
    
    #set initial indices
    t=cts.FTYPE(0.0) 
    ncollsteps  = 0
    storeidx = 0
    dx2over6=(fieldY.getdx()**2)/6.0

    sto.savingstep(storeidx, t, fundparameters, evolverparameters, fieldY, timeline, gammacol2, ncollsteps)

    #The process of evolution
    for h in range(1,evolverparameters[0]+1):
        #We adapt the timesteps
        maxpot=cts.FTYPE(np.maximum(np.amax(np.abs(pot.potentialc)),np.amax(np.abs(pot.potentialq))))
        if maxpot==0 or np.isnan(maxpot)==True:
            dt=cts.FTYPE(10**(-evolverparameters[3])*dx2over6)
        else:
            dt=cts.FTYPE(10**(-evolverparameters[3])*np.minimum(dx2over6,1/maxpot))
        
        #Kick-drift steps
        kickdrift(chi,dt,fundparameters,grids,fieldY,pot)
        #Normalization in imaginary time
        if evolverparameters[2]==True:
            fieldY.normalizer()
        #Collisions
        if fieldY.getcohfrac() < 1.0 and evolverparameters[2]==False and optparameters[1][0]==False:
            collisionstep(evolverparameters,optparameters,grids,fieldY,dt,gammacol2,ncollsteps)
        
        t += dt

        if h%(evolverparameters[1])==0:
            storeidx += 1
            sto.savingstep(storeidx, t, fundparameters, evolverparameters, fieldY, timeline, gammacol2, ncollsteps)
    
    return None

#Function that updates the velocities in the kickdrift evolver
def kick(chi, dt, fundparameters, grids, fieldY, pot):
    # Kick non-coherent
    if fieldY.getcohfrac() < 1 and len(fieldY.fieldnc[0]) > 0:
        w2 = fieldY.weight2()[:, None]
        acel_q, acel_mf = acceleration(grids, fieldY, pot)
        fieldY.fieldnc[1] += (acel_q * w2 + acel_mf) * dt
    # Kick coherent
    if fieldY.getcohfrac() > 0:
        #fieldY.fieldc *= np.exp(-chi * dt * pot.potentialc)
        np.multiply(fieldY.fieldc,np.exp(-chi * dt * pot.potentialc),out=fieldY.fieldc)
        fieldY._invalidate_phase_only()

    return None

#Function that updates the positions in the kickdrift evolver
def drift(chi, dt, grids, fieldY, pot):
    #Drift for non-coherent
    if fieldY.getcohfrac() < 1 and len(fieldY.fieldnc[0]) > 0:
        np.copyto(fieldY.x0buffer, fieldY.fieldnc[0])
        v_base = fieldY.fieldnc[1]
        w1_0 = fieldY.weight1()[:, None]
        fieldY.fieldnc[0] += v_base * w1_0 * dt
        fieldY.invalidate_positions_only()
        for _ in range(2): 
            fieldY.fieldnc[0] += fieldY.x0buffer
            fieldY.fieldnc[0] *= 0.5
            fieldY.fieldnc[0] = pcr.poscorrector(fieldY.fieldnc[0], fieldY.getunigrid())
            fieldY.invalidate_positions_only()
            w1_mid = fieldY.weight1()[:, None]
            np.copyto(fieldY.fieldnc[0], fieldY.x0buffer)  
            fieldY.fieldnc[0] += v_base * w1_mid * dt
            fieldY.invalidate_positions_only()
        fieldY.fieldnc[0] = pcr.poscorrector(fieldY.fieldnc[0], fieldY.getunigrid())
        fieldY.invalidate_for_potential()

    #Drift for coherent
    if fieldY.getcohfrac() > 0:
        fieldYk=fftn(fieldY.fieldc,workers=-1)
        fieldYk *= np.exp(-chi*0.5*grids[0]*dt).astype(cts.CTYPE)
        fieldY.fieldc = ifftn(fieldYk,workers=-1).astype(cts.CTYPE)
        fieldY.invalidate_fieldc_cache()

    pot.updatepotentials(fieldY, grids)
    cleargradientcache()
    if fieldY.getcohfrac() < 1 and len(fieldY.fieldnc[0]) > 0:
        _ = fieldY.getinterpdens()
    return None


#Kickdrift normal
def kickdrift(chi,dt,fundparameters,grids,fieldY,pot):
    kick(chi,0.5*dt,fundparameters,grids,fieldY,pot)
    drift(chi,dt,grids,fieldY,pot)
    kick(chi,0.5*dt,fundparameters,grids,fieldY,pot)

#Kickdrift fourth order
def kickdriftfourth(chi,dt,fundparameters,grids,fieldY,pot):
    w0 = -2**(1/3)/(2-2**(1/3))
    w1 = 1/(2-2**(1/3))
    kick(chi,0.5*w1*dt,fundparameters,grids,fieldY,pot)
    drift(chi,w1*dt,grids,fieldY,pot)
    kick(chi,0.5*w1*dt,fundparameters,grids,fieldY,pot)
    kick(chi,0.5*w0*dt,fundparameters,grids,fieldY,pot)
    drift(chi,w0*dt,grids,fieldY,pot)
    kick(chi,0.5*w0*dt,fundparameters,grids,fieldY,pot)
    kick(chi,0.5*w1*dt,fundparameters,grids,fieldY,pot)
    drift(chi,w1*dt,grids,fieldY,pot)
    kick(chi,0.5*w1*dt,fundparameters,grids,fieldY,pot)

#Calling for collisions
def collisionstep(evolverparameters,optparameters,grids,fieldY,dt,gammacol2,ncollsteps):
    #If there is only one non-coherent particle there is no collision
    if len(fieldY.fieldnc[0])>=2:
        #Create a random array to shuffle particles in adaptive cells
        seed = cts.ITYPE(np.random.randint(1, 2**31))
        randarr = np.array([cts.UITYPE(seed)], dtype=cts.UITYPE)
        #construct the adaptive cells
        subcells, subcellcoord, subcelllength = constructadaptivecells(fieldY,randarr)
        #Collision I2
        gncinterp = fieldY.g * fieldY.getinterpdens()
        ntilde = interp.interpolatorntilde(fieldY)
        totalomega = cts.FTYPE(np.sum(fieldY.weight1()))
        collisionrate = col.launchercolI2(fieldY, gncinterp, ntilde, totalomega, dt, subcells, subcellcoord, subcelllength,randarr)
        gammacol2 += collisionrate
        ncollsteps  += 1

    #R_grid = run_c12(fieldY, grids, dt)
    #fieldY.fieldc *= np.exp(- dt * R_grid)
    #fieldY.updatecohfrac()
    #print(fieldY.getcohfrac())
    #fieldY.invalidate_fieldc_cache()
            
    
   