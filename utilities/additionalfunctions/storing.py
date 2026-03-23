import numpy as np
from utilities.additionalfunctions import constants as cts
        
#Function that save in external files the outputs of field, time, energy, total number and potentials
def savingstep(indexstep, t, fundparameters, evolverparameters, fieldY, timeline, gammacol2, ncollsteps):
    filepath=fundparameters[4]
    timeline[indexstep]=t
    
    np.save(f'{filepath}'+'/evolution/fieldc_'+str(indexstep)+'.npy', fieldY.fieldc)
    np.save(f'{filepath}'+'/evolution/fieldnc_'+str(indexstep)+'.npy', fieldY.fieldnc)
        
    if indexstep==evolverparameters[0] // evolverparameters[1]:
        np.save(f'{filepath}'+'/timeline.npy', timeline)
        np.save(f'{filepath}'+'/gammacol2.npy', gammacol2 / max(ncollsteps, 1)) 
        
    return None