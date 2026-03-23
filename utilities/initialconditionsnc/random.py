import numpy as np
from utilities.additionalfunctions import constants as cts

def randomnc(fieldparameters,fieldnc,unigrid):
    fieldnc[0]=np.random.uniform(np.amin(unigrid),np.amax(unigrid),size=(fieldparameters[2][0],3)).astype(cts.FTYPE)
    module=np.random.uniform(0,fieldparameters[2][1],size=(fieldparameters[2][0],)).astype(cts.FTYPE)

    angleth=np.random.uniform(0,np.pi,fieldparameters[2][0]).astype(cts.FTYPE)
    angleph=np.random.uniform(0,2*np.pi,fieldparameters[2][0]).astype(cts.FTYPE)
    fieldnc[1][:,0]=module*np.sin(angleth)*np.cos(angleph)
    fieldnc[1][:,1]=module*np.sin(angleth)*np.sin(angleph)
    fieldnc[1][:,2]=module*np.cos(angleth)

    return fieldnc