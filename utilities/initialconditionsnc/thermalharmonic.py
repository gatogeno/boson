import numpy as np
from utilities.additionalfunctions import constants as cts

def thermalharmonic(fieldparameters,potparameters,fieldnc):
    w=np.array([potparameters[0][1], potparameters[0][2], potparameters[0][3]])
    fieldnc[0]=np.random.normal(0,1/(np.sqrt(fieldparameters[1][1])*w),size=(fieldparameters[2][0],3)).astype(cts.FTYPE)
    fieldnc[1]=np.random.normal(0,1/np.sqrt(fieldparameters[1][1]),size=(fieldparameters[2][0],3)).astype(cts.FTYPE)

    return fieldnc