import numpy as np
from utilities.additionalfunctions import constants as cts

def normal(fieldparameters,fieldnc):
    fieldnc[0]=np.random.normal(fieldparameters[1][1],fieldparameters[1][2],size=(fieldparameters[2][0],3)).astype(cts.FTYPE)
    fieldnc[1]=np.random.normal(fieldparameters[1][3],fieldparameters[1][4],size=(fieldparameters[2][0],3)).astype(cts.FTYPE)

    return fieldnc