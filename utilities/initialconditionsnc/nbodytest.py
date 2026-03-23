import numpy as np
from utilities.additionalfunctions import constants as cts

def nbodytest(fieldparameters,grids,fieldc,fieldnc,dx):
    N_part = fieldparameters[2][0]

    r = np.cbrt(np.random.rand(N_part))
    cos_theta = np.random.uniform(-1, 1, N_part)
    phi = np.random.uniform(0, 2 * np.pi, N_part)
    sin_theta = np.sqrt(1 - cos_theta**2)
    X = r * sin_theta * np.cos(phi)
    Y = r * sin_theta * np.sin(phi)
    Z = r * cos_theta
    fieldnc[0][:N_part, :3] = np.stack([X, Y, Z], axis=-1)

    vmag=cts.FTYPE(fieldparameters[1][1])

    cos_theta = np.ones((N_part))*0
    phi = np.ones((N_part))*np.pi/4
    sin_theta = np.sqrt(1 - cos_theta**2)
    X = vmag * sin_theta * np.cos(phi)
    Y = vmag * sin_theta * np.sin(phi)
    Z = vmag * cos_theta
    fieldnc[1][:N_part, :3] = np.stack([X, Y, Z], axis=-1)
    

    return fieldnc