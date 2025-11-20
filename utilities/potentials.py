import numpy as np
from scipy.fftpack import fftn, fftfreq, ifftn, fftshift, ifftshift

def harmonicpotential(parameters, grids):
    coord_indices = (2, 4, 6)[:parameters[0]]
    coords = [grids[i] for i in coord_indices if len(grids) > i]
    if len(coords) < parameters[0]:
        coords += [np.zeros_like(coords[0])] * (parameters[0] - len(coords))

    omegas = np.array(parameters[7][0][:parameters[0]], dtype=np.float32)
    return (0.5 * sum((omegas[i]**2) * coords[i]**2 for i in range(parameters[0]))).astype(np.float32)

def potential(parameters, grids, harmpot, fieldY, coherent=True):
    #Self-interaction part
    if coherent:
        fc, fnc = 1.0, 2.0
    else:
        fc, fnc = 2.0, 2.0
    selfintpot=((parameters[7][2][0]-parameters[7][3][0]/3.0)*(fc*fieldY.densityc+fnc*fieldY.densitync)).astype(np.float32)

    #Dipolar part
    zerokmask = np.abs(grids[1]) < 1e-16
    invksq=np.divide(1.0,grids[1],where=~zerokmask,out=np.zeros_like(grids[1],dtype=np.float32))
    if parameters[7][3][0]!=0:
        if parameters[0]==1:
            nk2=(parameters[7][3][1]*grids[3])**2
        elif parameters[0]==2:
            nk2=(parameters[7][3][1]*grids[3]+parameters[7][3][2]*grids[5])**2
        else:
            nk2=(parameters[7][3][1]*grids[3]+parameters[7][3][2]*grids[5]+parameters[7][3][3]*grids[7])**2
        dippotk=(-parameters[7][3][0]*nk2*fftn(fieldY.densityc+fieldY.densitync)*invksq).astype(np.complex64)
        dippot=np.real(ifftn(dippotk)).astype(np.float32)
    else:
        dippot=np.zeros_like(fieldY.densityc).astype(np.float32)
    
    #Gravitational part
    if parameters[7][1][0]==True:
        if parameters[0]==1:
            Sd=2
        elif parameters[0]==2:
            Sd=2*np.pi
        else:
            Sd=4*np.pi        
        potgravk=(-Sd*fftn(fieldY.densityc+fieldY.densitync)*invksq).astype(np.complex64)
        potgrav=np.real(ifftn(potgravk)).astype(np.float32)
    else:
        potgrav=np.zeros_like(fieldY.densityc).astype(np.float32)
    
    return (harmpot + selfintpot + dippot + potgrav).astype(np.float32)