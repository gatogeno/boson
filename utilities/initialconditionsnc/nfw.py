import numpy as np
from utilities.additionalfunctions import constants as cts

def nfw(fundparamters,fieldparameters,grids,fieldnc):
    c=fieldparameters[1][2]
    rs=fieldparameters[1][1]

    r=np.sqrt(grids[2]**2+grids[4]**2+grids[6]**2)
    n0=(1-fundparameters[3])*fundparameters[2]/(4*np.pi*rs**3*(np.log(1+c)-c/(1+c)))
    r[r == 0] = 1e-8
    rho = n0 / (r/rs * (1.0 + r/rs)**2)

    rho_flat = rho.flatten()
    rho_flat /= rho_flat.sum()
    indices = np.random.choice(len(rho_flat), size=fieldparameters[2][0], p=rho_flat)
    inds = np.unravel_index(indices, rho.shape)

    x = grids[2][inds]
    y = grids[4][inds]
    z = grids[6][inds]
    fieldnc[0,:,0] = cts.FTYPE(x)
    fieldnc[0,:,1] = cts.FTYPE(y)
    fieldnc[0,:,2] = cts.FTYPE(z)

    #module=np.random.uniform(0,parameters[6][2],size=(parameters[6][0],)).astype(np.float32)
    #if parameters[0]==1:
    #    fieldnc[1][:,0]=module*(2*np.random.randint(0,2,size=parameters[6][0])-1).astype(np.float32)
    #elif parameters[0]==2:
    #    angleph=np.random.uniform(0,2*np.pi,parameters[6][0]).astype(np.float32)
    #    fieldnc[1][:,0]=module*np.cos(angleph)
    #    fieldnc[1][:,1]=module*np.sin(angleph)
    #else:
    #    angleth=np.random.uniform(0,np.pi,parameters[6][0]).astype(np.float32)
    #    angleph=np.random.uniform(0,2*np.pi,parameters[6][0]).astype(np.float32)
    #    fieldnc[1][:,0]=module*np.sin(angleth)*np.cos(angleph)
    #    fieldnc[1][:,1]=module*np.sin(angleth)*np.sin(angleph)
    #    fieldnc[1][:,2]=module*np.cos(angleth)
    fieldnc[1]=np.random.normal(0,2,size=(fieldparameters[2][0],3)).astype(cts.FTYPE)
    
    return fieldnc