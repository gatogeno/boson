import sys
import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift
from utilities import storer, grid, fields, potentials

#Function that removes the comments with # in the initial file
def remove_comments(lines):
    new_lines = []
    for line in lines:
        if line.startswith("#"):
            continue
        line = line.split(" #")[0]
        if line.strip() != "":
            new_lines.append(line)
    return new_lines

#Function that returns the index in an array for a specific value:
def index_return(array,value):
    return [array.index(i) for i in array if f'{value}' in i][0]

#Function that checks if some parameter is in a list of parameters called container
def checkerlist(param, container):
    try:
        indicestore = next(i for i, item in enumerate(container) if param in item)
    except StopIteration:
        print(f'Parameter {param} is missing. Please enter this parameter in the initial file')
        sys.exit()

    element = container[indicestore].partition("=")[2].replace(" ", "").strip("\n")
    return element

#Function that generates a progress bar
def progbar(progress,iteration_number):
    size = 60
    status = ""
    progress = progress/iteration_number
    if progress >= 1.:
        progress = 1
    block = int(round(size * progress))
    text="\r{}|{:.0f}%".format("*"*block+""*(size - block),round(progress * 100, 0))
    sys.stdout.write(text)
    sys.stdout.flush()
        
#Functions common to all evolvers
#Function that save in external files the outputs of field, time, energy, total number and potentials
def savingstep(indexstep, parameters, grids, fieldY, potc, potnc, timeline, totalnumber, energy, totalcollisionsinbin):
    filepath=parameters[11]
    
    if parameters[8][3]==False:
            if parameters[4]==1:
                np.save(f'{filepath}'+'/evolution/fieldc_'+str(indexstep)+'.npy', fieldY.fieldc)
            elif parameters[4]==0:
                np.save(f'{filepath}'+'/evolution/fieldnc_'+str(indexstep)+'.npy', fieldY.fieldnc)
            else:
                np.save(f'{filepath}'+'/evolution/fieldc_'+str(indexstep)+'.npy', fieldY.fieldc)
                np.save(f'{filepath}'+'/evolution/fieldnc_'+str(indexstep)+'.npy', fieldY.fieldnc)
    else:
        if parameters[4]==1 and indexstep<1:
            np.save(f'{filepath}'+'/evolution/fieldc_'+str(indexstep)+'.npy', fieldY.fieldc)
        elif parameters[4]==0 and indexstep<1:
                np.save(f'{filepath}'+'/evolution/fieldnc_'+str(indexstep)+'.npy', fieldY.fieldnc)
        elif parameters[4]>0 and parameters[4]<1 and indexstep<1:
                np.save(f'{filepath}'+'/evolution/fieldc_'+str(indexstep)+'.npy', fieldY.fieldc)
                np.save(f'{filepath}'+'/evolution/fieldnc_'+str(indexstep)+'.npy', fieldY.fieldnc)
        
    if indexstep==parameters[8][1] // parameters[8][2]:
        np.save(f'{filepath}'+'/timeline.npy', timeline)
        np.save(f'{filepath}'+'/totalnumber.npy', totalnumber)
        np.save(f'{filepath}'+'/energy.npy', energy)
        np.save(f'{filepath}'+'/totalcollisionsinbin.npy', totalcollisionsinbin)
        if parameters[4]==1:
            np.save(f'{filepath}'+'/potentialc.npy', potc)
            np.save(f'{filepath}'+'/fieldc.npy', fieldY.fieldc)
        elif parameters[4]==0:
            np.save(f'{filepath}'+'/potentialnc.npy', potnc)
            np.save(f'{filepath}'+'/fieldnc.npy', fieldY.fieldnc)
        else:
            np.save(f'{filepath}'+'/potentialc.npy', potc)
            np.save(f'{filepath}'+'/potentialnc.npy', potnc)
            np.save(f'{filepath}'+'/fieldc.npy', fieldY.fieldc)
            np.save(f'{filepath}'+'/fieldnc.npy', fieldY.fieldnc)
        
    return None

#Function that computes kinetic energy
def kineticenergy(parameters, grids, fieldY, coherent=True):
    eps = np.float32(1e-16) #factor to divide in normalizing when the kinetic energy is zero
    if coherent:
        kineticen=(0.5*np.sum(np.real(fieldY.fieldc.conjugate()*ifftn(grids[1]*fftn(fieldY.fieldc))))*(grids[0]**parameters[0])).astype(np.float32)
        kineticen/=max(np.sum(fieldY.densityc)*(grids[0]**parameters[0]),eps)
    else:
        kineticen=(0.5*np.sum(fieldY.fieldnc[1]**2)).astype(np.float32)
        kineticen/=max(parameters[6][0],eps)

    return kineticen

#Auxiliary functions for potential energy for the non-coherent part
def _potential_energy_ngp(pot, indice, weighter, dim):
    flat_idx = indice.reshape(-1, dim)
    flat_w = weighter.ravel()
    if dim == 1:
        lin_idx = flat_idx[:, 0].astype(np.int32)
    else:
        lin_idx = np.ravel_multi_index(flat_idx.T, pot.shape[:dim]).astype(np.int32)

    weights_grid = np.zeros_like(pot.ravel(), dtype=float)
    np.add.at(weights_grid, lin_idx, flat_w)

    return np.sum(weights_grid.reshape(pot.shape) * pot).astype(np.float32)

def _potential_energy_cic(pot, indice, weighter, dim):
    Ntp = indice.shape[1]
    ncomb = indice.shape[0]
    flat_idx = indice.reshape(-1, dim)
    flat_w = weighter.ravel()

    if dim == 1:
        lin_idx = flat_idx[:, 0].astype(np.int32)
    else:
        lin_idx = np.ravel_multi_index(flat_idx.T, pot.shape[:dim]).astype(np.int32)

    weights_grid = np.zeros_like(pot.ravel(), dtype=float)
    np.add.at(weights_grid, lin_idx, flat_w)

    return np.sum(weights_grid.reshape(pot.shape) * pot).astype(np.float32)

#Function that computes potential energy
def potentialenergy(parameters, grids, fieldY, pot, coherent=True):
    eps = np.float32(1e-16) #factor to divide in normalizing when the kinetic energy is zero
    if np.amax(pot)==np.inf:
        potentialen=0

    if coherent:
        potentialen=np.sum(pot*fieldY.densityc)*(grids[0]**parameters[0]).astype(np.float32)
        potentialen/=max(np.sum(fieldY.densityc)*(grids[0]**parameters[0]),eps)
    else:
        if len(fieldY.fieldnc[0]) == 0:
            return np.float32(0.0)
        indice, weighter = fieldY.weightassign
        if parameters[6][1] == 'nearestgridpoint':
            potentialen = _potential_energy_ngp(pot, indice, weighter, parameters[0])
        elif parameters[6][1] == 'cloudincell':
            potentialen = _potential_energy_cic(pot, indice, weighter, parameters[0])
        else:
            return np.float32(0.0)

        potentialen/=max(parameters[6][0],eps)
    
    return potentialen

#Function that corrects the position of a particle if the value goes outside the box according to periodical boundary conditions
def poscorrector(x,grid):
    #incr=0.5*(grid[1]-grid[0])
    #return (x+np.abs(np.amin(grid)-incr))%(np.amax(grid)+incr+np.abs(np.amin(grid)-incr))-np.abs(np.amin(grid)-incr)
    Lmin, Lmax = np.min(grid), np.max(grid)
    L = np.float32(Lmax - Lmin + (grid[1] - grid[0]))
    return (Lmin + ((x - Lmin) % L)).astype(np.float32)

#Function that generate containers for the collision counts for radial spatial bins
def containerscoll(parameters,nbins):
    rmax = parameters[1]
    rbins = np.linspace(0, rmax, nbins + 1, dtype=np.float32)
    r_centers = 0.5 * (rbins[:-1] + rbins[1:])
    total_collisions_in_bin = np.zeros((4,nbins), dtype=np.float32)
    return rbins,total_collisions_in_bin
        