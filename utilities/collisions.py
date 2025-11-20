import numpy as np
from scipy.special import gamma
from itertools import product
from collections import defaultdict

#Function to bin velocities
def binvelocities(parameters,fieldY,velocities):
    dim = fieldY.getdim()
    dv = fieldY.getdv()
    vmax= parameters[6][2]
    nbins = int(2 * vmax / dv)
    
    # Index of velocity cell of the target velocity
    vel_idx = np.floor((velocities + vmax) / dv).astype(np.int32)
    vel_idx = np.clip(vel_idx, 0, nbins - 1)

    if dim == 1:
        v_flat = vel_idx[:, 0]
    elif dim == 2:
        v_flat = vel_idx[:, 0] + nbins * vel_idx[:, 1]
    else:
        v_flat = vel_idx[:, 0] + nbins * vel_idx[:, 1] + nbins**2 * vel_idx[:, 2]
    
    return v_flat

#Function to define the particles in a cell
def celldata(parameters,fieldY):
    dim = fieldY.getdim()
    dx = fieldY.getdx()
    unigrid = fieldY.getunigrid()
    Ngrid = len(unigrid)
    xmin = np.amin(unigrid)
    pos = fieldY.fieldnc[0]
    
    cell_idx = np.floor((pos - xmin) / dx).astype(np.int32) % Ngrid
    if dim == 1:
        idxflat = cell_idx[:,0]
    elif dim == 2:
        idxflat = cell_idx[:,0] + Ngrid * cell_idx[:,1]
    else:
        idxflat = cell_idx[:,0] + Ngrid * cell_idx[:,1] + Ngrid**2 * cell_idx[:,2]

    sortedidx = np.argsort(idxflat)
    sortedcells = idxflat[sortedidx]
    uncells, startidx = np.unique(sortedcells, return_index=True)
    endidx=np.append(startidx[1:],len(sortedidx))
    Npercell=endidx-startidx

    #Indices of velocity cell of the full particles
    vidxflat=binvelocities(parameters,fieldY,fieldY.fieldnc[1])

    return sortedidx.astype(np.int32), idxflat.astype(np.int32), vidxflat, uncells.astype(np.int32), startidx.astype(np.int32), endidx.astype(np.int32), Npercell.astype(np.int32)

#Function to select pairs of non-coherent particles in order to evaluate collisions
def generatepairs(sortedidx, uncells,startidx, endidx, Npercell):    
    mask = Npercell >= 2
    startidx = startidx[mask]
    endidx = endidx[mask]
    Npercell = Npercell[mask]
    uncells = uncells[mask]
    
    allpairs = []
    pairscounts=[]
    cellindex = []
    for cellid, start, end, Nincell in zip(uncells, startidx, endidx, Npercell):
        particles = sortedidx[start:end]
        perms=np.random.permutation(Nincell)
        Npairs = Nincell // 2
        pairs = particles[perms[:2*Npairs]].reshape(Npairs, 2)
        allpairs.append(pairs)
        pairscounts.append(np.full(Npairs, Nincell, dtype=int))
        cellindex.append(np.full(Npairs, cellid, dtype=int))

    if len(allpairs) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    return np.vstack(allpairs), np.concatenate(pairscounts), np.concatenate(cellindex)

#Function to interpolate values of an array in a grid to points inside of the grid
def interpolator(fieldY, objeto):
    dim = fieldY.getdim()
    dx = fieldY.getdx()
    pos = fieldY.fieldnc[0]
    unigrid = fieldY.getunigrid()
    Ngrid = len(unigrid)
    xmin = np.amin(unigrid)

    indicelow = np.floor((pos - xmin) / dx).astype(np.int32) % Ngrid
    indicehigh = (indicelow + 1) % Ngrid
    fhigh = ((pos - (xmin + indicelow * dx)) / dx).astype(np.float32)
    flow = (1.0 - fhigh).astype(np.float32)

    if dim == 1:
        i0 = indicelow[:, 0]
        i1 = indicehigh[:, 0]
        return flow[:, 0] * objeto[i0] + fhigh[:, 0] * objeto[i1]

    if dim == 2:
        i0, j0 = indicelow[:, 0], indicelow[:, 1]
        i1, j1 = indicehigh[:, 0], indicehigh[:, 1]

        wx0, wx1 = flow[:, 0], fhigh[:, 0]
        wy0, wy1 = flow[:, 1], fhigh[:, 1]

        return objeto[i0, j0] * wx0 * wy0 + objeto[i1, j0] * wx1 * wy0 + objeto[i0, j1] * wx0 * wy1 + objeto[i1, j1] * wx1 * wy1

    if dim == 3:
        i0, j0, k0 = indicelow[:, 0], indicelow[:, 1], indicelow[:, 2]
        i1, j1, k1 = indicehigh[:, 0], indicehigh[:, 1], indicehigh[:, 2]

        wx0, wx1 = flow[:, 0], fhigh[:, 0]
        wy0, wy1 = flow[:, 1], fhigh[:, 1]
        wz0, wz1 = flow[:, 2], fhigh[:, 2]

        return (
            objeto[i0, j0, k0] * wx0 * wy0 * wz0 +
            objeto[i1, j0, k0] * wx1 * wy0 * wz0 +
            objeto[i0, j1, k0] * wx0 * wy1 * wz0 +
            objeto[i0, j0, k1] * wx0 * wy0 * wz1 +
            objeto[i1, j1, k0] * wx1 * wy1 * wz0 +
            objeto[i1, j0, k1] * wx1 * wy0 * wz1 +
            objeto[i0, j1, k1] * wx0 * wy1 * wz1 +
            objeto[i1, j1, k1] * wx1 * wy1 * wz1
        )

#Function to compute R for the coherent part
def computeRdt(fieldY,Pnet):
    dim = fieldY.getdim()
    dx = fieldY.getdx()
    unigrid = fieldY.getunigrid()
    Ngrid = len(unigrid)
    xmin = np.amin(unigrid)
    shape = (Ngrid,) * dim
    dV = dx**dim

    if fieldY.densassign == 'nearestgridpoint':
        indice = np.round((fieldY.fieldnc[0] - xmin)/dx).astype(np.int32) % Ngrid
        linearidx = np.ravel_multi_index(tuple(indice.T), shape)
        Rnum = np.zeros(np.prod(shape))
        np.add.at(Rnum, linearidx, Pnet)
        Rnum = Rnum.reshape(shape)

    elif fieldY.densassign == 'cloudincell':
        indicelow = np.floor((fieldY.fieldnc[0] - xmin)/dx).astype(np.int32) % Ngrid
        indicehigh = (indicelow + 1) % Ngrid
        fhigh = ((fieldY.fieldnc[0] - (xmin + indicelow*dx))/dx).astype(np.float32)
        flow = (1.0 - fhigh).astype(np.float32)

        neighbors = np.array(list(product([0,1], repeat=dim)), dtype=np.int8)
        indice = (indicelow[None, :, :] * (1 - neighbors[:, None, :]) + indicehigh[None, :, :] * neighbors[:, None, :])
        weighter = np.prod(flow[None, :, :] * (1 - neighbors[:, None, :]) + fhigh[None, :, :] * neighbors[:, None, :], axis=2).astype(np.float32)
        indice = indice.astype(np.int32)

        didx = tuple(indice.reshape(-1, dim).T)
        weighterflat = ((weighter.ravel()) * np.repeat(Pnet, len(neighbors))).astype(np.float32)
        Rnum = np.zeros(shape)
        np.add.at(Rnum, didx, weighterflat)
        
    Rdt = 0.5*Rnum/dV

    return Rdt

#Function to compute the object of variance for the noise
def computematvar(fieldY,Psum,dt):
    dim = fieldY.getdim()
    dx = fieldY.getdx()
    unigrid = fieldY.getunigrid()
    Ngrid = len(unigrid)
    xmin = np.amin(unigrid)
    shape = (Ngrid,) * dim
    dV = dx**dim

    if fieldY.densassign == 'nearestgridpoint':
        indice = np.round((fieldY.fieldnc[0] - xmin)/dx).astype(np.int32) % Ngrid
        linearidx = np.ravel_multi_index(tuple(indice.T), shape)
        Dnum = np.zeros(np.prod(shape))
        np.add.at(Dnum, linearidx, Psum)
        Dnum = Dnum.reshape(shape)

    elif fieldY.densassign == 'cloudincell':
        indicelow = np.floor((fieldY.fieldnc[0] - xmin)/dx).astype(np.int32) % Ngrid
        indicehigh = (indicelow + 1) % Ngrid
        fhigh = ((fieldY.fieldnc[0] - (xmin + indicelow*dx))/dx).astype(np.float32)
        flow = (1.0 - fhigh).astype(np.float32)

        neighbors = np.array(list(product([0,1], repeat=dim)), dtype=np.int8)
        indice = (indicelow[None, :, :] * (1 - neighbors[:, None, :]) + indicehigh[None, :, :] * neighbors[:, None, :])
        weighter = np.prod(flow[None, :, :] * (1 - neighbors[:, None, :]) + fhigh[None, :, :] * neighbors[:, None, :], axis=2).astype(np.float32)
        indice = indice.astype(np.int32)

        didx = tuple(indice.reshape(-1, dim).T)
        weighterflat = ((weighter.ravel()) * np.repeat(Psum, len(neighbors))).astype(np.float32)
        Dnum = np.zeros(shape)
        np.add.at(Dnum, didx, weighterflat)
        
    D = 0.5 * Dnum / (dV * dt)

    return D

#Function to compute the distribution f(v3) and f(v4) with binned velocities
def fbinned(parameters,fieldY,idxflat,vidxflat,cellindex,vtarget):
    dim=fieldY.getdim()
    dx=fieldY.getdx()
    dv=fieldY.getdv()
    vmax=parameters[6][2]
    nbins=int(2*vmax / dv)
    prefactor = (((fieldY.gettotalnumber()-np.sum(np.abs(fieldY.fieldc)**2)*(dx**dim)) / len(fieldY.fieldnc[0])) * (2*np.pi)**dim / (dx**dim * dv**dim)).astype(np.float32)

    uniquecells, compactindices = np.unique(idxflat, return_inverse=True)
    idxflat = compactindices
    Ncells = len(uniquecells)

    cellmap = {orig: new for new, orig in enumerate(uniquecells)}
    cellindexcompact = np.array([cellmap[c] for c in cellindex]).astype(np.int32)
    
    globalidx = (idxflat * (nbins**dim) + vidxflat).astype(np.int64)
    totalbins = Ncells * (nbins**dim)
    counts = np.bincount(globalidx, minlength=totalbins)
    fcounts = counts.reshape(Ncells, nbins**dim)

    #Index of velocity cell of the target velocity
    vflat=binvelocities(parameters,fieldY,vtarget)

    fv = fcounts[cellindexcompact, vflat] * prefactor

    return fv

#Functions that computes the final velocities in the types of collisions
def finalvelocitiesnc(v1,v2,dv,fieldY):
    if fieldY.getdim() == 1:
        u = np.random.choice([-1, 1], size=(len(v1), 1))
    elif fieldY.getdim() == 2:
        theta = (2 * np.pi * np.random.rand(len(v1))).astype(np.float32)
        u = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    else:
        phi = (2 * np.pi * np.random.rand(len(v1))).astype(np.float32)
        cos_theta = (2 * np.random.rand(len(v1)) - 1).astype(np.float32)
        sin_theta = np.sqrt(1 - cos_theta**2).astype(np.float32)
        u = np.stack([sin_theta * np.cos(phi),sin_theta * np.sin(phi),cos_theta], axis=1).astype(np.float32)
    v3=(0.5*(v1+v2+dv[:, np.newaxis]*u)).astype(np.float32)
    v4=(0.5*(v1+v2-dv[:, np.newaxis]*u)).astype(np.float32)
    return v3, v4

def finalvelocitiescncin(parameters,g,densc,v2,v2norm,fieldY):
    if fieldY.getdim() == 1:
        v4perp = np.zeros((len(v2), 1))
    elif fieldY.getdim() == 2:
        v4perp=np.stack([-v2[:,1],v2[:,0]],axis=1)
        v4perp/=np.linalg.norm(v4perp,axis=1)[:,None]
        v4perp*=(parameters[6][2]/2)*np.sqrt(np.random.rand(len(v2)))[:, None]
    else:
        u=np.array([0,0,1])
        v4aux1=np.cross(v2,u)
        norms=np.linalg.norm(v4aux1, axis=1)
        zero_mask = norms < 1e-12
        if np.any(zero_mask):
            u2 = np.array([1, 0, 0])
            v4aux1[zero_mask] = np.cross(v2[zero_mask], u2)
        norms = np.linalg.norm(v4aux1, axis=1)
        v4aux1/=norms[:, None]
        v4aux2=np.cross(v2,v4aux1)
        v4aux2/=np.linalg.norm(v4aux2,axis=1)[:, None]
        phi = 2*np.pi*np.random.rand(len(v2))
        r = (parameters[6][2]) * np.sqrt(np.random.rand(len(v2)))
        v4perp = (np.cos(phi)[:, None]*v4aux1 + np.sin(phi)[:, None]*v4aux2) * r[:, None]

    v3=((1+g*densc/v2norm**2)[:, None]*v2+v4perp).astype(np.float32)
    v4=((g*densc/v2norm**2)[:, None]*v2+v4perp).astype(np.float32)
    return v3, v4

def finalvelocitiescncout(v,v2,fieldY):
    if fieldY.getdim() == 1:
        u = np.random.choice([-1, 1], size=(len(v2), 1))
    elif fieldY.getdim() == 2:
        theta = (2 * np.pi * np.random.rand(len(v2))).astype(np.float32)
        u = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    else:
        phi = (2 * np.pi * np.random.rand(len(v2))).astype(np.float32)
        cos_theta = (2 * np.random.rand(len(v2)) - 1).astype(np.float32)
        sin_theta = np.sqrt(1 - cos_theta**2).astype(np.float32)
        u = np.stack([sin_theta * np.cos(phi),sin_theta * np.sin(phi),cos_theta], axis=1).astype(np.float32)
    v3=(0.5*(v2+v[:, np.newaxis]*u)).astype(np.float32)
    v4=(0.5*(v2-v[:, np.newaxis]*u)).astype(np.float32)
    return v3, v4

#Function to compute the probability of the collisions between two non-coherent happen
def probabilitycollisionsnc(parameters,dt,dens,dv,v3,v4,fieldY,idxflat,vidxflat,cellindex):
    #First we determine the probability of collisions
    coef=np.float32(((2**(4-2*fieldY.getdim()))*(np.pi**(1-fieldY.getdim()/2.0)))/gamma(fieldY.getdim()/2.0))
    gammacoef=(fieldY.gettotalnumber()-np.sum(np.abs(fieldY.fieldc)**2)*(fieldY.getdx()**fieldY.getdim()))/len(fieldY.fieldnc[0])
    g2=(parameters[7][2][0]-parameters[7][3][0]/3.0)**2

    f3=fbinned(parameters,fieldY,idxflat,vidxflat,cellindex,v3)
    f4=fbinned(parameters,fieldY,idxflat,vidxflat,cellindex,v4)

    P2=(coef*gammacoef*g2*dens*dv**(fieldY.getdim()-2)*(1+f3)*(1+f4)*dt).astype(np.float32)
    return np.minimum(P2,np.float32(1.0))

#Function to compute the probability in and out of the collisions between coherent and non-coherent
def probabilitycollisionscnc(parameters,dt,fieldY,mask,v2norm,v,v2,v3out,v4out,v4in,idxflat,vidxflat):
    dim=fieldY.getdim()
    coefout=np.float32(((2**(4-2*dim))*(np.pi**(1-dim/2.0)))/gamma(dim/2.0))
    coefin=np.float32((2**(2-dim))*(np.pi**(1-dim)))
    gammacoef=(fieldY.gettotalnumber()-np.sum(np.abs(fieldY.fieldc)**2)*(fieldY.getdx()**dim))/len(fieldY.fieldnc[0])
    g2=(parameters[7][2][0]-parameters[7][3][0]/3.0)**2
    
    if dim == 1:
        Av=1.0
    else:
        Av=(np.pi**((dim-1)/2) / gamma((dim-1)/2 + 1)) * (parameters[6][2])**(dim-1)
    
    f3out=fbinned(parameters,fieldY,idxflat,vidxflat,idxflat,v3out)
    f4out=fbinned(parameters,fieldY,idxflat,vidxflat,idxflat,v4out)
    f4in=fbinned(parameters,fieldY,idxflat,vidxflat,idxflat,v4in)
    v2norm = np.maximum(v2norm, 1e-8)

    P1out = np.zeros_like(v2norm)
    P1in = np.zeros_like(v2norm)

    P1out[mask] = coefout * gammacoef * g2 * v[mask]**(dim - 2) * (1 + f3out[mask]+f4out[mask]) * dt
    P1in= coefin * Av * gammacoef * g2 * f4in* dt/v2norm
    #P1in[mask]= coefin * Av * gammacoef * g2 * f4in[mask]* dt/v2norm[mask]
    
    P1out = np.minimum(P1out, np.float32(1.0))
    P1in = np.minimum(P1in,np.float32(1.0))

    return P1out, P1in


#Function that defines if collisions between two non-coherent happen
def collisionsnc(parameters,dt,fieldY,hasitcol,ind,sortedidx,idxflat,vidxflat,uncells,startidx,endidx,Npercell):
    pairsarray, elemsincell, cellindex=generatepairs(sortedidx,uncells,startidx,endidx,Npercell)
    if len(pairsarray)==0:
        return hasitcol, np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1]))
        
    dens=elemsincell/(fieldY.getdx()**fieldY.getdim())
    v1 = fieldY.fieldnc[1][pairsarray[:, 0]]
    v2 = fieldY.fieldnc[1][pairsarray[:, 1]]
    dv = np.linalg.norm(v1 - v2, axis=1)

    v3, v4=finalvelocitiesnc(v1,v2,dv,fieldY)

    P2=probabilitycollisionsnc(parameters,dt,dens,dv,v3,v4,fieldY,idxflat,vidxflat,cellindex)
    randomnum=np.random.rand(len(P2))
    collides=randomnum<P2

    collidedpairs=pairsarray[collides]

    mask_valid = (hasitcol[collidedpairs[:, 0]] == 0) & (hasitcol[collidedpairs[:, 1]] == 0)
    valid_pairs = collidedpairs[mask_valid]

    #Changing the status of the objects that collided
    hasitcol[valid_pairs[:, 0]] = 3
    hasitcol[valid_pairs[:, 1]] = 4
    
    return hasitcol, v3[collides][mask_valid], v4[collides][mask_valid]
    
#Function that defines if collisions between coherent and non-coherent happen
def collisionscnc(parameters,dt,fieldY,hasitcol,ind,idxflat,vidxflat):
    if np.all(fieldY.densityc==0):
        shape = fieldY.densityc.shape
        return hasitcol, np.zeros(shape), np.zeros(shape), np.full(len(fieldY.fieldnc[1]), False), np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1])), np.zeros(len(fieldY.fieldnc[1]))
        
    dim=fieldY.getdim()
    g=parameters[7][2][0]-parameters[7][3][0]/3.0
    v2=fieldY.fieldnc[1]
    v2norm = np.linalg.norm(v2, axis=1)
    densc=interpolator(fieldY,fieldY.densityc)

    v = np.zeros(len(v2))
    mask = v2norm**2 > 4 * g * densc
    v[mask] = np.sqrt(v2norm[mask]**2 - 4 * g * densc[mask])
    v[~mask] = v2norm[~mask]

    v3out, v4out=finalvelocitiescncout(v,v2,fieldY)
    v3in, v4in=finalvelocitiescncin(parameters,g,densc,v2,v2norm,fieldY)

    #Computation of the probabilities of collision
    P1out,P1in=probabilitycollisionscnc(parameters,dt,fieldY,mask,v2norm,v,v2,v3out,v4out,v4in,idxflat,vidxflat)
    
    #Sum of both probabilities for use in the noise and Difference of both probabilities for R
    Psum=P1out+P1in
    Pnet=P1out-P1in
    D=computematvar(fieldY,Psum,dt)
    Rdt=computeRdt(fieldY,Pnet)

    #Determination of type of collision
    randomnum=np.random.rand(len(P1out))
    collideout = (randomnum < P1out*densc)
    collidein = (randomnum >= P1out*densc) & (randomnum < (P1out + P1in)*densc)

    collideoutidx = np.where(collideout)[0]
    collideinidx = np.where(collidein)[0]

    hasitcol[collideoutidx] = 1

    #Determination of particles that will be eliminated
    toeliminate = np.full(len(fieldY.fieldnc[1]), False)
    valid_mask_in = np.zeros(len(collideinidx), dtype=bool)
    if collideinidx.size > 0:
        available_particles_map = defaultdict(list)
        available_mask = (hasitcol == 0)
        available_indices = np.where(available_mask)[0]
        available_vel_cells = vidxflat[available_mask]
        for i, cell in enumerate(available_vel_cells):
            particle_idx = available_indices[i]
            available_particles_map[cell].append(particle_idx)
        target_v4_cells = binvelocities(parameters,fieldY,v4in[collideinidx])

        for i, global_idx in enumerate(collideinidx):
            target_cell = target_v4_cells[i]
            if target_cell in available_particles_map and available_particles_map[target_cell]:
                chosen_particle_idx = available_particles_map[target_cell].pop()
                toeliminate[chosen_particle_idx] = True
                valid_mask_in[i] = True
    
    hasitcol[collideinidx[valid_mask_in]] = 2

    return hasitcol, Rdt, D, toeliminate, v3out, v4out, v3in, v4in

#Function that eliminate, create the particles and change the velocities at the end of the collisions.
def collisionresults(parameters,fieldY,totalcollisionsinbin,rbins,hasitcol,toeliminate,v3out,v4out,v3in,v4in,v3,v4):
    if len(fieldY.fieldnc[0][hasitcol==1]) > 0:
        # Count the out collisions 1 by bin
        counts_out, _ = np.histogram(np.linalg.norm(fieldY.fieldnc[0][hasitcol==1], axis=1), bins=rbins)
        totalcollisionsinbin[2] += counts_out
        #Change the velocities of the out process in the collision 1 
        fieldY.fieldnc[1][hasitcol==1]=v3out[hasitcol==1]
    
    if len(fieldY.fieldnc[0][hasitcol==2]) > 0:
        # Count the in collisions 1 by bin
        counts_in, _ = np.histogram(np.linalg.norm(fieldY.fieldnc[0][hasitcol==2], axis=1), bins=rbins)
        totalcollisionsinbin[3] += counts_in
        #Change the velocities of the in process in the collision 1 
        fieldY.fieldnc[1][hasitcol==2]=v3in[hasitcol==2]
  
    if len(fieldY.fieldnc[0][hasitcol==3])>0 and len(fieldY.fieldnc[0][hasitcol==4])>0:
        # Count the collisions 2 by bin
        counts_coll, _ = np.histogram(np.linalg.norm(fieldY.fieldnc[0][hasitcol==3], axis=1), bins=rbins)
        totalcollisionsinbin[1] += counts_coll
        #Change the velocities in the collision 2
        fieldY.fieldnc[1][hasitcol==3]=v3
        fieldY.fieldnc[1][hasitcol==4]=v4

    #delete the particles selected and create the new ones
    if parameters[10][0]==True:
        to_keep = toeliminate==False
        survivor_positions = fieldY.fieldnc[0][to_keep].copy()
        survivor_velocities = fieldY.fieldnc[1][to_keep].copy()
    
        newparticlesposout = fieldY.fieldnc[0][hasitcol==1].copy()
        newparticlesvelout = v4out[hasitcol==1].copy()
    
        allnewpositions = np.vstack([survivor_positions,newparticlesposout])
        allnewvelocities = np.vstack([survivor_velocities,newparticlesvelout])
        fieldY.replacefieldnc(allnewpositions, allnewvelocities)
    
