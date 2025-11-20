#Load the modules needed for the program
import time
import gc
import numpy as np
import sys
from utilities import storer, grid, fields, potentials, evolvers, collisions

def main():
    #Read the argument -file name
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-path':
        path=args[1]
    
    #Initialize the counting of time execution
    start_time = time.time()

    #Load parameters
    parameters=storer.reader(path)

    #Create the grids
    grids=grid.creategrid(parameters)

    #Create the initial configuration of the fields and generate the potentials
    psi=fields.boson(parameters, grids)
    harmpot=potentials.harmonicpotential(parameters, grids)
    potc=potentials.potential(parameters,grids,harmpot,psi, coherent=True)
    potnc=potentials.potential(parameters,grids,harmpot,psi, coherent=False)

    #Evolve the system
    evolvedarray=evolvers.kickdrift(parameters,grids,psi,harmpot,potc,potnc)
    
    end_time = time.time()
    with open(f'{path}/timeelapsed.dat', "w") as time_file:
        time_file.write("Time elapsed: %s seconds" % (end_time - start_time))
    print("Time elapsed: %s seconds" % (end_time - start_time))

    #Delete objects to release the memory at the end
    del evolvedarray
    del psi
    del harmpot
    del potc
    del potnc
    del grids
    del parameters
    gc.collect()
    

if __name__ == "__main__":
    main()