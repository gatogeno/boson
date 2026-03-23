#Load the modules needed for the program
import time
import numpy as np
import sys
import os
from utilities import storer, grid, fields, potentials, evolvers
from utilities.additionalfunctions import parsing


def main():
    #Read the argument -file name
    args = sys.argv[1:]
    path=parsing.parseargs(args)
    
    #Initialize the counting of time execution
    start_time = time.time()

    #Creates the directory where it will save the files of evolution if it is not created
    isExist=os.path.exists(path+'/evolution')
    if not isExist:
        os.makedirs(path+'/evolution')

    #Load parameters
    fundparameters,fieldparameters,potparameters,evolverparameters,optparameters=storer.reader(path)

    #Create the grids
    grids=grid.creategrid(fundparameters)

    #Create the initial configuration of the fields and generate the potential for the coherent part
    psi=fields.boson(fundparameters, fieldparameters, potparameters, optparameters, grids)
    pot=potentials.potential(fundparameters, potparameters, grids, psi)

    #Evolve the system
    evolvers.evolver(fundparameters,fieldparameters,evolverparameters,optparameters,grids,psi,pot)
    
    end_time = time.time()
    with open(f'{path}/timeelapsed.dat', "w") as time_file:
        time_file.write("Time elapsed: %s seconds" % (end_time - start_time))
    print("Time elapsed: %s seconds" % (end_time - start_time))
    

if __name__ == "__main__":
    main()