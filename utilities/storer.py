import sys
import os
import shutil
import numpy as np
from utilities import additionalfunctions as adf

#Function that reads the initial file and pass the parameters to the program
def reader(filepath):
    #List of parameters needed to work
    parameterlist=['dimension','boxlength','sitesbylength','totalnumber','coherentfraction','evolver','smoothing','collisions1','collisions2']
    #List of initial conditions. Entry 0 in each sub-array contains the possible initial conditions, next entries are additional values needed for such initial condition.
    icclist=[['None'],['testing'],['random'],['from-file','initialfilec'],['centralsoliton','radc'],['gaussianpackets'],['soliton1D','velsol']]
    icnclist=[['None'],['testing'],['random'],['from-file','initialfilenc'],['normal','mupos','sigmapos','muvel','sigmavel'],['thermaltest'],['thermalharmonic','beta'],['zng','beta','mu']]
    #List of incoherent density assignments
    dalist=['nearestgridpoint','cloudincell']
    #List of parameters in the potential available. The parameter omega corresponds to the harmonical part and selfcoupling corresponds to the self-self-interaction.
    potentiallist=[['omegax','omegay','omegaz'],['gravitationalpotential'],['selfcoupling'],['dipolarcoupling','nx','ny','nz']]
    #List of evolvers
    evollist=['kickdrift']
    
    #Open the initial file. If it doesn't exist it generates an error
    try:
        f=open(f'{filepath}'+"/inifile.dat")
    except FileNotFoundError:
        print('Error opening input file')
    
    #Introduces the lines of the input file in a temporal vector called storertmp deleting empty lines and comments
    storertmp=[line for line in f.readlines() if not line.isspace()]
    storertmp=adf.remove_comments(storertmp)
    f.close()
    
    #create the array which will contain the parameters
    parameters=[]
    
    #Check if the parameters in the parameter list exist in the initial file and then they will be added
    for i in range(0,len(parameterlist)):
        element=adf.checkerlist(parameterlist[i],storertmp)
        
        #Check particular conditions in the parameters and it adds the values in the array parameters       
        #Dimension. Only works if D=1,2 or 3
        if i==0:
            if int(element)>3 or int(element)<1:
                print(f'Code not supported for d={int(element)}. Try with dimensions 1, 2 or 3.')
                break
            else:
                parameters.append(int(element))
        
        #Box length L, it is a cubic box
        if i==1:
            if np.float32(element)<=0:
                print(f'The length must be a positive number.')
                break
            else:
                parameters.append(np.float32(element))
        
        #Number of sites by length (N)
        if i==2:
            if int(element)<=0:
                print(f'The number of sites per length must be a positive number.')
                break
            else:
                parameters.append(int(element))
            
        #Total number Ntot in the system
        if i==3:
            if int(element)<=0:
                print(f'The total number must be a positive number.')
                break
            else:
                parameters.append(int(element))

        #Coherent fraction in the system, initial conditions and potential
        if i==4:
            if np.float32(element)<0 or np.float32(element)>1:
                print(f'The coherent fraction must be a number between 0 and 1 (both included).')
                break
            else:
                condfraction=np.float32(element)
                parameters.append(condfraction)
                
                #Now, it adds an array for the initial conditions for coherent and incoherent
                parameters.append([])
                parameters[5].append([]) #Array that will contain initial conditions for the coherent part
                parameters[5].append([]) #Array that will contain initial conditions for the incoherent part
                
                counterificc=0 #parameter that will turn into 1 at the end of this section when the coherent initial condition in the inifile matches with a name in icclist
                counterificnc=0 #parameter that will turn into 1 at the end of this section when the incoherent initial condition in the inifile matches with a name in icnclist
                indexicc=0
                indexicnc=0
                if condfraction==1:
                    elementicc=adf.checkerlist('initialcondc',storertmp)
                    elementicnc='None'
                    counterificnc=1
                elif condfraction==0:
                    elementicc='None'
                    elementicnc=adf.checkerlist('initialcondnc',storertmp)
                    counterificc=1
                else:
                    elementicc=adf.checkerlist('initialcondc',storertmp)
                    elementicnc=adf.checkerlist('initialcondnc',storertmp)

                for j in range(len(icclist)):
                    if elementicc==icclist[j][0]:
                        counterificc=counterificc+1
                        indexicc=j
                for j in range(len(icnclist)):
                    if elementicnc==icnclist[j][0]:
                        counterificnc=counterificnc+1
                        indexicnc=j

                #Action when there is no match between the parameter in inifile and the existent initial conditions
                if counterificc==0:
                    print('Initial condition for coherent part not available. Here is the list of available initial conditions:')
                    for j in range(1,len(icclist)):
                        print(icclist[j][0])
                    sys.exit()
                elif counterificnc==0:
                    print('Initial condition for incoherent part not available. Here is the list of available initial conditions:')
                    for j in range(1,len(icnclist)):
                        print(icnclist[j][0])
                    sys.exit()
                else:
                    parameters[5][0].append(elementicc)
                    parameters[5][1].append(elementicnc)

                #Here we add the parameters of the chosen initial condition
                for l in range(1,len(icclist[indexicc])):
                    elementparicc=adf.checkerlist(icclist[indexicc][l],storertmp)
                    if elementicc=='from-file':
                        parameters[5][0].append(elementparicc)
                    else:
                        parameters[5][0].append(np.float32(elementparicc))
                for l in range(1,len(icnclist[indexicnc])):
                    elementparicnc=adf.checkerlist(icnclist[indexicnc][l],storertmp)
                    if elementicnc=='from-file':
                        parameters[5][1].append(elementparicnc)
                    else:
                        parameters[5][1].append(np.float32(elementparicnc))

                #Now, it stores the number of test particles and the density assignment for the incoherent part
                parameters.append([])
                if condfraction<1:
                    counterda=0 #parameter that will turn into 1 at the end of this section when the density assignment in the inifile matches with a name in dalist
                    #This part stores the number of test particle number
                    elementpart=adf.checkerlist('testparticlenumber',storertmp)
                    if int(elementpart)<=0:
                        print(f'The number of test particles must be a positive number.')
                        sys.exit()
                    else:
                        parameters[6].append(int(elementpart))

                    #This part stores the name of the density assignment for the incoherent part
                    elementpart=adf.checkerlist('densityassignment',storertmp)
                    for r in range(len(dalist)):
                        if elementpart==dalist[r]:
                            counterda=counterda+1
                    if counterda==0:
                        print(f'Density assignment for incoherent part not available. Here is the list of available density asignments:')
                        for l in range(len(dalist)):
                            print(dalist[l])
                        sys.exit()
                    else:
                        parameters[6].append(elementpart)

                    #This part stores the maximum velocity of the particles
                    elementpart=adf.checkerlist('maxvel',storertmp)
                    if int(elementpart)<=0:
                        print(f'The maximum value for the velocity of the incoherent particles must be a positive number.')
                        sys.exit()
                    else:
                        parameters[6].append(np.float32(elementpart))
                    
                
                #Assignation by default if there is no incoherent part
                else:
                    parameters[6].append(0)
                    parameters[6].append('None')
                    parameters[6].append(0)
                     
                #Potential
                parameters.append([])
                #Here we add the parameters of the potential
                #first we add the harmonical part
                parameters[7].append([])
                for k in range(parameters[0]):
                    elementpot=adf.checkerlist(potentiallist[0][k],storertmp)
                    parameters[7][0].append(np.float32(elementpot))
                #then we add the gravitational potential
                parameters[7].append([])
                elementpot=adf.checkerlist(potentiallist[1][0],storertmp)
                if elementpot.lower()=='true':
                    parameters[7][1].append(True)
                elif elementpot.lower()=='false':
                    parameters[7][1].append(False)
                else:
                    print(f'Parameter {elementpot} only can be True or False')
                    sys.exit() 
                #then we add the self interaction and dipolar cases
                for k in range (2,len(potentiallist)):
                    parameters[7].append([])
                    elementpot=adf.checkerlist(potentiallist[k][0],storertmp)
                    parameters[7][k].append(np.float32(elementpot))
                #It adds the direction vector n components if there is dipolar interaction
                for k in range(parameters[0]):
                    if parameters[7][3][0]!=0:
                        element2=adf.checkerlist(potentiallist[3][k],storertmp)
                        parameters[7][3].append(np.float32(element2))
                    else:
                        parameters[7][3].append(np.float32(0))
                                    
                        
        #Evolver
        if i==5:
            parameters.append([])
            counterevol=0
            for j in range(0,len(evollist)):
                if element==evollist[j]:
                    counterevol=counterevol+1
            if counterevol==0:
                print('Evolver not available. Here is the list of available evolvers:')
                for j in range(0,len(evollist)):
                    print(evollist[j][0])
                sys.exit()
            else:
                parameters[8].append(element)
                element2=adf.checkerlist('iterations',storertmp)
                parameters[8].append(int(element2))
                element2=adf.checkerlist('recordevery',storertmp)
                parameters[8].append(int(element2))
                if element==evollist[0]:
                    element3=adf.checkerlist('imagprop',storertmp)
                    if element3.lower()=='true':
                        parameters[8].append(True)
                    elif element3.lower()=='false':
                        parameters[8].append(False)
                    else:
                        print('Please indicate True if the evolution is in imaginary time or False if the evolution is in real time')
                        sys.exit()
                    isExist=os.path.exists(filepath+'/evolution')
                    if not isExist:
                        os.makedirs(filepath+'/evolution')
                element2=adf.checkerlist('factordt',storertmp)
                parameters[8].append(int(element2))                

        #Smoothing of the densities
        if i==6:
            parameters.append([])
            if element.lower()=='true':
                parameters[9].append(True)
                element2=adf.checkerlist('smoothfactor',storertmp)
                if float(element2)>0:
                    parameters[9].append(np.float32(element2))
                else:
                    print('The parameter smoothfactor must be a positive number')
                    sys.exit()
            elif element.lower()=='false':
                parameters[9].append(False)
            else:
                print(f'Parameter {element} only can be True or False')
                sys.exit()

        #Allowing collisions
        if i==7:
            parameters.append([])
            if element.lower()=='true':
                parameters[10].append(True)
            elif element.lower()=='false':
                parameters[10].append(False)
            else:
                print(f'Parameter {element} only can be True or False')
                sys.exit()
        if i==8:
            if element.lower()=='true':
                parameters[10].append(True)
            elif element.lower()=='false':
                parameters[10].append(False)
            else:
                print(f'Parameter {element} only can be True or False')
                sys.exit()
            
    parameters.append(filepath)
    
    
    return parameters