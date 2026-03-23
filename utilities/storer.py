import sys
import shutil
import numpy as np
from utilities.additionalfunctions import constants as cts
from utilities.additionalfunctions import parsing as prs

#Function that reads the initial file and pass the parameters to the program
def reader(filepath):
    #List of fundamental parameters needed to work
    fundparameterlist=['boxlength','sitesbylength','totalnumber','coherentfraction']
    #List of initial conditions. Entry 0 in each sub-array contains the possible initial conditions, next entries are additional values needed for such initial condition.
    icclist=[['None'],['testing'],['random'],['from-file','initialfilec'],['centralsoliton','radc'],['gaussianpackets'],['soliton1D','velsol']]
    icnclist=[['None'],['testing'],['random'],['from-file','initialfilenc'],['normal','mupos','sigmapos','muvel','sigmavel'],['thermaltest'],['thermalharmonic','beta'],['nfw','rs','cparameter'],['nbodytest','velgroup']]
    #List of parameters in the potential available. The parameter omega corresponds to the harmonical part and selfcoupling corresponds to the self-self-interaction.
    potentiallist=[['harmonicpotential','omegax','omegay','omegaz','vxcenter','vycenter','vzcenter'],['contactinteraction','selfcoupling']]
    #List of optional flags
    optionalflags=['smoothing','nocollisions']
    
    #Open the initial file. If it doesn't exist it generates an error
    try:
        f=open(f'{filepath}'+"/inifile.dat")
    except FileNotFoundError:
        print('Error opening input file')
    
    #Introduces the lines of the input file in a temporal vector called storertmp deleting empty lines and comments
    storertmp=[line for line in f.readlines() if not line.isspace()]
    storertmp=prs.remove_comments(storertmp)
    f.close()
    
    #create the arrays which will contain the parameters
    fundparameters=[]
    fieldparameters=[]
    potparameters=[]
    evolverparameters=[]
    optparameters=[]

    #Check if the parameters in the fundamental parameter list exist in the initial file and then they will be added to the array fundamentalparameters
    for i in range(0,len(fundparameterlist)):
        element=prs.checkerlist(fundparameterlist[i],storertmp)

        #Box length L, it is a cubic box
        if i==0:
            if cts.FTYPE(element)<=0:
                print(f'The length must be a positive number.')
                break
            else:
                fundparameters.append(cts.FTYPE(element))

        #Number of sites by length (N)
        if i==1:
            if cts.ITYPE(element)<=0:
                print(f'The number of sites per length must be a positive number.')
                break
            else:
                fundparameters.append(cts.ITYPE(element))

        #Total number Ntot in the system
        if i==2:
            if cts.ITYPE(element)<=0:
                print(f'The total number must be a positive number.')
                break
            else:
                fundparameters.append(cts.ITYPE(element))

        #Coherent fraction in the system, initial conditions and potential
        if i==3:
            if cts.FTYPE(element)<0 or cts.FTYPE(element)>1:
                print(f'The coherent fraction must be a number between 0 and 1 (both included).')
                break
            else:
                condfraction=cts.FTYPE(element)
                fundparameters.append(condfraction)
        
    #The path where the objects are saved is stored in the array fundamentalparameters      
    fundparameters.append(filepath)


    #According to the coherent fraction choice it stores the parameters for the field.
    fieldparameters.append([]) #Array that will contain initial conditions for the coherent part
    fieldparameters.append([]) #Array that will contain initial conditions for the incoherent part
    fieldparameters.append([]) #Array that will contain parameters required for the incoherent part
    counterificc=0 #parameter that will turn into 1 at the end of this section when the coherent initial condition in the inifile matches with a name in icclist
    counterificnc=0 #parameter that will turn into 1 at the end of this section when the incoherent initial condition in the inifile matches with a name in icnclist
    indexicc=0
    indexicnc=0
    if fundparameters[3]==1:
        elementicc=prs.checkerlist('initialcondc',storertmp)
        elementicnc='None'
        counterificnc=1
    elif fundparameters[3]==0:
        elementicc='None'
        elementicnc=prs.checkerlist('initialcondnc',storertmp)
        counterificc=1
    else:
        elementicc=prs.checkerlist('initialcondc',storertmp)
        elementicnc=prs.checkerlist('initialcondnc',storertmp)

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
        fieldparameters[0].append(elementicc)
        fieldparameters[1].append(elementicnc)

    #Here we add the parameters of the chosen initial condition
    for l in range(1,len(icclist[indexicc])):
        elementparicc=prs.checkerlist(icclist[indexicc][l],storertmp)
        if elementicc=='from-file':
            fieldparameters[0].append(elementparicc)
        else:
            fieldparameters[0].append(cts.FTYPE(elementparicc))
    for l in range(1,len(icnclist[indexicnc])):
        elementparicnc=prs.checkerlist(icnclist[indexicnc][l],storertmp)
        if elementicnc=='from-file':
            fieldparameters[1].append(elementparicnc)
        else:
            fieldparameters[1].append(cts.FTYPE(elementparicnc))

    if fundparameters[3]<1:
        counterda=0 #parameter that will turn into 1 at the end of this section when the density assignment in the inifile matches with a name in dalist
        #This part stores the number of test particle number
        elementpart=prs.checkerlist('testparticlenumber',storertmp)
        if int(elementpart)<=0:
            print(f'The number of test particles must be a positive number.')
            sys.exit()
        else:
            fieldparameters[2].append(int(elementpart))

        #This part stores the maximum velocity of the particles
        elementpart=prs.checkerlist('maxvel',storertmp)
        if int(elementpart)<=0:
            print(f'The maximum value for the velocity of the incoherent particles must be a positive number.')
            sys.exit()
        else:
            fieldparameters[2].append(cts.FTYPE(elementpart))
                                  
    #Assignation by default if there is no incoherent part
    else:
        fieldparameters[2].append(0)
        fieldparameters[2].append(0)


    #The parameters for the potential are stored in the array potparameters
    for i in range(0,len(potentiallist)):
        potparameters.append([])
        try:
            indicestore = next(k for k, item in enumerate(storertmp) if potentiallist[i][0] in item)
            elementpot=prs.checkerlist(potentiallist[i][0],storertmp)
            if elementpot.lower()=='true':
                potparameters[i].append(True)
                for k in range (1,len(potentiallist[i])):
                    elementpot2=prs.checkerlist(potentiallist[i][k],storertmp)
                    potparameters[i].append(cts.FTYPE(elementpot2))
            elif elementpot.lower()=='false':
                potparameters[i].append(False)
                for k in range (1,len(potentiallist[i])):
                    potparameters[i].append(cts.FTYPE(0))     
            else:
                print(f'Parameter {elementpot} only can be True or False')
                sys.exit() 
        except:
            potparameters[i].append(False)
            for k in range (1,len(potentiallist[i])):
                potparameters[i].append(cts.FTYPE(0))


    #The parameters for the evolver are stored in the array evolverparameters
    element2=prs.checkerlist('iterations',storertmp)
    evolverparameters.append(int(element2))
    element2=prs.checkerlist('recordevery',storertmp)
    evolverparameters.append(int(element2))
    element2=prs.checkerlist('imagprop',storertmp)
    if element2.lower()=='true':
        evolverparameters.append(True)
    elif element2.lower()=='false':
        evolverparameters.append(False)
    else:
        print('Please indicate True if the evolution is in imaginary time or False if the evolution is in real time')
        sys.exit()
    element2=prs.checkerlist('factordt',storertmp)
    evolverparameters.append(int(element2))


    #Additional options stored in the array optparameters.
    for i in range(0,len(optionalflags)):
        optparameters.append([])
        try:
            indicestore = next(k for k, item in enumerate(storertmp) if optionalflags[i] in item)
            optparameters[i].append(True)
            if optionalflags[i]=='smoothing':
                element2=prs.checkerlist('smoothfactor',storertmp)
                if float(element2)>0:
                    optparameters[i].append(cts.FTYPE(element2))
                else:
                    print('The parameter smoothfactor must be a positive number')
                    sys.exit()              
        except:
            optparameters[i].append(False)
    
    
    return fundparameters,fieldparameters,potparameters,evolverparameters,optparameters
    