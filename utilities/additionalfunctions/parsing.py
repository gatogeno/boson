import sys
import numpy as np

#Function that parse the arguments in main
def parseargs(argv):
    path = None
    for i, arg in enumerate(argv):
        if arg == '-path':
            path = argv[i + 1]
    if path is None:
        raise ValueError("-path argument is required.")
    return path

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

#Function that checks if some parameter is in a list of parameters called container
def checkerlist(param, container):
    try:
        indicestore = next(i for i, item in enumerate(container) if param in item)
        element = container[indicestore].partition("=")[2].replace(" ", "").strip("\n")
    except StopIteration:
        print(f'Parameter {param} is missing. Please enter this parameter in the initial file')
        sys.exit()
 
    return element
        