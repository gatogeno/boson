import numpy as np
from utilities.additionalfunctions import constants as cts

def randomc(grid_shape):
    field=(np.random.random(grid_shape)+1.0j*np.random.random(grid_shape)).astype(cts.CTYPE)

    return field