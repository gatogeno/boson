import numpy as np

CTYPE = np.complex64
FTYPE = np.float32
ITYPE = np.int64
UITYPE = np.uint64
MINVAL  = FTYPE(np.finfo(FTYPE).tiny)
FEPS  = FTYPE(np.finfo(FTYPE).eps)
UIMAX = UITYPE(np.iinfo(UITYPE).max)