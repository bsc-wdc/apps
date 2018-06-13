#Imports
    # General Imports
import itertools
    # PyCOMPSs Imports
from pycompss.api.api import compss_wait_on

def dict_compss_wait_on(dicc, dimension_perms):
    for comb in itertools.product(*dimension_perms):
        dicc[comb] = compss_wait_on(dicc[comb])
    return dicc

