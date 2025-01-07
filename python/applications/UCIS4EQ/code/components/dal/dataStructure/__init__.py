"""This is init module."""

################################################################################
# Module imports

from ucis4eq.misc.factory import Provider
from . import martaSalvusData as MartaSalvusFormat

# Define the supported list of repositories
formats = Provider()

# Register each data repository
formats.registerBuilder('Marta_Salvus', MartaSalvusFormat.MartaSalvusData)
