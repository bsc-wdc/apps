#!/usr/bin/env python3
#
# Abstract Base Class for Launchers
# This module is part of the Smart Center Control (SSC) solution
#
# Author:  Juan Esteban Rodríguez, Josep de la Puente
# Contact: juan.rodriguez@bsc.es, josep.delapuente@bsc.es
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

################################################################################
# Module imports
#from __future__ import annotations
from abc import ABC, abstractmethod

################################################################################
# Methods and classes

# Class in charge of running a process either Slurm or bash 
class RunnerABC(ABC):

    # Some attributes
    type = "sh"

    # Initialization method
    def __init__(self, task):
        # Task name
        self.task = task
        self.setup = None

    # Method for defining the "getRules" required method
    @abstractmethod    
    def getRules(self, stage):
        raise NotImplementedError("Error: 'getRules' method should be implemented")

    # Method for obtaining the spefific MPI command (srun, mpirun, etc...)           
    @abstractmethod    
    def getMPICommand(self):
        raise NotImplementedError("Error: 'getMPICommand' method should be implemented")
        
    # Method for obtaining environment setup (module loads, PATH, conda environmnet, etc ...)
    @abstractmethod    
    def getEnvironmentSetup(self, stage):
        raise NotImplementedError("Error: 'getMPICommand' method should be implemented")

    # Method for defining the "run" required method
    def run(self):
        raise NotImplementedError("Error: 'run' method should be implemented")


        
        
    
