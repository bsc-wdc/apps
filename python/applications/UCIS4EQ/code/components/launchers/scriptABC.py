#!/usr/bin/env python3
#
# RESTful server for event treatment
# This module is part of the Automatic Alert System (AAS) solution
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
import os
import time
import subprocess
import threading
from abc import ABC, abstractmethod

import ucis4eq.launchers as launchers

################################################################################
# Methods and classes
    
# Abstract class for defining a common interface between all workflow stages
class ScriptABC(ABC):

    #@property
    #@classmethod
    #@abstractmethod

    # Initialization method
    def __init__(self, type='LOCAL'):
        self.lines = []
        self.type = type
        self.time = 1
        
        # Create the specific launcher 
        self.launcher = launchers.launchers.create(type, **launchers.config)
        
        # Obtain the script's filename
        self.fileName = self._className() + "." + self.type + "." + self.launcher.type        
        
    # Method for defining the "buildScript" required method
    @abstractmethod        
    def build(self):
        raise NotImplementedError("Error: 'build' method should be implemented")

    # Method for defining the "buildScript" required method
    def getMPICommand(self):
        return self.launcher.getMPICommand()

    # Just a class method for obtaining the class name
    @classmethod
    def _className(cls):
        return(cls.__name__)

    # Obtain common script header 
    def _getHeader(self):
        self.lines.append("#!/bin/bash")
        self.lines.append("")

    # Build specific slurm rules
    def _getRules(self, stage):

        self.lines = self.lines \
                     + self.launcher.getRules(stage) \
                     + self.launcher.getEnvironmentSetup(stage)
                     
        self.lines.append("")                                                                                      

    # Generate script
    def _saveScript(self, path=""):

        # Generate the specific script
        file = path + self.fileName
        with open(file, 'w') as f:
            for item in self.lines:
                f.write("%s\n" % item)

        # Assign execution permisions to the script
        os.chmod(file, 0o777)
        
        return file

    # Method in charge of run and wait the script
    def run(self, path):

        # Build the script filename 
        script = path + '/' + self.fileName

        # Launch and wait for results
        self.launcher.run(path, script)
