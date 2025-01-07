#!/usr/bin/env python3
#
# Slurm specialization launcher
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
import os
import time
import subprocess
import threading

# Internal
import ucis4eq.dal as dal
from ucis4eq.dal import staticDataMap
from ucis4eq.dal.dataStructure.dataStructure import DataStructureABC

################################################################################
# Methods and classes
@staticDataMap.build
class MartaSalvusData(DataStructureABC):

    # Initialization method
    def __init__(self):
        # Select the database
        self.format = "Marta_Salvus"
        
        # Obtain format structure
        col = dal.database['DataStructure']
        query = { "id": {"$eq": self.format} }
        self.structure = col.find_one(query, {'_id': False, 'id': False})    
        
        # Obtain directories
        self.structureFileMap = None
        self.pathValidation = False
    
    # Obtain the path to an specific file    
    def prepare(self, region):
        # Obtain regions's files
        self.structureFileMap = self.filePing[region]
        
        if isinstance(self.structureFileMap, list):
            self.pathValidation = True
                    
    # Obtain the path to an specific file
    def getPathTo(self, attribute, selectors=None):
        
        path = None
        
        if not attribute == 'root' and attribute in self.structure.keys():
            attributeName = self.structure[attribute]['name']
            
            if self.pathValidation:
                matches = [match for match in self.structureFileMap if attributeName in match]
                if selectors and isinstance(selectors, list):
                    for selector in selectors:
                        generalMatches = matches
                        matches = [match for match in generalMatches if selector in match]
                        
                if matches:
                    path = matches[-1]
                                    
            else:
                path = self.structure[attribute]['name'] + "/"
            
        # Return the set of instructions
        return path
