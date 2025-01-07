#!/usr/bin/env python3
#
# General abstract factory 
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
class Factory():

    def __init__(self):
        """
        Factory initialization
        """
        self._builders = {}
        
    # Method for registering builders    
    def registerBuilder(self, key, builder):
        #print("Registering builder for '" + key + "' builder")
        self._builders[key] = builder
            
    # Method for choosing the best builder
    def selectFrom(self, options):
        """
        Select a builder from a set 
        """
        
        if not options.keys():
            raise Exception('There are no builder to select')
        else:
            name = list(options)[0]
            
        return name, options[name]
        
    # Method to obtain an specific builder
    def __getitem__(self, key):
        return self._builders[key]
            
    # Method for look and returning the correct builder for a concrete instance
    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        
        if not builder:
            raise ValueError(key)
            
        return builder(**kwargs)

class Provider(Factory):
    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)  
