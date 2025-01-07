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
from ucis4eq.misc.factory import Provider
from abc import ABC, abstractmethod

################################################################################
# Methods and classes

# Class in charge of dealing with data structure formats
class DataStructureABC(ABC):


    # Method for defining the "getPathTo" required method
    @abstractmethod    
    def prepare(self, region):
        raise NotImplementedError("Error: 'attribute' method should be implemented")

    # Method for defining the "getPathTo" required method
    @abstractmethod    
    def getPathTo(self, attribute):
        raise NotImplementedError("Error: 'attribute' method should be implemented")
