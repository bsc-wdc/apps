#!/usr/bin/env python3
#
# Abstract factory for accessing storage repositories
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

import ucis4eq
from ucis4eq.misc.factory import Factory 

################################################################################
# Methods and classes

class SDAFactory(Factory):
    
    # Method for choosing best repository
    def selectFrom(self, repositories, repo):
        """
        Select a repository from a set 
        """
 
        # TODO: Add some mechanism for choosing a repository,
        #       load balancing, etc...
        if repo in repositories.keys():
            repository = repo
        elif len(repositories.keys()) == 1:
            repository = list(repositories)[0]
        elif ucis4eq.dal.repository in repositories.keys():
            repository = ucis4eq.dal.repository
        else:
            print("WARNING: No repository found for current file")
            return None, None
                           
        return repository, repositories[repository]

class SDAProvider(SDAFactory):
    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)

class RepositoryABC(ABC):
    """
    The Product interface declares the operations that all concrete repositories
    must implement.
    """
    
    @abstractmethod
    def authenticate(self):
        """
        Authenticates against a repository
        """
        pass

    @abstractmethod
    def mkdir(self, rpath):
        """
        Creates a new directory
        """        
        pass
        
    @abstractmethod
    def tree(self, rpath):
        """
        Obtains the list of files in a remote directory
        """        
        pass        

    @abstractmethod
    def downloadFile(self):
        """
        Downloads a file
        """        
        pass
        
    @abstractmethod
    def uploadFile(self):
        """
        Upload a file
        """        
        pass
        
