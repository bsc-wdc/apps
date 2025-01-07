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
from ucis4eq.dal.staticDataAccess.staticDataAccess import RepositoryABC

################################################################################
# Methods and classes

class LocalRepository(RepositoryABC):
    def __init__(self, location):
        self._location = location

    def authenticate(self):
        pass
    
    def mkdir(self):
        print("Creating directory in Local")
        
    def tree(self):
        print("Obtaining list of files from Local")
        
    def downloadFile(self):
        print("Downloading file from Local")
        pass
        
    def uploadFile(self):
        print("Uploading file to Local")
        pass    
    
class LocalRepositoryBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, localClientKey, localClientSecret, **_ignored):
        if not self._instance:
            accessCode = self.authorize(
                localClientKey, localClientSecret)
            self._instance = LocalRepository(accessCode)
        return self._instance

    def authorize(self, key, secret):
        return 'LOCAL_ACCESS_CODE'
