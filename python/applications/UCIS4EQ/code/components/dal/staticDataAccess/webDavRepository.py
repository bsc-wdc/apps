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

# Load NextCloud API
from webdav3.client import Client

################################################################################
# Methods and classes

class WebDavRepository(RepositoryABC):
    def __init__(self, user, passwd, url):
        
        # Store the username
        self.user = user
            
        # Create a client
        options = {
         'webdav_hostname': url,
         'webdav_login':    user,
         'webdav_password': passwd,
         'verbose': True
        }
        self._client = Client(options)
            
    def authenticate(self):
        pass
    
    def mkdir(self, rpath):
        # Create remote folder
        self._client.mkdir(rpath)
        
    def tree(self, rpath):
        # List files from a remote path
        print("WARNING: Obtaining list of files from WebDavRepository is unsuported yet")
        return None

    def downloadFile(self, remote, local):
        # Download file
        print("Downloading webdav " + str(remote) + " to " + str(local) )
        self._client.download_sync(remote_path=remote, local_path=local)
        
    def uploadFile(self, remote, local):
        # Upload file
        print(f"Uploading '{local}' to '{remote}'")
        self._client.upload_sync(remote_path=remote, local_path=local)
        
    def __del__(self):
        pass
        # TODO: Figure out why the following line is failing
        #free_size = self._client.free()
    
class BSCB2DROPRepositoryBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, BSC_B2DROP, **_ignored):
        user = BSC_B2DROP["user"]
        passw = BSC_B2DROP["pass"]
        url = BSC_B2DROP["url"]
        
        if not self._instance:
            accessCode = self.authorize(
                user, passw)
            self._instance = WebDavRepository(user, passw, url)
        return self._instance

    def authorize(self, key, secret):
        return 'B2DROP_ACCESS_CODE'
