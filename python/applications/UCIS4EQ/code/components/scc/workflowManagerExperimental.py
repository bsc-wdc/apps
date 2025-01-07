#!/usr/bin/env python3
#
# Workflow Manager
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
# along with this program. If not, see <https://www.gnu.org/licenses/>.

################################################################################
# Module imports
# System
import requests
import json

# Third parties
from flask import jsonify
from dask.distributed import Client, progress

# Internal
from ucis4eq.misc import config, microServiceABC


# TODO Remove this
import time
import random

def inc(x):
    time.sleep(random.random())
    return x + 1

def double(x):
    time.sleep(random.random())
    return 2 * x

def add(x, y):
    time.sleep(random.random())
    return x + y

################################################################################
# Methods and classes
        
class DaskWorkflowManager(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize the sourceType component implementation    
        """
        # Set the microServices URL
        self.urls = {}
        
        self.urls["microServices"] = "http://127.0.0.1:5000"
        self.urls["thirdParties"] = "http://127.0.0.1:5002"
        
        # Select the database
        self.db = config.database
        
        # Event ID
        self.eid = None
        
        # Set the Dask client
        self.client = Client()

    # Service's entry point definition
    @config.safeRun
    def entryPoint(self, body):
        """
        Workflow manager based on the Dask scheduler
        """
        # Just a naming convention
        event = body
        
        zs = []
        for i in range(25):
            x = self.client.submit(inc, i)     # x = inc(i)
            y = self.client.submit(double, x)  # y = inc(x)
            z = self.client.submit(add, x, y)  # z = inc(y)
            zs.append(z)
            
        total = self.client.submit(sum, zs)
        
        #print(total, flush=True)
        
        print(total.result(), flushh=True)
                    
        # Obtain the Event Id. (useful during all the workflow livecycle)
        # TODO: This task belong to the branch "Urgent computing" (It runs in parallel)
        
        #r  = self.client.submit(self._request, "http://127.0.0.1:5000", "eventCountry", event)
        
        #foo = self.client.submit(config.checkPostRequest, r)
        
        #event['country'] = r.json()['result']
                            
        # Return list of Id of the newly created item
        return jsonify(result = "Event notified", response = 201)
        #return jsonify(result = "Event with UUID " + str(body['uuid']) + " notified for region " + str(region['id']), response = 201)
        
    # Request wrapper for simplification
    def _request(self, url, service, data):
        """
        Request wrapper for simplification
        """
                
        #return requests.post(url + "/eventCountry", json=data)

         
            
