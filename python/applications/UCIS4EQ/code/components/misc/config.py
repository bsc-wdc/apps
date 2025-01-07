#!/usr/bin/env python3
#
# Configure app to connect with database
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
import json
import traceback
import sys
from bson import ObjectId

# Third parties
from flask import jsonify

# Class for convert ObjectId to string        
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

# Method for formatting exceptions
def printException():
    print('#'*80)
    print("Exception in code:")
    traceback.print_exc(file=sys.stdout)
    print('#'*80, flush=True)
    
# Decorator for protect a service execution
def safeRun(func):
    def func_wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)

        except Exception as e:

            printException()

            # Return error code and message
            return jsonify(result = str(e), response = 501)

    return func_wrapper

# Check post request
def checkPostRequest(r):
    if r.json()['response'] == 501:
        raise Exception(r.json()['result'])
        
