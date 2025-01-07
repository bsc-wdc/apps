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

# ###############################################################################

import sys
import json
import ast
import traceback
from functools import wraps

# Flask (WSGI) utils
from flask import Flask, request, jsonify

# Load micro-services implemented components
from ucis4eq.scc.workflowManager import PyCommsWorkflowManager
from ucis4eq.scc.workflowManager_baseline import WorkflowManagerEmulator
#from ucis4eq.scc.workflowManagerExperimental import DaskWorkflowManager

# ###############################################################################
# Dispatcher App creation
# ###############################################################################
workflowManagerServiceApp = Flask(__name__)


# POST request decorator
def postRequest(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):

        """
        Function to dispatch new events.
        """
        # Create new events
        try:
            #print("INFO: Received JSON: " + str(request.get_json()))
            body = ast.literal_eval(json.dumps(request.get_json()))
        except:
            # Bad request as request body is not available
            # MPC printing information for debugging
            print("Error in JSON request. Received the following JSON: " + str(request.get_json()))
            traceback.print_exc()
            return "Error in JSON request format " + str(request.get_json()), 400

        # Call the decorated method passing it the input JSON
        return fn(body)

    return wrapped


# Base root of the micro-services Hub
@workflowManagerServiceApp.route("/")
def get_initial_response():
    """Welcome message for the API."""
    # Message to the user
    message = {
        'apiVersion': 'v1.0',
        'status': '200',
        'message': 'Welcome to the UCIS4EQ Workflow Manager service for PD1'
    }
    # Making the message looks good
    resp = jsonify(message)
    # Returning the object
    return resp

# ###############################################################################
# Services definition
# ###############################################################################


# Determine the kind of source for the simulation
@workflowManagerServiceApp.route("/WMEmulator", methods=['POST'])
@postRequest
def workflowManagerEmulatorService(body):
    """
    Call component implementing this micro service
    """
    return WorkflowManagerEmulator().entryPoint(body)


# Determine the kind of source for the simulation
@workflowManagerServiceApp.route("/PyCOMPSsWM", methods=['POST'])
@postRequest
def PyCommsWorkflowManagerService(body):
    """
    Call component implementing this micro service
    """
    return PyCommsWorkflowManager().entryPoint(body)


# ###############################################################################
# Start the micro-services aplication
# ###############################################################################

if __name__ == '__main__':
    # Running app in debug mode
    workflowManagerServiceApp.run(host="0.0.0.0", debug=False, port=5001)
