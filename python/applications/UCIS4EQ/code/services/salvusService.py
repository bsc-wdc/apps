#!/usr/bin/env python3
#
# RESTful server for event treatment
# This module is part of the Automatic Alert System (AAS) solution
#
# Author: Juan Esteban Rodríguez
# Contributor: Josep de la Puente <josep.delapuente@bsc.es>
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

from flask import Flask, request, jsonify

# Load Salvus service implemented components
from ucis4eq.scc.salvus import SalvusPrepare, SalvusRun, SalvusPost,\
    SalvusPlots, SalvusPing, SalvusPostSwarm

# ###############################################################################
# Dispatcher App creation
# ###############################################################################
salvusServiceApp = Flask(__name__)


def postRequest(fn):
    # POST request decorator

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
            # MPC: add info message for debugging purposes
            print("Error in JSON request. Received the following JSON: " + str(request.get_json()))
            traceback.print_exc()
            return "Error in JSON request format " + str(request.get_json()), 400

        # Call the decorated method passing it the input JSON
        return fn(body)

    return wrapped


@salvusServiceApp.route("/")
def get_initial_response():
    # Base roor of the micro-services Hub
    """Welcome message for the API."""
    # Message to the user
    message = {
        'apiVersion': 'v1.0',
        'status': '200',
        'message': 'Welcome to the UCIS4EQ Salvus service for PD1'
    }
    # Making the message looks good
    resp = jsonify(message)
    # Returning the object
    return resp

# ###############################################################################
# Services definition
# ###############################################################################


# Generate input parameters for Salvus
@salvusServiceApp.route("/SalvusPrepare", methods=['POST'])
@postRequest
def SalvusPrepareService(body):
    """
    Call component implementing this service
    """

    return SalvusPrepare().entryPoint(body)


@salvusServiceApp.route("/SalvusRun", methods=['POST'])
@postRequest
def SalvusRunService(body):
    # Call Salvus for a trial
    """
    Call component implementing this service
    """

    return SalvusRun().entryPoint(body)


# Call Salvus post-processing
@salvusServiceApp.route("/SalvusPost", methods=['POST'])
@postRequest
def SalvusPostService(body):
    """
    Call component implementing this service
    """
    return SalvusPost().entryPoint(body)


# Call Salvus post-processing
@salvusServiceApp.route("/SalvusPostSwarm", methods=['POST'])
@postRequest
def SalvusPostSwarmService(body):
    """
    Call component implementing this service
    """
    return SalvusPostSwarm().entryPoint(body)


# Call Salvus post-processing
@salvusServiceApp.route("/SalvusPlots", methods=['POST'])
@postRequest
def SalvusPlotsService(body):
    """
    Call component implementing this service
    """
    return SalvusPlots().entryPoint(body)


# Call Salvus post-processing
@salvusServiceApp.route("/SalvusPing", methods=['POST'])
@postRequest
def SalvusPingService(body):
    """
    Call component implementing this service
    """
    return SalvusPing().entryPoint(body)

# ###############################################################################
# Start the micro-services aplication
# ###############################################################################

if __name__ == '__main__':
    # Running app in debug mode
    salvusServiceApp.run(host="0.0.0.0", debug=True, port=5003)
