#!/usr/bin/env python3
#
# RESTful server for event treatment
# This module is part of the Automatic Alert System (AAS) solution
#
# Author:  Juan Esteban Rodríguez
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

import sys, os, json
import ast
import traceback
from functools import wraps

# Flask (WSGI) utils
from flask import Flask, request, jsonify

# Load micro-services implemented components
from ucis4eq.misc import config
from ucis4eq.scc.event import EventRegistration, EventRegion, EventCountry, EventSetState, EventSetup
from ucis4eq.scc.CMTCalculation import CMTCalculation, CMTInputs, CMTSeisEnsMan
from ucis4eq.scc.sourceAssesment import SourceType, PunctualSource
from ucis4eq.scc.inputBuilder import InputParametersBuilder
from ucis4eq.scc.indexPriority import IndexPriority
from ucis4eq.scc.resources import ComputeResources
import ucis4eq
import ucis4eq.dal as dal
from ucis4eq.dal import dynamicDataAccess

# ###############################################################################
# Dispatcher App creation
# ###############################################################################
microServicesApp = Flask(__name__)


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
        except Exception as e:
            # Bad request as request body is not available
            # MPC printing information for debugging
            print("Data:" +str(request.data))
            print(e)
            import traceback
            traceback.format_exc()
            return str(e), 400

        # Call the decorated method passing it the input JSON
        return fn(body)

    return wrapped


# Task to run before first request
@microServicesApp.before_first_request
def initializeDALData():
    dynamicDataAccess.DAL().entryPoint()


# Base root of the micro-services Hub
@microServicesApp.route("/")
def get_initial_response():
    """Welcome message for the API."""
    # Message to the user
    message = {
        'apiVersion': 'v1.0',
        'status': '200',
        'message': 'Welcome to the UCIS4EQ Micro-Services Hub for PD1'
    }
    # Making the message looks good
    resp = jsonify(message)
    # Returning the object
    return resp

# ###############################################################################
# Services definition
# ###############################################################################


# Initialize DAL Data
@microServicesApp.route("/dal", methods=['POST'])
@postRequest
def initializeDALService(body):
    """
    Call component implementing this micro service
    """
    return dynamicDataAccess.DAL().entryPoint()


# CMT Input generation
@microServicesApp.route("/precmt", methods=['POST'])
@postRequest
def CMTInputsService(body):
    """
    Call component implementing this micro service
    """
    return CMTInputs().entryPoint(body)


# CMT Aproximation
@microServicesApp.route("/cmt", methods=['POST'])
@postRequest
def CMTCalculationService(body):
    """
    Call component implementing this micro service
    """
    return CMTCalculation().entryPoint(body)


# CMT SeisEnsMan
@microServicesApp.route("/cmtSeisEnsMan", methods=['POST'])
@postRequest
def CMTSeisEnsManService(body):
    """
    Call component implementing this micro service
    """

    return CMTSeisEnsMan().entryPoint(body)


# Index Priority
@microServicesApp.route("/indexPriority", methods=['POST'])
@postRequest
def indexPriorityService(body):
    """
    Call component implementing this micro service
    """
    return IndexPriority().entryPoint(body)


# Determine the kind of source for the simulation
@microServicesApp.route("/sourceType", methods=['POST'])
@postRequest
def sourceTypeService(body):
    """
    Call component implementing this micro service
    """
    return SourceType().entryPoint(body)


# Calculate a punctual source for an event
@microServicesApp.route("/punctualSource", methods=['POST'])
@postRequest
def punctualSourceService(body):
    """
    Call component implementing this micro service
    """
    return PunctualSource().entryPoint(body)


# Calculate the punctual source for an event
@microServicesApp.route("/inputParametersBuilder", methods=['POST'])
@postRequest
def YAMLBuilderService(body):
    """
    Call component implementing this micro service
    """
    return InputParametersBuilder().entryPoint(body)


# Incomming event registration
@microServicesApp.route("/eventRegistration", methods=['POST'])
@postRequest
def eventDispatcherService(body):
    """
    Call component implementing this micro service
    """
    return EventRegistration().entryPoint(body)


# Incomming event registration
@microServicesApp.route("/eventSetState", methods=['POST'])
@postRequest
def eventSetStateService(body):
    """
    Call component implementing this micro service
    """
    return EventSetState().entryPoint(body)


# Event domains detection
@microServicesApp.route("/eventRegion", methods=['POST'])
@postRequest
def eventRegionService(body):
    """
    Call component implementing this micro service
    """
    return EventRegion().entryPoint(body)


# Event domains detection
@microServicesApp.route("/eventSetup", methods=['POST'])
@postRequest
def eventSetupService(body):
    """
    Call component implementing this micro service
    """
    return EventSetup().entryPoint(body)


# Event region detection
@microServicesApp.route("/eventCountry", methods=['POST'])
@postRequest
def eventCountryService(body):
    """
    Call component implementing this micro service
    """
    return EventCountry().entryPoint(body)


# Computing resources service
@microServicesApp.route("/computeResources", methods=['POST'])
@postRequest
def computeResourcesService(body):
    """
    Call component implementing this micro service
    """
    return ComputeResources().entryPoint(body)

# ###############################################################################
# Start the micro-services aplication
# ###############################################################################

if __name__ == '__main__':
    # Running app in debug mode
    microServicesApp.run(host="0.0.0.0", debug=True)
