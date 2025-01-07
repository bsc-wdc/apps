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
import uuid
import concurrent.futures
import time
import os

# Third parties
from flask import jsonify

from pycompss.api.parameter import *
from pycompss.api.http import http
from pycompss.api.api import compss_wait_on, compss_barrier, compss_cancel_group, TaskGroup
from pycompss.api.task import task
from pycompss.api.on_failure import on_failure
from pycompss.api.compss import compss
from pycompss.api.constraint import constraint
from pycompss.streams.distro_stream import ObjectDistroStream

from pycompss.util.serialization import serializer
serializer.FORCED_SERIALIZER = 4

# Internal
import ucis4eq
from ucis4eq.misc import config, microServiceABC
from ucis4eq.stream.update_information import process_update, EventUpdate

################################################################################
# Methods and classes
HPC_RUN_PYCOMPSS=os.environ.get("HPC_RUN_PYCOMPSS", "False").lower()

class PyCommsWorkflowManager(microServiceABC.WMServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize -the workflow manager
        """
        super().__init__()

    # Service's entry point definition
    @config.safeRun
    def entryPoint(self, body):
        """
        PyCOMPSs workflow manager
        """

        print("__ Running PyCOMPSs workflow __")
        event = body
        #print(body)

        # Set event's name
        basename = "event_" + event['uuid']
    
        # Obtain the Event Id. (useful during all the workflow livecycle)
        eid = register_event(event)

        ensemble = 'statisticalCMT'
        if 'ensemble' in body:
            ensemble = body['ensemble']

        # Obtain the region where the event occured
        region = get_region(eid)

        # Obtain the setup depending on the incoming event
        setup = get_setup(eid,ensemble)

        # Wait for future to check if continue or abort
        region = compss_wait_on(region)
        eid = compss_wait_on(eid)

        if not region:
            _ = set_event_state(eid, "REJECTED")

            # MPC: improve error message about which EQ is not simulated
            # MPC: try to always print info from IRIS for consistency unless not available

            IRIS_idx = next((index for (index, d) in enumerate(event['alerts']) if d["agency"] == "IRIS"), None)

            if IRIS_idx:
                print_info = event['alerts'][IRIS_idx]
            else:
                print_info = event['alerts'][0]

            error_message = f" \n WARNING: There is not enough information for simulating the following event: \n " \
                             f"Time: {print_info['time']}, \n " \
                             f"Description: {print_info['description']}, \n " \
                             f"Latitude: {print_info['latitude']:.1f}, \n " \
                             f"Longitude: {print_info['longitude']:.1f}, \n " \
                             f"Magnitude: {print_info['magnitude']} \n"

            return error_message, 500

            
        # Calculate computational resources for the given domain
        resources = compute_resources(eid, region)
        # Obtain rupture generator's setup
        gpsetup = graves_pitarka_setup(eid, region, setup)
        # Wait for GP Setup
        gpsetup = compss_wait_on(gpsetup)
        resources = compss_wait_on(resources)
        if gpsetup == None:
            _ = set_event_state(eid, "REJECTED")
            error_message = f" \n ERROR: When generating Graves-Pitarka setup for the event. \n " 
            return error_message, 500
        
        if ensemble == "seisEnsMan":
            print("__ Reading SeisEnsMan data __")            	
        	# Read the json that has the outputs of SeisEnsMan
            inputsSeisEnsMan = input_seisensman(eid, region, resources, setup)
            inputsSeisEnsMan = compss_wait_on(inputsSeisEnsMan) 
            cmts = convert_to_cmts(inputsSeisEnsMan)
            # Compute and launch a simulation per each input given in the SeisEnsMan json
            all_results = []
            alert_num = 0
            for alert in event['alerts']:
                monitor_event(alert_num, cmts, ods)
                # For each calculated or provided CMT
                for cmt in cmts.keys():
                    # For each GP defined trial launch a simulation
                    with TaskGroup(gen_group_name(alert_num, str(cmt)), False):
                        # Call Graves-Pitarka's rupture generator for seisensman
                        all_results.append(self.launch_simulation(eid, alert, basename, cmts[cmt], region, gpsetup, resources, ensemble, cmt, slip, alert_num))
                alert_num = alert_num + 1
        else: 		   
	        # Calculate the CMT input parameters
            precmt = build_cmt_input(eid, region, resources, setup)	
        	
            # Compute alerts
            all_results = []
            alert_num = 0
            ods = ObjectDistroStream(alias="updates")
            for alert in event['alerts']:	
                # Calculating CMTs
                cmts = calculate_cmt(alert, eid, region, precmt)
                # Wait for calculated CMTs
                cmts = compss_wait_on(cmts)	
                # Monitor new events
                monitor_event(alert_num, cmts, ods)
                # For each calculated or provided CMT
                for cmt in cmts.keys():
                    # For each GP defined trial launch a simulation
                    with TaskGroup(gen_group_name(alert_num, str(cmt)) , False):
                        for slip in range(1, gpsetup['trials']+1):
                            all_results.append(self.launch_simulation(eid, alert, basename, cmts[cmt], region, gpsetup, resources, ensemble, cmt, slip, alert_num))
                alert_num = alert_num + 1
            # Cancel or launch new simulation according to event updates
            self.update_simulations(ods, eid, event['alerts'], basename, region, gpsetup, resources, ensemble, all_results)
        result = self.launch_post_swarm(eid, region, basename, resources, all_results)

        # Set the event with SUCCESS state  
        eid = set_event_state(eid, "SUCCESS")

        # Wait for the workflow to finish
        compss_barrier(no_more_tasks=True)

        # Return list of Id of the newly created item
        return jsonify(result = "Event with UUID " + str(body['uuid']), response = 201)
    

    def update_simulations(self, ods, eid, alerts, basename, region, gpsetup, resources, ensemble, all_results):
       print("******** Checking Updates ************")
       print("**************************************")
       while not ods.is_closed():
           alert_updates = ods.poll(timeout=10000)
           print(" * Checking updates ")
           for update in alert_updates:
               if (update.type == 'NEW'):
                   print("* LAUNCHING NEW simulation for " + update.source)
                   with TaskGroup(gen_group_name(update.alert, update.source), False):
                       for slip in range(1, gpsetup['trials']+1):	
                           all_results.append(self.launch_simulation(eid, alerts[update.alert], basename, update.cmt, region, gpsetup, resources, ensemble, update.source, slip, update.alert))
               elif (update.type == 'CANCEL'):
                   print("* CANCELLING simulation for "+ update.source)
                   compss_cancel_group(gen_group_name(update.alert, update.source))
                   self.register_simulation(eid + "_alert_" + str(update.alert) + "_" + update.source, eid, resources, basename + "/trial_" + ".".join([update.source, "slip"+str(slip)]), "CANCELED")
               else:
                   print("*** WARNING: Undefined update ***")
       time.sleep(5) 
       print("******** Stream is closed ************")
       print("**************************************")
    
    
    def launch_simulation(self, eid, alert, basename, data, region, gpsetup, resources, ensemble, source, slip, alert_num):
       path = basename + "/trial_" + ".".join([source, "slip"+str(slip)])
       if HPC_RUN_PYCOMPSS == 'true':
           rupture = '"'+ path + '/scratch/outdata/rupture/rupture.srf"'
           # Call input parameters builder
           inputs = build_input_parameters( eid, alert, data, rupture, region, resources, gpsetup, ensemble)
           # Call prepare
           path = basename + "/trial_" + ".".join([source, "slip"+str(slip)])
           outs = prepare_simulation(eid, alert, path, data, region, gpsetup, inputs, resources, ensemble)
           result = run_simulation(*outs)
           self.register_simulation(eid + "_alert_" + str(alert_num) + "_" + source, eid, resources, path, "RUNNING")
       else:
           # Call Graves-Pitarka's rupture generator
           rupture = compute_graves_pitarka(eid, alert, path, data, region, gpsetup, resources, ensemble)
           # Call input parameters builder
           inputs = build_input_parameters(eid, alert, data, rupture, region, resources, gpsetup, ensemble)
           # Build the Salvus input parameter file (remotely)
           salvus_inputs = build_salvus_parameters( eid, path, inputs, resources)
           # Build the Salvus input parameter file (remotely)
           salvus_result = run_salvus( eid, path, salvus_inputs,resources)
           # Call Salvus post
           result = run_salvus_post(eid, salvus_result, path, resources)
       return result
    
    def launch_post_swarm(self, eid, region, basename, resources, all_results):
       if HPC_RUN_PYCOMPSS == 'true':
           result = post_simulation(eid, region, basename, resources, all_results)
           result = compss_wait_on(result)
           self.success_running(eid)
       else:
           # Call postprocessing swarm
           output_swarm = run_salvus_post_swarm(eid, basename, resources)
    
           # General post-processing for generating plots
           result = run_salvus_plots(eid, output_swarm, region, basename, resources)
           result = compss_wait_on(result)
       return result

def convert_to_cmts(inputsSeisEnsMan):
    cmts={}
    for i in range(len(inputsSeisEnsMan)):
        source = "seisEnsMan"+str(i)
        data_seisEnsMan = inputsSeisEnsMan[i]['data']
        cmts[source] = data_seisEnsMan
    return cmts

def gen_group_name(alert_num, source):
    return "alert_" + str(alert_num) + "_" + source.replace("-","_")

@constraint(is_local=True)
@task(obs=STREAM_OUT)
def monitor_event(alert,cmts,obs):
    updates = process_update(cmts, alert)
    print(str(obs))
    for update in updates:
        print(" **** Publishing: " + str(update), flush=True)
        obs.publish(update)
        time.sleep(2)
    print("*** closing obs")
    obs.close()

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="eventRegistration", service_name="microServices",
      payload="{{event}}", produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def register_event(event):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="eventRegion", service_name="microServices",
      payload='{ "id" : {{event_id}} }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def get_region(event_id):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="eventSetup", service_name="microServices",
      payload='{ "id" : {{event_id}}, "ensemble" : "{{ensemble}}" }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def get_setup(event_id, ensemble):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="eventSetState", service_name="microServices",
      payload='{ "id" : "{{event_id}}", "state": "{{state}}" }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def set_event_state(event_id, state):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="cmtSeisEnsMan", service_name="microServices",
      payload='{ "id" : "{{event_id}}", "region": {{region}}, \
                 "resources": {{resources}}, "setup": {{setup}} }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def input_seisensman(event_id, region, resources, setup):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="precmt", service_name="microServices",
      payload='{ "id" : "{{event_id}}", "region": {{region}}, \
                 "resources": {{resources}}, "setup": {{setup}} }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def build_cmt_input(event_id, region, resources, setup):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
#, "base" : {{base_name}}, "resources" : {{resources}}
@http(request="POST", resource="cmt", service_name="microServices",
      payload='{ "event" : {{alert}}, "id" : "{{event_id}}", \
                 "region" : {{region}}, "setup" : {{precmt}} }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def calculate_cmt(alert, event_id, region, precmt):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="computeResources", service_name="microServices",
      payload='{ "id" : "{{event_id}}", "region": {{region}} }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def compute_resources(event_id, region):
    """
    """
    pass

if HPC_RUN_PYCOMPSS == 'true':
    preGP_service_name="simulation"
else :
    preGP_service_name="slipgen"

 #@on_failure(management='IGNORE', returns=0)
@on_failure(management ='CANCEL_SUCCESSORS')
@http(request="POST", resource="preGraves-Pitarka", service_name=preGP_service_name,
      payload='{ "id" : "{{event_id}}", "region": {{region}}, "setup": {{setup}} }',
      produces='{"result" : "{{return_0}}" }')
@task(returns=1)
def graves_pitarka_setup(event_id, region, setup):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="Graves-Pitarka", service_name="slipgen",
      payload='{ "event" : {{alert}}, "id" : "{{event_id}}", "CMT" : {{cmt}}, \
                 "trial" : "{{path}}", "region": {{region}}, "setup" : {{setup}}, \
                 "resources" : {{resources}} , "ensemble" : "{{ensemble}}" }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def compute_graves_pitarka(event_id, alert, path, cmt, region, setup, resources, ensemble):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@http(request="POST", resource="inputParametersBuilder", service_name="microServices",
      payload='{ "id" : "{{event_id}}", "event" : {{alert}}, "CMT" : {{cmt}}, \
                 "rupture" : {{rupture}}, "region" : {{region}}, \
                 "resources" : {{resources}}, "setup" : {{setup}}, "ensemble" : "{{ensemble}}" }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def build_input_parameters(event_id, alert, cmt, rupture, region, resources, setup, ensemble):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@on_failure(management ='CANCEL_SUCCESSORS')
@http(request="POST", resource="SalvusPrepare", service_name="salvus",
      payload='{ "id" : "{{event_id}}", "trial" : "{{trial}}", \
                 "input" : {{input}}, "resources" : {{resources}} }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def build_salvus_parameters(event_id, trial, input, resources):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@on_failure(management ='CANCEL_SUCCESSORS')
@http(request="POST", resource="SalvusRun", service_name="salvus",
      payload='{ "id" : "{{event_id}}", "trial" : "{{trial}}", \
                 "input" : {{input}}, "resources" : {{resources}} }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def run_salvus(event_id, trial, input, resources):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@on_failure(management ='CANCEL_SUCCESSORS')
@http(request="POST", resource="SalvusPost", service_name="salvus",
      payload='{ "id" : "{{event_id}}", "trial" : "{{trial}}", \
                 "resources" : {{resources}} }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def run_salvus_post(event_id, salvus_result, trial, resources):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@on_failure(management ='CANCEL_SUCCESSORS')
@http(request="POST", resource="SalvusPostSwarm", service_name="salvus",
      payload='{ "id" : "{{event_id}}", "base" : "{{base}}", "resources" : {{resources}} }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1)
def run_salvus_post_swarm(event_id, base, resources):
    """
    """
    pass

#@on_failure(management='IGNORE', returns=0)
@on_failure(management ='CANCEL_SUCCESSORS')
@http(request="POST", resource="SalvusPlots", service_name="salvus",
      payload='{ "id" : "{{event_id}}", "region" : {{region}}, "base" : "{{base}}", \
                 "resources" : {{resources}} }',
      produces='{"result" : "{{return_0}}"}')
#@task(returns=1, results=COLLECTION_IN)
@task(returns=1)
def run_salvus_plots(event_id, salvus_post_results, region, base, resources):
    """
    """
    pass    
    

@on_failure(management='IGNORE')
@http(request="POST", resource="simulation-run", service_name="simulation",
      payload='{ "event" : {{alert}}, "id" : "{{event_id}}", "CMT" : {{cmt}}, \
                 "trial" : "{{trial}}", "region": {{region}}, "setup" : {{setup}}, \
                 "input" : {{input}}, "resources" : {{resources}}, "ensemble" : "{{ensemble}}"  }',
      produces='{"result" : { "workflow_dir" : "{{return_0}}", \
                              "exec_dir" : "{{return_1}}", \
                              "arguments" : "{{return_2}}"} }')
@task(returns=3)
def prepare_simulation(event_id, alert, trial, cmt, region, setup, input, resources, ensemble):
    """
    """
    pass

@on_failure(management='IGNORE', returns=0)
@constraint(computing_units=48)
@compss(app_name="{{workflow_path}}/hpc_workflow.py", args="{{flags}}", worker_in_master="worker", flags="--env_script={{workflow_path}}/env.sh --io_executors=1", computing_nodes=11, working_dir="{{working_dir}}")
@task(time_out=3600, returns=1)
def run_simulation(workflow_path=None, working_dir=None, flags=""):
    pass



@on_failure(management='CANCEL_SUCCESSORS')
@http(request="POST", resource="simulation-post", service_name="simulation",
      payload='{ "id" : "{{event_id}}", "region" : {{region}}, "base" : "{{base}}", \
                 "resources" : {{resources}} }',
      produces='{"result" : "{{return_0}}"}')
@task(returns=1, all_results=COLLECTION_IN)
def post_simulation(event_id, region, base, resources, all_results):
    """
    """
    pass

