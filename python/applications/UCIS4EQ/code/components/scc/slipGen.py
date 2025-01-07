#!/usr/bin/env python3
#
# Slip generation
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
import sys
import os
import traceback
import json
import uuid
import random
from bson.json_util import dumps

# Third parties
from flask import jsonify

# Internal
import ucis4eq
import ucis4eq.dal as dal
from ucis4eq.misc import config, microServiceABC
from ucis4eq.launchers.scriptABC import ScriptABC
from ucis4eq.dal import staticDataMap, staticDataAccess, dataStructure

################################################################################
# Methods and classes

class SlipGenSubmision(ScriptABC):

    # Build the submission script
    def build(self, image, path, args, stage):
        # Obtain Header
        self._getHeader()

        # Obtain Slurm rules

        self._getRules(stage)

        # Build command
        cmd = []
        cmd.append("singularity exec --bind $PWD:/workspace --pwd /workspace " + image + " /opt/scripts/launcher.sh "
                   + args)

        self.lines.append(" ".join(cmd))

        # Additional instructions
        self.lines.append("")

        self.lines.append("")
        self.lines.append("touch SUCCESS")

        # Save the script to disk
        return self._saveScript(path)

@staticDataMap.build
class SlipGenGPSetup(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize the component implementation
        """

    # Service's entry point definition
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        # Select GP setup depending on the fmax policy
        policy = body['setup']['fmax_policy']
        if policy == "min":
            freq = float(min(body['region']['available_fmax']))
        elif policy == "max":
            freq = float(max(body['region']['available_fmax']))
        elif policy.isnumeric() and policy in body['region']['available_fmax']:
            freq = float(policy)
        elif body['region']['available_fmax']:
            freq = float(body['region']['available_fmax'][0])
        else:
            raise Exception("Can't decide the simulation frequency for region" +\
                            body['region']['id'] )

        # Create the data structure
        dataFormat = dataStructure.formats[body['region']['file_structure']]()
        dataFormat.prepare(body['region']['id'])

        # Select an available ensamble
        path = self.fileMapping[body['region']['id']]
        parametersFileName = path + "/" + dataFormat.getPathTo('slip_gen') + "/" + str(freq) + \
                             "Hz_GPsetup.json"

        # Read the input from file
        with open(parametersFileName, 'r') as f:
            inputParameters = json.load(f)

        # Add the frequency and local data path
        inputParameters['freq'] = freq
        inputParameters['path'] = path

        return jsonify(result = inputParameters, response = 201)

@staticDataMap.build
class SlipGenGP(microServiceABC.MicroServiceABC):

    # Mai-Beroza relations
    def _MaiBerozaRelations(self, mw, lat, lon, depth):

        log_Mo = (mw*1.5)+9.05

        relations = {}

        relations['FAULT_LENGTH'] = 10**(-6.13 + (0.39*log_Mo)) #32.84
        relations['FAULT_WIDTH'] =  10**(-5.05 + (0.32*log_Mo)) #27.8
        relations['DEPTH_TO_TOP'] =  depth/1000 - (relations['FAULT_WIDTH']/2) #10.666
        relations['LAT_TOP_CENTER'] = lat
        relations['LON_TOP_CENTER'] = lon
        relations['HYPO_ALONG_STK'] = 0.0 #-2.72
        relations['HYPO_DOWN_DIP'] = relations['FAULT_WIDTH']/2 #16.24

        return relations


    def _SeisEnsManRelations(self, mw, lat, lon, depth, length, area):


        relations = {}

        relations['FAULT_LENGTH'] = float(length)
        relations['FAULT_WIDTH'] =  float(area)/relations['FAULT_LENGTH']

        if float(depth) < relations['FAULT_WIDTH']/2:
            relations['DEPTH_TO_TOP'] =  0
            relations['HYPO_DOWN_DIP'] =  float(depth)
        else:
            relations['DEPTH_TO_TOP'] = float(depth) - (relations['FAULT_WIDTH']/2)
            relations['HYPO_DOWN_DIP'] = relations['FAULT_WIDTH']/2

        relations['LAT_TOP_CENTER'] = float(lat)
        relations['LON_TOP_CENTER'] = float(lon)
        relations['HYPO_ALONG_STK'] = 0.0

        return relations


    # Service's entry point definition
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        Call the Graves-Pitarka slip generator
        """
        # Initialization
        result = {}

        # Select target machine
        machine = body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Create the data structure
        dataFormat = dataStructure.formats[body['region']['file_structure']]()
        dataFormat.prepare(body['region']['id'])

        # Creating the repository instance for data transfer
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Create local directories
        path =  ucis4eq.workSpace + "/scratch/outdata/" + body['trial'] + "/"
        cmt = body['CMT']
        event = body['event']

        os.makedirs(path, exist_ok=True)

        # Create the input parameter for slip-gen
        image = self.filePing['slipgen.singularity']

        # Define Source file
        source = path + "inputs.src"

        # Generate GP source file
        if body['ensemble'] == "statisticalCMT":
            srcParams = self._MaiBerozaRelations(event['magnitude'],
                                                 event['latitude'],
                                                 event['longitude'],
                                                 event['depth'])
            srcParams['MAGNITUDE'] = event['magnitude']
            srcParams['STRIKE'] = cmt['strike']
            srcParams['RAKE'] = cmt['rake']
            srcParams['DIP'] = cmt['dip']
            setup = body['setup']
            srcParams['DWID'] = setup['dwid']
            srcParams['DLEN'] = setup['dlen']
            srcParams['CORNER_FREQ'] = setup['corner_freq']

        elif body['ensemble'] == "seisEnsMan":
            srcParams = self._SeisEnsManRelations(cmt['magnitude'],
                                                  cmt['latitude'],
                                                  cmt['longitude'],
                                                  cmt['depth'],
                                                  cmt['faultLength'],
                                                  cmt['faultArea'])
            srcParams['MAGNITUDE'] = cmt['magnitude']
            srcParams['STRIKE'] = cmt['strike']
            srcParams['RAKE'] = cmt['rake']
            srcParams['DIP'] = cmt['dip']
            setup = body['setup']
            srcParams['DWID'] = setup['dwid']
            srcParams['DLEN'] = setup['dlen']
            srcParams['CORNER_FREQ'] = setup['corner_freq']
        else:
            raise Exception('Not recognized ensemble')

        if 'seed' in body.keys():
            srcParams['SEED'] = body['seed']
        else:
            srcParams['SEED'] = random.randint(1, 9999999)

        with open(source, 'w') as fd:
            for key in srcParams.keys():
                fd.write(key + ' = ' + str(srcParams[key]) + '\n')

        # Prepare script's arguments
        initSlip = ""
        if "initSlip" in cmt.keys():
            initSlip = " -i " + \
                        os.path.basename(self.fileMapping[cmt['initSlip']]) \
                        + " -a 0.9"

        #args = "-o " + result['id'] + " -v " + self.fileMapping[setup['model']] + " -s " + source + " -i " + self.fileMapping["initSlip21june2000"] + " -a 0.99 > /dev/null 2>&1"

        args = "-o rupture "  + "--dt " + str(setup['dt']) + " -v " \
                + setup['model'] \
                + " -s " + os.path.basename(source) + initSlip

        # Submission instance
        submission = SlipGenSubmision(machine['id'])

        script = submission.build(image, path, args, "SlipGenGP")

        # Create the remote working directory
        rworkpath = body['trial'] + "/" + "slipgen"
        dataRepo.mkdir(rworkpath)

        # Transfer source, script, model and init slip (if there is one)
        dataRepo.uploadFile(rworkpath, script)
        dataRepo.uploadFile(rworkpath, source)
        if "initSlip" in cmt.keys():
            dataRepo.uploadFile(rworkpath, self.fileMapping[cmt['initSlip']])

        modelFileName = setup['path'] + "/" + \
                        dataFormat.getPathTo('slip_gen') + "/" + \
                        setup['model']
        dataRepo.uploadFile(rworkpath, modelFileName)

        # Submit and wait for finish
        submission.run(dataRepo.path + "/" + rworkpath)

        # Get the path to the generated rupture
        result = dataRepo.path + "/" + rworkpath + "/scratch/outdata/rupture/rupture.srf"

        # Return list of Id of the newly created item
        # TODO: Return the SRF in plain text
        return jsonify(result = result, response = 201)

        # TODO: Instead of running the GP rupture generator, build the submission script
        #os.system("/bin/bash -c '/root/scripts/launcher.sh " + args + "'")
        #
        ## Read the generated SRF
        #with open(path + result['id'] + ".srf", encoding="utf-8") as f:
        #    #result['slipmodel'] = f.readlines();
        #    result['slipmodel'] = f.read().splitlines()
