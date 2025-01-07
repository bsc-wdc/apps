#!/usr/bin/env python3
#
# Salvus (Wave propagation simulator)
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
import yaml
from bson.json_util import dumps

# Third parties
from flask import jsonify

# Internal
from ucis4eq.misc import config, microServiceABC
from ucis4eq.launchers import scriptABC
import ucis4eq.dal as dal
import ucis4eq as ucis4eq
from ucis4eq.dal import staticDataMap, staticDataAccess, dataStructure


################################################################################
# Methods and classes

class SalvusWrapperSubmision(scriptABC.ScriptABC):

    # Build the submission script
    def build(self, commands, path, stage, additional = []):
        # Obtain Header
        self._getHeader()

        # TODO: Change this depending of the target environment queue system
        self._getRules(stage)

        for line in additional:
            self.lines.append(line)

        # Build command
        for command in commands:
            self.lines.append("")
            self.lines.append(command[0] + " " + command[1])

        self.lines.append("")
        self.lines.append("touch SUCCESS")

        # Save the script to disk
        return self._saveScript(path)

@staticDataMap.build
class SalvusPrepare(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize SalvusRun instance
        """

    # Service's entry point definition
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        Call the Salvus-Flow Marta's wrapper
        """

        # Initialization
        result = {}

        # Select target machine
        machine = body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Creating the repository instance for data transfer
        # TODO: Select the repository from the DB 'Resources' document
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Enable multiline writting
        yaml.SafeDumper.org_represent_str = yaml.SafeDumper.represent_str

        def repr_str(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')
            return dumper.org_represent_str(data)

        yaml.add_representer(str, repr_str, Dumper=yaml.SafeDumper)

        # Create the directory for the current execution
        workSpace = "/workspace/runs/" + body["trial"] + "/"
        os.makedirs(workSpace, exist_ok=True)

        # Write the YAML file
        inputPyaml = yaml.safe_dump(body["input"])
        pfile = workSpace + "/" + "salvus_input.yaml"
        with open(pfile, "w") as f:
           f.write(inputPyaml)

        # Prepare script's arguments
        commands = []
        binary = "python "  + self.filePing['salvus_wrapper']
        args = "--ucis4eq salvus_input.yaml " +\
               "--salvus " + self.filePing['salvus_setup']

        commands.append((binary, args))

        # Solver dependencies
        additional = []
        additional.append("export PYTHONPATH=" +\
                           os.path.dirname(self.filePing['salvus_wrapper']) +\
                           ":$PYTHONPATH")

        # Submission instance
        submission = SalvusWrapperSubmision(machine['id'])

        script = submission.build(commands, workSpace, "SalvusPrepare", additional)

        # Create the remote working directory
        rworkpath = body['trial'] + "/" + "salvus_wrapper"
        dataRepo.mkdir(rworkpath)

        # Transfer input parameter file and script
        dataRepo.uploadFile(rworkpath, script)
        dataRepo.uploadFile(rworkpath, pfile)

        # Submit and wait for finish
        submission.run(dataRepo.path + "/" + rworkpath)

        # Get the path to the Salvus input parameters
        result['salvus_input'] = dataRepo.path + "/" + rworkpath + "/salvus_input_rupture.toml"

        # Return list of Id of the newly created item
        return jsonify(result = result['salvus_input'], response = 201)

class SalvusRunSubmision(scriptABC.ScriptABC):

    # Build the submission script
    def build(self, commands, path, stage, additional = []):
        # Obtain Header
        self._getHeader()

        # TODO: Change this depending of the target environment queue system
        self._getRules(stage)

        for line in additional:
            self.lines.append(line)

        # Build command
        for command in commands:
            self.lines.append("")
            self.lines.append(command[0] + " " + command[1])

        self.lines.append("")
        self.lines.append("touch SUCCESS")

        # Save the script to disk
        return self._saveScript(path)

@staticDataMap.build
class SalvusRun(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize SalvusRun instance
        """

    # Service's entry point definition
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        Call the Salvus-Compute on the remote machine
        """

        # Initialization
        result = {}

        # Select target machine
        machine =  body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Instantiate Salvus submission
        submission = SalvusRunSubmision(machine['id'])

        # Creating the repository instance for data transfer
        # TODO: Select the repository from the DB 'Resources' document
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Create the directory for the current execution
        workSpace = "/workspace/runs/" + body["trial"] + "/"
        os.makedirs(workSpace, exist_ok=True)

        # Obtain remote location of Salvus binary
        binary = self.filePing['salvus_compute']

        # Prepare script's arguments
        commands = []

        # ... build command for job monitoring
        #binary = "python "  + self.filePing['salvus_job_tracker']
        #args = body['input'] + " $PWD 10 snapshots &"
        #commands.append((binary, args))

        # ... build command for the modeling
        binary = submission.getMPICommand() + self.filePing['salvus_compute'] +\
                " compute "
        args = body['input']
        commands.append((binary, args))


        script = submission.build(commands, workSpace, "SalvusRun", [])

        # Create the remote working directory
        rworkpath = body['trial'] + "/" + "salvus"
        dataRepo.mkdir(rworkpath)

        # Transfer input parameter file and script
        dataRepo.uploadFile(rworkpath, script)

        # Submit and wait for finish
        submission.run(dataRepo.path + "/" + rworkpath)

        # Return list of Id of the newly created item
        return jsonify(result = workSpace, response = 201)

@staticDataMap.build
class SalvusPost(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize the CMT statistical component implementation
        """

        # Select the database
        self.db = ucis4eq.dal.database

    # Service's entry point definition
    @config.safeRun
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        Call the Salvus-Flow Marta's wrapper (Postprocessing scripts)
        """

        # Initialization
        result = {}

        # Select target machinef
        machine = body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Creating the repository instance for data transfer
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Create the directory for the current execution
        workSpace = "/workspace/runs/" + body["trial"] + "/"
        os.makedirs(workSpace, exist_ok=True)

        pattern = ""
        root = dataRepo.path + "/" + body["trial"]
        wworkpath = root + "/" + "salvus_wrapper/"
        sworkpath = root + "/" + "salvus/"

        # Prepare script's arguments
        commands = []
        # ... for postprocessing
        binary = "python "  + self.filePing['salvus_wrapper_post']
        args = "--ucis4eq " + wworkpath + "salvus_input.yaml " +\
               "--salvus "  + self.filePing['salvus_setup'] + " " +\
               "--coordsdata " +  wworkpath + " " +\
               "--rawdata " + sworkpath
        commands.append((binary, args))

        # Solver dependencies
        additional = []
        additional.append("export PYTHONPATH=" +\
                           os.path.dirname(self.filePing['salvus_wrapper_post']) +\
                           ":$PYTHONPATH")

        # Submission instance
        submission = SalvusWrapperSubmision(machine['id'])

        script = submission.build(commands, workSpace, "SalvusPost", additional)

        # Create the remote working directory
        rworkpath = body['trial'] + "/" + "salvus_post"
        dataRepo.mkdir(rworkpath)

        # Transfer input parameter file and script
        dataRepo.uploadFile(rworkpath, script)

        # Submit and wait for finish
        submission.run(dataRepo.path + "/" + rworkpath)

        # Return list of Id of the newly created item
        return jsonify(result = result, response = 201)

@staticDataMap.build
class SalvusPostSwarm(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize the CMT statistical component implementation
        """

        # Select the database
        self.db = ucis4eq.dal.database

    # Service's entry point definition
    @config.safeRun
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        Call the Salvus-Flow Marta's wrapper (Postprocessing scripts)
        """

        # Initialization
        result = None

        # Select target machinef
        machine = body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Creating the repository instance for data transfer
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Create the directory for the current execution
        bpath = body['base'] + "/"
        workSpace = "/workspace/runs/" + bpath
        os.makedirs(workSpace, exist_ok=True)

        dname = "salvus_post_swarm"
        pattern = "/trial_" + "*/"
        root = dataRepo.path + "/"
        pworkpath = root + bpath + "/" + dname
        sworkpath = root + body['base'] + pattern + "salvus/"

        # Prepare script's arguments
        commands = []
        # ... for postprocessing
        binary = "python "  + self.filePing['salvus_wrapper_post_swarm']
        args = "--processeddata '" + sworkpath + "' " +\
               "--output " + pworkpath
        commands.append((binary, args))

        # Solver dependencies
        additional = []
        additional.append("export PYTHONPATH=" +\
                           os.path.dirname(self.filePing['salvus_wrapper_post_swarm']) +\
                           ":$PYTHONPATH")

        # Submission instance
        submission = SalvusWrapperSubmision(machine['id'])

        script = submission.build(commands, workSpace, "SalvusPostSwarm", additional)

        # Create the remote working directory
        rworkpath = bpath + dname
        dataRepo.mkdir(rworkpath)

        # Transfer input parameter file and script
        dataRepo.uploadFile(rworkpath, script)

        # Submit and wait for finish
        submission.run(dataRepo.path + "/" + rworkpath)

        # Return list of Id of the newly created item
        return jsonify(result = result, response = 201)

@staticDataMap.build
class SalvusPlots(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize the CMT statistical component implementation
        """

        # Select the database
        self.db = ucis4eq.dal.database

    # Service's entry point definition
    @config.safeRun
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        Call the Salvus-Flow Marta's wrapper (Postprocessing scripts)
        """

        # Initialization
        result = {}

        # Select target machinef
        machine = body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Creating the repository instance for data transfer
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Create the data structure
        regionName = body['region']['id'] + "_DATA"
        dformat =  body['region']['file_structure']
        dataFormat = dataStructure.formats[dformat]()
        dataFormat.setMainRepository(machine['repository'])
        dataFormat.prepare(regionName)

        # Create the directory for the current execution
        bpath = body['base'] + "/"
        workSpace = "/workspace/runs/" + bpath
        os.makedirs(workSpace, exist_ok=True)


        dname = "salvus_plots"
        pattern = "/" + "salvus_post_swarm"
        root = dataRepo.path + "/"
        pworkpath = root + bpath + "/" + dname
        sworkpath = root + body['base'] + pattern + "/"

        # Prepare script's arguments
        commands = []

        topogrid = dataFormat.getPathTo('topography', ["grid"])
        # ... for plotting
        binary = "python "  + self.filePing['salvus_wrapper_plot']

        args = "--processeddata '" + sworkpath + "' "

        if topogrid:
               args = args + "--topogrid " + topogrid

        commands.append((binary, args))

        # ... move to proper directory
        binary = "cd"
        args = pworkpath
        commands.append((binary, args))

        # ... create symbolic link
        binary = "ln"
        args = "-s " + root + body['base'] + " " + body['base']
        commands.append((binary, args))

        # ... create symbolic link
        binary = "tar "
        args = "czvf  " + body['base'] + ".tar.gz " + body['base'] +\
                pattern + "/*.png"
        commands.append((binary, args))

        # ... create symbolic link
        binary = "rm  "
        args = "-fr " + body['base']
        commands.append((binary, args))

        # Solver dependencies
        additional = []
        additional.append("export PYTHONPATH=" +\
                           os.path.dirname(self.filePing['salvus_wrapper_plot']) +\
                           ":$PYTHONPATH")

        # Submission instance
        submission = SalvusWrapperSubmision(machine['id'])

        script = submission.build(commands, workSpace, "SalvusPlots", additional)

        # Create the remote working directory
        rworkpath = bpath + dname
        dataRepo.mkdir(rworkpath)

        # Transfer input parameter file and script
        dataRepo.uploadFile(rworkpath, script)

        # Submit and wait for finish
        submission.run(dataRepo.path + "/" + rworkpath)

        # Download results from HPC machine
        dataRepo.downloadFile(rworkpath + "/" + body['base'] + ".tar.gz",
                              workSpace + body['base'] + ".tar.gz")

        # Upload results to B2DROP
        b2drop = staticDataAccess.repositories.create(dal.repository, **dal.config)
        outputPath = dal.config[dal.repository]["output"]

        rpath = outputPath + "/" + body["base"] + "_" + body['region']['id']  + "/"
        lpath = workSpace

        ## Create remote path
        #input = "/input/"
        #b2drop.mkdir(rpath + input)
        b2drop.mkdir(rpath)

        # Upload results
        file = body['base'] + ".tar.gz"
        b2drop.uploadFile(rpath + file, lpath + file)

        # Return list of Id of the newly created item
        # TODO: Return the SRF in plain text
        return jsonify(result = result, response = 201)

@staticDataMap.build
class SalvusPing(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize SalvusRun instance
        """

    # Service's entry point definition
    @config.safeRun
    def entryPoint(self, body):
        """
        Call the Salvus-Compute on the remote machine
        """

        # Initialization
        result = {}
        progress = {}

        # Select target machinefgetRules
        machine = body['resources']

        # Set repository
        self.setMainRepository(machine['repository'])

        # Creating the repository instance for data transfer
        # TODO: Select the repository from the DB 'Resources' document
        dataRepo = staticDataAccess.repositories.create(machine['repository'], **dal.config)

        # Create the directory for the current execution
        workSpace = "/workspace/runs/" + body["trial"] + "/"
        os.makedirs(workSpace, exist_ok=True)

        # Create the remote working directory
        filename = "progress.json"
        rfile = body['trial'] + "/salvus/" + filename
        lfile = workSpace + filename

        # Download results from HPC machine
        dataRepo.downloadFile(rfile, lfile)

        # Read local json file
        try:
            with open(lfile, 'r') as f:
                progress = json.load(f)
        except FileNotFoundError:
            # MPC jsonify(result=None) throws an error later, so print a more meaningful message
            print("ERROR: file %s that was meant to be downloaded from %s was not found." %(lfile, rfile) )
            progress = None

        # Return list of Id of the newly created item
        return jsonify(result = progress, response = 201)
