#!/usr/bin/env python3
#
# Slurm specialization launcher
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
import os
import time
import subprocess

from ucis4eq.launchers.runnerABC import RunnerABC

################################################################################
# Methods and classes

class LocalRunner(RunnerABC):

    # Some attributes
    time = 0
    
    # Initialization method    
    def __init__(self, user, url, path):
        
        # Base command 
        self.baseCmd = "ssh" 
        
        # Store the username, url and base path
        self.user = user
        self.url = url
        self.path = path     
            
    # Obtain the spefific rules for a local script    
    def getRules(self, stage):
        
        # Return the set of instructions
        return []

    # Obtain the spefific MPI command (srun, mpirun, etc...)           
    def getMPICommand(self):
        pass
        
    # Obtain environment setup (module loads, PATH, conda environmnet, etc ...)
    def getEnvironmentSetup(self, stage):
        pass
        
    # Run locally and wait
    def run(self, cmd):

        command_trials = 0
        while True:
            # Run the command and wait
            process = subprocess.run(cmd, stdout=subprocess.PIPE,
                                     stdin=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     encoding='utf8')
            if process.returncode != 0:
                if command_trials < 3:
                    print("\nWARNING: Command '" + cmd
                          + "' failed. Error: " + process.stderr + "Retrying in 15 seconds.", flush=True)
                    time.sleep(15)
                    command_trials += 1
                    continue
                else:
                    raise Exception("Command '" + cmd
                                   + "' failed three times in a row. Error: " + process.stderr)
            else:
                break
        
class LocalRunnerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, LOCAL, **_ignored):
        user = LOCAL["user"]
        url = LOCAL["url"]
        path = LOCAL["path"]
        if not self._instance:
            self._instance = LocalRunner(user, url, path)
            
        return self._instance
