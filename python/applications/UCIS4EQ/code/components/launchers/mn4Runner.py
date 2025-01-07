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
import uuid
import subprocess
import threading

from ucis4eq.launchers.slurmRunnerABC import SlurmRunnerABC

################################################################################
# Methods and classes

class MN4SlurmRunner(SlurmRunnerABC):
            
    # Obtain the spefific rules for a slurm Script    
    def getRules(self, stage):
        
        # Initialize list of instructions
        lines = []
        
        # Select the resources for the given stage
        # TODO: Control that provided stage exist
        args = self.resources[stage]
        
        # Check if a job name was provided
        if "name" in args.keys():
            name = args['name']
        else:
            name = stage

        # Add rules to the slurm script
        lines.append("#SBATCH --job-name=" + name)        
        lines.append("#SBATCH --time=" + time.strftime('%H:%M:%S', time.gmtime(args['wtime'])))
        lines.append("#SBATCH --nodes=" + str(args['nodes']))
        lines.append("#SBATCH --tasks-per-node=" + str(args['tasks-per-node']))
        lines.append("#SBATCH --ntasks=" + str(int(args['nodes']) * int(args['tasks-per-node'])))
        lines.append("#SBATCH --error=" + stage + ".e")
        lines.append("#SBATCH --output=" + stage + ".o")
        lines.append("#SBATCH --qos=" + args['qos'])
        
        if "constraint" in args.keys():
            lines.append("#SBATCH --constraint=" + args['constraint'])
            
        #lines.append("#SBATCH --partition=main")
        #lines.append("#SBATCH --reservation=ChEESE21_test")

        lines.append("")
        lines.append("cd $SLURM_SUBMIT_DIR")
        lines.append("")        
        
        # Return the set of instructions
        return lines

    # Method for obtaining environment setup (module loads, PATH, conda environmnet, etc ...)
    def getEnvironmentSetup(self, stage):
        # Initialize list of instructions
        lines = []
        
        # TODO: Control that provided stage exist
        args = self.resources[stage]
        
        add_lines = args['environment']

        for line in add_lines:
            lines.append(line)
        
        return lines
    
class MN4RunnerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, MN4, **_ignored):
        user = MN4["user"]
        url = MN4["url"]
        path = MN4["path"]
        if "setup" in MN4.keys():
            setup = MN4["setup"]           
        else:
            setup = None
        resources = MN4["resources"]
        
        # WARNING!!!: We dont want to do this as a Singleton
        #if not self._instance:
        #    self._instance = SlurmRunner(user, url, path)
            
        #return self._instance
        return MN4SlurmRunner(user, url, path, resources, setup)
