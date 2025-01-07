"""This is init module."""

################################################################################
# Module imports

import json
import os

from ucis4eq.misc.factory import Provider 
from . import localRunner
from . import mn4Runner
from . import pdRunner
from . import PyCOMPSsRunner

# Create the launcher's registry
launchers = Provider()

# Register each data repository
launchers.registerBuilder('LOCAL', localRunner.LocalRunnerBuilder())
HPC_RUN_PYCOMPSS=str(os.environ.get("HPC_RUN_PYCOMPSS", "False")).lower()
if HPC_RUN_PYCOMPSS == "true":
    launchers.registerBuilder('MN4', PyCOMPSsRunner.PyCOMPSsRunnerBuilder())
    launchers.registerBuilder('DAINT', PyCOMPSsRunner.PyCOMPSsRunnerBuilder())
else:
    launchers.registerBuilder('MN4', mn4Runner.MN4RunnerBuilder())
    launchers.registerBuilder('DAINT', pdRunner.PDRunnerBuilder())

# Load computational sites setting
config = None
with open("/opt/Sites.json", 'r') as f:
    config = json.load(f)
