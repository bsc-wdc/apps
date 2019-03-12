# Functions that are not in the notebook

import os
import sys
import tempfile
import scipy
import numpy
import fnmatch
import subprocess
from collections import OrderedDict


__version__ = "0.3.6"


class Trace(object):
    def __init__(self, path, processes):
        self.path = path
        self.processes = processes
    def get_path(self):
        return self.path
    def get_processes(self):
        return self.processes


def which(cmd):
    """Returns path to cmd in path or None if not available."""
    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        cmd_path = os.path.join(path, cmd)
        if os.path.isfile(cmd_path) and os.access(cmd_path, os.X_OK):
            return cmd_path
    return None


def human_readable(size, precision=1):
    """Converts a given size in bytes to the value in human readable form."""
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1
        size = size/1024.0
    return "%.*f%s"%(precision,size,suffixes[suffixIndex])


def check_installation(debug):
    """Check if Dimemas and paramedir are in the path."""

    if not which('Dimemas'):
        raise Exception('Could not find Dimemas. Please make sure Dimemas is correctly installed and in the path.')
    if not which('paramedir'):
        raise Exception('Could not find paramedir. Please make sure Paraver is correctly installed and in the path.')

    if debug:
        print('==DEBUG== Using', __version__)
        print('==DEBUG== Using', sys.executable, ".".join(map(str, sys.version_info[:3])))

        try:
            print('==DEBUG== Using', 'SciPy', scipy.__version__)
        except NameError:
            print('==DEBUG== SciPy not installed.')

        try:
            print('==DEBUG== Using', 'NumPy', numpy.__version__)
        except NameError:
            print('==DEBUG== NumPy not installed.')

        print('==DEBUG== Using', which('Dimemas'))
        print('==DEBUG== Using', which('paramedir'))
        print('')


def run_command(cmd, debug):
    """Runs a command and forwards the return value."""
    if debug:
        print('==DEBUG== Executing:', ' '.join(cmd))

    #In debug mode, keep the output. Otherwise, redirect it to devnull.
    if debug:
        out = tempfile.NamedTemporaryFile(suffix='.out', prefix=cmd[0]+'_', dir='./', delete=False)
        err = tempfile.NamedTemporaryFile(suffix='.err', prefix=cmd[0]+'_', dir='./', delete=False)
    else:
        out = open(os.devnull, 'w')
        err = open(os.devnull, 'w')

    return_value = subprocess.call(cmd, stdout=out, stderr=err)

    out.close
    err.close

    if return_value == 0:
        if debug:
            os.remove(out.name)
            os.remove(err.name)
    else:
        print('==ERROR== ' + ' '.join(cmd) + ' failed with return value ' + str(return_value) + '!')
        print('See ' + out.name + ' and ' + err.name + ' for more details.')

    return return_value


def save_remove(path, debug):
    """Wraps os.remove with a try clause."""
    try:
        os.remove(path)
    except:
        if debug:
            print('==DEBUG== Failed to remove ' + path + '!')

#######################################################################

def get_traces_from_args(trace_list):
    """Filters the given list to extract traces, i.e. matching *.prv and sorts
    the traces in ascending order based on the number of processes in the trace.
    Excludes all files other than *.prv and ignores also simulated traces from
    this script, i.e. *.sim.prv
    Returns list of trace paths and dictionary with the number of processes.
    """
    trace_path_list = [x for x in trace_list if fnmatch.fnmatch(x, '*.prv') if not fnmatch.fnmatch(x, '*.sim.prv')]
    trace_path_list = sorted(trace_path_list, key=get_num_processes)

    if not trace_list:
        raise Exception('==Error== could not find any traces matching "', ' '.join(trace_path_list))

    traces = OrderedDict()
    for trace in trace_path_list:
        trace_name = os.path.basename(trace)
        traces[trace_name] = Trace(trace, get_num_processes(trace))

    print_overview(traces)
    return traces

def get_num_processes(prv_file):
    """Gets the number of processes in a trace from the according .row file.
    The number of processes in a trace is always stored at the fourth position
    in the first line of the according *.row file.
    Please note: return value needs to be integer because this function is also
    used as sorting key.
    """
    cpus = open( prv_file[:-4] + '.row' ).readline().rstrip().split(' ')[3]
    return int(cpus)

def print_overview(traces): # trace_list, trace_processes):
    """Prints an overview of the traces that will be processed."""
    print('Running Model Factors for the following traces:')
    for k, v in traces.items():
        line = k
        line += ', ' + str(v.get_processes()) + ' processes'
        line += ', ' + human_readable(os.path.getsize(v.get_path()))
        print(line)
    print('')
