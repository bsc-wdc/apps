# Functions that are not in the notebook

import os
import sys
import tempfile
import scipy
import numpy
import fnmatch
import subprocess
from collections import OrderedDict


__version__ = "0.3.6 + PyCOMPSs"


###############################################################################
########################### CLASSES ###########################################
###############################################################################

class Trace(object):
    def __init__(self, path, processes):
        self.path = path
        self.processes = processes
    def get_path(self):
        return self.path
    def get_processes(self):
        return self.processes

###############################################################################
############################ CREATION FUNCTIONS ###############################
###############################################################################

def create_raw_data(trace_name, raw_data_doc):
    """Creates 2D dictionary of the raw input data and initializes with zero.
    The raw_data dictionary has the format: [raw data key][trace].
    """
    raw_data = {}
    for key in raw_data_doc:
        trace_dict = {}
        trace_dict[trace_name] = 0
        raw_data[key] = trace_dict
    return raw_data

def create_empty_raw_data(raw_data_doc):
    """Creates 2D dictionary of the raw input data with empty dictionaries for the traces.
    The raw_data dictionary has the format: [raw data key]{}.
    """
    raw_data = {}
    for key in raw_data_doc:
        trace_dict = {}
        raw_data[key] = trace_dict
    return raw_data

def create_mod_factors(trace_name, mod_factors_doc):
    """Creates 2D dictionary of the model factors and initializes with an empty string.
    The mod_factors dictionary has the format: [mod factor key][trace].
    """
    mod_factors = {}
    for key in mod_factors_doc:
        trace_dict = {}
        trace_dict[trace_name] = 0.0
        mod_factors[key] = trace_dict
    return mod_factors

def create_empty_mod_factors(mod_factors_doc):
    """Creates 2D dictionary of the model factors with empty dictionaries for the traces.
    The mod_factors dictionary has the format: [mod factor key]{}.
    """
    mod_factors = {}
    for key in mod_factors_doc:
        trace_dict = {}
        mod_factors[key] = trace_dict
    return mod_factors

###############################################################################
########################### AUXILIAR FUNCTIONS ################################
###############################################################################

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

###############################################################################
############################# HELPER FUNCTIONS ################################
###############################################################################

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

def get_list_proc(traces):
    """Retrieve the traces and their processes"""
    trace_list = traces.keys()
    trace_processes = {}
    for trace_name in trace_list:
        trace_processes[trace_name] = traces[trace_name].get_processes()
    return trace_list, trace_processes

def print_overview(traces):
    """Prints an overview of the traces that will be processed."""
    print('Running Model Factors for the following traces:')
    for k, v in traces.items():
        line = k
        line += ', ' + str(v.get_processes()) + ' processes'
        line += ', ' + human_readable(os.path.getsize(v.get_path()))
        print(line)
    print('')

def read_mod_factors_csv(debug, project, mod_factors_doc):
    """Reads the model factors table from a csv file."""
    delimiter = ';'
    file_path = project

    # Read csv to list of lines
    if os.path.isfile(file_path) and file_path[-4:] == '.csv':
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
    else:
        raise Exception('==ERROR==', file_path, 'is not a valid csv file.')

    # Get the number of processes of the traces
    processes = lines[0].split(delimiter)
    processes.pop(0)

    # Create artificial trace_list and trace_processes
    trace_list = []
    trace_processes = {}
    for process in processes:
        trace_list.append(process)
        trace_processes[process] = int(process)

    # Create empty mod_factors handle
    mod_factors = create_mod_factors(trace_list)

    # Get mod_factor_doc keys
    mod_factors_keys = list(mod_factors_doc.items())

    # Iterate over the data lines
    for index, line in enumerate(lines[1:len(mod_factors_keys)+1]):
        key = mod_factors_keys[index][0]
        line = line.split(delimiter)
        for index, trace in enumerate(trace_list):
            mod_factors[key][trace] = float(line[index+1])

    if debug:
        print_mod_factors_table(mod_factors, trace_list, trace_processes)

    return mod_factors, trace_list, trace_processes


def print_raw_data_table(raw_data, traces, raw_data_doc):
    """Prints the raw data table in human readable form on stdout."""
    print('Overview of the collected raw data:')

    trace_list, trace_processes = get_list_proc(traces)

    longest_name = len(sorted(raw_data_doc.values(), key=len)[-1])

    line = ''.rjust(longest_name)
    for trace in trace_list:
        line += ' | '
        line += str(trace_processes[trace]).rjust(15)
    print(line)

    print(''.ljust(len(line),'='))

    for data_key in raw_data_doc:
        line = raw_data_doc[data_key].ljust(longest_name)
        for trace in trace_list:
            line += ' | '
            line += str(raw_data[data_key][trace]).rjust(15)
        print(line)
    print('')


def print_mod_factors_table(mod_factors, traces, mod_factors_doc):
    """Prints the model factors table in human readable form on stdout."""
    print('Overview of the computed model factors:')

    longest_name = len(sorted(mod_factors_doc.values(), key=len)[-1])

    trace_list, trace_processes = get_list_proc(traces)

    line = ''.rjust(longest_name)
    for trace in trace_list:
        line += ' | '
        line += str(trace_processes[trace]).rjust(10)
    print(line)

    print(''.ljust(len(line),'='))

    for mod_key in mod_factors_doc:
        line = mod_factors_doc[mod_key].ljust(longest_name)
        if mod_key in ['speedup','ipc','freq']:
            for trace in trace_list:
                line += ' | '
                try:
                    line += ('{0:.2f}'.format(mod_factors[mod_key][trace])).rjust(10)
                except ValueError:
                    #except NaN
                    line += ('{}'.format(mod_factors[mod_key][trace])).rjust(10)
        else:
            for trace in trace_list:
                line += ' | '
                try:
                    line += ('{0:.2f}%'.format(mod_factors[mod_key][trace])).rjust(10)
                except ValueError:
                    # except NaN
                    line += ('{}'.format(mod_factors[mod_key][trace])).rjust(10)
        print(line)
        # Print empty line to separate values
        if mod_key in ['global_eff','freq_scale']:
            line = ''.ljust(longest_name)
            for trace in trace_list:
                line += ' | '
                line += ''.rjust(10)
            print(line)
    print('')


###############################################################################
########################### REDUCTION FUNCTIONS ###############################
###############################################################################

def merge_reduce(function, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    dataNew = data[:]
    q = deque(list(range(len(dataNew))))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            dataNew[x] = function(dataNew[x], dataNew[y])
            q.append(x)
        else:
            return dataNew[x]

def merge_reduce_accum(function, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    dataNew = data[:]
    q = deque(list(range(len(dataNew))))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            function(dataNew[x], dataNew[y])
            q.append(y)
        else:
            return dataNew[x]

###############################################################################
############################### WIDGETS #######################################
###############################################################################

import ipywidgets as widgets

class wdgts(object):
    style = {'description_width': 'initial'}

    # List of traces to process. Accepts wild cards and automatically filters for valid traces
    w_trace_folder = widgets.Text(value=os.getcwd() + os.path.sep + 'traces/gromacs_jesus/',
                                  description='List of traces:',
                                  layout={'width':'60%'})
    # Increase output verbosity to debug level
    w_debug = widgets.Checkbox(value=False,
                               description='Debug')
    # Define whether the measurements are weak or strong scaling (default: auto)
    w_scaling = widgets.ToggleButtons(options=['auto', 'weak','strong'],
                                      description='Scaling',
                                      button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                      tooltips=['Automatic measurements scaling', 'weak measurements scaling', 'Strong measurements scaling'])
    # Run only the projection for the given modelfactors.csv (default: false)
    w_project = widgets.Text(value='false',
                             placeholder='modelfactors.csv',
                             description='CSV projection file path:',
                             style=style,
                             layout={'width':'60%'})
    # Limit number of cores for the projection (default: 10000)
    w_limit = widgets.IntText(value=10000,
                              description='Projection # cores:',
                              style=style,
                              layout={'width':'60%'})
    # Select model for prediction (default: amdahl)
    w_model = widgets.ToggleButtons(options=['amdahl','pipe','linear'],
                                    description='Model',
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltips=['Amdahl model prediction', 'Pipe model prediction', 'Linear model prediction'])
    # Set bounds for the prediction (default: yes)
    w_bounds = widgets.Checkbox(value=True,
                                description='Prediction bounds')
    # Set error restrains for prediction (default: first). first: prioritize smallest run; equal: no priority; decrease: decreasing priority for larger runs
    w_sigma = widgets.ToggleButtons(options=['first','equal','decrease'],
                                    description='Sigma',
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltips=['Prioritize smallest run', 'No priority', 'Decreasing priority for larger runs'])
    # Path of the configuration files
    w_cfgs = widgets.Text(value=os.getcwd() + os.path.sep + 'cfgs',
                          placeholder='cfgs',
                          description='Configuration files path:',
                          style=style,
                         layout={'width':'60%'})
    # Path of matplotlib output file
    w_gp_out = widgets.Text(value=os.getcwd() + os.path.sep + 'results.gp',
                            placeholder='Output_file.gp',
                            description='Gnuplot output file:',
                            style=style,
                            layout={'width':'60%'})
    # Path of matplotlib output file
    w_mpl_out = widgets.Text(value=os.getcwd() + os.path.sep + 'results.png',
                             placeholder='Output_file.png',
                             description='Matplotlib Output file:',
                             style=style,
                             layout={'width':'60%'})
    # Path of csv output file
    w_csv = widgets.Text(value=os.getcwd() + os.path.sep + 'results.csv',
                         placeholder='Output_file.csv',
                         description='CSV output file:',
                         style=style,
                         layout={'width':'60%'})
    # Choose reduction strategy
    w_reduction = widgets.ToggleButtons(options=['Accumulate', 'Reduce', 'MergeReduce', 'MergeReduceAccumulate'],
                                        description='Reductions',
                                        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltips=['Accummulate in the same loop', 'Simple reduce function', 'Reduce in pairs', 'Reduce in pairs accumulating'])
