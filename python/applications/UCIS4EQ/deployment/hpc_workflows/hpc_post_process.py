from pycompss.api.task import task
from pycompss.api.binary import binary
from pycompss.api.container import container
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.parameter import *

import os

from salvus_urgent_wrapper.salvus_preprocess.salvus_preprocess import pre_process
from salvus_urgent_wrapper.salvus_postprocess.helpers import process_outputs_grid

from salvus_urgent_wrapper.salvus_postprocess.process_outputs_swarm import process_swarm
from salvus_urgent_wrapper.salvus_plot.plot_outputs import generate_plots

@task(salvus_post_swarm_dir=DIRECTORY_INOUT)
def salvus_post_swarm(input_data, salvus_post_swarm_dir, stations):
    # os.mkdir(salvus_post_swarm_dir)
    print("Processing: " + input_data)
    process_swarm(input_data, salvus_post_swarm_dir, stations) 

@task(salvus_post_swarm_dir=DIRECTORY_INOUT)
def salvus_plot(salvus_post_swarm_dir, grid_topo_to_plot, scale, path_to_maxall_netcdf):
    generate_plots(salvus_post_swarm_dir, grid_topo_to_plot, scale, path_to_maxall_netcdf)
   
@binary(binary="tar", args="-czvf {{plots_file}} {{salvus_post_swarm_dir}}/*.png", fail_by_exit_value=True)
@task(salvus_post_swarm_dir=DIRECTORY_INOUT, plots_file=FILE_OUT)
#@task(salvus_post_swarm_dir=DIRECTORY_INOUT)
def compress(salvus_post_swarm_dir,plots_file):
    pass

def compute_swarm_and_plots(event_id, input_data, output_path, stations, grid_topo_to_plot, scale, path_to_maxall_netcdf):
    salvus_post_swarm_dir = output_path + "/salvus_post_swarm"
    if not os.path.exists(salvus_post_swarm_dir):
        os.mkdir(salvus_post_swarm_dir)
    salvus_plot_dir = output_path + "/salvus_plot"
    if not os.path.exists(salvus_plot_dir):
        os.mkdir(salvus_plot_dir)
    print("Salvus plots dir: " + salvus_plot_dir)
    print("Input data to treat:" + input_data)    
    print("Event id: " + event_id)
    if "/" in event_id:
        plots_file = salvus_plot_dir + "/" + event_id.split("/")[-2]+ ".tar.gz"
    else:
        plots_file = salvus_plot_dir + "/" + event_id + ".tar.gz"
    print("plots file: " + plots_file)
    print("post swarm dir: " + salvus_post_swarm_dir)
    salvus_post_swarm(input_data, salvus_post_swarm_dir, stations)
    salvus_plot(salvus_post_swarm_dir, grid_topo_to_plot, scale, path_to_maxall_netcdf)
    compress(salvus_post_swarm_dir, plots_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--event_id", dest='event_id', required=True,
                        help="the pattern of the pathnames of the folder(s) with the netcdf files with proxies; this needs "
                             "to be a set of paths for simulations where only the source parameters vary (swarm run)")
    parser.add_argument("-p", "--processeddata", dest='paths_to_processed_netcdf', required=True,
                        help="the pattern of the pathnames of the folder(s) with the netcdf files with proxies; this needs "
                             "to be a set of paths for simulations where only the source parameters vary (swarm run)")
    parser.add_argument("-o", "--output", dest='processed_output_path', required=True,
                        help="the path to the folder where the netcdf with swarm run processed outputs is dumped")
    parser.add_argument("-st", "--stations", action="store_true",
                        help="the script processes the proxies for the gridded outputs by default, passing the --stations "
                             "argument means that it will process the proxies for the specified stations in the domain")
    parser.add_argument("-t", "--topogrid", dest='grid_topo_to_plot',  required=False,
                        help="(optional) the path to the netcdf file with the grid for the topography; if not "
                             "provided, topography is downloaded on the fly and therefore outgoing "
                             "internet connection is required")
    parser.add_argument("-sc", "--scale", action="store_true", required=False,
                        help="scale all to max value of all")
    parser.add_argument("-m", "--maxvalues", dest="path_to_maxall_netcdf", required=False,
                        help="Path to the netcdf file with max of all values. Used to generate all other plots and scale "
                             "them to the maximum of all. Useful for arranging outputs in grids and comparing.")

    args = parser.parse_args()
    compute_swarm_and_plots(args.event_id, args.paths_to_processed_netcdf, args.processed_output_path, args.stations, args.grid_topo_to_plot, args.scale, args.path_to_maxall_netcdf)
