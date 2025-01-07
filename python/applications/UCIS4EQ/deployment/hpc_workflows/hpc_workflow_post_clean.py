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

from utils import create_swarm_out_dirs
from utils import parse_swarm_arguments

@task(salvus_post_swarm_dir=DIRECTORY_INOUT)
def salvus_post_swarm(input_data, salvus_post_swarm_dir, stations):
    process_swarm(input_data, salvus_post_swarm_dir, stations) 

@task(salvus_post_swarm_dir=DIRECTORY_INOUT)
def salvus_plot(salvus_post_swarm_dir, grid_topo_to_plot, scale, path_to_maxall_netcdf):
    generate_plots(salvus_post_swarm_dir, grid_topo_to_plot, scale, path_to_maxall_netcdf)
   
@binary(binary="tar", args="-czvf {{plots_file}} {{salvus_post_swarm_dir}}/*.png", fail_by_exit_value=True)
@task(salvus_post_swarm_dir=DIRECTORY_INOUT, plots_file=FILE_OUT)
#@task(salvus_post_swarm_dir=DIRECTORY_INOUT)
def compress(salvus_post_swarm_dir,plots_file):
    pass

def main():
    args = parse_args()
    salvus_post_swarm_dir, salvus_plot_dir, plots_file = create_swarm_out_dirs(args.processed_output_path, args.event_id)
    salvus_post_swarm(args.input_data, salvus_post_swarm_dir, args.stations)
    salvus_plot(salvus_post_swarm_dir, args.grid_topo_to_plot, args.scale, args.path_to_maxall_netcdf)
    compress(salvus_post_swarm_dir, plots_file)
    
if __name__ == '__main__':
    main()
