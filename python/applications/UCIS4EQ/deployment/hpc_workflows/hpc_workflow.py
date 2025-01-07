from pycompss.api.task import task
from pycompss.api.binary import binary
from pycompss.api.mpi import mpi
from pycompss.api.container import container
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.parameter import *

import os
import argparse

from salvus_urgent_wrapper.salvus_preprocess.salvus_preprocess import pre_process
from salvus_urgent_wrapper.salvus_postprocess.helpers import process_outputs_grid
from utils.job_tracker import run_track

def create_output_dirs():
    import os
    trial_dir = os.getcwd()

    slipgen_dir = trial_dir + "/slipgen"
    if not os.path.exists(slipgen_dir):
        os.mkdir(slipgen_dir)
    salvus_wrapper_dir = trial_dir + "/salvus_wrapper"
    if not os.path.exists(salvus_wrapper_dir):
        os.mkdir(salvus_wrapper_dir)
    salvus_dir = trial_dir + "/salvus"
    if not os.path.exists(salvus_dir):
        os.mkdir(salvus_dir)
    salvus_post_dir = trial_dir + "/salvus_post"
    if not os.path.exists(salvus_post_dir):
        os.mkdir(salvus_post_dir)
    return slipgen_dir, salvus_wrapper_dir, salvus_dir, salvus_post_dir

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-si", "--salvus_input", dest='input_yaml_path', required=True,
                        help="Path to the Salvus simulation input YAML file.")
    parser.add_argument("-src", "--slip_src", dest='slip_input_src', required=True,
                        help="Path to the slip input src file")
    parser.add_argument("-id", "--slip_id", dest="slip_id", required=True,
                        help="Slip identifier")
    parser.add_argument("-ss", "--salvus_setup", dest='salvus_setup',  required=True,
                        help="Path to the Salvus Setup YAML file")
    parser.add_argument("-r", "--region", dest='region_fk1d', required=True,
                        help="Path to region fk1d file")
    parser.add_argument("-d", "--dt", dest="dt", required=True,
                        help="Dt parameter for slip generation")
    args = parser.parse_args()
    dt = float(args.dt)
    return args.slip_id, args.input_yaml_path, args.salvus_setup, args.slip_input_src, args.region_fk1d, dt


@container(engine="SINGULARITY", image="$SLIPGEN_IMAGE", options="-e --bind {{workingdir}}:/workspace/ --pwd /workspace")
@binary(binary="/opt/scripts/launcher.sh", args="-o rupture --dt {{dt}} -v {{fk_file}} -s {{input_src}}" , working_dir="{{workingdir}}")
@task(input_src=FILE_IN, fk_file=FILE_IN, returns=1)
def slipgen(input_src, dt, fk_file, workingdir):
    pass

@task(input_data=FILE_IN, salvus_setup=FILE_IN, returns=1)
def salvus_prepare(input_data, rupture, salvus_setup, working_dir, slip_gen_result)
    rupture_file = rupture + "/scratch/outdata/rupture/rupture.srf"
    os.chdir(working_dir)
    pre_process(input_data, rupture_file, salvus_setup, working_dir)
    return true

@mpi(runner="mpirun", binary="$SALVUS_BINARY", args="compute {{prepare_path}}/salvus_input_rupture.toml", processes="$SALVUS_PROCESSES" , processes_per_node= "$SALVUS_PPN", working_dir="{{working_dir}}")
@task(returns=1)
def salvus_run(prepare_path, working_dir, prepare_result):
    pass

@task(UC_input = FILE_IN, salvus_setup= FILE_IN)
def salvus_post(UC_input, salvus_setup, grid_coordinates, simu_folder, output_path, salvus_res):
    process_outputs_grid( UC_input, salvus_setup,  grid_coordinates, simu_folder, output_path=output_path)

@io()
@task()
def job_track(prepare_path, salvus_dir, track_interval, prepare_result):
    run_track(prepare_path+"/salvus_input_rupture.toml", salvus_dir, track_interval, salvus_dir+"/snapshots")

if __name__ == "__main__":

    slip_id, input_path, salvus_setup, slip_input_src, region_fk1d, dt = parse_arguments()
    slipgen_dir, salvus_wrapper_dir, salvus_dir, salvus_post_dir = create_output_dirs()
    
    sg_out = slipgen(slip_input_src, dt, region_fk1d, slipgen_dir)
    sp_out = salvus_prepare(input_path, slipgen_dir, salvus_setup, salvus_wrapper_dir, sg_out)
    job_track(salvus_wrapper_dir, salvus_dir, 10, sg_out)
    sw_out = salvus_run(salvus_wrapper_dir, salvus_dir, sp_out)
    salvus_post(input_path, salvus_setup, salvus_wrapper_dir, salvus_dir, salvus_post_dir, sw_out) 


