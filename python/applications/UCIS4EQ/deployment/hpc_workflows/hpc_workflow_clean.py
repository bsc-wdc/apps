
@container(engine="SINGULARITY", image="$SLIPGEN_IMAGE", options="-e --bind {{workingdir}}:/workspace/ --pwd /workspace")
@binary(binary="/opt/scripts/launcher.sh", args="-o rupture --dt {{dt}} -v {{fk_file}} -s {{input_src}}" , working_dir="{{workingdir}}")
@task(input_src=FILE_IN, fk_file=FILE_IN, workingdir=DIRECTORY_INOUT)
def slipgen(input_src, dt, fk_file, workingdir):
    pass

@task(input_data=FILE_IN, rupture=DIRECTORY_IN, salvus_setup=FILE_IN, working_dir=DIRECTORY_INOUT)
def salvus_prepare(input_data, rupture, salvus_setup, working_dir):
    os.chdir(working_dir)
    pre_process(input_data, rupture + "/scratch/outdata/rupture/rupture.srf", salvus_setup, working_dir) 

@mpi(runner="mpirun", binary="$SALVUS_BINARY", args="compute {{prepare_path}}/salvus_input_rupture.toml", processes="$SALVUS_PROCESSES" , processes_per_node= "$SALVUS_PPN", working_dir="{{working_dir}}")
@task(prepare_path=DIRECTORY_IN, working_dir= DIRECTORY_INOUT)
def salvus_run(prepare_path, working_dir):
    pass

@task(UC_input = FILE_IN, salvus_setup= FILE_IN, grid_coordinates= DIRECTORY_IN, simu_folder=DIRECTORY_IN, output_path=DIRECTORY_INOUT)
def salvus_post(UC_input, salvus_setup, grid_coordinates, simu_folder, output_path):
    process_outputs_grid( UC_input, salvus_setup,  grid_coordinates, simu_folder, output_path=output_path)

def main():
    slip_id, input_path, salvus_setup, slip_input_src, region_fk1d, dt = parse_arguments()
    slipgen_dir, salvus_wrapper_dir, salvus_dir, salvus_post_dir = create_output_dirs()
    
    slipgen(slip_input_src, dt, region_fk1d, slipgen_dir)
    salvus_prepare(input_path, slipgen_dir, salvus_setup, salvus_wrapper_dir)
    salvus_run(salvus_wrapper_dir, salvus_dir)
    salvus_post(input_path, salvus_setup, salvus_wrapper_dir, salvus_dir, salvus_post_dir) 

if __name__ == '__main__':
    main()
