sh clean.sh
. env.sh
event_id="event_001_test"
exec_dir="$PWD/$event_id"
mkdir -p ${exec_dir}
slip_id="ClustMedian-1.slip1"
input_yaml_path="$PWD/salvus_input.yaml"
slip_input_src="$PWD/inputs.src"
salvus_setup="/gpfs/projects/bsc44/earthquake/UCIS4EQ/salvus_urgent_simulations_setup/data/salvus_parameters_template/salvus_parameters.yaml"
region_fk1d="$PWD/crust1p0.1d"
dt=0.005
enqueue_compss --qos=debug --num_nodes=11 --exec_time=60 -d --job_execution_dir=${exec_dir} --log_dir=$PWD --worker_working_dir=$PWD $PWD/hpc_workflow.py --slip_id ${slip_id} --salvus_input ${input_yaml_path} --slip_src ${slip_input_src} --salvus_setup ${salvus_setup} --region ${region_fk1d} --dt ${dt} 

