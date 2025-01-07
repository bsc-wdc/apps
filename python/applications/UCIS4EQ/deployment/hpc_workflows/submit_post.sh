. env.sh
event_id="event_001_test"
exec_dir="$PWD/${event_id}"
mkdir -p ${exec_dir}
slip_id="ClustMedian-1.slip1"
processed_data="'${exec_dir}/trial_*/salvus_post/'"

rm -rf ${exec_dir}/salvus_post_swarm
rm -rf ${exec_dir}/salvus_plot

enqueue_compss --qos=debug --num_nodes=1 --exec_time=60 -d --job_execution_dir=${exec_dir} --log_dir=$PWD --worker_working_dir=$PWD $PWD/hpc_post_process.py -e ${event_id} --processeddata ${processed_data} --output ${exec_dir} -t /gpfs/projects/bsc44/earthquake/UCIS4EQ/data/MED_SAMOS_IZMIR/TOPOGRAPHY/grid_topo_file_1p5Hz.nc

