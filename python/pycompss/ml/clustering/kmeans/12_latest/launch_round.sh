# AUTOMATIC LAUNCHER                                                                                                  
# ------------------                                                                                                  
# Launches a set of experiments using the launch.sh script                                                            
# Takes the dependeny id of the last launch and uses it for the next job.                                           


wait_and_get_jobID() {
  sleep 6
  jobID=$(bjobs | tail -n 1 | cut -c -7)
  echo "jobID = $jobID"
}

wait_and_get_jobID_MT() {
  sleep 6
  jobID=$(squeue | grep $(whoami) | sort -n - | tail -n 1 | awk '{ print $1 }')
  echo
}
                                                                                                                    
jobID=None

NUM_EXPERIMENTS=3

POINTS=(768000 1536000 3072000)
DIMENSIONS=(50 50 50)
ITERATIONS=(100 100 100)
FRAGMENTS=(512 512 512)

NUM_NODES_LEN=4
NUM_NODES=(2 3 5 9 17 33)

for (( tp=0; tp<$NUM_NODES_LEN; tp++))
do
  for (( i=0; i<$NUM_EXPERIMENTS; i++))
  do
    ./launch.sh $jobID ${NUM_NODES[tp]} 60 16 false ${POINTS[i]} ${DIMENSIONS[i]} ${ITERATIONS[i]} ${FRAGMENTS[i]}
    echo $jobID ${NUM_NODES[tp]} 60 16 false ${POINTS[i]} ${DIMENSIONS[i]} ${ITERATIONS[i]} ${FRAGMENTS[i]}
    wait_and_get_jobID_MT
  done
done





