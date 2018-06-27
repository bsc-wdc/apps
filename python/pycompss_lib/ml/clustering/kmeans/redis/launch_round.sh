# AUTOMATIC LAUNCHER                                                                                                  
# ------------------                                                                                                  
# Launches a set of experiments using the launch.sh script                                                            
# Takes the dependeny id of the last launch and uses it for the next job.                                           


wait_and_get_jobID() {
  sleep 2
  jobID=$(bjobs | tail -n 1 | cut -c -7)
  echo "jobID = $jobID"
}

wait_and_get_jobID_MT() {
  sleep 6
  jobID=$(squeue | grep $(whoami) | sort -n - | tail -n 1 | awk '{ print $1 }')
  echo
}
                                                                                                                    
jobID=None

NUM_EXPERIMENTS=1

POINTS=(128000000)
DIMENSIONS=(50)
ITERATIONS=(6)
FRAGMENTS=(512)

NUM_NODES_LEN=1
NUM_NODES=(2 4 8 16)

for (( i=0; i<1; i++))
do
  for (( tp=0; tp<$NUM_NODES_LEN; tp++))
  do
    for (( i=0; i<$NUM_EXPERIMENTS; i++))
    do
      ./launch.sh $jobID ${NUM_NODES[tp]} 60 48 false ${POINTS[i]} ${DIMENSIONS[i]} ${ITERATIONS[i]} ${FRAGMENTS[i]}
      echo $jobID ${NUM_NODES[tp]} 60 48 false ${POINTS[i]} ${DIMENSIONS[i]} ${ITERATIONS[i]} ${FRAGMENTS[i]}
      wait_and_get_jobID_MT
    done
  done

done
