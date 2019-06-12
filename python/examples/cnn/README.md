This is the Readme for:
convolutional NN

[Name]: cnn
[Contact Person]: support-compss@bsc.es
[Access Level]: private
[License Agreement]: Apache2
[Platform]: COMPSs

[Body]
== Description ==
cnn, using tensorflow and mnist datset

== Execution instructions ==

* Usage in local machine:

   ./run_local.sh <TRACING> <BASE_PATH> <NUM_MODELS>

   - Where:
       <TRACING>............... Enable or disable tracing ( true | false )
       <BASE_PATH>............. Path where model files should be stored
       <NUM_MODELS>............ Number of models to be trained in parallel

    - Example: ./run_local.sh false . 2

* Usage in supercomputer:

   ./launch.sh <JOB_DEPENDENCY> <NUM_NODES> <EXECUTION_TIME> <TRACING> <BASE_PATH> <NUM_MODELS>

   - Where:
       <JOB_DEPENDENCY>........ Job dependency (run after the given jobid - None if not needed)
       <NUM_NODES>............. Number of nodes for the reservation
       <EXECUTION_TIME>........ Walltime
       <TRACING>............... Enable or disable tracing ( true | false )
       <BASE_PATH>............. Path where model files should be stored
       <NUM_MODELS>............ Number of models to be trained in parallel

   - Example: ./launch.sh None 2 5 false . 2


== Build ==
No build is required
