This is the Readme for:

max_norm

[Name]: max_norm
[Contact Person]: support-compss@bsc.es
[Access Level]: public
[License Agreement]: Apache2
[Platform]: COMPSs

[Body]
== Description ==
max


== Execution instructions ==
Usage:
max.py <numP> <dim> <numFrag>

where:
                 * - numP: number of points
                 * - dim: dimensions
                 * - numFrag: number of fragments

* Usage in local machine:

    ./run_local.sh <TRACING> <POINTS> <DIM> <FRAGMENTS>

    - Where:
        <TRACING>............... Enable or disable tracing ( true | false )
        <POINTS>................ Number of points
        <DIMENSIONS>............ Number of dimensions
        <FRAGMENTS>............. Number of fragments

     - Example: ./run_local.sh false 16000 3 16

* Usage in supercomputer:

    ./launch.sh <JOB_DEPENDENCY> <NUM_NODES> <EXECUTION_TIME> <TRACING> <POINTS> <DIM> <FRAGMENTS>

    - Where:
        <JOB_DEPENDENCY>........ Job dependency (run after the given jobid - None if not needed)
        <NUM_NODES>............. Number of nodes for the reservation
        <EXECUTION_TIME>........ Walltime
        <TRACING>............... Enable or disable tracing ( true | false )
        <POINTS>................ Number of points
        <DIMENSIONS>............ Number of dimensions
        <FRAGMENTS>............. Number of fragments

    - Example: ./launch.sh None 2 5 false 16000 3 16

== Build ==
No build is required
