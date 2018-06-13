#
#  Copyright 2.02-2017 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#PyCOMPSs Mathematical Library: Clustering: DBSCAN
#=================================================
#   This file contains different test run commands. 

# Running with the help flag will output the following:
# runcompss DBSCAN.py --help
# usage: DBSCAN.py [-h] [--is_mn] [--print_times] epsilon min_points datafile
#
# DBSCAN Clustering Algorithm implemented within the PyCOMPSs framework. For a
# detailed guide on the usage see the user guide provided.
#
# positional arguments:
#   epsilon        Radius that defines the maximum distance under which
#                  neighbors are looked for.
#   min_points     Minimum number of neighbors for a point to be considered core
#                  point.
#   datafile       Numeric identifier for the dataset to be used. For further
#                  information see the user guide provided.
#
# optional arguments:
#   -h, --help     show this help message and exit
#   --is_mn        If set to true, this tells the algorithm that you are running
#                  the code in the MN cluster, setting the correct paths to the
#                  data files and setting the correct parameters. Otherwise it
#                  assumes you are running the code locally.
#   --print_times  If set to true, the timing for each task will be printed
#                  through the standard output. NOTE THAT THIS WILL LEAD TO
#                  EXTRA BARRIERS INTRODUCED IN THE CODE. Otherwise only the
#                  total time elapsed is printed.
#

# DBSCAN Local execution: -d for debugging, -t for tracing and -g for the dependency graph.
runcompss \
    --lang=python \
    ./DBSCAN.py 0.1 10 1
