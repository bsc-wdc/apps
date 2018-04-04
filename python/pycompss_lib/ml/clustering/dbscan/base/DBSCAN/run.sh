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
#@author: csegarra
#
#PyCOMPSs Mathematical Library: Clustering: DBSCAN
#=================================================
#    This file contains different test run commands.

#Running without COMPSs. Comment all the @task, compss_wait_on and COMPSs imports.
#python ./launchDBSCAN.py ./data/moons.txt 3 0.015 10 1 2D

#Local execution: -d for debugging, -t for tracing and -g for the dependency graph.
runcompss --lang=python -t ./launchDBSCAN.py ./data/blobs_small.txt 8 0.015 10 4 2D

#Running on a Cluster with COMPSs installed.
#enqueue_compss --lang=python --num_nodes=6 --exec_time=10 --worker_working_dir=gpfs/home/bsc19/bsc19685/tmp/ ./launchDBSCAN.py ./data/blobs.txt 8 0.015 10 24

