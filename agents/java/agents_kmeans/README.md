<!-- LOGOS AND HEADER -->
<h1 align="center">
  <br>
  <a href="https://www.bsc.es/">
    <img src="files/logos/bsc_logo.png" alt="Barcelona Supercomputing Center" height="60px">
  </a>
  <a href="https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar/">
    <img src="files/logos/COMPSs_logo.png" alt="COMP Superscalar" height="60px">
  </a>
  <br>
  <br>
  Agent Examples - KMeans
  <br>
</h1>

<!-- SECTIONS -->

<!-- SOURCES STRUCTURE -->
# Folder Structure
  - **src** contains a maven project with the source code of the application.
  - **scripts** contains useful execution scripts
    - **deploy_agents.sh** script to create a new deployment of COMPSs agents able to run the application
    - **launch_app.sh** script to request the master agent the execution of the application

<!-- Execution -->
# Execution
## Application compilation
To compile the application get into the application folder and create the application package using maven

```bash
$ cd application
$ mvn clean package
```

## Agents deployment
To create a new deployment for the application, run the deploy_app.sh script indicating the number of agents to deploy and the base port.
The base port will be used to compute the ports for the rest and comm interfaces (rest_port = base_port+X01; comm_port=base_port+X02)  
By default, the script deploys 2 agents based on port 46000.

```bash
$ cd scripts/
$ ./deploy_agents.sh             # deploys 2 agents based on port 46000
$ ./deploy_agents.sh 3           # deploys 3 agents based on port 46000
$ ./deploy_agents.sh 3 45000     # deploys 3 agents based on port 45000
```

## Compute request
To request the execution of the main method, call the ./launch_app script indicating the master ip (127.0.0.1) and its rest port.

```bash
$ ./launch_app.sh 127.0.0.1 46101
```

<!-- CONTACT -->
# Contact

:envelope: COMPSs Support <support-compss@bsc.es> :envelope:

Workflows and Distributed Computing Group (WDC)

Department of Computer Science (CS)

Barcelona Supercomputing Center (BSC) 


<!-- LINKS -->
[1]: http://compss.bsc.es
