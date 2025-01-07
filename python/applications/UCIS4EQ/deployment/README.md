# Installation guide
_________________
### 1- Install software Dependencies

* Docker (Community Edition)
  ```
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER
  ```
  Find more installation options in:
  https://docs.docker.com/install/linux/docker-ce/ubuntu/

* Docker compose
  ```
  sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  ```
  Find more installation options in:
  https://docs.docker.com/compose/install/

### 2 - Install image on the local host

Docker images being stored in the Docker registry of https://gitlab.bsc.es repo, logging into the docker reistry is necessary.

```
  docker login registry.gitlab.bsc.es
```

##### Option 1: Obtain the docker-composer.yml file from the GitLab repository

```
  curl --request GET --header 'PRIVATE-TOKEN: ER9nSBQo8xsiczs47pAn' 'https://gitlab.bsc.es/api/v4/projects/2703/repository/files/deployment%2Fdockers%2Fdocker-compose.yml/raw?ref=migration-fixes' >| docker-compose.yml
```

```
  docker-compose up
```

##### Option 2: Get and run the docker containers manually

1 - Pull UCIS4EQ images from the registry, at registry.gitlab.bsc.es
```
  docker pull registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/listener
  docker pull registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/dispatcher
  docker pull registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/slip-gen
```

2 - Indicate the configuration file of the ucis4eq listener service
```
  export UCIS4EQ_LISTENER_CONFIG=/path/to/config_events.json
```

3 - Start the Docker containers

```
  docker run -ti --net="host" -v "$PWD:/workspace" registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/dispatcher
  docker run -ti --net="host" -v "$UCIS4EQ_LISTENER_CONFIG:/root/services/config.json" -v "$PWD:/workspace" registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/listener
  docker run -ti -v "$PWD:/workspace" registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/slip-gen
```

# Additional actions
_________________
### Generate the ucis4eq-service Docker local image

This steps is just for users who wants to create the Docker image from scratch.

1 - Obtain the Dockerfiles with the Docker images setup
```
  curl --request GET --header 'PRIVATE-TOKEN: ER9nSBQo8xsiczs47pAn' 'https://gitlab.bsc.es/api/v4/projects/2703/repository/files/deployment%2Fdockers%2FDockerfile-listener/raw?ref=migration-fixes' >| Dockerfile-listener
  curl --request GET --header 'PRIVATE-TOKEN: ER9nSBQo8xsiczs47pAn' 'https://gitlab.bsc.es/api/v4/projects/2703/repository/files/deployment%2Fdockers%2FDockerfile-dispatcher/raw?ref=migration-fixes' >| Dockerfile-dispatcher
```

2 - Building the images with the environement for ucis4eq services

```
  docker build -f Dockerfile-listener -t registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/listener .
  docker build -f Dockerfile-dispatcher -t registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/dispatcher .
```

3 - Register the built images
```
  docker push registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/listener
  docker push registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/dispatcher
```


### Start container in a terminal

```
  docker run -ti --net="host" -v "$PWD:/workspace" --entrypoint=bash registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/dispatcher
  docker run -ti --net="host" -v "$PWD:/workspace" --entrypoint=bash registry.gitlab.bsc.es/wavephenomenagroup/ucis4eq/listener
```
