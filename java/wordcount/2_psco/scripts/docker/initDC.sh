#!/bin/bash

# sudo service docker start
# docker load -i dataclay_java

export COMPOSE_HTTP_TIMEOUT=300

docker-compose stop
docker-compose rm --all
 
docker-compose up -d ds1postgres ds2postgres lmpostgres
sleep 10
docker-compose up -d logicmodule
sleep 10
docker-compose --verbose up -d ds1java ds2java 

