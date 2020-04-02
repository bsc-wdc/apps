# Hello world Application

This application presents the basic usage of persistent storage with PyCOMPSs.

To this end, shows how to declare tasks and persistent objects and how
they can be used trasnparently as task parameters.

This application is composed of two main files:

```
src
  |- utils
  |    |- __init__.py
  |    |- classes.py
  |
  |- hello_world.py
```

More in detail, this application declares three tasks within the
```src/hello_world.py``` file:

1. **create_greeting**: Receives an string and instantiates a persistent object
with the string.
2. **greet**: Receives the persistent object and retrieves its content using an
object method.
3. **check_greeting**: Compares the initial message and the persistent object
content.

And the persistent object declaration can be found in the ```src/utils/classes.py```.

In addition, this application also contains a set of scripts to submit the
```hello_world.py``` application within the <ins>MN4 supercomputer</ins>
queuing system for the three currently supported persistent storage frameworks,
which are: **Redis**, **dataClay** and **Hecuba**.
The following commands submit the execution *without a job dependency*,
requesting *2 nodes*, with *5 minutes* walltime and with *tracing and graph
generation disabled*.

> Please, check the **requirements** before using the following commands.

* Launch with dataClay:
```bash
./launch_with_dataClay.sh None 2 5 false $(pwd)/src/hello_world.py
```

* Launch with Hecuba:
```bash
./launch_with_Hecuba.sh None 2 5 false $(pwd)/src/hello_world.py
```
* Launch with Redis:
```bash
./launch_with_redis.sh None 2 5 false $(pwd)/src/hello_world.py
```

And also, contains a script to run the ```hello_world.py``` application
<ins>locally</ins> with **Redis**:

```bash
./run_with_redis.sh
```
