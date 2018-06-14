This is the Readme for:
Discrete

[Name]: Discrete
[Contact Person]: support-compss@bsc.es
[Access Level]: public
[License Agreement]: Apache2
[Platform]: COMPSs

[Body]
 == Description ==
Discrete application

== Execution instructions ==
Usage:
runcompss discrete.Discrete <...>

where:
		 * - debug: Set the debug mode on

Example:
runcompss discrete.Discrete <...>

== Build ==
=== Option 1: Native java ===
cd ~/workspace_java/discrete/; javac src/main/java/discrete/*.java
cd src/main/java/; jar cf discrete.jar discrete/
cd ../../../; mv src/main/java/discrete.jar jar/

=== Option 2: Maven ===
cd ~/workspace_java/discrete/
mvn clean package
