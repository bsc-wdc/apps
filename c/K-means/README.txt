Build:

compile <flags> kmeans

<flags> should be --ompss if you are building the OmpSs version, and --ompss --cuda if you are building a version that uses Cuda.

The same is applied for OmpSs-2 version, if you want more information use the flag -h.

Run:

To run the CPU version, use the queue_cpu script, for the GPU the queue_gpu. Consider the machine where you are going to run the application and modify them correctly.

Generating data:

See README.txt in generator/ folder. 

