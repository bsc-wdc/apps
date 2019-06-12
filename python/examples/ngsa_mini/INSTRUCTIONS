# ngsa-mini-py

Test stub input:

	
        $ runcompss -d -g --pythonpath=/home/compss/apps/ngsa-mini-py/bin /home/compss/apps/ngsa-mini-py/bin/workflow.py ./input/stub_input/bwa_db/reference.fa ./input/stub_input/seq_contig.md ./input/stub_input/reference.fa ./input/stub_input/reference.fa.fai ./input/stub_input/00-read-rank/ 4

To repeat:

		$ rm -r workflow_*



Provar l'original

        mpiexec -n 2 /home/compss/apps/ngsa-mini-master/bin/workflow /data/bwa_db/reference.fa ~/dataset/seq_contig.md ~/dataset/reference.fa ~/dataset/reference.fa.fai ~/dataset/mini_dataset/


Provar el py

        runcompss -d -g --lang=python --pythonpath=/home/compss/apps/ngsa-mini-py/bin/ /home/compss/apps/ngsa-mini-py/bin/workflow.py /data/bwa_db/reference.fa ~/dataset/seq_contig.md ~/dataset/reference.fa ~/dataset/reference.fa.fai ~/dataset/mini_dataset/ 2


Returns amb els stub scripts:

Return de l'original:
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_01.sh /home/compss/dataset/mini_dataset//0 ./workflow_MED/0 /data/bwa_db/reference.fa /home/compss/dataset/seq_contig.md
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_01.sh /home/compss/dataset/mini_dataset//1 ./workflow_MED/1 /data/bwa_db/reference.fa /home/compss/dataset/seq_contig.md
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_02.sh 0 ./workflow_MED/0 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_02.sh 1 ./workflow_MED/1 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_03.sh 1 2 ./workflow_MED/1 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_03.sh 0 2 ./workflow_MED/0 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_04.sh ./workflow_MED/0 /home/compss/dataset/reference.fa.fai
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_04.sh ./workflow_MED/1 /home/compss/dataset/reference.fa.fai
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_05.sh ./workflow_MED/0 ./workflow_OUT /home/compss/dataset/reference.fa 
Call: /home/compss/apps/ngsa-mini-master/bin//workflow_05.sh ./workflow_MED/1 ./workflow_OUT /home/compss/dataset/reference.fa


Return del py:
compss@bsc:~/.COMPSs/workflow.py_16/jobs$ cat job*_NEW.out | grep Call | sort
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_01.sh /home/compss/dataset/mini_dataset//0 ./workflow.py_MED/0 /data/bwa_db/reference.fa /home/compss/dataset/seq_contig.md
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_01.sh /home/compss/dataset/mini_dataset//1 ./workflow.py_MED/1 /data/bwa_db/reference.fa /home/compss/dataset/seq_contig.md
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_02.sh 0 ./workflow.py_MED/0 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_02.sh 1 ./workflow.py_MED/1 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_03.sh 0 2 ./workflow.py_MED/0 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_03.sh 1 2 ./workflow.py_MED/1 ../SH_GLOBAL
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_04.sh ./workflow.py_MED/0 /home/compss/dataset/reference.fa.fai
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_04.sh ./workflow.py_MED/1 /home/compss/dataset/reference.fa.fai
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_05.sh ./workflow.py_MED/0 ./workflow.py_OUT /home/compss/dataset/reference.fa 
Call: /home/compss/apps/ngsa-mini-py/bin/workflow_05.sh ./workflow.py_MED/1 ./workflow.py_OUT /home/compss/dataset/reference.fa 

