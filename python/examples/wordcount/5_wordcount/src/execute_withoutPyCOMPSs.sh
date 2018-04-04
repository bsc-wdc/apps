sed -e '/@task/s/^#*/#/' wordcount.py > wordcount_wop.py

sed -e '/pycompss/s/^#*/#/' -i wordcount_wop.py

sed -e '/compss_wait_on/s/^#*/#/' -i wordcount_wop.py


#python wordcount_wop.py /gpfs/projects/bsc19/COMPSs_APPS/wordcount/data/all_8x.txt 10000000 
python wordcount_wop.py /gpfs/projects/bsc19/COMPSs_APPS/wordcount/data/text_102400.txt False 10000
python wordcount_wop.py /gpfs/projects/bsc19/COMPSs_APPS/wordcount/data/small/ True
