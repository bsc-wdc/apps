sed -e '/@task/s/^#*/#/' sort.py > sort_wop.py

sed -e '/pycompss/s/^#*/#/' -i sort_wop.py

sed -e '/compss_wait_on/s/^#*/#/' -i sort_wop.py


python sort_wop.py 20 50 4 20 2 4 12345 false
