#!/bin/bash -x
#ESM.sh script for submit a job ensemble on a cluster
#$1 configuration file 
#$2 directory file of log example:  /var/www/cgi-bin/grbstar/etc/grb_users/admin

function usage()
{
echo "Usage: $0 "
echo "-f <conf file>: configuration file"
echo "-o <experiment log dir>: log of the ensemble experiment"
echo "-u <user>: job owner"
echo "-i <subjob index>: subjob index"
}

basedir=`echo $0 | sed -e "s|[^/]*$||"`

while getopts  "f:i:o:u:" flag
do
  case "$flag" in
        f) conf_file=$OPTARG
           ;;
        i) index=$OPTARG
           ;;
        o) logpath=$OPTARG
           ;;
        u) user=$OPTARG
           ;;
        :) usage
           exit 1;;
        ?) usage
           exit 1
           ;;
  esac
done

start_date=`grep start_date: $conf_file | sed "s/start_date://" | sed "s/^ *//" | sed "s/* $//"`

duration=`grep duration: $conf_file | sed "s/duration://" | sed "s/^ *//" | sed "s/* $//"`

if [ -z $start_date ]; then
        echo starting date not specified: $startdate
        usage
        exit 1
fi

if [ -z $resubmit ]; then
	resubmit=-1
fi


period=$duration

pyear=`echo $period | cut -b1-2`
pmonth=`echo $period | cut -b3-4`
echo $pmonth

if [ -z $pmonth ]; then
        pmonth=12
fi
pday=`echo $period | cut -b5-6`
echo $pday
 
if [ -z $pday ]; then
        pday=31
fi

totalnum=$((pyear*12 + pmonth))

durationmonth=1
#durationday=1

start_year=`echo $start_date | cut -b1-4`
start_month=`echo $start_date | cut -b5-6`
if [ -z $start_month ]; then
        start_month="01"
fi
start_day=`echo $start_date | cut -b7-8`
if [ -z $start_day ]; then
        start_day="01"
fi

final_year=`expr $pyear + $start_year`
final_month=`expr $pmonth + $start_month`
if [ $final_month -gt 12 ]; then
        echo "Duration of the run has been specified in a format not supported"
        exit 1
fi
if [ $final_month -lt 10 ]; then
   final_month=0$final_month
fi

final_day=`expr $pday + $start_day`
if [ $final_day -gt 31 ]; then
        echo "Duration of the run has been specified in a format not supported"
        exit 1
fi
if [ $final_day -lt 10 ]; then
   final_day=0$final_day
fi

time=`date +%s`
model_name=toy
computing_host=calypso
site_name="CMCC - Italy"
public_ip=193.204.78.27  #used for geo-referencing the site 



if [ $durationmonth -ne 0 ]; then
        d="${durationmonth}m"
elif [ $durationday -ne 0 ]; then
        d="${durationday}d"
else
        echo "Restart period is not valid"
        exit 1
fi


now=`date +%Y%m%d%H%M`

if [ ! -f $logpath/log_$index.log ]; then
  echo "startdate: ${start_year}${start_month}${start_day}" >> $logpath/log_$index.log
  echo "enddate: ${final_year}${final_month}${final_day}" >> $logpath/log_$index.log
  echo "restartperiod: $d " >> $logpath/log_$index.log
  echo "hostname: $computing_host " >> $logpath/log_$index.log
  echo "PublicIP: $public_ip" >> $logpath/log_$index.log
  echo "SiteName: $site_name" >> $logpath/log_$index.log
  echo "Beginning of Experiment: `date +%Y%m%d%H%M`" >> $logpath/log_$index.log
fi

#./toy_model $logpath/log_$index.log $total_num $start_month $start_year

logfile=$logpath/log_$index.log
pend_delay_av=31
run_delay_av=16

year=$start_year
month=$start_month

for i in `seq 1 $total_num`; do

	jobid=$RANDOM

	pend_time=`date +%Y%m%d%H%M`
	printf "${year}%02d01; pending; $pend_time; $i\n" $month >> $logfile

	pend_delay=$(($pend_delay_av + $RANDOM%60 - 30))
	sleep $pend_delay

	start_time=`date +%Y%m%d%H%M`
	printf "${year}%02d01; start; $start_time; $i; $jobid\n" $month >> $logfile

	run_delay=$(($run_delay_av + $RANDOM%30 - 15))
	sleep $run_delay

	finish_time=`date +%Y%m%d%H%M`
	printf "${year}%02d01; step_done; $start_time; $i; $jobid; $run_delay; $pend_delay\n" $month >> $logfile

	month=$(($month % 12 + 1))
	if [ $month == 1 ]; then
		   year=$(($year + 1))
	   fi
   done
	
   printf "\n" >> $logfile

   if [ $? != 0 ]; then
	exit 1
   fi

