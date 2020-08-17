#Gnuplot template for the projection functions

#Prepare the axes
#REPLACE_BY_XRANGE
set xlabel "Number of Processes"
set logscale x
set yrange [0:105]
set ylabel "Efficiency"
set ytics ( 0, "10%%" 10, "20%%" 20, "30%%" 30, "40%%" 40, "50%%" 50, "60%%" 60, "70%%" 70, "80%%" 80, "90%%" 90, "100%%" 100 )
set grid ytics

set style line 1 lt 7 lw 1.5 lc rgb "#0E3D59"
set style line 2 lt 7 lw 1.5 lc rgb "#88A61B"
set style line 3 lt 7 lw 1.5 lc rgb "#F29F05"
set style line 4 lt 7 lw 1.5 lc rgb "#F25C05"
set style line 5 lt 7 lw 1.5 lc rgb "#D92525"

set key left bottom Left reverse

#REPLACE_BY_PARA_FUNCTION
#REPLACE_BY_LOAD_FUNCTION
#REPLACE_BY_COMM_FUNCTION
#REPLACE_BY_COMP_FUNCTION
#REPLACE_BY_GLOB_FUNCTION

plot para(x) title "Parallel Efficiency" ls 1,\
     load(x) title "Load Balance" ls 2,\
     comm(x) title "Communication Efficiency" ls 3,\
     comp(x) title "Computation Scalability" ls 4,\
     glob(x) title "Global Efficiency" ls 5,\
     '-' with points notitle ls 1,\
     '-' with points notitle ls 2,\
     '-' with points notitle ls 3,\
     '-' with points notitle ls 4,\
'-' with points notitle ls 5
