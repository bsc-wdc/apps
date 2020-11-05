
#!/bin/sh
set -e

/usr/bin/aclocal
/usr/bin/automake -a -c
/usr/bin/autoconf
./configure  --with-cs-prefix=/opt/COMPSs//Bindings/c  CXXFLAGS=" -I/home/jorgee/git/compss/framework/tests/sources/sc/apps/c/matmul/matmul_objects/worker/gsbuild" CFLAGS=" -I/home/jorgee/git/compss/framework/tests/sources/sc/apps/c/matmul/matmul_objects/worker/gsbuild" LDFLAGS=" -L." 
