
#!/bin/sh
set -e

/usr/bin/aclocal
/usr/bin/automake -a -c
/usr/bin/autoconf
./configure --with-cs-prefix=/opt/COMPSs/Bindings/c

