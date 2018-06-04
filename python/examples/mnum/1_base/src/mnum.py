#!/usr/bin/python
#
#  Copyright 2002-2018 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

import numpy as np
from decimal import getcontext, setcontext, ExtendedContext
digits = 8192
#prec = (digits*np.log(10.0) / np.log(2.0) + 1.0)

if __name__ == "__main__":
    from pycompss.api.api import compss_wait_on
    from src.function import Function, fun2, fun6, cos,exp
    from src.error import AError
    from src.method import SMethodP, SMethodP2
    from decimal import Decimal
    import time
    import sys
    prec = 4096
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        prec = int(sys.argv[2])
    else:
        print "Default Params"
    step = float((2.0-1.5)/n)
    
    #prec = 4096
    setcontext(ExtendedContext)
    getcontext().prec = prec
    
    
    f = Function(fun6,prec)
    e = AError(digits,prec)

    #m = SMethodP(f,e,prec,3)
    m = SMethodP(f,e,prec,n)
    print("Main: ",getcontext().prec)
    initial_aprox = list(np.arange(1.5,2.0,step))
    start = time.time()
    r = m.solution(initial_aprox)
    #print(m.solution([1.5,1.55,1.6,1.65,1.70,1.75,1.8,1.85,1.9,1.95]))
    print "Ellapsed Total Time {} (s)".format(time.time()-start)
    #m.plot()
    
    #f2 = Function(fun2,prec)
    #e2 = AError(digits,prec)
    #m2 = KMethodP(f2,e2,prec,10)
    #print(m2.solution([1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95]))
    
    
