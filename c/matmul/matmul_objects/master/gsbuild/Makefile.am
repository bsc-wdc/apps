#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
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

bin_PROGRAMS = Matmul

ACLOCAL_AMFLAGS=-I m4

Matmul_CPPFLAGS = -I../../src -I../../include -Wno-write-strings -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux/ -I$(CS_HOME)/../bindings-common/include -I$(CS_HOME)/include -Wall

Matmul_SOURCES = Matmul-stubs.cc \
        Matmul.cc

Matmul_LDFLAGS =

Matmul_LDADD = -L../../lib -L$(LIBJAVAFOLDER) -ljvm -ldl -L$(CS_HOME)/../bindings-common/lib/$(TARGET_HOST) -lbindings_common -L$(CS_HOME)/lib/$(TARGET_HOST) -lcbindings -L$(BOOST_LIB) -lboost_iostreams -lboost_serialization -lpthread -lboost_system

COMPSSGEN_FLAGS =

Matmul-executor.cc Matmul-stubs.cc Matmul.h: Matmul.idl
	$(COMPSSGEN) $(COMPSSGEN_FLAGS) Matmul.idl
	
