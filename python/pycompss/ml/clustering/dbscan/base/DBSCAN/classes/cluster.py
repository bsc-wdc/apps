#
#  Copyright 2.02-2017 Barcelona Supercomputing Center (www.bsc.es)
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
"""
@author: csegarra

PyCOMPSs Mathematical Library: Clustering: DBSCAN
=================================================
    This file contains the Cluster class definition.
"""

import numpy as np
class Cluster(object):

	def __init__(self,  *args, **kwargs):
        	super(Cluster, self).__init__(*args, **kwargs)
        	self.square=[]
        	self.points=[]

	def add(self, points, square):
		self.points=np.asarray(points)
		self.square=square

	def addPoint(self,p):
		self.points=np.vstack((self.points, np.asarray(p)))

	def addSquare(self,s):
		self.square=list(set(self.square) | set(s))

	def merge(self, q):
                if len(self.points):
                    for p in q.points:
                            self.addPoint(p)
                    self.addSquare(q.square)
                else: 
                    tmp=(q.points, q.square)
                    self.add(*tmp)
