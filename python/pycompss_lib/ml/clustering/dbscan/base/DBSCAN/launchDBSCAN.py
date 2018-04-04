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
    This file contains the DBSCAN launcher.
    :input dataFile:    path to the file where the dataset is stored. Loading is performed by
                        numpy.loadtxt().
    :input fragSize:    parameter to determine the number of chunks in which the dataset will
                        be split.
    :input epsilon:     maximum distance under which two points are considered neighbors.
    :input minPoints:   minimum number of neighbors for a point to be considered a core point.
    :input numParts:    scale parameter.
    :input dim:         optional parameter, 2D/3D for 2D/3D dataset plotting
    :output outDBSCAN:  text file with each cluster and its points.
"""

import sys
import numpy as np
import DBSCAN as DBSCAN

def main(dataFile, fragSize, epsilon, minPoints, numParts, *args):
    [defCluster, fragVec] = DBSCAN.DBSCAN(dataFile, int(fragSize), float(epsilon), int(minPoints),                                            int(numParts))
    newName = dataFile[dataFile.find('.')+1:dataFile.rfind('.')]
    newName = '.'+newName+'n.txt'
    dataset = np.loadtxt(newName)
    if len(args) > 0 and args[0] == '2D':
        import matplotlib.pyplot as plt
        from matplotlib import colors	
        fig, ax = plt.subplots()
        plt.hlines(fragVec[1],0,1, 'k', 'dashdot', linewidths = 0)
        plt.vlines(fragVec[0],0,1, 'k', 'dashdot', linewidths = 0) 
        ax.scatter([p[0] for p in dataset], [p[1] for p in dataset], s = 1)
        plt.savefig('dataset.png')
        plt.close()
        colours = [hex for (name, hex) in colors.cnames.iteritems()]
        fig, ax = plt.subplots()
        plt.hlines(fragVec[1],0,1, 'k', 'dashdot', linewidths = 0.1)
        plt.vlines(fragVec[0],0,1, 'k', 'dashdot', linewidths = 0.1) 
        for i,key in enumerate(defCluster):
                ax.scatter([p[0] for p in defCluster[i]], [p[1] for p in defCluster[i]],                                        color=colours[i], s=1)
        plt.savefig('clusters.png')
        plt.close()
    elif len(args) > 0 and args[0] == '3D':
        import matplotlib.pyplot as plt
        from matplotlib import colors	
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter([p[0] for p in dataset], [p[1] for p in dataset], [p[2] for p in dataset], s = 1)
        plt.grid()
        plt.savefig('plot/dataset.png')
        plt.close()
        colours = [hex for (name, hex) in colors.cnames.iteritems()]
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        for c in defCluster:
                ax.scatter([p[0] for p in defCluster[c]], [p[1] for p in defCluster[c]], [p[2] for                              p in defCluster[c]], color=colours[c], s=1)
        plt.show()
    f = open('outDBSCAN.txt', 'w')
    for num,lista in enumerate(defCluster):
        f.write('Cluster ' + str(num) + ':\n')
        for point in defCluster[num]:
                f.write(str(point) + '\n')	
    f.close()
    
if __name__=='__main__':
    main(sys.argv[1],float(sys.argv[2]),float(sys.argv[3]), int(sys.argv[4]), sys.argv[5],                                     sys.argv[6] if len(sys.argv) >= 7 else 0)
