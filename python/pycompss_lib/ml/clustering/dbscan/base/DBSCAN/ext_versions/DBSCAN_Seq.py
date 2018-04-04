# Reference paper: http://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
# Ideas:
# 1) RegionQuery por fragmentos de puntos y reduce para juntar todos los puntos resultantes.
#    Posible problema: expandCluster tambien podria hacerse por fragmentos de datos.
#    Pero entonces regionQuery no podria ser tarea (nesting no soportado)
# 2) El calculo mas costoso son las distancias. Precalcular las distancia por fragmentos de datos.
#    consultar esos datos para el resto de calculos. Otra opcion es usar kd-trees para evitar el calculo de distancias.
# 3) Para regionQuery, subdividir el espacio de puntos y llamar a regionQuery solo con bloques contiguos
from collections import defaultdict
import numpy as np
import random
visited = defaultdict(bool)
noise = defaultdict(bool)
cluster = defaultdict(list)
points = defaultdict(int)


def normalizeData(dataFile):
	"""
	Given a dataset, divide each dimension by its maximum obtaining data in the range [0,1]^n
	:param	dataFile: name of the original data file.
	:return newName: new name of the data file containing the normalized data.
	"""
        dataset=np.loadtxt(dataFile)
        normData=np.where(np.max(dataset, axis=0)==0, dataset, dataset*1./np.max(dataset, axis=0))
        newName=dataFile[dataFile.find('.')+1:dataFile.rfind('.')]
	newName='.'+newName+'n.txt'
	np.savetxt(newName,normData)
	return newName

def DBSCAN(dataset, epsilon, minPoints):
    """
    Density-Based algorithm for discovering clusters.
    :param dataset: Dataset of points
    :param epsilon: minimum distance
    :param minPoints: min number of points to not be considered noise
    """
    c = 0
    for point in dataset:
        if visited[str(point)]:
            continue
        visited[str(point)] = True
        neighborPts = regionQuery(dataset, point, epsilon)
        if len(neighborPts) < minPoints:
            noise[str(point)] = True
        else:
            c += 1  # next cluster
            expandCluster(point, neighborPts, c, epsilon, minPoints)


def expandCluster(point, neighborPts, c, epsilon, minPoints):
    """
    Find the points that belong to the same cluster as the given point.
    :param point: point
    :param neighborPts: neighbors for the point
    :param c: index of the cluster
    :param epsilon: minimum distance
    :param minPoints: min number of points to not be considered noise
    """
    cluster[c].append(point)  # add point to the cluster
    points[str(point)] = c    # assign the point to the cluster
    for p in neighborPts:
        if not visited[str(p)]:
            visited[str(p)] = True
            neighborPts2 = regionQuery(dataset, p, epsilon)  # find neighbours
            if len(neighborPts2) >= minPoints:
                neighborPts += neighborPts2
        if str(p) not in points:  # if the point is not assigned to another cluster...
            cluster[c].append(p)
            points[str(p)] = c


def regionQuery(dataset, point, epsilon):
    """
    Find the neighbors for a given point.
    :param dataset: Dataset of points
    :param point: point to check distances
    :param epsilon: minimum distance to become a neighbour
    :return: point neighbors
    """
    distance = [np.linalg.norm(point-p) for p in dataset]  # improve: symmetric distance
    # (p-point).all() check if the distance is not zero (because if it's 0 is the same point)
    neighborPts = [p for i, p in enumerate(dataset) if (p-point).all() and distance[i] < epsilon]
    return neighborPts


if __name__ == "__main__":
    numV = 1000     # Number of points
    dim = 2         # Number of dimensions
    k = 4           # Number of center (only used to generate data)
    epsilon = 0.1   # Minimum distance to belong to the same cluster
    minPoints = 10  # Minimum number of point to not be considered noise
    plot = False 
    dataFile = 'data/10k.txt'		

    dataset = np.loadtxt(normalizeData(dataFile)) 

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter([p[0] for p in dataset], [p[1] for p in dataset])
        plt.show()
        plt.savefig('dbscan.png')


    DBSCAN(dataset, epsilon, minPoints)
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import colors
        colours = [hex for (name, hex) in colors.cnames.iteritems()]
        fig, ax = plt.subplots()
        for c in cluster:
            ax.scatter([p[0] for p in cluster[c]], [p[1] for p in cluster[c]], color=colours[c])
        plt.show()
        plt.savefig('dbscan2.png')


