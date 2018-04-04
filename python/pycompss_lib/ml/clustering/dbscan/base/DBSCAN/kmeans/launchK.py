import numpy as np
import time

if __name__=='__main__':
    time.sleep(3)
    data=np.loadtxt(sys.argv[1])
    import kmeans
    mu = kmeans.kmeans(data, int(sys.argv[2]), int(sys.argv[2]))
    import matplotlib.pyplot as plt
    from matplotlib import colors
    fig,ax = plt.subplots()
    a=[p for p in data if np.linalg.norm(p-mu[0]) > np.linalg.norm(p-mu[1])]
    b=[p for p in data if np.linalg.norm(p-mu[0]) < np.linalg.norm(p-mu[1])]
    c=[a,b]
    colours = [hex for (name, hex) in colors.cnames.iteritems()]
    for i,vec in enumerate(c):
        ax.scatter([p[0] for p in vec], [p[1] for p in vec], color = colours[i], s=1)
    plt.savefig('moonsKMEANS.png')
    plt.close()
