import sys

from pycompss.api.parameter import *
from pycompss.api.task      import task
from pycompss.api.api       import compss_wait_on

from block                  import Block

@task(returns=Block)
def initBlock (blockSize):
    return Block(blockSize = blockSize, rand = 1)

@task(returns=Block, b00=IN, b01=IN, b02=IN, b10=IN, b11=IN, b12=IN,b20=IN, b21=IN, b22=IN)
def updateBlock(b00, b01, b02, b10, b11, b12, b20, b21, b22, aFactor, blockSize):
    subStateA = []
    subStateB = []

    #subStateA
    for i in range (0, 3):
        subStateA.append([])

    for i in range (0, 3):
        for j in range (0, 3):
            subStateA[i].append(Block(blockSize))

    #subStateB
    subStateB = [[Block(ref = b00), Block(ref = b01), Block(ref = b02)],
                    [Block(ref = b10), Block(ref = b11), Block(ref = b12)],
                    [Block(ref = b20), Block(ref = b21), Block(ref = b22)]]

    #iterations
    for t in range(aFactor, -1 , -1):
        subStateC = subStateA
        subStateA = subStateB
        subStateB = subStateC

        for i in range(blockSize - t, 2 * blockSize + t):
            for j in range(blockSize -t, 2 * blockSize + t):

                count = 0

                #count
                for off_i in range(-1, 2):
                    for off_j in range(-1, 2):
                        if off_i != 0 or off_j != 0:
                            p = subStateA[(i + off_i) / blockSize][(j + off_j) / blockSize]
                            if p.get((i + off_i) % blockSize, (j + off_j) % blockSize) == 1:
                                count = count + 1

                #Rules
                p = subStateA[i / blockSize][j / blockSize];
                q = subStateB[i / blockSize][j / blockSize];
                mod_i = i % blockSize;
                mod_j = j % blockSize;

                if p.get(mod_i, mod_j) == 1:
                    if count == 2 or count == 3:
                        q.set(mod_i, mod_j, 1)
                    else:
                        q.set(mod_i, mod_j, 0)
                else:
                    if count == 3:
                        q.set(mod_i, mod_j, 1)
                    else:
                        q.set(mod_i, mod_j, 0)
    

    return Block(ref = subStateB[1][1])