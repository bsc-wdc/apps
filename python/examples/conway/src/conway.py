import sys
from time import time

from pycompss.api.parameter import *
from pycompss.api.task      import task
from pycompss.api.api       import compss_wait_on
from pycompss.api.api       import compss_barrier

from block                  import Block
from updateBlock            import updateBlock
from updateBlock            import initBlock

#Define auxiliar functions
def initMatrix(widthNumBlocks, lengthNumBlocks, blockSize):
    res = []
    for i in range(0, widthNumBlocks):
        a = []
        for j in range(0, lengthNumBlocks):
            a.append(initBlock(blockSize))
        res.append(a)
    return res

def initMatrixVoid(widthNumBlocks, lengthNumBlocks, blockSize):
    res = []
    for i in range(0, widthNumBlocks):
        a = []
        for j in range(0, lengthNumBlocks):
            a.append(None)
        res.append(a)
    return res

DEBUG = 1

def main_program():
    def usage():
        print("[ERROR] Bad number of parameters.")
        print("    Usage: simple <W, L, numIterations, blockSize, aFactor>")

    # Initialize constants
    if len(sys.argv) != 6:
        usage()
        exit(-1)

    widthElements = int(sys.argv[1])
    lengthElements = int(sys.argv[2])
    numIterations = int(sys.argv[3])
    blockSize = int(sys.argv[4])
    aFactor = int(sys.argv[5])

    widthNumBlocks = widthElements / blockSize
    lengthNumBlocks = lengthElements / blockSize

    if DEBUG == 1:
        print("Application parameters:");
        print("- Elements Width: {}".format(widthElements));
        print("- Elements Length: {}".format(lengthElements));
        print("- Num. Iterations: {}".format(numIterations));
        print("- Block size: {}".format(blockSize));
        print("- A factor: {}".format(aFactor));

    # Timing
    startTime = time()

    # Initialize state
    stateA = initMatrix(widthNumBlocks, lengthNumBlocks, blockSize)
    # Initialize swap state (only structure, blocks will be copied)
    stateB = initMatrixVoid(widthNumBlocks, lengthNumBlocks, blockSize)

    # Iterations
    for iter in range(0, numIterations/(aFactor+1)):
        if DEBUG == 1:
            print("Running iteration {}".format(iter))

        #Swap states
        if iter != 0:
            if DEBUG == 1:
                print("- Swapping starting states...")
            for i in range(0, widthNumBlocks):
                for j in range(0, lengthNumBlocks):
                    stateA[i][j] = stateB[i][j]

        #Update blocks
        if DEBUG == 1:
            print("- Updating block states...")

        for i in range(0, widthNumBlocks):
            for j in range(0, lengthNumBlocks):
                stateB[i][j] = updateBlock(
                                            stateA[(i - 1 + widthNumBlocks) % widthNumBlocks][(j - 1 + lengthNumBlocks) % lengthNumBlocks],
                                            stateA[(i - 1 + widthNumBlocks) % widthNumBlocks][(j + 0 + lengthNumBlocks) % lengthNumBlocks],
                                            stateA[(i - 1 + widthNumBlocks) % widthNumBlocks][(j + 1 + lengthNumBlocks) % lengthNumBlocks],
                                            
                                            stateA[(i + 0 + widthNumBlocks) % widthNumBlocks][(j - 1 + lengthNumBlocks) % lengthNumBlocks],
                                            stateA[(i + 0 + widthNumBlocks) % widthNumBlocks][(j + 0 + lengthNumBlocks) % lengthNumBlocks],
                                            stateA[(i + 0 + widthNumBlocks) % widthNumBlocks][(j + 1 + lengthNumBlocks) % lengthNumBlocks],
                                            
                                            stateA[(i + 1 + widthNumBlocks) % widthNumBlocks][(j - 1 + lengthNumBlocks) % lengthNumBlocks],
                                            stateA[(i + 1 + widthNumBlocks) % widthNumBlocks][(j + 0 + lengthNumBlocks) % lengthNumBlocks],
                                            stateA[(i + 1 + widthNumBlocks) % widthNumBlocks][(j + 1 + lengthNumBlocks) % lengthNumBlocks],
                                            
                                            aFactor, blockSize)

    # Results
    for i in range(0, widthNumBlocks):
        for j in range(0, lengthNumBlocks):
            stateB[i][j] = compss_wait_on(stateB[i][j])

    if DEBUG == 1:
        print("Results:")

        for i in range(0, widthNumBlocks):
            for j in range(0, lengthNumBlocks):
                print("Block [{},{}] = {}".format(i, j, stateB[i][j]))

    # Timing
    compss_barrier()
    endTime = time()
    print("Total execution time: {} s".format(endTime-startTime))

if __name__ == "__main__":
    main_program()