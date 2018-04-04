from numpy import log2, ceil, pad, zeros, eye

minSize = 4

class arraySlice(object):
    TYPE_ZEROS = 0
    TYPE_ONES = 1
    TYPE_DENSE = 2
    #@task(...)
    def __init__(self, data, finalShape, specialType=None,*args, **kwargs):
        print finalShape
        print data
        if specialType == self.TYPE_ZEROS and (len(data) == 0 or len(data[0]) == 0):
            # Zeros
            self.type = self.TYPE_ZEROS
            self.finalShape = finalShape
        elif specialType == self.TYPE_ONES and (len(data) == 0 or len(data[0]) == 0):
            # Identity
            self.type = self.TYPE_ONES
            self.finalShape = finalShape
        else:
            from numpy import array
            self.type = self.TYPE_DENSE
            self.block = array(data, *args, **kwargs)
            if self.block.shape != finalShape:
                shapeAux = self.block.shape
                self.block = pad(self.block,
                                 ((0, finalShape[0]-shapeAux[0]), (0, finalShape[1]-shapeAux[1])),
                                 mode='constant',
                                 constant_values=0)
                if specialType == self.TYPE_ONES:
                    startIndex = min(shapeAux[0], shapeAux[1])
                    finalIndex = min(finalShape[0], finalShape[1])
                    increment = finalShape[1]
                    self.block.flat[increment * startIndex + startIndex:(finalIndex+1)*(finalIndex+1):increment + 1] = 1
            print self.block

    def __str__(self):
        if self.type == self.TYPE_ZEROS:
            return str(zeros(self.finalShape))
        elif self.type == self.TYPE_ONES:
            return str(eye(self.finalShape[0], self.finalShape[1]))
        else:
            return str(self.block)

    def printStoredStructure(self, referenceString):
        print(referenceString)
        if self.type == self.TYPE_ZEROS:
            print("ZERO")
        elif self.type == self.TYPE_ONES:
            print("IDENTITY")
        elif self.type == self.TYPE_DENSE:
            print(str(self.block))
        else:
            raise AttributeError("ArraySlice " + str(referenceString) + " unknown type")


"""
class arrayDist(object):

    blockSize = 4

    def __init__(self, original, shape, specialType = None, *args, **kwargs):
        if shape[0] == self.minSize and shape[1] == self.minSize:
            self.data = [arraySlice(original, shape, (self.minSize, self.minSize))]
            print(str(self.data))
        elif shape[0] < self.minSize and shape[1] < self.minSize:
            finalShape = (self.minSize, self.minSize)
            self.data = [arraySlice(original, finalShape)]
            print(str(self.data))
        elif shape[0] > self.minSize or shape[1] > self.minSize:
            self.data = self.sliceBlock(shape, original)
            print(str(self.data))
        else:
            print(shape[0])

    def sliceBlock(self, shape, data):
        blockSize = int(ceil(log2(max(shape))) - 1)
        minCoord = lambda m, axis: min(m * 2**blockSize, shape[axis])

        b11 = [i[0:minCoord(1, 1)] for i in data[0:minCoord(1, 0)]]
        b12 = [i[minCoord(1, 1):minCoord(2, 1)] for i in data[0:minCoord(1,0)]]
        b21 = [i[0:minCoord(1,1)] for i in data[minCoord(1,0):minCoord(2,0)]]
        b22 = [i[minCoord(1,1):minCoord(2,1)] for i in data[minCoord(1,0):minCoord(2,0)]]

        blocks = [arrayDist(b, (max(len(b), len(b[0])), max(len(b), len(b[0])))) for b in [b11, b12, b21, b22]]
        return blocks

    #def __getitem__(self, item):
    #    pass

    def __str__(self):
        if type(self.data) == arraySlice:
            return str(self.data)
        else:
            return str([str(d) for d in self.data])

    #Function to debug and show the structure of the class
    def printStoredStructure(self, referenceString = ""):
        if referenceString != "":
            referenceString = referenceString + "_"
        i = 0
        for block in self.data:
            i += 1
            block.printStoredStructure(referenceString + str(i))

    def getRealStructure(self):
        import numpy as np
        #if self.data.__class__
        #return np.bmat([[self.data[],],[,]])

    #def printRealStructure(self):


"""
class arrayDist(object):

    def __init__(self, original, shape, specialType = None, *args, **kwargs):
        #Exactly the size of the block
        if shape[0] == minSize and shape[1] == minSize:
            self.data = [arraySlice(original, shape, (minSize, minSize))]
        elif shape[0] <= minSize and shape[1] <= minSize:
            #finalShape = (self.minSize, self.minSize)
            self.data = [arraySlice(original, shape)]
        elif shape[0] > minSize or shape[1] > minSize:
            self.data = arraySlice(shape, original)

    def __getitem__(self, item):
        pass

    def __str__(self):
        if type(self.data) == arraySlice:
            return str(self.data)
        else:
            return str([str(d) for d in self.data])

    #Function to debug and show the structure of the class
    def printStoredStructure(self, referenceString = ""):
        if referenceString != "":
            referenceString = referenceString + "_"
        i = 0
        for block in self.data:
            i += 1
            block.printStoredStructure(referenceString + str(i))

    def getRealStructure(self):
        import numpy as np
        #if self.data.__class__
        #return np.bmat([[self.data[],],[,]])

    #def printRealStructure(self):


