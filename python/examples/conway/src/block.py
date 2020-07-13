class Block (object):
    def __init__(self, blockSize = None, ref = None, rand = 0):
        from random import randint
        if(ref == None):
            if rand == 1:
                self._blockSize = blockSize
                self._matrix = []
                for i in range(0, self._blockSize):
                    a = []
                    for j in range(0, self._blockSize):
                        a.append(randint(0, 1))
                    self._matrix.append(a)
            else:
                self._blockSize = blockSize
                self._matrix = []
                for i in range(0, self._blockSize):
                    a = []
                    for j in range(0, self._blockSize):
                        a.append(0)
                    self._matrix.append(a)
        else:
            self._blockSize = ref._blockSize
            self._matrix = []
            for i in range(0, self._blockSize):
                a = []
                for j in range(0, self._blockSize):
                    a.append(ref._matrix[i][j])
                self._matrix.append(a)

    def set (self, i, j, val):
        self._matrix[i][j] = val

    def get (self, i, j):
        return self._matrix[i][j]