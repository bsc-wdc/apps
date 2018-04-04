import sys
sys.path.append('~/svn/bar/apps/python/matlib/matlib/')
sys.path.append('~/svn/bar/apps/python/matlib/')
import matlib as np

if __name__ == '__main__':

    # m x n matrix
    m = 4
    n = 2
    ones = [[(i+1)*(j+1) for i in range(0,n)] for j in range(0,m)]

    print("ONES:")
    uns = np.ones((4,4))

    print("ORIGINAL MATRIX:")
    print(uns)

    print("CUSTOM ARRAY GENERATION:")
    b = np.array([2,3,1,0])
    c = b*b
    print("Variable b: " + str(b) + " of kind " + str(type(b)))
    print("Begin sin computation")
    newArray = np.sin(b)
    print("Sin array " + str(newArray) + " of kind " + str(type(newArray)))
    mulArray = np.dot(newArray,newArray)
    mulArray2 = np.multiply(newArray, newArray)
    mulArray3 = newArray * newArray
    sumArray = newArray + newArray
    print("Previous array: " + str(newArray) + " of kind " + str(type(newArray)))
    print("Mul array: " + str(mulArray) + " of kind " + str(type(mulArray)))
    print("Mul array2: " + str(mulArray2) + " of kind " + str(type(mulArray2)))
    print("Mul array3: " + str(mulArray3) + " of kind " + str(type(mulArray3)))
    print("Sum array: " + str(sumArray) + " of kind " + str(type(sumArray)))
    print("Component sum of b: " + str(np.sum(b)) + " of kind " + str(type(b)))
    print("Component sum of newArray: " + str(np.sum(newArray)) + " of kind " + str(type(newArray)))

    # print("QR EXAMPLE:")
    # originalMatrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # print("Original matrix: " + str(originalMatrix) + " of kind " + str(type(originalMatrix)))
    # resultQR = np.linalg.qr(originalMatrix)
    # print("Printing all the output items")
    # for item in resultQR:
    #  print("Item: " + str(item) + " of kind " + str(type(item)))
    #
    # b.printStoredStructure()
    print("---------------------------------------------------------------------------------------------")
    m = 9
    n = 1
    ones = [[(i+1)*(j+1) for i in range(0,n)] for j in range(0,m)]
    print(ones)
    b = np.arrayDist(ones, (m, n))
    print(b)
    b.getRealStructure()
    b.printStoredStructure()
    #
    # exemple = np.arraySlice([[1,2,3],[4,5,6]], (4,4))
    # print(exemple)
