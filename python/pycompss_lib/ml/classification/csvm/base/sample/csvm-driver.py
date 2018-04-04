import argparse
import numpy as np
from cascadesvm import CascadeSVM
from time import time
import csv
import os
from sklearn.datasets import load_svmlight_file

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-cr", "--centralized_read", help="read the whole CSV file at the master", action="store_true")
    parser.add_argument("--libsvm", help="read files in libsvm format", action="store_true") 
    parser.add_argument("-k", metavar="KERNEL", type=str, help="linear or rbf (default is rbf)", choices=["linear", "rbf"], default='rbf')
    parser.add_argument("-a", metavar="CASCADE_ARITY", type=int, help="default is 2", default=2)
    parser.add_argument("-n", metavar="N_CHUNKS", type=int, help="number of chunks in which to divide the dataset (default is 4)", default=4)
    parser.add_argument("-i", metavar="MAX_ITERATIONS", type=int, help="default is 5", default=5)
    parser.add_argument("-g", metavar="GAMMA", type=float, help="(only for rbf kernel) default is 1 / n_features", default=None)
    parser.add_argument("-c", metavar="C", type=float, help="default is 1", default=1)    
    parser.add_argument("-f", metavar="N_FEATURES", type=int, help="mandatory if --libsvm option is used and train_data is a directory (optional otherwise)", default=None)    
    parser.add_argument("-t", metavar="TEST_FILE_PATH", help="test CSV file path", type=str, required=False)       
    parser.add_argument("-o", metavar="OUTPUT_FILE_PATH", help="output file path", type=str, required=False)      
    parser.add_argument("train_data", help="CSV file or directory containing CSV files (if a directory is provided N_CHUNKS is ignored)", type=str)    
    args = parser.parse_args()
    
    train_data = args.train_data
    
    csvm = CascadeSVM(kernel=args.k, C=args.c, cascade_arity=args.a, n_chunks=args.n, gamma=args.g, cascade_iterations=args.i)
    
    if args.centralized_read:
        if args.libsvm:
            x, y = load_svmlight_file(train_data)            
        else:
            train = np.loadtxt(train_data, delimiter=",", dtype=float)
            
            x = train[:, :-1]
            y = train[:, -1]        
        
        csvm.fit(X=x, y=y)        
    elif args.libsvm:           
        csvm.fit(path=train_data, data_format="libsvm", n_features=args.f)
    else:
        csvm.fit(path=train_data, n_features=args.f)
        
    out = [args.k, args.a, args.n, csvm._clf_params["gamma"], args.c, csvm.iterations, csvm.converged, csvm.read_time, csvm.fit_time]    
        
    if os.path.isdir(train_data):
        n_files = os.listdir(train_data)        
        out.append(len(n_files))
    
    if args.t: 
        if args.libsvm:
            testx, testy = load_svmlight_file(args.t, args.f)
            out.append(csvm.score(testx, testy))
        else:
            test = np.loadtxt(args.t, delimiter=",", dtype=float)            
            out.append(csvm.score(test[:, :-1], test[:, -1]))
    
    if args.o:
        with open(args.o, "ab") as f:
            wr = csv.writer(f)
            wr.writerow(out)
    else:
        print(out)            
    
      
if __name__ == "__main__":
    main()
