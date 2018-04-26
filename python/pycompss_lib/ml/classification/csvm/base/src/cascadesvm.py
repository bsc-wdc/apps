from __future__ import print_function

from exceptions import AttributeError
from pycompss.api.task import task
from pycompss.api.parameter import *
from sklearn import metrics

from collections import deque
from sklearn.svm import SVC

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.api import barrier
from pycompss.api.api import compss_delete_object
from operator import itemgetter
import csv
from itertools import islice,imap
import mmap
from time import time
from time import sleep
from copy import copy
import os
from sklearn.datasets import load_svmlight_file
from scipy.sparse import vstack, issparse, lil_matrix


class CascadeSVM(object):

    name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self, cascade_arity=2, n_chunks=4, cascade_iterations=5, C=1.0, kernel='rbf', gamma=None, convergence=10**-3, exec_time='total'):
        """
        :param exec_time: defines how to compute execution times. Accepted values are 'total' to compute overall execution time, and 'detailed' to compute read and fit time separately. Default is 'total'.
        """

        self.iterations = 0
        self.read_time = 0
        self.fit_time = 0
        self.total_time = 0
        self.converged = False                

        try:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel['rbf'])

        self._cascade_arity = cascade_arity        
        self._max_iterations = cascade_iterations

        self._nchunks = n_chunks
        self._convergence_margin = convergence       

        self._last_W = None
        self._clf = None
        self._exec_time = exec_time        

        assert (gamma is None or type(gamma) == float or type(float(gamma)) == float), "Gamma is not a valid float"
        assert (kernel is None or kernel in self.name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (kernel, self.name_to_kernel.keys())
        assert (C is None or type(C) == float or type(float(C)) == float), \
            "Incorrect C type [%s], type : %s" % (C, type(C))
        assert self._cascade_arity > 1, "Cascade arity must be greater than 1"      
        assert self._max_iterations > 0, "Max iterations must be greater than 0"
        
        self._clf_params = {"gamma": gamma, "C": C, "kernel": kernel}

        print(str(self))

    def __str__(self):
        return (" * CascadeSVM\n"
                "\t- Gamma:  %s\n"
                "\t- Kernel: %s\n"
                "\t- Arity:  %s\n"                
                "\t- Chunks:  %s\n"
                "\t- Max iterations: %s\n"
                "\t- C: %s\n") \
               % (self._clf_params["gamma"],
                  self._clf_params["kernel"],
                  self._cascade_arity,                  
                  self._nchunks,
                  self._max_iterations,
                  self._clf_params["C"])       

    def fit(self, X=None, y=None, path=None, n_features=None, data_format=None):
        """
        Fit a model with training data. The model is stored in self._clf
        :param X: input vectors
        :param y: input labels
        :param path: a file or a directory containing files with input vectors. If path is defined, X and y are ignored and input data is read from path. 
        :param data_format: defines the format of the data in path. Default (None) is CSV with the label in the last column. Alternative formats are: libsvm
        """
        
        
        s_time = time()
        t_time = time()
        
        # WARNING: when partitioning the data it is not guaranteed that all chunks contain vectors from both classes               
        if path and os.path.isdir(path):
            chunks = self._read_dir(path, data_format, n_features)
        elif path:
            chunks = self._read_file(path, data_format, n_features)
        else:
            chunks = self._read_data(X, y)      
            
        if self._exec_time == 'detailed':
            barrier()
            self.read_time = time() - s_time
            s_time = time()
        
        self._do_fit(chunks)
        
        if self._exec_time == 'detailed':
            self.fit_time = time() - s_time
            
        self.total_time = time() - t_time     
        
    def _read_dir(self, path, data_format, n_features):
        files = os.listdir(path)                      
        
        if data_format == "libsvm":
            assert n_features > 0, "Number of features is required to read from multiple files using libsvm format"
        elif not n_features:
            n_features = self._count_features(os.path.join(path, files[0]), data_format)                         
        
        if not self._clf_params["gamma"]:
            self._check_and_set_gamma(n_features) 
            
        self._nchunks = len(files)                    
        
        chunks = []     
        
        for f in files:    
            chunks.append(read_chunk(os.path.join(path, f), data_format=data_format, n_features=n_features))
            
        return chunks            
              
    def _read_file(self, path, data_format, n_features):        
        n_lines = self._count_lines(path) 
        
        assert n_lines > self._nchunks, "Not enough vectors to divide into %s chunks\n" \
                                                " - Minimum required elements: %s\n" \
                                                " - Vectors available: %s\n" % \
                                                (self._nchunks, self._nchunks, n_lines)
               
        if not n_features:
            n_features = self._count_features(path, data_format)   
               
        if not self._clf_params["gamma"]:                
            self._check_and_set_gamma(n_features)                
            
        steps = np.linspace(0, n_lines + 1, self._nchunks + 1, dtype=int)     
        chunks = []       
        
        for s in range(len(steps)  - 1):    
            chunks.append(read_chunk(path, steps[s], steps[s+1], data_format=data_format, n_features=n_features))                    
            
        return chunks
        
    def _read_data(self, X, y):
        """
        Use the training data to fit a model. The model
        is represented by the self._clf parameter.
        :param X: training examples
        :param y: training labels        
        """                
        chunks = self._get_chunks(X, y)
        
        if not self._clf_params["gamma"]:            
            self._check_and_set_gamma(X.shape[1])          
            
        return chunks
        
    def _do_fit(self, chunks):
        iteration = 0                
        q = deque()
        clf = None 
        feedback = None
        
        while iteration < self._max_iterations and not self.converged:      
            start_time = time()
            
            if len(chunks) > 1:
                for chunk in chunks:
                    data = filter(None, [chunk, feedback])                    
                    q.append(train(False, *data, **self._clf_params))                    
            else:
                # we jump to the last train
                data = [chunks[0]]        
                                                        
            while q:
                data = []
                
                while q and len(data) < self._cascade_arity:
                    data.append(q.popleft())                    
                
                if q:
                    q.append(train(False, *data, **self._clf_params))                                                       
                                          
                    # delete partial results
                    for d in data:
                        compss_delete_object(d)
                    
            sv, sl, clf = compss_wait_on(train(True, *data, **self._clf_params))
            feedback = sv, sl
            
            iteration += 1
            print("Checking convergence...")
            self._check_convergence_and_update_w(sv, sl, clf)
            end_time = time()
            print(" - Iteration %s/%s: converged?: %s\n - Iteration time %s\n\n" %
                  (iteration, self._max_iterations, self.converged, (end_time - start_time)))
               
        self._clf = clf        
        self.iterations = iteration
        
        print(" - Model: %s" % self._clf)
        print(" - Iterations performed %s, convergence was achieved?: %s" % (self.iterations, self.converged))
   
    def predict(self, X):
        """
        Predict the labels associated with the X examples with
        the fit model of the class
        :param X: examples
        :return: predicted labels
        """

        if self._clf:
            return self._clf.predict(X)
        else:
            raise Exception("Calling predict method before fit (aka model is not initialized)")
            return
        
    def decision_function(self, X):
        if self._clf:
            return self._clf.decision_function(X)
        else:
            raise Exception("Calling predict method before fit (aka model is not initialized)")
            return

    def score(self, X, y):
        """
        Score the perfomance of the class model with the
        given testing examples and labels
        :param X: testing examples
        :param y: testing labels
        :return: report of the classification as given by metrics.classification_report
        """
        if self._clf:
            #predicted = self._clf.predict(X)
            #status = metrics.classification_report(predicted, y)

            #return status
            return self._clf.score(X, y)
        else:
            raise Exception("Calling score method before fit (aka model is not initialized)")
            return
        

    def _lagrangian_fast(self, SVs, sl, coef):
        set_sl = set(sl)
        assert len(set_sl) == 2, "Only binary problem can be handled"
        new_sl = sl.copy()
        new_sl[sl == 0] = -1

        if issparse(coef):
            coef = coef.todense()          

        C1, C2 = np.meshgrid(coef, coef)
        L1, L2 = np.meshgrid(new_sl, new_sl)
        double_sum = C1 * C2 * L1 * L2 * self._kernel_f(SVs)
        double_sum = double_sum.sum()
        W = -0.5 * double_sum + coef.sum()

        return W   

    def _rbf_kernel(self, x):
        self._check_and_set_gamma(x.shape[1])
        
        # Trick: || x - y || ausmultipliziert
        sigmaq = -1 / (2 * self._clf_params["gamma"])
        n = x.shape[0]  
        K = x.dot(x.T) / sigmaq            
            
        if issparse(K):             
            K = K.todense()   
            
        d = np.diag(K).reshape((n, 1))
        K = K - np.ones((n, 1)) * d.T / 2
        K = K - d * np.ones((1, n)) / 2
        K = np.exp(K)
        return K

    def _check_and_set_gamma(self, n_features):
        if self._clf_params["gamma"] == None:
            print("Gamma was not set. Will use 1 / n_features =", 1. / n_features)
            self._clf_params["gamma"] = 1. / n_features
        

    def _check_convergence_and_update_w(self, sv, sl, clf):
        self.converged = False
        if clf:
            W = self._lagrangian_fast(sv, sl, clf.dual_coef_)
            print(" - Computed W %s" % W)

            if self._last_W:
                delta = np.abs((W - self._last_W) / self._last_W)
                if delta < self._convergence_margin:
                    print(" - Converged with delta: %s " % (delta))
                    self.converged = True
                else:
                    print(" - No convergence with delta: %s " % (delta))
            else:
                print(" - First iteration, not testing converge.")
            self._last_W = W        

    def _get_chunks(self, X, y):          
        chunks = []       
            
        steps = np.linspace(0, X.shape[0], self._nchunks + 1, dtype=int)                         
        
        for s in range(len(steps)  - 1):    
            chunkx = X[steps[s]:steps[s + 1]]            
            chunky = y[steps[s]:steps[s + 1]]
            
            chunks.append((chunkx, chunky)) 
                
        return chunks

    #@staticmethod
    #def precisionCheckDiff(s0, s1):
        #""""
        #Method used to check how many vectors of s0 and s1 are equal or very similar (to check numerical issues)
        #"""
        #for i in range(0, s0.shape[0]):
            #for j in range(i, s1.shape[0]):
                ## print(s0[i] - s1[j])
                #print("vm=%s" % (np.linalg.norm(s0[i] - s1[j])))
                #if 0.5 > np.linalg.norm(s0[i] - s1[j]) > 0:
                    #print("mode warning")
                #if np.linalg.norm(s0[i] - s1[j]) == 0:
                    #print("zero mode")

    @staticmethod
    def _count_lines(filename):
        f = open(filename, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0    
        line = buf.readline()            
        lines += 1    
        readline = buf.readline
        
        while readline():        
            lines += 1    
            
        f.close()
            
        return lines
    
    @staticmethod
    def _count_features(filename, data_format=None):
        if data_format == "libsvm":
            X, y = load_svmlight_file(filename)
            features = X.shape[1]            
        else:        
            f = open(filename, "r+")
            buf = mmap.mmap(f.fileno(), 0)        
            line = buf.readline()    
            features = len(line.split(",")) - 1
            f.close()
        
        return features
    
    @staticmethod
    def _linear_kernel(x1):
        return np.dot(x1, x1.T)

@task(returns=tuple)
def train(return_classifier, *args, **kwargs):    
    if len(args) > 1:
        X, y = merge(*args)
    else:        
        X, y = args[0]
    
    clf = SVC(random_state=1, **kwargs)
    clf.fit(X, y)       
   
    sv = X[clf.support_]    
    sl = y[clf.support_]
    
    if return_classifier:
        return sv, sl, clf
    else:
        return sv, sl

@task(filename=FILE, returns=tuple)
def read_chunk(filename, start=None, stop=None, data_format=None, n_features=None):
    if data_format == "libsvm":
        X, y = load_svmlight_file(filename, n_features)
        
        if start and stop:            
            X = X[start:stop]
            y = y[start:stop]
    else:
        with open(filename) as f:                
            vecs = np.genfromtxt(islice(f, start, stop), delimiter=",")                     
        
        X, y = vecs[:, :-1], vecs[:, -1]    
    
    return X, y

def merge(*args):
    if issparse(args[0][0]):
        return merge_sparse(*args)
    else:
        return merge_dense(*args)

def merge_dense(*args):            
    sv1, sl1 = args[0]
    sv1 = np.concatenate((sv1, sl1[:, np.newaxis]), axis=1)
    
    for t2 in args[1:]:        
        sv2, sl2 = t2
        sv2 = np.concatenate((sv2, sl2[:, np.newaxis]), axis=1)                
        sv1 = np.concatenate((sv1, sv2))       
    
    sv1 = np.unique(sv1, axis=0)
    
    return sv1[:, :-1], sv1[:, -1] 
        
def merge_sparse(*args):
    sv1, sl1 = args[0]
    sv1 = sv1.asformat('lil')    
    
    rows = sv1.rows.tolist()
    data = sv1.data.tolist()

    for t2 in args[1:]:
        sv2, sl2 = t2
        sv2 = sv2.asformat('lil')
        nrows = len(sv2.rows)
        
        for i in range(nrows):            
            duplicate = False
        
            for j in range(len(rows)):  
                # check if column indices are the same
                if set(sv2.rows[i]) == set(rows[j]):
                    
                    # check if data is the same
                    data1 = data[j]
                    data2 = sv2.data[i]
                    
                    if np.allclose(data1, data2):
                        duplicate = True
                        break
                    
            if not duplicate:
                rows.append(sv2.rows[i])
                data.append(sv2.data[i])
                sl1 = np.concatenate((sl1, sl2[i, np.newaxis]))      
       
    svs = lil_matrix((len(rows), sv1.shape[1]))
    svs.rows = np.array(rows, dtype=list)
    svs.data = np.array(data, dtype=list)
    
    return svs.asformat("csr"), sl1
