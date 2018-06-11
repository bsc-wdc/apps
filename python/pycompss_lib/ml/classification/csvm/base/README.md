# COMPSs machine learning repository

## Folder structure

├── **bin** : contains executable files  
├── **data** : data for sample executions  
├── **README.md** : this MarkDown readme file  
├── **scripts** : launch scripts examples  
└── **src** : Cascade SVM sources  

## Cascade SVM

Cascade support vector machines (C-SVM) are an extension of classic support vector machines that can be parallelized 
efficiently allowing both faster training and less memory consumption on large datasets. C-SVM split the dataset into 
chunks which are then optimized separately. The results are grouped forming new chunks and optimized again until one 
result group is left. This creates a 'Cascade' of SVMs which is repeated, feeding the resulting support vectors to the 
next initial layer, until global optimum is reached. The CSVM can be run on multiple processors with small 
communication overhead and far less memory consumption because the kernel matrices are much smaller.

Usage of COMPSs implemented Cascade SVM mirrors sklearn.svm.SVC class. It has four methods: load_data, fit, 
predict and score.

    from pycompss_lib.ml.classification import CascadeSVM

    svm = CascadeSVM() # empty constructor will use default parameters
    svm.load_data(train_X, train_y)
    svm.fit()

    acc = svm.predict(test_X, test_y)
        
    print(" - Accuracy: %s" % acc)
    
### Considerations

C-SVM assumes that input data contains observations from 2 classes. If a CSV file or a directory are passed to
load_data, C-SVM assumes that all input files contain observations from 2 classes AND that the input files have been
randomly shuffled previously. In other words, when splitting the input dataset, C-SVM does not check that all chunks
contain observations from 2 classes, and will raise an error if they do not.

