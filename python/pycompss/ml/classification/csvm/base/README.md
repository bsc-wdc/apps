# COMPSs machine learning repository

## Folder structure

├── **bin** : contains executable files  
├── **data** : data for sample executions  
├── **README.md** : this MarkDown readme file  
├── **sample** : launch scripts examples  
└── **svm** : Cascade SVM sources  

## Cascade SVM

Cascade support vector machines (C-SVM) are an extension of classic support vector machines that can be parallelized efficiently allowing both faster training and less memory consumption on large datasets. C-SVM split the dataset into chunks which are then optimized separately. The results are grouped forming new chunks and optimized again until one result group is left. This creates a 'Cascade' of SVMs which is repeated, feeding the resulting support vectors to the next initial layer, until global optimum is reached. The CSVM can be run on multiple processors with small communication overhead and far less memory consumption because the kernel matrices are much smaller.

Usage of COMPSs implemented Cascade SVM mirrors sklearn.svm.SVC class. It has three methods: fit, predict and score.

    from compssml.svm import CascadeSVM

    svm = CascadeSVM() # empty constructor will use default parameters
    svm.fit(train_X, train_y)

    predicted = svm.predict(test_X)

    acc = accuracy_score(test_y, predicted)
    print(" - Accuracy: %s" % acc)
    
Score method is a thin wrapper which actually calls the predict method and passes the results to accuracy_score.

