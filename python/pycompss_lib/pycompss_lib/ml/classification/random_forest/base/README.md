# COMPSs machine learning repository

## Folder structure

├── **sample** : launch scripts examples    
├── **src** : RandomForest sources    
├── **tests** : comparing with sklearn     
├── **utils** : testing utilities    
├── **data** : commands to generate test data    
└── **README.md** : this MarkDown readme file      

## Random Forest 

Random forests are an ensemble learning method for classification, regression and other tasks, that operate by
constructing a multitude of decision trees at training time and agreggating the results of the individual trees.

Usage of COMPSs implemented RandomForestClassifier and RandomForestRegressor mirrors the classes
sklearn.ensemble.RandomForestClassifier and sklearn.ensemble.RandomForestRegressor.


    from pycompss_lib.ml.classification import RandomForestClassifier

    rf = RandomForestClassifier() # empty constructor will use default parameters
    rf.fit(train_X, train_y)

    reuslt = rf.predict(test_X)
        
    print(reuslt)

### Considerations

The construction of each tree is parallelized. The sklearn.tree module is used for the trees. The data must fit into memory
in order to build each tree.

## License

This module is a derivative work of the sklearn.ensemble.src module released under BSD 3 clause license. Original note:

Authors: Gilles Louppe <g.louppe@gmail.com>  
         Brian Holt <bdholt1@gmail.com>  
         Joly Arnaud <arnaud.v.joly@gmail.com>  
         Fares Hedayati <fares.hedayati@gmail.com>  

License: BSD 3 clause
