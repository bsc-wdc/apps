# COMPSs machine learning repository

## Folder structure

├── **sample** : launch scripts
├── **src** : RandomForest sources
├── **test** : Unit testing
├── **utils** : data generators
├── **data** : commands to generate test data   
└── **README.md** : this MarkDown readme file    

## Random Forest 

Random forests are an ensemble learning method for classification, regression and other tasks, that operate by
constructing a multitude of decision trees at training time and aggregating the results of the individual trees.

This module includes the classes DecisionTreeClassifier and RandomForestClassifier.


    from pycompss_lib.ml.classification.random_forest.full.src.decision_tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(path_in, n_instances, n_features, path_out, name_out, max_depth, distr_depth)
    tree.fit()

---


    from pycompss_lib.ml.classification.random_forest.full.src.forest import RandomForestClassifier

    forest = RandomForestClassifier(path_in, n_instances, n_features, path_out, n_estimators, max_depth, distr_depth)
    forest.fit()

### Considerations

Each attribute must fit into memory to build the trees. The construction of each tree is internally parallelized.
