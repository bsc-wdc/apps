# COMPSs machine learning repository

## Folder structure

├── **sample** : launch scripts examples   
├── **src** : RandomForest sources   
├── **utils** : testing utilities   
├── **data** : commands to generate test data   
└── **README.md** : this MarkDown readme file    

## Random Forest 

Random forests are an ensemble learning method for classification, regression and other tasks, that operate by
constructing a multitude of decision trees at training time and agreggating the results of the individual trees.

This module includes the classes DecisionTree and RandomForestClassifier.


    from pycompss_lib.ml.classification.random_forest.full.src.decision_tree import DecisionTree

    tree = DecisionTree(args.path_in, args.n_instances, args.n_features, args.path_out, args.name_out, args.max_depth)
    tree.fit()

---


    from pycompss_lib.ml.classification.random_forest.full.src.forest import RandomForestClassifier

    tree = DecisionTree(path_in, n_instances, n_features, path_out, name_out, max_depth)
    tree.fit()

    forest = RandomForestClassifier(path_in, n_instances, n_features, path_out, n_estimators, max_depth)
    forest.fit()

### Considerations

The input data must be divided by attribute. Each attribute must fit into memory to build the trees. The construction of
each tree is internally parallelized.
