# Constraint Generation for MRCs

The implementation of MRC-CG and MRC-LP can be found in the `Main` folder. The library for SVM-CG can be found in the `Libraries` folder which is forked from this repository. The datasets are also available as zip in the `Datasets` folder with functions to easily load them an numpy matrices in `load.py`. All the experimental results can be reproduced using the python scripts - 

```
param_eps.py : Experiments to study the influence of epsilon parameter and show the monotonic decrease in the worst-case error probability.
param_n_max.py : Experiment to analyze the effect of the parameter n_max and show improved efficiency over MRC-LP.
scalability.py : This set of experiments show the scalability of the different algorithms with increasing number of features.
comparison.py : Experiments to compare our method with the state of the art techniques in terms of error and training time.
feature_selection.py : Experiment to compare our method as feature selection approach. The experiment is performed for binary classification datasets.
```

To reproduce the any experimental result for any dataset, run the following command - 

```
python3 <scriptname> <dataset>
```

The result corresponding to that experiment and dataset will be saved in the `Results/` folder in the respective folder of the experiment and the dataset.
For instance, the experiment for comparing the scalability of different algorithms for the `Ovarian` dataset can be performed as follows - 

```
python3 scalability.py Ovarian
```
and the corresponding results will be saved in `Results/Scalability/Ovarian/` as CSV.

The datasets available in this repository are as follows - 

Dataset | Variables | Samples | Classes
--- | --- | --- | --- 
Arcene | 10000 | 200 | 2 
