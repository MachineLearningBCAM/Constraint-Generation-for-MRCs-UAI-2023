# Constraint Generation for MRCs

This repository provides efficient learning algorithm for Minimax Risk Classifiers (MRCs) in high dimensions. The presented algorithm utilizes the constraint generation approach for the MRC linear program. The implementation of the presented algorithm (MRC-CG) can be found in the `Main` folder. Using multiple experiments, we compare our algorithm against directly solving the MRC linear program (MRC-LP) and other similar technique for SVM (SVM-CG). The MRC-LP is implemented in the `Main` folder using GUROBI optimizer. We use the library for SVM-CG (available in the `Libraries` folder) which is forked from [this](https://github.com/wanghaoyue123/Column-and-constraint-generation-for-L1-SVM-and-cousins) repository. We use multiple high-dimensional biological datasets to perform experiments as follows - 

Dataset | Variables | Samples | Classes
--- | --- | --- | --- 
Arcene | 10000 | 200 | 2 
Colon | 2000 | 62 | 2
CLL_SUB_111 | 11340 | 111 | 3
Dorothea | 100000 | 1150 | 2
GLI_85 | 22283 | 85 | 2
GLIOMA | 4434 | 50 | 4
Leukemia | 7129 | 72 | 3
Lung | 12600 | 203 | 5
MLL | 12582 | 72 | 3
Ovarian | 15154 | 253 | 2
Prostate_GE | 5966 | 102 | 2
SMK_CAN_187 | 19993 | 187 | 2
tox_171 | 5748 | 171 | 4

The datasets are available as zip files in this repository in the `Datasets` folder with functions to easily load them an numpy matrices in `load.py`. All the experimental results can be reproduced using the python scripts - 

```
param_eps.py : Experiments to study the influence of epsilon parameter and show the monotonic decrease in the worst-case error probability.
param_n_max.py : Experiment to analyze the effect of the parameter n_max and show improved efficiency over MRC-LP.
scalability.py : Experiments to show the scalability of the different algorithms with increasing number of features.
comparison.py : Experiments to compare our method with the state of the art techniques in terms of error and training time.
feature_selection.py : Experiment to compare our method as feature selection approach. The experiment is performed for binary classification datasets.
```

To reproduce any experimental result for any dataset, run the following command - 

```
python3 <scriptname> <dataset>
```

The result corresponding to that experiment and dataset will be saved in the `Results/` folder in the respective folder of the experiment and the dataset.
For instance, the experiment for comparing the scalability of different algorithms for the `Ovarian` dataset can be performed as follows - 

```
python3 scalability.py Ovarian
```

and the corresponding results will be saved in `Results/Scalability/Ovarian/` as CSV.