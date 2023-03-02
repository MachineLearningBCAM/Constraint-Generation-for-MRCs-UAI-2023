"""
Experiments to study the influence of epsilon parameter and
show the monotonic decrease in the worst-case error probability.
"""

import sys
import numpy as np
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from MRCpy import MRC

sys.path.append('./Main/')
from main import mrc_cg
from mrc_lp_gurobi import mrc_lp_model_gurobi
sys.path.append('./')

sys.path.append('./Datasets/')
from load import *
sys.path.append('./')

sys.path.append('./Libraries/')
from myRandomPhi import *
sys.path.append('./')

sys.path.append('./Libraries/L1SVM_CG/')
from Our_methods import *
from Benchmarks import *
sys.path.append('./')

import random

if __name__ == '__main__':

	# Get the command line arguments.
	warnings.simplefilter("ignore")
	path = "./Results/Hyperparameter eps/"
	init_seed = 2022
	dataset_name = sys.argv[1]
	tol = 1e-4
	random.seed(init_seed)
	s = 1

	n_max = 100
	k_max = 500

	eps_arr = [0.01, 0.001, 0.0001]

#---> Loading the dataset
	load_dataset = 'load_' + dataset_name

	X, y = eval(load_dataset +  "(return_X_y=True)")
	n, d = X.shape
	n_classes = len(np.unique(y))
	one_hot = False
	fit_intercept = False

	print('Dataset ' + str(dataset_name) + ' loaded. The dimensions are : ' + str(n) + ', ' + str(d))

	# Save all the data that is being computed.
	save = True
	seed = 2021

	X = StandardScaler().fit_transform(X, y)

	phi_ob = BasePhi(n_classes=n_classes,
					 fit_intercept=fit_intercept,
					 one_hot=one_hot).fit(X, y)

	# Get the upper bound for the whole dataset as we dont need testing data to get upper bound
	# So, we can get the exact upper bound instead of approximating it.

	R_k_error_mat = []

	# Compute the true upper bound R*
	mrc = MRC(phi=phi_ob,
			  loss='0-1',
			  solver='cvx',
			  s=s,
			  one_hot=one_hot,
			  random_state=seed,
			  fit_intercept=fit_intercept).fit(X, y)

	R_ = mrc.get_upper_bound()

	print('###### The true worst-case error probability is : ', R_)
	for eps in eps_arr:
		print('######### Running MRC-CG using eps : ', eps)
		mu, nu, R, I, R_k, mrc_cg_time, init_mrc_cg_time = mrc_cg(X,
                                                                  y,
                                                                  phi_ob,
                                                                  s,
                                                                  n_max,
                                                                  k_max,
                                                                  eps)


		R_k_error_mat.append(R_ - R_k)

	R_k_error_mat = np.asarray(R_k_error_mat)

	for i, eps in enumerate(eps_arr):
		print('###### The eps value is : ', eps)
		print('###### The convergence of the error is : \n', R_k_error_mat[i])

	# Save the data
	np.savetxt(path + str(dataset_name) + '/R_k_error_mat.csv', R_k_error_mat, delimiter=",", fmt='%s')
	np.savetxt(path + str(dataset_name) + '/eps_arr.csv', eps_arr, delimiter=",", fmt='%s')
	print('Data saved successfully in ', path + str(dataset_name))
