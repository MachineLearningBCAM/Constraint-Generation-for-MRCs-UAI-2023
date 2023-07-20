"""
Experiment to analyze the effect of the parameter n_max and 
show improved efficiency over MRC-LP.
"""

import sys
import numpy as np
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools as it
import scipy.special as scs

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
	path = "./Results/Hyperparameter n_max/"
	init_seed = 42
	dataset_name = sys.argv[1]
	eps = 1e-4
	k_max = 200
	random.seed(init_seed)
	s = 1

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

	X = StandardScaler().fit_transform(X, y)
	n_max_arr = [1, 10, 15, 25, 40, 50, 63, 100, 200, 300, 400, 500, 600, 800, 1000, 1585, d]
	time_arr = np.zeros(len(n_max_arr))

	n_reps = 50
	for j, n_max in enumerate(n_max_arr):
		print('######### Running MRC-CG using n_max : ', n_max)

		for i in range(n_reps):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
			phi_ob = BasePhi(n_classes=n_classes,
					 fit_intercept=fit_intercept,
					 one_hot=one_hot).fit(X_train, y_train)

		#--> Time taken by MRC-LP to solve the optimization.
			if n_max == d:
				n = X_train.shape[0]
				phi_ = phi_ob.eval_x(X_train)
				tau_ = phi_ob.est_exp(X, y)
				lambda_ = s * (phi_ob.est_std(X, y)) / np.sqrt(X.shape[0])

				F = np.vstack((np.sum(phi_[:, S, ], axis=1)
							   for numVals in range(1, phi_ob.n_classes + 1)
							   for S in it.combinations(np.arange(phi_ob.n_classes), numVals)))

				cardS = np.arange(1, phi_ob.n_classes + 1).\
							repeat([n * scs.comb(phi_ob.n_classes, numVals)
							for numVals in np.arange(1, phi_ob.n_classes + 1)])

				# Constraint coefficient matrix
				M = F / (cardS[:, np.newaxis])

				# The bounds on the constraints
				h = 1 - (1 / cardS)

				initTime = time.time()
				# Solve the LP.
				mrc_model   = mrc_lp_model_gurobi(M, h, tau_, lambda_)
				total_time = time.time() - initTime

			else:

		#--> Time taken by MRC-CG to solve the optimization.
				mu, nu, R, I, R_k, total_time, init_mrc_cg_time = mrc_cg(X_train,
																		 y_train,
																		 phi_ob,
																		 s,
																		 n_max,
																		 k_max,
																		 eps)
			time_arr[j] = time_arr[j] + total_time

	time_arr = np.asarray(time_arr) / n_reps
	# Calculate the relative time defined as
	# the time taken by mrc-cg divided by
	# the time taken to solve the mrc-lp problem.
	rel_time_arr = time_arr / time_arr[-1]
	print('###### The relative times are as follows : \n', rel_time_arr)

	# Save the data
	np.savetxt(path + str(dataset_name) + '/rel_time.csv', rel_time_arr, delimiter=",", fmt='%s')
	np.savetxt(path + str(dataset_name) + '/n_max_arr.csv', n_max_arr, delimiter=",", fmt='%s')
	print('Data saved successfully in ', path + str(dataset_name))
