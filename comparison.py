"""
Experiments to compare our method with the state of the art techniques
in terms of error and training time.
"""

import sys
import numpy as np
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import numpy as np
import random
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

def predict_proba(phi_te, mu, nu, n_classes):
	"""
	Predict the classification probabilities of 0-1 MRC for the given solution.

	Parameters
	----------
	phi_te : array-like of shape (no_of_instances, n_classes, no_of_features)
		One-hot encoded feature mappings of the testing data.

	mu : array-like of shape (no_of_features)
		Solution of the classifier.

	nu : float
		Solution of the classifier.

	Returns
	-------
	hy_x : array-like of shape (no_of_instances, n_classes)
		Prediction probabilities of the given testing data.

	"""

	hy_x = np.clip(1 + np.dot(phi_te, mu) + nu, 0., None)
	c = np.sum(hy_x, axis=1)
	
	# check when the sum is zero
	zeros = np.isclose(c, 0)
	c[zeros] = 1
	hy_x[zeros, :] = 1 / n_classes
	c = np.tile(c, (n_classes, 1)).transpose()
	hy_x = hy_x / c
	return hy_x

if __name__ == '__main__':

	# Get the command line arguments.
	warnings.simplefilter("ignore")
	path = "./Results/Comparison with state of the art/"
	dataset_name = sys.argv[1]
	eps = 1e-4
	s = 1
	n_max = 100
	k_max = 20

#---> Loading the dataset
	load_dataset = 'load_' + dataset_name

	X, y = eval(load_dataset +  "(return_X_y=True)")
	n, d = X.shape
	n_classes = len(np.unique(y))
	if n_classes == 2:
		one_hot = False
	else:
		one_hot = True

	fit_intercept = True

	print('Dataset ' + str(dataset_name) + ' loaded. The dimensions are : ' + str(n) + ', ' + str(d))

	use_mrc_cg = True
	use_svm_cg = True

	seed = 42

	# Times
	mrc_cg_time_arr = []
	svm_cg_time_arr = []

	# Errors
	mrc_cg_error_arr = []
	svm_cg_error_arr = []

	n_splits = 10
	X = StandardScaler().fit_transform(X, y)

	cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

	phi_ob = BasePhi(n_classes=n_classes,
					 fit_intercept=fit_intercept,
					 one_hot=one_hot).fit(X, y)

	# Get the upper bound for the whole dataset 
	# as we dont need testing data to get upper bound
	# So, we can get the exact upper bound instead of approximating it.
	if use_mrc_cg:
		mu, nu, R_true, I, R_k, _, _ = mrc_cg(X,
                                         	  y,
	                                     	  phi_ob,
	                                     	  s,
	                                     	  n_max,
	                                     	  k_max,
	                                     	  eps)

	i = 0
	# Paired and stratified cross-validation
	for train_index, test_index in cv.split(X, y):

		print('Cross validation iteration : ', i)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		n_train = X_train.shape[0]
		d = X.shape[1]
		n_est = n_train

		# Linear feature mapping 
		phi_ob = BasePhi(n_classes=n_classes,
						 fit_intercept=fit_intercept,
						 one_hot=one_hot).fit(X_train, y_train)

#---> MRC_CG (Constraint generation for MRCs)

		if use_mrc_cg:
			mu, nu, R, I, R_k, mrc_cg_time, init_mrc_cg_time = mrc_cg(X_train,
                                                                      y_train,
                                                                      phi_ob,
                                                                      s,
                                                                      n_max,
                                                                      k_max,
                                                                      eps)

			# Remove the values of mu close to zero
			ind_to_remove = np.where(np.isclose(mu[I], 0))[0]
			I = list(set(I) - set(np.asarray(I)[ind_to_remove]))
			mu[ind_to_remove] = 0

			# Predict the error
			phi_te = phi_ob.eval_x(X_test)
			proba = predict_proba(phi_te, mu, nu, n_classes)
			# Deterministic classifier
			y_pred = np.argmax(proba, axis=1)
			mrc_cg_error = np.average(y_test != y_pred)

			# Add to the array to be averaged
			mrc_cg_time_arr.append(mrc_cg_time)
			mrc_cg_error_arr.append(mrc_cg_error)


#---> SVM CG (Constraint generation for SVM)
		if use_svm_cg:
			initTime = time.time()
			beta0_arr = np.zeros(n_classes)
			beta_arr = np.zeros((X_train.shape[1], n_classes))

			y_train1 = y_train.copy()
			if n_classes == 2:
				n_classif = 1
			else:
				n_classif = n_classes

			for y_i in range(n_classif):

				y_train1[y_train1 != y_i] = -1
				y_train1[y_train1 == y_i] = 1
				lam_max = np.max(np.sum( np.abs(X_train), axis=0))
				lam = 0.01*lam_max
				obj, time_total, time_CG, beta, beta0, support = use_FOM_CG(X_train,
																			y_train1,
																			lam=lam,
																			tau_max=0.1,
																			tol=eps)

				beta0_arr[y_i] = beta0
				beta_arr[:, y_i] = beta

				y_train1 = y_train.copy()

			proba = X_test @ beta_arr + beta0_arr
			y_pred = np.argmax(proba, axis=1)
			svm_cg_error = np.average(y_pred != y_test)
			svm_cg_time = time.time() - initTime

			# Add to the array to be averaged.
			svm_cg_time_arr.append(svm_cg_time)
			svm_cg_error_arr.append(svm_cg_error)

		i = i + 1

	# Print and save the data
	if use_mrc_cg:
		avg_time_mrc_cg = np.asarray([np.average(mrc_cg_time_arr),
									  np.std(mrc_cg_time_arr)])

		avg_error_mrc_cg = np.asarray([np.average(mrc_cg_error_arr),
									  np.std(mrc_cg_error_arr)])

		# Time
		print('Average time taken by MRC_CG : \t' + str(avg_time_mrc_cg[0]) \
											    + ' +/- ' + str(avg_time_mrc_cg[1]))
		np.savetxt(path + str(dataset_name) + '/mrc_cg_time.csv', avg_time_mrc_cg, delimiter=",", fmt='%s')

		# Worst-case error probability
		print('The worst-case error probability of MRC-CG : \t' + str(R_true))

		# Error
		print('Average error for MRC-CG : \t' + str(avg_error_mrc_cg[0]) \
													+ ' +/- ' + str(avg_error_mrc_cg[1]))
		np.savetxt(path + str(dataset_name) + '/mrc_cg_error.csv', mrc_cg_error_arr, delimiter=",", fmt='%s')

	if use_svm_cg:
		print('\n\n')

		avg_time_svm_cg = np.asarray([np.average(svm_cg_time_arr),
									  np.std(svm_cg_time_arr)])

		avg_error_svm_cg = np.asarray([np.average(svm_cg_error_arr),
									   np.std(svm_cg_error_arr)])

		print('Average time taken by SVM-CG : \t' + str(avg_time_svm_cg[0]) \
											   + ' +/- ' + str(avg_time_svm_cg[1]))
		np.savetxt(path + str(dataset_name) + '/svm_cg_time.csv', avg_time_svm_cg, delimiter=",", fmt='%s')

		print('Average error for SVM-CG : \t' + str(avg_error_svm_cg[0]) \
											 + ' +/- ' + str(avg_error_svm_cg[1]))
		np.savetxt(path + str(dataset_name) + '/svm_cg_error.csv', avg_error_svm_cg, delimiter=",", fmt='%s')

	print('Data saved successfully in ', path + str(dataset_name))
		