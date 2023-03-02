"""
Experiment to compare our method as feature selection approach.
The experiment is performed for binary classification problem.
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
	path = "./Results/Feature selection/"
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
	fit_intercept = True

	print('Dataset ' + str(dataset_name) + ' loaded. The dimensions are : ' + str(n) + ', ' + str(d))

	use_mrc_cg = True
	use_svm_cg = True
	use_rfe = True

	seed = 42

	mrc_cg_time_arr = []
	svm_cg_time_arr = []
	rfe_time_arr = []

	# Errors
	lr_mrc_cg_error_arr = []
	dt_mrc_cg_error_arr = []
	lr_svm_cg_error_arr = []
	dt_svm_cg_error_arr = []
	lr_rfe_error_arr = []
	dt_rfe_error_arr = []

	# Feature arrays 
	n_feats_mrc_cg = []
	n_feats_svm_cg = []

	n_splits = 10
	X = StandardScaler().fit_transform(X, y)

	cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

	i = 0
	# Paired and stratified cross-validation
	for train_index, test_index in cv.split(X, y):

		print('Cross validation iteration : ', i)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		n_train = X_train.shape[0]
		d = X.shape[1]
		n_est = n_train

		# Feature mapping
		phi_ob = BasePhi(n_classes=n_classes,
						 fit_intercept=fit_intercept,
						 one_hot=False).fit(X_train, y_train)

#---> MRC_CG (Constraint generation for MRCs)

		if use_mrc_cg:
			mu, nu, R, I, R_k, mrc_cg_time, _ = mrc_cg(X_train,
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

			# Map the features to original space
			indices = []
			for ind in I:
				if (ind - 1) >= 0:
					indices.append(ind - 1)

			# Using features from MRC-CG to perform the classification using LR and DT
			lr_classif = LogisticRegression(random_state=seed, penalty='none').fit(X_train[:, indices], y_train)
			y_pred = lr_classif.predict(X_test[:, indices])
			lr_mrc_cg_error = np.average(y_test != y_pred)

			dt_clf = tree.DecisionTreeClassifier(random_state=seed).fit(X_train[:, indices], y_train)
			y_pred = dt_clf.predict(X_test[:, indices])
			dt_mrc_cg_error = np.average(y_test != y_pred)

			# Add the data to the array to be averaged
			mrc_cg_time_arr.append(mrc_cg_time)
			lr_mrc_cg_error_arr.append(lr_mrc_cg_error)
			dt_mrc_cg_error_arr.append(dt_mrc_cg_error)
			n_feats_mrc_cg.append(len(I))

#---> SVM CG (Constraint generation for L1 SVM)
		if use_svm_cg:
			initTime = time.time()
			y_train1 = y_train.copy()
			y_test1 = y_test.copy()
			y_train1[y_train1 == 0] = -1
			y_train1[y_train1 == 1] = 1
			y_test1[y_test1 == 0] = -1
			y_test1[y_test1 == 1] = 1
			lam_max = np.max(np.sum( np.abs(X_train), axis=0))
			lam = 0.01*lam_max
			obj, time_total, time_CG, beta, beta0, support = use_FOM_CG(X_train, y_train1, lam=lam, tau_max=0.1, tol=eps)

			# Using features from SVM-CG to perform the classification using LR and DT
			lr_classif = LogisticRegression(random_state=seed, penalty='none').fit(X_train[:, support], y_train)
			y_pred = lr_classif.predict(X_test[:, support])
			lr_svm_cg_error = np.average(y_test != y_pred)

			dt_clf = tree.DecisionTreeClassifier(random_state=seed).fit(X_train[:, support], y_train)
			y_pred = dt_clf.predict(X_test[:, support])
			dt_svm_cg_error = np.average(y_test != y_pred)
			svm_cg_time = time.time() - initTime

			# Add the data to the array to be averaged
			svm_cg_time_arr.append(svm_cg_time)
			lr_svm_cg_error_arr.append(lr_svm_cg_error)
			dt_svm_cg_error_arr.append(dt_svm_cg_error)
			n_feats_svm_cg.append(len(support.tolist()))

#---> SVM RFE
		if use_rfe:
			initTime = time.time()
			estimator = LinearSVC(penalty='l1', random_state=seed, dual=False)
			selector = RFE(estimator, n_features_to_select=len(I))
			selector = selector.fit(X_train, y_train)
			support = selector.get_support(indices=True).tolist()
			rfe_time = time.time() - initTime
			lr_est = LogisticRegression(random_state=seed, penalty='none').fit(X_train[:, support], y_train)
			y_pred = lr_est.predict(X_test[:, support])
			lr_rfe_error = np.average(y_test != y_pred)
			dt_clf = tree.DecisionTreeClassifier(random_state=seed).fit(X_train[:, support], y_train)
			y_pred = dt_clf.predict(X_test[:, support])
			dt_rfe_error = np.average(y_test != y_pred)

			# Add the data to the array to be averaged
			rfe_time_arr.append(rfe_time)
			lr_rfe_error_arr.append(lr_rfe_error)
			dt_rfe_error_arr.append(dt_rfe_error)

		i = i + 1


	# Print and save the data
	if use_mrc_cg:
		avg_time_mrc_cg = np.asarray([np.average(mrc_cg_time_arr),
									  np.std(mrc_cg_time_arr)])

		avg_error_lr = np.asarray([np.average(lr_mrc_cg_error_arr),
									  np.std(lr_mrc_cg_error_arr)])

		avg_error_dt = np.asarray([np.average(dt_mrc_cg_error_arr),
								   np.std(dt_mrc_cg_error_arr)])

		avg_feats = np.asarray([np.average(n_feats_mrc_cg),
								np.std(n_feats_mrc_cg)])

		# Time
		print('Average time taken by MRC-CG : \t' + str(avg_time_mrc_cg[0]) \
											    + ' +/- ' + str(avg_time_mrc_cg[1]))
		np.savetxt(path + str(dataset_name) + '/mrc_cg_time.csv', avg_time_mrc_cg, delimiter=",", fmt='%s')

		# LR error
		print('Average error for LR classifier using MRC-CG features : \t' + str(avg_error_lr[0]) \
													+ ' +/- ' + str(avg_error_lr[1]))
		np.savetxt(path + str(dataset_name) + '/lr_mrc_cg_error.csv', avg_error_lr, delimiter=",", fmt='%s')

		# DT error
		print('Average error for DT classifier using MRC-CG features : \t' + str(avg_error_dt[0]) \
													+ ' +/- ' + str(avg_error_dt[1]))
		np.savetxt(path + str(dataset_name) + '/dt_mrc_cg_error.csv', avg_error_dt, delimiter=",", fmt='%s')

		# Average number of features selected
		print('Average number of features selcted using MRC-CG : \t' + str(avg_feats[0]) \
													+ ' +/- ' + str(avg_feats[1]))
		np.savetxt(path + str(dataset_name) + '/mrc_cg_n_feats.csv', avg_feats, delimiter=",", fmt='%s')

	if use_svm_cg:
		print('\n\n')

		avg_time_svm_cg = np.asarray([np.average(svm_cg_time_arr),
									  np.std(svm_cg_time_arr)])

		avg_error_lr = np.asarray([np.average(lr_svm_cg_error_arr),
									  np.std(lr_svm_cg_error_arr)])

		avg_error_dt = np.asarray([np.average(dt_svm_cg_error_arr),
								   np.std(dt_svm_cg_error_arr)])

		avg_feats = np.asarray([np.average(n_feats_svm_cg),
								np.std(n_feats_svm_cg)])

		# Time
		print('Average time taken by SVM-CG : \t' + str(avg_time_mrc_cg[0]) \
											    + ' +/- ' + str(avg_time_mrc_cg[1]))
		np.savetxt(path + str(dataset_name) + '/svm_cg_time.csv', avg_time_svm_cg, delimiter=",", fmt='%s')

		# LR error
		print('Average error for LR classifier using SVM-CG features : \t' + str(avg_error_lr[0]) \
													+ ' +/- ' + str(avg_error_lr[1]))
		np.savetxt(path + str(dataset_name) + '/lr_svm_cg_error.csv', avg_error_lr, delimiter=",", fmt='%s')

		# DT error
		print('Average error for DT classifier using SVM-CG features : \t' + str(avg_error_dt[0]) \
													+ ' +/- ' + str(avg_error_dt[1]))
		np.savetxt(path + str(dataset_name) + '/dt_svm_cg_error.csv', avg_error_dt, delimiter=",", fmt='%s')

		# Average number of features selected
		print('Average number of features selcted using SVM-CG : \t' + str(avg_feats[0]) \
													+ ' +/- ' + str(avg_feats[1]))
		np.savetxt(path + str(dataset_name) + '/svm_cg_n_feats.csv', avg_feats, delimiter=",", fmt='%s')

	if use_rfe:
		avg_time_rfe = np.asarray([np.average(rfe_time_arr),
									  np.std(rfe_time_arr)])

		avg_error_lr = np.asarray([np.average(lr_rfe_error_arr),
									  np.std(lr_rfe_error_arr)])

		avg_error_dt = np.asarray([np.average(dt_rfe_error_arr),
								   np.std(dt_rfe_error_arr)])

		# Time
		print('Average time taken by RFE : \t' + str(avg_time_rfe[0]) \
											    + ' +/- ' + str(avg_time_rfe[1]))
		np.savetxt(path + str(dataset_name) + '/rfe_time.csv', avg_time_rfe, delimiter=",", fmt='%s')

		# LR error
		print('Average error for LR classifier using RFE features : \t' + str(avg_error_lr[0]) \
													+ ' +/- ' + str(avg_error_lr[1]))
		np.savetxt(path + str(dataset_name) + '/lr_rfe_error.csv', avg_error_lr, delimiter=",", fmt='%s')

		# DT error
		print('Average error for DT classifier using RFE features : \t' + str(avg_error_dt[0]) \
													+ ' +/- ' + str(avg_error_dt[1]))
		np.savetxt(path + str(dataset_name) + '/dt_rfe_error.csv', avg_error_dt, delimiter=",", fmt='%s')

		print('Note that the number of features selected by RFE is equal to the number of features selected by SVM-CG')

	print('Data saved successfully in ', path + str(dataset_name))
