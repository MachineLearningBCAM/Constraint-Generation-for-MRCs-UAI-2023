"""
    This set of experiments show the scalability of the different algorithms
    with increasing number of features.
"""

import sys
import numpy as np
import warnings
import time
from sklearn.preprocessing import StandardScaler
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
    path = './Results/Scalability/'
    dataset_name = sys.argv[1]
    eps = 1e-4
    n_max = 100
    k_max = 20    
    s = 1
    one_hot = False
    fit_intercept = False

    n_feats = np.asarray([100, 500, 1000, 5000, 10000, 15000, 20000, 25000])

    # Loading the dataset
    load_dataset = 'load_' + dataset_name

    X, y = eval(load_dataset +  "(return_X_y=True)")
    n, d = X.shape
    n_classes = len(np.unique(y))

    # For 20 random split
    n_seeds = 50

    use_mrc_cg = True
    use_svm_cg = True
    use_mrc_lp = True

    mrc_cg_time_arr = np.zeros(n_feats.shape[0])
    svm_cg_time_arr = np.zeros(n_feats.shape[0])
    mrc_lp_time_arr = np.zeros(n_feats.shape[0])

    for j, k in enumerate(n_feats):

        print('\n\n Using ' + str(k) + ' number of RFF .......... ')
        seeds = random.sample(range(1000, 2000), n_seeds)

        seed_i = 0

        avg_mrc_cg_time = 0
        avg_svm_cg_time = 0
        avg_mrc_lp_time = 0

        for seed in seeds:

            # Setting the seed
            np.random.seed(seed)
            random.seed(seed)

            print('\n\n\n ###################### Random data partition ' + str(seed_i) + ' ..... ##################')
            X = StandardScaler().fit_transform(X, y)
            n_train = X.shape[0]
            d = X.shape[1]

            print('Dataset ' + str(dataset_name) + ' loaded. The dimensions are : ' + str(n_train) + ', ' + str(d))

            # Build the feature mapping.
            sigma = np.asarray([np.sqrt(d/2)])
            phi_ob = myRandomPhi(n_classes=n_classes,
                                 sigma=sigma,
                                 fit_intercept=fit_intercept,
                                 n_components=k,
                                 random_state=seed,
                                 one_hot=one_hot).fit(X, y)

            phi_ = phi_ob.eval_x(X)
            phi_ = np.unique(phi_, axis=0)

        #--> MRC-CG (Constraint generation for MRCs)
            if use_mrc_cg:

                mu, nu, R, I, R_k, mrc_cg_time, init_mrc_cg_time = mrc_cg(X,
                                                                          y,
                                                                          phi_ob,
                                                                          s,
                                                                          n_max,
                                                                          k_max,
                                                                          eps)

                print('The worst-case error probability using MRC-CG is : ', R)
                avg_mrc_cg_time = avg_mrc_cg_time + mrc_cg_time

        #--> SVM-CG (Constraint generation for SVMs)
            if use_svm_cg:
                y1 = y.copy()
                y1[y == 0] = -1
                y1[y == 1] = 1
                initTime = time.time()
                lam_max = np.max(np.sum( np.abs(phi_ob.transform(X)), axis=0))
                lam = 0.01*lam_max
                obj, _, _, beta, beta0, support = use_FOM_CG(phi_ob.transform(X), y1, lam=lam, tau_max=0.1, tol=eps)
                svm_cg_time = time.time() - initTime

                avg_svm_cg_time = avg_svm_cg_time + svm_cg_time

        #--> MRC-LP (Solving the LP formulation of MRCs using GUROBI optimizer)
            if use_mrc_lp and k<5000:
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
                mu_plus     = [(mrc_model.getVarByName("mu_+_" + str(i))).x for i in I]
                mu_minus    = [(mrc_model.getVarByName("mu_-_" + str(i))).x for i in I]
                nu_pos      = mrc_model.getVarByName("nu_+")
                nu_neg      = mrc_model.getVarByName("nu_-")
                mu          = np.asarray(mu_plus) - np.asarray(mu_minus)
                nu          = nu_pos - nu_neg
                mrc_upper   = mrc_model.objVal
                print('The worst-case error probability using MRC-LP is : ', mrc_upper)

                mrc_lp_time = time.time() - initTime

                avg_mrc_lp_time = avg_mrc_lp_time + mrc_lp_time

            print('\n ###################### Completed random data partition ' + str(seed_i) + ' ..... ##################')
            seed_i = seed_i + 1

        #---> Save all the data in an array for different number of features.
        # Saving the times
        print('The number of features are : ', k)
        if use_mrc_cg:
            avg_mrc_cg_time = avg_mrc_cg_time / seed_i
            mrc_cg_time_arr[j] = avg_mrc_cg_time
            print('Average time taken by MRC-CG : ', avg_mrc_cg_time)

        if use_svm_cg:
            avg_svm_cg_time = avg_svm_cg_time / seed_i
            svm_cg_time_arr[j] = avg_svm_cg_time
            print('Average time taken by SVM-CG : ', avg_svm_cg_time)

        if use_mrc_lp and k<5000:
            avg_mrc_lp_time = avg_mrc_lp_time / seed_i
            mrc_lp_time_arr[j] = avg_mrc_lp_time
            print('Average time taken by MRC-LP : ', avg_mrc_lp_time)

    print('The increasing number of features are : \n', n_feats)
    print('The average training times for MRC-CG : \n', mrc_cg_time_arr)
    print('The average training times for SVM-CG : \n', svm_cg_time_arr)
    print('The average training times for MRC-LP : \n', mrc_lp_time_arr)

    # Save the data
    if use_mrc_cg:
        np.savetxt(path + str(dataset_name) + '/mrc_cg_time.csv', mrc_cg_time_arr, delimiter=",", fmt='%s')

    if use_svm_cg:
        np.savetxt(path + str(dataset_name) + '/svm_cg_time.csv', svm_cg_time_arr, delimiter=",", fmt='%s')

    if use_mrc_lp:
        np.savetxt(path + str(dataset_name) + '/mrc_lp_time.csv', mrc_lp_time_arr, delimiter=",", fmt='%s')

    print('Data saved successfully in ', path + str(dataset_name))

      
