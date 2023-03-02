import csv
from os.path import dirname, join

import zipfile
import scipy.io
import sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch
from sklearn.preprocessing import OneHotEncoder


def normalizeLabels(origY):
	"""
	Normalize the labels of the instances in the range 0,...r-1 for r classes
	"""

	# Map the values of Y from 0 to r-1
	domY = np.unique(origY)
	Y = np.zeros(origY.shape[0], dtype=int)

	for i, y in enumerate(domY):
		Y[origY == y] = i

	return Y

def load_Colon(return_X_y=False):
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 2
	Samples per class               383, 307]
	Samples total                         690
	Dimensionality                         15
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	with zipfile.ZipFile(join(module_path, 'colon.csv.zip')) as zObject:
		with open(zObject.extract('colon.csv')) as csv_file:
			data_file = csv.reader(csv_file)
			n_samples = 62
			n_features = 2000
			data = np.empty((n_samples, n_features), dtype=float)
			target = []

			for i, ir in enumerate(data_file):
				data[i] = np.asarray([float(val) for val in ir[:-1]], dtype=float)
				target.append(ir[-1])

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if return_X_y:
		return data, normalizeLabels(np.asarray(target))

	return Bunch(data=data, target=normalizeLabels(np.asarray(target)),
				 target_names=target_names,
				 feature_names=[])

def load_Leukemia(return_X_y=False):
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 2
	Samples per class               383, 307]
	Samples total                         690
	Dimensionality                         15
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	with zipfile.ZipFile(join(module_path, 'leukemia.csv.zip')) as zObject:
		with open(zObject.extract('leukemia.csv')) as csv_file:
			data_file = csv.reader(csv_file)
			n_samples = 72
			n_features = 7129
			data = np.empty((n_samples, n_features), dtype=float)
			target = []

			for i, ir in enumerate(data_file):
				data[i] = np.asarray([float(val) for val in ir[:-1]], dtype=float)
				target.append(ir[-1])

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if return_X_y:
		return data, normalizeLabels(np.asarray(target))

	return Bunch(data=data, target=normalizeLabels(np.asarray(target)),
				 target_names=target_names,
				 feature_names=[])

def load_Arcene(return_X_y=False):
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 2
	Samples per class               383, 307]
	Samples total                         690
	Dimensionality                         15
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	with zipfile.ZipFile(join(module_path, 'arcene.csv.zip')) as zObject:
		with open(zObject.extract('arcene.csv')) as csv_file:
			data_file = csv.reader(csv_file)
			n_samples = 200
			n_features = 10001
			data = np.empty((n_samples, n_features), dtype=float)
			target = []

			for i, ir in enumerate(data_file):
				data[i] = np.asarray([float(val) for val in ir[:-1]], dtype=float)
				target.append(ir[-1])

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if return_X_y:
		return data, normalizeLabels(np.asarray(target))

	return Bunch(data=data, target=normalizeLabels(np.asarray(target)),
				 target_names=target_names,
				 feature_names=[])

def load_Ovarian(return_X_y=False):
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 2
	Samples per class               383, 307]
	Samples total                         690
	Dimensionality                         15
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	with zipfile.ZipFile(join(module_path, 'ovarian.csv.zip')) as zObject:
		with open(zObject.extract('ovarian.csv')) as csv_file:
			data_file = csv.reader(csv_file)
			n_samples = 253
			n_features = 15154
			data = np.empty((n_samples, n_features), dtype=float)
			target = []

			for i, ir in enumerate(data_file):
				data[i] = np.asarray([float(val) for val in ir[:-1]], dtype=float)
				target.append(ir[-1])

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if return_X_y:
		return data, normalizeLabels(np.asarray(target))

	return Bunch(data=data, target=normalizeLabels(np.asarray(target)),
				 target_names=target_names,
				 DESCR=None,
				 feature_names=[])

def load_Dorothea(return_X_y=False):
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 2
	Samples per class               383, 307]
	Samples total                         690
	Dimensionality                         15
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	with zipfile.ZipFile(join(module_path, 'dorothea.csv.zip')) as zObject:
		with open(zObject.extract('dorothea.csv')) as csv_file:
			data_file = csv.reader(csv_file)
			n_samples = 1150
			n_features = 100000
			data = np.empty((n_samples, n_features), dtype=float)
			target = []

			for i, ir in enumerate(data_file):
				data[i] = np.asarray([float(val) for val in ir[:-1]], dtype=float)
				target.append(ir[-1])

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if return_X_y:
		return data, normalizeLabels(np.asarray(target))

	return Bunch(data=data, target=normalizeLabels(np.asarray(target)),
				 target_names=target_names,
				 DESCR=None,
				 feature_names=[])

def load_MLL(return_X_y=False):
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 2
	Samples per class               383, 307]
	Samples total                         690
	Dimensionality                         15
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	with zipfile.ZipFile(join(module_path, 'MLL.csv.zip')) as zObject:
		with open(zObject.extract('MLL.csv')) as csv_file:
			data_file = csv.reader(csv_file)
			ids = next(data_file)
			n_samples = 72
			n_features = 12582
			data = np.empty((n_samples, n_features), dtype=float)
			target = []

			j = 0
			for i, ir in enumerate(data_file):
				if len(ir) < n_features:
					continue
				else:
					data[j] = np.asarray([float(val) for val in ir[1:-1]], dtype=float)
					target.append(ir[-1])
					j = j+1

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if return_X_y:
		return data, normalizeLabels(np.asarray(target))

	return Bunch(data=data, target=normalizeLabels(np.asarray(target)),
				 target_names=target_names,
				 DESCR=None,
				 feature_names=[])

def load_Prostate_GE(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'Prostate_GE.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('Prostate_GE.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])
		
	return X, normalizeLabels(y)

def load_GLI_85(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'GLI_85.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('GLI_85.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])

	return X, normalizeLabels(y)


def load_SMK_CAN_187(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'SMK_CAN_187.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('SMK_CAN_187.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])

	return X, normalizeLabels(y)

def load_T0X_171(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'TOX_171.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('TOX_171.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])

	return X, normalizeLabels(y)

def load_Lung(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'lung.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('lung.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])

	return X, normalizeLabels(y)

def load_GLIOMA(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'GLIOMA.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('GLIOMA.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])

	return X, normalizeLabels(y)

def load_CLL_SUB_111(return_X_y=False):
	module_path = dirname(__file__)
	with zipfile.ZipFile(join(module_path, 'CLL_SUB_111.mat.zip')) as zObject:
		mat = scipy.io.loadmat(zObject.extract('CLL_SUB_111.mat'))
		X = np.asarray(mat['X'])
		y = np.asarray(mat['Y'])
		y = np.asarray([i[0] for i in y])

	return X, normalizeLabels(y)