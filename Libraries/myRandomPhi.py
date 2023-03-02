from MRCpy.phi import BasePhi
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_random_state
from sklearn.utils import check_array, check_X_y
import statistics

class myRandomPhi(BasePhi):

    def __init__(self, n_classes, sigma, fit_intercept=True,
                 n_components=600, random_state=None, one_hot=False):

        super().__init__(n_classes=n_classes, fit_intercept=fit_intercept,
                         one_hot=one_hot)
        self.n_components = n_components
        self.random_state = random_state
        self.sigma = sigma

    def fit(self, X, Y=None):

        X = check_array(X, accept_sparse=True)

        d = X.shape[1]

        self.random_weights_ = np.ones((X.shape[1],1))

        for sigma_i in self.sigma:
            # Obtain the random weight from a normal distribution.
            self.random_state = check_random_state(self.random_state)
            self.random_weights_ = \
                np.hstack((self.random_weights_, self.random_state.normal(0, 1 / sigma_i,
                                         size=(d, int(self.n_components / 2)))))

        self.random_weights_ = np.delete(self.random_weights_, 0, axis=1)

        # Sets the length of the feature mapping
        super().fit(X, Y)

        return self

    def transform(self, X):

        check_is_fitted(self, ["random_weights_", "is_fitted_"])
        X = check_array(X, accept_sparse=True)

        X_trans = X @ self.random_weights_
        X_feat = (1 / np.sqrt(int(self.n_components / 2))) * \
            np.hstack((np.cos(X_trans), np.sin(X_trans)))

        return X_feat