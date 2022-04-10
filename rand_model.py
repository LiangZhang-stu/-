import numpy as np
from sklearn import preprocessing


class rand_select:
    def __init__(self, rand_type):
        self.rand_type = rand_type
    
    def fit(self, x_train,y_train):
        self._input_train=x_train
        self._output_train=y_train
        self.scale = preprocessing.StandardScaler().fit(y_train)
        return self

    def _predict(self, x):
        if self.rand_type == 'uniform':
            return np.random.rand(self._output_train.shape[1])
        elif self.rand_type == 'gaussian':
            return np.random.randn(self._output_train.shape[1])
    
    def predict(self,X_predict):
        y_predict=[self._predict(x1) for x1 in X_predict]
        return np.array(y_predict)

    def __repr__(self):
        return 'knn(k=%d):'%self.k
 
    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return sum(y_predict==y_test)/len(x_test)