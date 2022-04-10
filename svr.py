from cv2 import sepFilter2D
from matplotlib.pyplot import axis
import numpy as np
from collections import Counter
from sklearn import svm

class SVR:
    def __init__(self,kernel='rbf',C=10, gamma = 0.01):
        self.clf_x = svm.SVR(kernel = kernel, C = C, gamma = gamma)
        self.clf_y = svm.SVR(kernel = kernel, C = C, gamma = gamma)
        self._input_train=None
        self._output_train=None
        # self.pred_type = pred_type
 
    def fit(self,x_train,y_train):
        self._input_train=x_train
        self._output_train=y_train
        self.clf_x.fit(x_train,y_train[:,0])
        self.clf_y.fit(x_train,y_train[:,1])
        # return self
 
    def predict(self, X_predict):
        # y_predict=[self._predict(x1) for x1 in X_predict]
        x = self.clf_x.predict(X_predict)
        y = self.clf_y.predict(X_predict)

        return np.column_stack((x, y))