# from cv2 import sepFilter2D
# from matplotlib.pyplot import axis
import numpy as np
# from collections import Counter

class KNNClassifier:
    def __init__(self,k, pred_type):
        assert k>=1,'k must be valid'
        self.k=k
        self._input_train=None
        self._output_train=None
        self.pred_type = pred_type
 
    def fit(self,x_train,y_train):
        self._input_train=x_train
        self._output_train=y_train
        return self
 
    def _predict(self,x):
        d=[np.sqrt(np.sum((x_i-x)**2)) for x_i in self._input_train]
        nearest=np.argsort(d)
        top_k=[self._output_train[i] for i in nearest[:self.k]]
        if self.pred_type == 'mean':
            top_k = np.row_stack(top_k)
            result = top_k.mean(axis=0)
        elif self.pred_type == 'weight':
            top_k = np.row_stack(top_k)
            weight = np.array([10**(-d[i]) for i in nearest[:self.k]])
            weight = (weight / sum(weight)).reshape(-1, 1)
            result = (top_k * weight).mean(axis=0)
            
        return result
 
    def predict(self,X_predict):
        y_predict=[self._predict(x1) for x1 in X_predict]
        return np.array(y_predict)
 
    def __repr__(self):
        return 'knn(k=%d):'%self.k
 
    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return sum(y_predict==y_test)/len(x_test)