# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 10:18:23 2017

@author: lenovo
"""
from collections import Counter
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import logging

class Ensemble():    
    # In[] n_subset number of subset, version:'easy_ensemble' or 'balance_cascade'
    def __init__(self,version='easy_ensemble',random_seed=None,n_subset=10,n_tree=10):
        self.random_seed = random_seed
        self.version = version
        self.n_subset = n_subset
        self.ensemble = {}
        self.ensemble['adclf'] = []
        self.ensemble['bias'] = []
        self.logger = logging.getLogger(__name__)
        
    # In[] find the class statistics before sampling
    def fit(self,x,y):               
        self.min_c = None
        self.maj_c = None
        self.stats_c = {}
        self.x_shape = None
        self.logger.info('compute classes statistics...')
        self.x_shape = x.shape
        self.stats_c = Counter(y)
        self.min_c = min(self.stats_c,key=self.stats_c.get)
        self.maj_c = max(self.stats_c,key=self.stats_c.get)
        self.logger.info('%s classes detected: %s'
                         %(len(self.stats_c), ','.join(str(self.stats_c))))
    
    # In[] x,y: subset need to be sampled not all set
    def under_sample(self,x,y,num_samples):
        random_state = np.random.RandomState(self.random_seed)
        sel_id = random_state.choice(range(len(x)),size=num_samples)
        return x[sel_id],y[sel_id]
    
    # In[] tune bias to be false positive 
    def bias_tune(self,false_positive,adaclf,x,y):
        estimators = adaclf.estimators_
        weights = adaclf.estimator_weights_
        res = np.zeros_like(y)
        ind = np.where(y==self.maj_c)
        for i,clf in enumerate(estimators):
            res = res+clf.predict(x)*weights[i]
        res = res[ind]
        res = sorted(res,reversed=True)
        bias = res[round(false_positive*len(ind))]
        return bias
        
    # In[]
    def sample(self,x,y):
        num_samples = self.stats_c[self.min_c]
        x_min = x[y==self.min_c]
        y_min = y[y==self.min_c]
        x_maj = x[y==self.maj_c]
        y_maj = y[y==self.maj_c]
        x_sample = x_min.copy()
        y_sample = y_min.copy()
        samples = []
        for _ in range(self.n_subset):
            x_under_samp,y_under_sample = self.under_sample(x_maj,y_maj,num_samples)
            x_samp = np.vstack((x_sample,x_under_samp))
            y_samp = np.hstack((y_sample,y_under_sample))
            samples.append((x_samp,y_samp))      
        for x_samp,y_samp in samples:                 
            adaboost_clf = AdaBoostClassifier(base_estimator='DecisionTreeClassifier',
                                              n_estimators=20)
            adaboost_clf.fit(x_samp,y_samp)      
            if self.version == 'easy_ensemble': 
                self.ensemble['adclf'].append(adaboost_clf)
            else:
                t = np.power((self.stats_c[self.min_c]*1.0/self.stats_c[self.maj_c]),
                             1.0/(self.n_subset+1))
                adaboost_clf = AdaBoostClassifier(base_estimator='DecisionTreeClassifier',
                                                  n_estimators=20)
                adaboost_clf.fit(x_samp,y_samp)
                bias = self.bias_tune(t,adaboost_clf,x_samp,y_samp)
                self.ensemble['adclf'].append(adaboost_clf)
                self.ensemble['bias'].append(bias)
        return self.ensemble                
