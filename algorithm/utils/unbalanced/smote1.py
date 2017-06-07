# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 17:39:48 2017

@author: lenovo
"""
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


class Smote():
    # version: original, borderline1,borderline2
    def __init__(self,random_seed=None,n_neighbors=5,m_neighbors=5,version='original'):
        self.random_seed = random_seed
        self.n_neighbors = n_neighbors
        self.m_neighbors = m_neighbors
        self.version = version
        self.logger = logging.getLogger(__name__)
    def _make_samples(self,x_gene,x_all,nns,num_samples):
        x_new = np.zeros((num_samples,x_all.shape[1])) 
        random_state = np.random.RandomState(self.random_seed)
        # randomly select x_min and its neighbor to generate samples
        samples = random_state.randint(
            low=0, high=len(nns.flatten()), size=num_samples)
        for i, n in enumerate(samples):
            line,col = divmod(n,nns.shape[1])
            x_new[i] = x_gene[line] - (x_gene[line] - x_all[nns[line, col]])*random_state.uniform()
        y_new = np.array([self.min_c] * len(x_new))
        return x_new,y_new
    
    def danger_samples(self,nns,y):
        danger_id = []        
        for i,line in enumerate(nns):
            num_maj = 0
            for k, neighbor in enumerate(line):
                if y[neighbor] == self.maj_c:
                    num_maj+=1
            # is noise ?
            if num_maj == self.m_neighbors:
                continue
            # is safe ?
            elif num_maj <= self.m_neighbors/2.0:
                continue
            else:
                danger_id.append(i)
        return danger_id
                
    def _validate_estimator(self):
        # to validate the input
        if isinstance(self.n_neighbors,int):
            self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        else:
            raise ValueError('n_neighbor should be int')
        if self.version not in ['original','borderline1','borderline2']:
            raise ValueError('version should be original,borderline1,borderline2')
        else:
            pass 
    
    def fit(self,x,y):
        # find statistics of dataset
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
    
    def sample(self,x,y):
        self._validate_estimator()
        self.fit(x,y)
        x_min = x[y==self.min_c]
        y_min = y[y==self.min_c]
        num_samples = self.stats_c[self.maj_c] - self.stats_c[self.min_c]    
        if self.version == 'original':
            self.nn_.fit(x_min)
            nns = self.nn_.kneighbors(x_min,return_distance=False)
            x_new,y_new = self._make_samples(x_min,x_min,nns,num_samples)
        elif self.version == 'borderline1':
            nn_m_ = NearestNeighbors(self.m_neighbors)
            nn_m_.fit(x)
            nns_m = nn_m_.kneighbors(x_min,return_distance=False)
            danger_id = self.danger_samples(nns_m,y)
            x_sam = x_min[danger_id]
            self.nn_.fit(x_min)
            nns = self.nn_.kneighbors(x_sam, return_distance=False) 
            x_new,y_new = self._make_samples(x_sam,x_min,nns,num_samples)
            
        else:
            # percentage for generate according maj and min,it's assigned arbitrary by me
            # cause I don't find the method to assign it in literature
            alpha = 0.6
            nn_m_ = NearestNeighbors(self.m_neighbors)
            nn_m_.fit(x)
            nns_m = self.nn_m_.kneighbors(x_min,return_distance=False)
            danger_id = self.danger_samples(nns_m,y)
            x_sam = x_min[danger_id]
            nn_min_ = NearestNeighbors(self.n_neighbors)
            nn_maj_ = NearestNeighbors(self.n_neighbors)
            nn_maj_.fit(x[y==self.maj_c])
            nn_min_.fit(x_min)
            nns_min = self.nn_min_.kneighbors(x_sam, return_distance=False)
            nns_maj = self.nn_maj_.kneighbors(x_sam, return_distance=False)
            num_samples_min = int(num_samples*alpha)+1
            num_samples_max = num_samples*(1-alpha)
            x_new_min,y_new_min = self._make_samples(x_sam,x_min,nns_min,num_samples_min)
            x_new_maj,y_new_maj = self._make_samples(x_sam,x[y==self.maj_c],nns_maj,num_samples_max)
            x_new = np.vstack((x_new_maj,x_new_min))
            y_new = np.hstack((y_new_maj,y_new_min))
        return np.vstack((x,x_new)),np.hstack((y,y_new))
