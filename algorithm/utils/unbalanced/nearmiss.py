# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 12:06:20 2017

@author: lenovo
"""
import logging
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors


class NearMiss():
    
    '''
        version:1,2,3, which represent nearmiss1,2 or 3 
    '''
    
    def __init__(self,random_state = None,version = 1,n_neighbors = 3):
        self.random_state = random_state
        self.version = version
        self.n_neighbors = n_neighbors
        self.logger = logging.getLogger(__name__)
        
    def _validate_estimator(self):
        # to validate the input
        if isinstance(self.n_neighbors,int):
            self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        else:
            raise ValueError('n_neighbor should be int')
        if self.version not in [1,2,3]:
            raise ValueError('version should be 1,2 or 3')
        else:
            pass
        
    def fit(self,x,y):        
        # find the class statistics before sampling
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
                         %(len(self.stats_c), ','.join(self.stats_c)))
        return self
    
    def select_dist_based(self,x,y,dist,key,num_sample,way='nearest'):
        dist_avg = np.mean(dist[:,-self.n_neighbors:],axis=1)
        if way == 'nearest':
            sort_way = False
        elif way == 'farthest':
            sort_way = False
        else:
            raise ValueError('way should be nearest or farthest')
        sorted_id = sorted(range(len(dist_avg)),
                           key=dist_avg.__getitem__,reverse=sort_way)
        if len(sorted_id) < num_sample:
            self.logger.info('warn: num_sample is too large')
        sel_id = sorted_id[:num_sample]
        
        return x[y==key][sel_id],y[y==key][sel_id]    
        
    def sample(self,x,y):    
        self._validate_estimator()
        self.fit(x,y)
        x_min = x[y==self.min_c]
        y_min = y[y==self.min_c]
        x_sample = x_min.copy()
        y_sample = y_min.copy()
        num_sample = self.stats_c[self.min_c]
        self.nn_.fit(x_min)
        for key in self.stats_c.keys():
            if key == self.min_c:
                continue
            else:
                sub_samples_x = x[y==key]               
                if self.version == 1:
                    dist, ind = self.nn_.kneighbors(sub_samples_x) 
                    sel_x,sel_y = self.select_dist_based(x,y,dist,key,num_sample,way='nearest')
                elif self.version == 2:
                    dist, ind = self.nn_.kneighbors(sub_samples_x,n_neighbors=self.stats_c[self.min_c])
                    sel_x,sel_y = self.select_dist_based(x,y,dist,key,num_sample,way='nearest')          
                else:
                    self.ver3_nn_ = NearestNeighbors(self.n_neighbors)
                    self.ver3_nn_.fit(sub_samples_x)
                    dist_maj, ind_maj = self.ver3_nn_.kneighbors(x_min)                
                    ind_maj = np.unique(ind_maj.reshape(-1))
                    sub_samples_x = sub_samples_x[ind_maj,:]
                    dist,ind = self.nn_.kneighbors(sub_samples_x)
                    sel_x,sel_y = self.select_dist_based(x,y,dist,key,num_sample,
                                                         way='farthest')
                    
                    x_sample = np.vstack((x_sample,sel_x))
                    y_sample = np.hstack((y_sample,sel_y))
        return x_sample,y_sample
