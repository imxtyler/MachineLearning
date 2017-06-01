#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas
from smote import Smote

if __name__ == "__main__":
    #file_fullpath='/home/login01/Workspaces/python/dataset/module_data_stg1/sample'
    file_fullpath='/home/login01/Workspaces/python/dataset/cs.csv'
    cs=pandas.read_csv(file_fullpath,sep=',',index_col=0,na_values='NA',low_memory=False)
    cs_mean_MonthlyIncome = cs.MonthlyIncome.mean(skipna=True)
    cs_mean_NumberOfDependents = cs.NumberOfDependents.mean(skipna=True)
    cs.ix[:, 'MonthlyIncome'] = cs.MonthlyIncome.fillna(cs_mean_MonthlyIncome, inplace=False)
    cs.ix[:, 'NumberOfDependents'] = cs.NumberOfDependents.fillna(cs_mean_NumberOfDependents, inplace=False)
    ismote=Smote(cs,20,6)
    print(ismote.n_samples)
    print(ismote.n_attrs)
    mysample=ismote.over_sampling()
    print(mysample)