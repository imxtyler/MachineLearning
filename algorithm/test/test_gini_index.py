#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas
from gini_index import GiniIndex

if __name__ == "__main__":
    file_fullpath = '/home/login01/Workspaces/python/dataset/cs.csv'
    #df = pandas.read_csv(file_fullpath,sep=',',index_col=0,na_values='NA',dtype=object,low_memory=False)
    #df = pandas.read_csv(file_fullpath,sep=',',index_col=0,na_values='NA',low_memory=False)
    df = pandas.read_csv(file_fullpath,sep=',',na_values='NA',low_memory=False)
    attribute=["RevolvingUtilizationOfUnsecuredLines","age","NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse","NumberOfDependents"]
    target_key = "SeriousDlqin2yrs"
    df[target_key] = df[target_key].fillna(0)
    target = df[target_key]
    #pandas.set_option('display.max_rows', None)
    #print(target)
    gini = GiniIndex(df,attribute,target_key,target)
    #mygini_index = gini.gini_index()
    #print("mygini_score:", mygini_index)
    mygini_index_dict = gini.gini_index()
    gini_list = sorted(mygini_index_dict.items(),key=lambda item:item[1])
    print("Gini index of each attribute:")
    #for key,val in mygini_index_dict.items():
    #    print("%s:%s" % (key,val))
    for item in gini_list:
        print(item)