#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas
from preprocessing import DataPreprocessing

def binning(col,cut_points,labels=None):
    minval = col.min()
    maxval = col.max()
    break_points = [minval] + cut_points + [maxval]

if __name__ == "__main__":
    file_fullpath = '/home/login01/Workspaces/python/dataset/cs.csv'
    #df = pandas.read_csv(file_fullpath,sep=',',index_col=0,na_values='NA',dtype=object,low_memory=False)
    #df = pandas.read_csv(file_fullpath,sep=',',index_col=0,na_values='NA',low_memory=False)
    df = pandas.read_csv(file_fullpath,sep=',',na_values='NA',low_memory=False)
    attributes=["RevolvingUtilizationOfUnsecuredLines","age","NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse","NumberOfDependents"]
    target_key = "SeriousDlqin2yrs"
    #print(df['age'].value_counts())
    #df.info()

    #cut_points = [30,50,70]
    #binning(df['age'],cut_points=cut_points)

    mypreprocessing = DataPreprocessing(df,attributes,target_key)
    age_bins = mypreprocessing.single_attr_binning('age',bin_num=5)
    print(type(age_bins))
    print(pandas.value_counts(age_bins))
    groups = df.groupby(mypreprocessing.single_attr_binning('age',bin_num=5))
    print(type(groups))
    print(list(groups))

