#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas

class GiniIndex:
    def __init__(self,df,attribute,key,y):
        '''
        :param df: the dataframe containing the attriute and target
        :param attribute: the attribute that need to be calculated the gini score and is type 1
        :param key: the target's label in the classification problem
        :param y: the target in the classification problem with value 0 or 1
        :return: the gini score of the attribute
        '''
        self.df = df
        self.attribute = attribute
        self.key = key
        self.y = y

    def gini_index(self):
        values = list(set(self.df[self.attribute]))
        attr_gini_index = []
        attr_gini_index_dict = {}
        for v in values:
            gini_index = 0
            #distinct_vals = list(set(self.df[v]))
            #if len(distinct_vals)!=self.df.shape[0]:
            wi = self.df[v].groupby(self.df[v]).count()/self.df.shape[0]
            pi_pos = self.df.groupby(self.df[v])[self.key].sum()/self.df.groupby(self.df[v])[v].count()
            pi_neg = 1-pi_pos
            ginii_score = 1-(pi_pos**2+pi_neg**2) #ginii_score's type is pandas.core.series.Series
            for key,value in ginii_score.to_dict().items():
                gini_index+=wi.to_dict().get(key)*value
            attr_gini_index.append(gini_index)
            attr_gini_index_dict.setdefault(v,gini_index)
        #return attr_gini_index
        return attr_gini_index_dict

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
    print("Gini index of each attribute:")
    for key,val in mygini_index_dict.items():
        print("%s:%s" % (key,val))
