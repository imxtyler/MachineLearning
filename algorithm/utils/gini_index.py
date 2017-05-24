import pandas
import numpy as np

class GiniIndex:
    def __init__(self,df,attribute,y):
        self.df = df
        self.attribute = attribute
        self.y = y

    def gini_index(self):
        '''
        :param df: the dataframe containing the attriute and target
        :param attribute: the attribute that need to be calculated the gini score and is type 1
        :param y: the target in the classification problem with value 0 or 1
        :return: the gini score of the attribute
        '''
        values = list(set(self.df[self.attribute]))
        GiniScore = 0
        for v in values:
            val_nums = list(set(self.df[v]))
            if len(val_nums) != self.df.shape[0]:
                for val in val_nums:
                    if val == self.df[v]:
                        pass # fixme
            #subset = self.df.loc[self.df[self.attribute] == v]
            #w = subset.shape[0]*1.0/self.df.shape[0]
            #y1 = sum(subset[self.y])*1.0/subset.shape[0]
            #y2 = 1-y1
            #gini = 1-(y1**2+y2**2)
            #GiniScore += gini*w
        return GiniScore

if __name__ == "__main__":
    file_fullpath = '/home/login01/Workspaces/python/dataset/cs.csv'
    df = pandas.read_csv(file_fullpath,sep=',',index_col=0,na_values='NA',low_memory=False)
    attribute=["RevolvingUtilizationOfUnsecuredLines","age","NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse","NumberOfDependents"]
    target = df["SeriousDlqin2yrs"]
    target.fillna(0, inplace=False)
    #print(target)
    gini = GiniIndex(df,attribute,target)
    mygini_index = gini.gini_index()
    print("mygini_score:", mygini_index)