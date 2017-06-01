#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import sys,pprint
import pandas
from preprocessing import DataPreprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class random_forest_model():
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


if __name__ == "__main__":
    #pprint.pprint(sys.path)
    file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_app_hl_calls_stg1/app_hl_stg1.csv'
    df = pandas.read_csv(file_fullpath,sep=',',na_values='NA',low_memory=False)
    attributes=[
        "id",
        "method",
        #"object_id",
        #"true_name",
        #"id_card_no",
        #"phone",
        "status",
        #"create_time",
        #"update_time",
        #"real_name",
        #"idcard",
        "sex",
        "age",
        "in_net_time",
        #"gmt_create",
        #"reg_time",
        "allavgamt",
        "12avgamt",
        "9avgamt",
        "6avgamt",
        "3avgamt",
        "planavgamt",
        #"phone_balance",
        "12zhuavgcount",
        "12zhutime",
        "9zhuavgcount",
        "9zhutime",
        "6zhuavgcount",
        "6zhutime",
        "3zhuavgcount",
        "3zhutime",
        "12beiavgcount",
        "12beitime",
        "9beiavgcount",
        "9beitime",
        "6beiavgcount",
        "6beitime",
        "3beiavgcount",
        "3beitime",
        "12hutongavgcount",
        "12hutongtime",
        "9hutongavgcount",
        "9hutongtime",
        "6hutongavgcount",
        "6hutongtime",
        "3hutongavgcount",
        "3hutongtime",
        "12receiveavg",
        "9receiveavg",
        "6receiveavg",
        "3receiveavg",
        "12sendavg",
        "9sendavg",
        "6sendavg",
        "3sendavg",
        "12avgflow",
        "9avgflow",
        "6avgflow",
        "3avgflow",
        "12avgnettime",
        "9avgnettime",
        "6avgnettime",
        "3avgnettime",
        "12contactavgcount",
        "12contacttime",
        "9contactavgcount",
        "9contacttime",
        "6contactavgcount",
        "6contacttime",
        "3contactavgcount",
        "3contacttime",
        "12zhuplace",
        "9zhuplace",
        "6zhuplace",
        "3zhuplace",
        #"user_own_overdue", #y, target
        #"user_own_overdue_yet",
        #"user_own_fpd_overdue_order",
        #"user_own_ninety_overdue_order",
        #"user_own_sixty_overdue_order",
        #"user_own_thirty_overdue_order",
        #"user_own_ninety_overdue_num",
        #"user_own_sixty_overdue_num",
        #"user_own_thirty_overdue_num",
        #"user_credit_ninety_overdue",
        "1zhuplace"
    ]
    target_key="user_own_overdue"
    validation_size = 0.20
    seed = 7
    X = df[attributes]
    Y = df[target_key]
    X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
    X_train_datapreprocessing = DataPreprocessing(X_train,attributes,target_key)
    X_train = X_train_datapreprocessing.data_pre_process()
    Y_train = Y_train.fillna(value=-1)

    scoring = 'accuracy'
    models = []
    models.append(('LR',LogisticRegression()))
    models.append(('RF',RandomForestClassifier()))

    results = []
    names = []
    for name,model in models:
        kfold = model_selection.KFold(n_splits=10,random_state=seed)
        cv_results = model_selection.cross_val_score(model,X_train,Y_train,scoring=scoring,cv=kfold)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
        print(msg)

    #fig = plt.figure()
    #fig.suptitle('Algorithm Comparison')
    #ax = fig.add_subplot(111)
    #plt.boxplot(results)
    #ax.set_xticklabels(names)
    #plt.show()

    # Make predictions on validation dataset
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    X_validation_datapreprocessing = DataPreprocessing(X_validation,attributes,target_key)
    X_validation = X_validation_datapreprocessing.data_pre_process()
    Y_validation = Y_validation.fillna(value=-1)
    predictions = lr.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
