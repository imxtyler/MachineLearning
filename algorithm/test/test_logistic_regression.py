#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import multiprocessing
import data_read
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from preprocessing import DataCheck,DataPreprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from ChicagoBoothML_Helpy.EvaluationMetrics import bin_classif_eval

def data_preprocess(data_path,form=0,attributes=None,all_labels=None,target_key=None,to_binary_attrs=None,area_attrs=None,show=True,stats=True,stats_file_path='/tmp',test_size=0.3,cut_point=0,random_state=99):
    '''
    :param data_path: string, the data's path
    :param form: int, indicate that what type of data to process:
            0: data isn't split to train and test set, one whole file
            1: data has been split to train and test set, two files
            2: data is split to multiple directories, each directory exists one or multiple files
    :param attributes: list of string, labels of X
    :param all_labels: list of string, labels of data, including all X and y, even the label isn't used, should use when form=2
    :param target_key: string, label of target
    :param to_binary_attrs: list of string, attributes to be converted to binary value:1 or 0
    :param area_attrs: list of string, attributes about china area, like province, city, etc.
    :param show: bool, indicate that the data's summary information should be printed
    :param stats: bool, indicate that the data's detail statistical information should be done
    :param stats_file_path: string, the path of statistical files
    :param test_size: float, the size ratio of test
    :param cut_point: int, indicate that which files are used to train model, which files are used to test model, it should be used when form=2
    :param random_state: int, random state
    :return: composite of DataFrame: X_train,X_test,y_train,y_test
    '''
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    try:
        df,train,test = data_read.data_read(data_path,form,all_labels,test_size,cut_point,random_state)
        if attributes is None:
            attributes = list(df.columns.values)
        if show==True:
            print("BEFORE DATA PREPROCESS, SUMMARY INFORMATION OF THE DATA:")
        file_path_stats = stats_file_path+'/'+'bef_data_statistics.csv'

        datacheck = DataCheck(df,target_key)
        df = datacheck.check_type(show=show,stats=stats,file_path_stats=file_path_stats)
        df_datapreprocessing = DataPreprocessing(df,attributes,target_key)
        df = df_datapreprocessing.discard_trivial_attrs()

        datacheck = DataCheck(train,target_key)
        train = datacheck.check_type(show=show,stats=stats,file_path_stats=file_path_stats)
        train_datapreprocessing = DataPreprocessing(train,attributes,target_key)
        train = train_datapreprocessing.discard_trivial_attrs()

        datacheck = DataCheck(test,target_key)
        test = datacheck.check_type(show=show,stats=stats,file_path_stats=file_path_stats)
        test_datapreprocessing = DataPreprocessing(test,attributes,target_key)
        test = test_datapreprocessing.discard_trivial_attrs()

        if to_binary_attrs is not None:
            train_datapreprocessing.transform_x_to_binary(to_binary_attrs)
            train_datapreprocessing.transform_x_dtype(to_binary_attrs,d_type=[int],uniform_type=True)
        resource_dir = '../resources'
        if area_attrs is not None:
            train_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
            train_datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
        X_train = train_datapreprocessing.x_dummies_and_fillna()

        if to_binary_attrs is not None:
            test_datapreprocessing.transform_x_to_binary(to_binary_attrs)
            test_datapreprocessing.transform_x_dtype(to_binary_attrs,d_type=[int],uniform_type=True)
        resource_dir = '../resources'
        if area_attrs is not None:
            test_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
            test_datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
        X_test = test_datapreprocessing.x_dummies_and_fillna()

        y_train = train[target_key]
        y_test = test[target_key]

        if show==True:
            print("AFTER DATA PREPROCESS, SUMMARY INFORMATION OF THE DATA:")
        file_path_stats = stats_file_path+'/'+'aft_statistics.csv'
        datacheck = DataCheck(df,target_key)
        datacheck.data_summary(show,stats,file_path_stats)
        print("Please see the statistical files in directory %s" % stats_file_path)

    except Exception as e:
        print(e)
    finally:
        return X_train,X_test,y_train,y_test

def train_test(X_train,X_test,y_train,y_test):
    #-----------------------------Find the best parameters' combination of the model------------------------------
    #param_test1 = {'n_estimators': range(20, 600, 20)}
    #gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=200,
    #                                                         min_samples_leaf=2, max_depth=5, max_features='sqrt',
    #                                                         random_state=19),
    #                        param_grid=param_test1, scoring='roc_auc', cv=5)
    #gsearch1.fit(X_train,y_train)
    #for item in gsearch1.grid_scores_:
    #    print(item)
    #print(gsearch1.best_params_)
    #print(gsearch1.best_score_)
    #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch2.best_score_,'\n')
    # print('-----------------------------------------------------------------------------------------------------')
    # param_test2 = {'max_depth': range(2, 16, 2), 'min_samples_split': range(20, 200, 20)}
    # gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=300,
    #                                                          min_samples_leaf=2, max_features='sqrt', oob_score=True,
    #                                                          random_state=19),
    #                         param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(X_train,y_train)
    # for item in gsearch2.grid_scores_:
    #    print(item)
    # print(gsearch2.best_params_)
    # print(gsearch2.best_score_)
    # print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_,'\n')
    ##-----------------------------Find the best parameters' combination of the model------------------------------
    B = 300
    RANDOM_SEED = 99
    model = \
        RandomForestClassifier(
            n_estimators=B,
            #criterion='entropy',
            criterion='gini',
            #max_depth=None,  # expand until all leaves are pure or contain < MIN_SAMPLES_SPLIT samples
            max_depth=4,
            min_samples_split=80,
            min_samples_leaf=2,
            min_weight_fraction_leaf=0.0,
            #max_features=None, # number of features to consider when looking for the best split; None: max_features=n_features
            max_features="sqrt",
            max_leaf_nodes=None,  # None: unlimited number of leaf nodes
            bootstrap=True,
            oob_score=True,  # estimate Out-of-Bag Cross Entropy
            n_jobs=multiprocessing.cpu_count() - 4,  # paralellize over all CPU cores minus 4
            class_weight=None,  # our classes are skewed, but but too skewed
            random_state=RANDOM_SEED,
            verbose=0,
            warm_start=False)

    kfold = model_selection.KFold(n_splits=5,random_state=RANDOM_SEED)
    eval_standard = ['accuracy','recall_macro','precision_macro','f1_macro']
    results = []
    for scoring in eval_standard:
        cv_results = model_selection.cross_val_score(model,X_train,y_train,scoring=scoring,cv=kfold)
        results.append(cv_results)
        msg = "%s: %f (%f)" % (scoring,cv_results.mean(),cv_results.std())
        print(msg)
    # Make predictions on validation dataset
    model.fit(X_train,y_train)
    print('oob_score: %f' % (model.oob_score_))
    #default evaluation way
    print('-------------------default evaluation----------------------')
    rf_pred_probs = model.predict(X=X_test)
    result_probs = np.column_stack((rf_pred_probs,y_test.as_matrix()))
    #for item in result_probs:
    #  print(item)
    print("confusion_matrix:\n",metrics.confusion_matrix(y_test, rf_pred_probs))
    print("accuracy_score:",metrics.accuracy_score(y_test, rf_pred_probs))
    print("recall_score:",metrics.recall_score(y_test, rf_pred_probs))
    print("precision_score:",metrics.precision_score(y_test, rf_pred_probs))
    print("f1_score:",metrics.f1_score(y_test, rf_pred_probs))
    print("roc_auc_score:",metrics.roc_auc_score(y_test, rf_pred_probs))
    print("classification_report:\n",metrics.classification_report(y_test, rf_pred_probs))

    rf_pred_probs = model.predict_proba(X=X_test)
    result_probs = np.column_stack((rf_pred_probs,y_test.as_matrix()))
    #for item in result_probs:
    #   print(item)
    result_df = DataFrame(result_probs,columns=['pred_neg','pred_pos','real'])
    #fpr,tpr,thresholds = metrics.roc_curve(result_df['real'],result_df['pred_pos'],pos_label=2)
    fpr,tpr,_ = metrics.roc_curve(result_df['real'],result_df['pred_pos'])
    # good model's auc should > 0.5
    print("auc:",metrics.auc(fpr,tpr))
    # good model's ks should > 0.2
    print("ks:",max(tpr-fpr))
    # good model's gini should > 0.6
    print("gini:",2*metrics.auc(fpr,tpr)-1)

    #self-defined evaluation way
    print('-------------------self-defined evaluation----------------------')
    low_prob = 1e-6
    high_prob = 1 - low_prob
    log_low_prob = np.log(low_prob)
    g_low_prob = np.log(low_prob)
    log_high_prob = np.log(high_prob)
    log_prob_thresholds = np.linspace(start=log_low_prob,stop=log_high_prob,num=100)
    prob_thresholds = np.exp(log_prob_thresholds)
    rf_pred_probs = model.predict_proba(X=X_test)
    #result_probs = np.column_stack((rf_pred_probs,y_test))
    #for item in result_probs:
    #    print(item)
    #for item in rf_pred_probs[:,1]:
    #    print(item)

    ## histogram of predicted probabilities
    ##n,bins,patches = plt.hist(rf_pred_probs[:1],10,normed=1,facecolor='g',alpha=0.75)
    ##plt.xlabel('Predicted probability of diabetes')
    ##plt.ylabel('Frequency')
    ##plt.title('Histogram of predicted probabilities')
    ###plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    ##plt.axis([0,1,0,1])
    ##plt.grid(True)

    #print(type(rf_pred_probs))
    #print(type(rf_pred_probs[:,1]))
    #print(rf_pred_probs[:,1])
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.hist(rf_pred_probs[:,1], bins=20)
    #plt.xlim(0,1)
    #plt.title('Histogram of predicted probabilities')
    #plt.xlabel('Predicted probability of diabetes')
    #plt.ylabel('Frequency')
    #plt.show()

    model_oos_performance = bin_classif_eval(rf_pred_probs[:,1],y_test,pos_cat=1,thresholds=prob_thresholds)
    #print(type(model_oos_performance))
    #for item in model_oos_performance.recall:
    #    print(item)
    recall_threshold = .74
    idx = next(i for i in range(100) if model_oos_performance.recall[i] <= recall_threshold) - 1
    print("idx = %d" % idx)
    selected_prob_threshold = prob_thresholds[idx]
    print("selected_prob_threshold:", selected_prob_threshold)
    print(model_oos_performance.iloc[idx,:])

def train_test1(X_train,X_test,y_train,y_test):
    B=10
    parameters = {'n_estimators':[10],'criterion':['gini']}
    best_parameters = {}
    model = \
        LogisticRegression(
            class_weight='balanced',
            penalty='L1'
            )
    grid_cv = GridSearchCV(model,parameters)
    grid_cv.fit(X_train,y_train)
    print('best_score_:',grid_cv.best_score_)
    print('best_estimator_:',grid_cv.best_estimator_)
    rf_pred_probs = grid_cv.predict(X=X_test)
    result_probs = np.column_stack((rf_pred_probs,y_test.as_matrix()))
    #for item in result_probs:
    #  print(item)
    print("confusion_matrix:\n",metrics.confusion_matrix(y_test, rf_pred_probs))
    print("accuracy_score:",metrics.accuracy_score(y_test, rf_pred_probs))
    print("recall_score:",metrics.recall_score(y_test, rf_pred_probs))
    print("precision_score:",metrics.precision_score(y_test, rf_pred_probs))
    print("f1_score:",metrics.f1_score(y_test, rf_pred_probs))
    print("roc_auc_score:",metrics.roc_auc_score(y_test, rf_pred_probs))
    print("classification_report:\n",metrics.classification_report(y_test, rf_pred_probs))

    rf_pred_probs = grid_cv.predict_proba(X=X_test)
    result_probs = np.column_stack((rf_pred_probs,y_test.as_matrix()))
    #for item in result_probs:
    #   print(item)
    result_df = DataFrame(result_probs,columns=['pred_neg','pred_pos','real'])
    #fpr,tpr,thresholds = metrics.roc_curve(result_df['real'],result_df['pred_pos'],pos_label=2)
    fpr,tpr,_ = metrics.roc_curve(result_df['real'],result_df['pred_pos'])
    # good model's auc should > 0.5
    print("auc:",metrics.auc(fpr,tpr))
    # good model's ks should > 0.2
    print("ks:",max(tpr-fpr))
    # good model's gini should > 0.6
    print("gini:",2*metrics.auc(fpr,tpr)-1)

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    attributes_dir = '../resources/attributes'

    #attribute_file_path = attributes_dir+'/'+'user_portrait_info_v1'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2_tt'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2'

    #attribute_file_path = attributes_dir+'/'+'user_portrait_info_v1_20170608'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_20170608'

    attribute_file_path = attributes_dir+'/'+'user_portrait_info_v2_20170612'
    data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2_20170612'

    attribute_file = open(attribute_file_path,'r')
    attributes = []
    all_labels = []
    while True:
        line = attribute_file.readline()
        if not line:
            break
        matchObj = re.match(r'\#(.*)',line.strip('\n'),re.M | re.I)
        if matchObj:
            all_labels.append(matchObj.group(1))
        else:
            attributes.append(line.strip('\n'))
            all_labels.append(line.strip('\n'))
    target_key="user_own_overdue"
    #target_key="user_own_ninety_overdue_order"

    to_binary_attrs = ['user_live_address','user_rela_name','user_relation','user_rela_phone','user_high_edu','user_company_name']
    area_attrs = ['user_live_province','user_live_city']
    #X_train,X_test,y_train,y_test = data_preprocess(data_path=data_path,form=1,attributes=attributes,target_key=target_key,to_binary_attrs=to_binary_attrs,area_attrs=area_attrs,show=False)
    X_train,X_test,y_train,y_test = data_preprocess(data_path=data_path,form=2,attributes=attributes,all_labels=all_labels,target_key=target_key,to_binary_attrs=to_binary_attrs,area_attrs=area_attrs,show=False,cut_point=6)
    #train_test(X_train,X_test,y_train,y_test)
    train_test1(X_train,X_test,y_train,y_test)
