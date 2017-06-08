#!/usr/bin/env python
#-*- coding:utf-8 -*-

import random
import pandas
import numpy
import multiprocessing
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from preprocessing import DataPreprocessing
from gini_index import GiniIndex
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from ChicagoBoothML_Helpy.EvaluationMetrics import bin_classif_eval
from nearmiss1 import NearMiss
#from smote1 import Smote
from smote import Smote
from ensemble import Ensemble

if __name__ == "__main__":
    pandas.set_option('display.max_rows', None)
    #pprint.pprint(sys.path)
    #file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_app_hl_calls_stg1/app_hl_stg1.csv'
    #file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/md_data_stg2.csv'
    #file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/md_data_stg2_tmp.csv'
    #train_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/md_data_stg2_tmp.csv'
    train_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/tmp_train.csv'
    test_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/tmp_test.csv'
    #test_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/tmp_train.csv'
    #attributes=[
    #    "id",
    #    "method",
    #    #"object_id",
    #    #"true_name",
    #    #"id_card_no",
    #    #"phone",
    #    "status",
    #    #"create_time",
    #    #"update_time",
    #    #"real_name",
    #    #"idcard",
    #    "sex",
    #    "age",
    #    "in_net_time",
    #    #"gmt_create",
    #    #"reg_time",
    #    "allavgamt",
    #    "12avgamt",
    #    "9avgamt",
    #    "6avgamt",
    #    "3avgamt",
    #    "planavgamt",
    #    #"phone_balance",
    #    "12zhuavgcount",
    #    "12zhutime",
    #    "9zhuavgcount",
    #    "9zhutime",
    #    "6zhuavgcount",
    #    "6zhutime",
    #    "3zhuavgcount",
    #    "3zhutime",
    #    "12beiavgcount",
    #    "12beitime",
    #    "9beiavgcount",
    #    "9beitime",
    #    "6beiavgcount",
    #    "6beitime",
    #    "3beiavgcount",
    #    "3beitime",
    #    "12hutongavgcount",
    #    "12hutongtime",
    #    "9hutongavgcount",
    #    "9hutongtime",
    #    "6hutongavgcount",
    #    "6hutongtime",
    #    "3hutongavgcount",
    #    "3hutongtime",
    #    "12receiveavg",
    #    "9receiveavg",
    #    "6receiveavg",
    #    "3receiveavg",
    #    "12sendavg",
    #    "9sendavg",
    #    "6sendavg",
    #    "3sendavg",
    #    "12avgflow",
    #    "9avgflow",
    #    "6avgflow",
    #    "3avgflow",
    #    "12avgnettime",
    #    "9avgnettime",
    #    "6avgnettime",
    #    "3avgnettime",
    #    "12contactavgcount",
    #    "12contacttime",
    #    "9contactavgcount",
    #    "9contacttime",
    #    "6contactavgcount",
    #    "6contacttime",
    #    "3contactavgcount",
    #    "3contacttime",
    #    "12zhuplace",
    #    "9zhuplace",
    #    "6zhuplace",
    #    "3zhuplace",
    #    #"user_own_overdue", #y, target
    #    #"user_own_overdue_yet",
    #    #"user_own_fpd_overdue_order",
    #    #"user_own_ninety_overdue_order",
    #    #"user_own_sixty_overdue_order",
    #    #"user_own_thirty_overdue_order",
    #    #"user_own_ninety_overdue_num",
    #    #"user_own_sixty_overdue_num",
    #    #"user_own_thirty_overdue_num",
    #    #"user_credit_ninety_overdue",
    #    "1zhuplace"
    #]
    #target_key="user_own_overdue"

    attributes=[
        #"create_date",
        #"user_name",
        #"user_phone",
        "user_age",
        "user_sex",
        #"user_id_card",
        "user_live_province",
        "user_live_city",
        "user_live_address",
        #"user_regi_address",
        #"user_mailbox",
        "user_marriage",
        "user_rela_name",
        "user_relation",
        "user_rela_phone",
        "user_high_edu",
        #"user_indu_type",
        "user_company_name",
        #"user_company_phone",
        #"user_work_time",
        #"user_work_phone",
        "user_income_range",
        "user_last_consume",
        "user_ave_six_consume",
        "user_ave_twelve_consume",
        "user_house_mortgage",
        "user_car_mortgage",
        "user_base_fund",
        "user_credit_limit",
        "user_other_overdue",
        #"user_own_overdue", #y, target
        #"user_other_overdue_yet",
        "user_own_overdue_yet",
        "user_own_fpd_overdue_order",
        #"user_own_ninety_overdue_order", #optional y, target
        "user_own_sixty_overdue_order",
        "user_own_thirty_overdue_order",
        "user_own_ninety_overdue_num",
        "user_own_sixty_overdue_num",
        "user_own_thirty_overdue_num",
        "user_credit_ninety_overdue",
        "user_loan_pass",
        "user_loan_amount",
        "user_four_ident",
        "user_face_ident",
        "user_base_fund_ident",
        "user_center_ident",
        "user_card_ident",
        "user_loan_ident"
    ]
    target_key="user_own_overdue"
    #target_key="user_own_ninety_overdue_order"
    RANDOM_SEED = 99

    ##############################################################################################################
    # Way two, cross-validation, using KFold spliting the source data set into train and test, repeat k times, the default evaluation
    train_df = pandas.read_csv(train_fullpath,sep=',',na_values='NA',low_memory=False)
    #for item in train_df.columns.values:
    #    pandas.to_numeric(train_df[item])
    X_train = train_df[attributes]
    y_train = train_df[target_key]
    train_datapreprocessing = DataPreprocessing(pandas.concat([X_train,y_train],axis=1),attributes,target_key)
    #train_datapreprocessing.data_summary()
    binary_transform_attrs = ['user_live_address','user_rela_name','user_relation','user_rela_phone','user_high_edu','user_company_name']
    X_train = train_datapreprocessing.transform_x_to_binary(binary_transform_attrs)
    X_train = train_datapreprocessing.transform_x_dtype(binary_transform_attrs,d_type=[int],uniform_type=True)
    area_attrs = ['user_live_province','user_live_city']
    resource_dir = '../resources'
    X_train = train_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
    X_train = train_datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
    X_train = train_datapreprocessing.x_dummies_and_fillna()
    train_datapreprocessing.data_summary()

    #Gini_DF = pandas.concat([X_train,y_train],axis=1)
    ##gini_attrs = Gini_DF.axes[1]
    #gini_attrs = list(X_train.columns.values)
    #gini = GiniIndex(Gini_DF,gini_attrs,target_key,Gini_DF[target_key])
    #gini_index_dict = gini.gini_index()
    #gini_list = sorted(gini_index_dict.items(),key=lambda item:item[1])
    #new_attributes = []
    #new_attribues_num = X_train.columns.values
    #i = 0
    #for item in gini_list:
    #    #print(type(item))
    #    #print(item)
    #    if i < new_attribues_num:
    #        new_attributes.append(str(item[0]))
    #    i = i+1
    #X_train = X_train[new_attributes]

    # Begin: smote
    #new_train_df = pandas.concat([X_train,y_train],axis=1)
    #smote_processor = Smote(new_train_df[new_train_df[target_key]==1],N=400,k=5)
    #train_df_sample = smote_processor.over_sampling()
    ##X_sample,y_sample = smote_processor.over_sampling()
    #sample = DataFrame(train_df_sample,columns=new_train_df.columns.values)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pandas.concat([X_train,sample[X_train.columns.values]],axis=0)
    #y_train = pandas.concat([y_train.to_frame().rename(columns={0:target_key}),sample[target_key].to_frame().rename(columns={0:target_key})],axis=0)[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pandas.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    # End: smote
    ## Begin: nearmiss
    #nearmiss_processor = NearMiss(random_state=RANDOM_SEED,n_neighbors=5)
    #X_sample,y_sample = nearmiss_processor.sample(X_train.as_matrix(),y_train.as_matrix())
    #sample = pandas.concat([DataFrame(X_sample,columns=X_train.columns.values),Series(y_sample).to_frame().rename(columns={0:target_key})],axis=1)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pandas.concat([X_train,DataFrame(X_sample,columns=X_train.columns.values)])
    #y_train = pandas.concat([y_train.to_frame(),sample[target_key].to_frame()])[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pandas.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    ## End: nearmiss
    ## Begin: smote1
    #smote_processor = Smote(random_seed=RANDOM_SEED,n_neighbors=5,m_neighbors=5)
    #X_sample,y_sample = smote_processor.sample(X_train.as_matrix(),y_train.as_matrix())
    #sample = pandas.concat([DataFrame(X_sample,columns=X_train.columns.values),Series(y_sample).to_frame().rename(columns={0:target_key})],axis=1)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pandas.concat([X_train,DataFrame(X_sample,columns=X_train.columns.values)])
    #y_train = pandas.concat([y_train.to_frame(),sample[target_key].to_frame()])[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pandas.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    ## End: smote1
    ## Begin: ensemble
    #ensemble_processor = Ensemble(random_seed=RANDOM_SEED,n_subset=10,n_tree=12)
    #X_sample,y_sample = ensemble_processor.sample(X_train.as_matrix(),y_train.as_matrix())
    #sample = pandas.concat([DataFrame(X_sample,columns=X_train.columns.values),Series(y_sample).to_frame().rename(columns={0:target_key})],axis=1)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pandas.concat([X_train,DataFrame(X_sample,columns=X_train.columns.values)])
    #y_train = pandas.concat([y_train.to_frame(),sample[target_key].to_frame()])[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pandas.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    ## End: ensemble

    #X_train.describe()
    #X_train.info()
    #print(X_train.head(5))

    #-----------------------------Find the best parameters' combination of the model------------------------------
    tuned_parameters = [{'C': [0.001,0.01,0.1,1,10,100,1000]}]
    gsearch1 = GridSearchCV(LogisticRegression(),tuned_parameters,cv=5)
    gsearch1.fit(X_train, y_train)
    print(gsearch1.best_estimator_)
    for item in gsearch1.grid_scores_:
        print(item)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    #-----------------------------Find the best parameters' combination of the model------------------------------
#    model = LogisticRegression(
#        penalty='l2',
#        dual=False,
#        tol=0.0001,
#        C=1.0,
#        fit_intercept=True,
#        intercept_scaling=1,
#        class_weight={0:0.2,1:0.8},
#        random_state=RANDOM_SEED,
#        solver='liblinear',
#        max_iter=100,
#        multi_class='ovr',
#        verbose=0,
#        warm_start=False,
#        n_jobs=1
#    )
#
#    kfold = model_selection.KFold(n_splits=5,random_state=RANDOM_SEED)
#    eval_standard = ['accuracy','recall_macro','precision_macro','f1_macro']
#    results = []
#    for scoring in eval_standard:
#        cv_results = model_selection.cross_val_score(model,X_train,y_train,scoring=scoring,cv=kfold)
#        results.append(cv_results)
#        msg = "%s: %f (%f)" % (scoring,cv_results.mean(),cv_results.std())
#        print(msg)
#    # Make predictions on validation dataset
#    test_df = pandas.read_csv(test_fullpath,sep=',',na_values='NA',low_memory=False)
#    #for item in test_df.columns.values:
#    #    pandas.to_numeric(test_df[item])
#    X_validation = test_df[attributes]
#    y_validation = test_df[target_key]
#    validation_datapreprocessing = DataPreprocessing(pandas.concat([X_validation,y_validation],axis=1),attributes,target_key)
#    #validation_datapreprocessing.data_summary()
#    X_validation = validation_datapreprocessing.transform_x_to_binary(binary_transform_attrs)
#    X_validation = validation_datapreprocessing.transform_x_dtype(binary_transform_attrs,d_type=[int],uniform_type=True)
#    X_validation = validation_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
#    X_validation = validation_datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
#    X_validation = validation_datapreprocessing.x_dummies_and_fillna(allnull=True,nullvalue=random.randint(0,2))
#    #validation_datapreprocessing.data_summary()
#    model.fit(X_train,y_train)
#    #default evaluation way
#    print('-------------------default evaluation----------------------')
#    #X_validation = X_validation[new_attributes]
#    #rf_pred_probs = model.predict_proba(X=X_validation)
#    rf_pred_probs = model.predict(X=X_validation)
#    result_probs = numpy.column_stack((rf_pred_probs,y_validation.as_matrix()))
#    #for item in result_probs:
#    #    print(item)
#    print(metrics.confusion_matrix(y_validation, rf_pred_probs))
#    print(metrics.accuracy_score(y_validation, rf_pred_probs))
#    print(metrics.precision_score(y_validation, rf_pred_probs))
#    print(metrics.f1_score(y_validation, rf_pred_probs))
#    print(metrics.classification_report(y_validation, rf_pred_probs))
#
#    #self-defined evaluation way
#    print('-------------------self-defined evaluation----------------------')
#    low_prob = 1e-6
#    high_prob = 1 - low_prob
#    log_low_prob = numpy.log(low_prob)
#    g_low_prob = numpy.log(low_prob)
#    log_high_prob = numpy.log(high_prob)
#    log_prob_thresholds = numpy.linspace(start=log_low_prob,stop=log_high_prob,num=100)
#    prob_thresholds = numpy.exp(log_prob_thresholds)
#    rf_pred_probs = model.predict_proba(X=X_validation)
#    #result_probs = numpy.column_stack((rf_pred_probs,y_validation))
#    #for item in result_probs:
#    #    print(item)
#    #for item in rf_pred_probs[:,1]:
#    #    print(item)
#
#    ## histogram of predicted probabilities
#    ##n,bins,patches = plt.hist(rf_pred_probs[:1],10,normed=1,facecolor='g',alpha=0.75)
#    ##plt.xlabel('Predicted probability of diabetes')
#    ##plt.ylabel('Frequency')
#    ##plt.title('Histogram of predicted probabilities')
#    ###plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#    ##plt.axis([0,1,0,1])
#    ##plt.grid(True)
#
#    #print(type(rf_pred_probs))
#    #print(type(rf_pred_probs[:,1]))
#    #print(rf_pred_probs[:,1])
#    #fig = plt.figure()
#    #ax = fig.add_subplot(111)
#    #ax.hist(rf_pred_probs[:,1], bins=20)
#    #plt.xlim(0,1)
#    #plt.title('Histogram of predicted probabilities')
#    #plt.xlabel('Predicted probability of diabetes')
#    #plt.ylabel('Frequency')
#    #plt.show()
#
#    model_oos_performance = bin_classif_eval(rf_pred_probs[:,1],y_validation,pos_cat=1,thresholds=prob_thresholds)
#    #print(type(model_oos_performance))
#    #for item in model_oos_performance.recall:
#    #    print(item)
#    recall_threshold = .74
#    idx = next(i for i in range(100) if model_oos_performance.recall[i] <= recall_threshold) - 1
#    print("idx = %d" % idx)
#    selected_prob_threshold = prob_thresholds[idx]
#    print("selected_prob_threshold:", selected_prob_threshold)
#    print(model_oos_performance.iloc[idx,:])