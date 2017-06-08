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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from ChicagoBoothML_Helpy.EvaluationMetrics import bin_classif_eval
from nearmiss1 import NearMiss
#from smote1 import Smote
from smote import Smote
from ensemble import Ensemble

if __name__ == "__main__":
    pandas.set_option('display.max_rows', None)
    #pprint.pprint(sys.path)
    train_fullpath = '/home/login01/Workspaces/python/dataset/qiubin/stg2_checked_dummied_train.csv'
    test_fullpath = '/home/login01/Workspaces/python/dataset/qiubin/stg2_checked_dummied_test.csv'
    attributes=[
        'user_phone',
        'user_age',
        'user_mailbox',
        'user_rela_name',
        'user_rela_phone',
        'user_high_edu',
        'user_company_name',
        'user_income_range',
        'user_last_consume',
        'user_ave_six_consume',
        'user_ave_twelve_consume',
        'user_house_mortgage',
        'user_car_mortgage',
        'user_base_fund',
        'user_credit_limit',
        'user_other_overdue',
        #'user_own_overdue', # y
        'user_own_overdue_yet',
        'user_own_ninety_overdue_order',
        'user_own_sixty_overdue_order',
        'user_own_thirty_overdue_order',
        'user_own_ninety_overdue_num',
        'user_own_sixty_overdue_num',
        'user_own_thirty_overdue_num',
        'user_credit_ninety_overdue',
        'user_loan_pass',
        'user_loan_amount',
        'user_four_ident',
        'user_face_ident',
        'user_base_fund_ident',
        'user_center_ident',
        'user_card_ident',
        'user_loan_ident',
        'city_False',
        'user_live_province_上海市',
        'user_live_province_云南省',
        'user_live_province_内蒙古自治区',
        'user_live_province_北京市',
        'user_live_province_吉林省',
        'user_live_province_四川省',
        'user_live_province_天津市',
        'user_live_province_宁夏自治区',
        'user_live_province_安徽省',
        'user_live_province_山东省',
        'user_live_province_山西省',
        'user_live_province_广东省',
        'user_live_province_广西自治区',
        'user_live_province_新疆自治区',
        'user_live_province_江苏省',
        'user_live_province_江西省',
        'user_live_province_河北省',
        'user_live_province_河南省',
        'user_live_province_浙江省',
        'user_live_province_海南省',
        'user_live_province_湖北省',
        'user_live_province_湖南省',
        'user_live_province_甘肃省',
        'user_live_province_福建省',
        'user_live_province_西藏自治区',
        'user_live_province_贵州省',
        'user_live_province_辽宁省',
        'user_live_province_重庆市',
        'user_live_province_陕西省',
        'user_live_province_青海省',
        'user_live_province_黑龙江省',
        'user_live_province_nan',
        'user_marriage_0',
        'user_marriage_1',
        'user_marriage_2',
        'user_marriage_nan',
        'user_relation_同事',
        'user_relation_同学',
        'user_relation_子女',
        'user_relation_朋友',
        'user_relation_父母',
        'user_relation_配偶',
        'user_relation_nan',
        'user_indu_type_住宿及旅游',
        'user_indu_type_其他',
        'user_indu_type_制造业',
        'user_indu_type_建筑工程',
        'user_indu_type_政府及事业单位',
        'user_indu_type_物流运输',
        'user_indu_type_软件及咨询服务',
        'user_indu_type_金融业',
        'user_indu_type_零售业',
        'user_indu_type_餐饮业',
        'user_indu_type_nan',
        'user_sex_女',
        'user_sex_男'
    ]
    target_key="user_own_overdue"
    #target_key="user_own_ninety_overdue_order"
    RANDOM_SEED = 99

    ##############################################################################################################
    ## Way one, using train_test_split spliting the source data set into train and test
    #df = pandas.read_csv(file_fullpath,sep=',',na_values='NA',low_memory=False)
    #df.convert_objects(convert_numeric=True)
    #X = df[attributes]
    #y = df[target_key]
    #validation_size = 0.20
    #X_train,X_validation,y_train,y_validation = model_selection.train_test_split(X,y,test_size=validation_size,random_state=RANDOM_SEED)
    ##X_train.info()
    ##print(X_train.head(5))

    ##Gini_DF = pandas.concat([X_train,y_train],axis=1)
    ###gini_attrs = Gini_DF.axes[1]
    ##gini_attrs = list(Gini_DF.columns.values)
    ##gini = GiniIndex(Gini_DF,gini_attrs,target_key,Gini_DF[target_key])
    ##gini_index_dict = gini.gini_index()
    ##gini_list = sorted(gini_index_dict.items(),key=lambda item:item[1])
    ##for item in gini_list:
    ##    print(item)

    #B = 400
    #rf_model = \
    #    RandomForestClassifier(
    #        n_estimators=B,
    #        criterion='entropy',
    #        max_depth=None,  # expand until all leaves are pure or contain < MIN_SAMPLES_SPLIT samples
    #        min_samples_split=200,
    #        min_samples_leaf=100,
    #        min_weight_fraction_leaf=0.0,
    #        max_features=None,
    #        # number of features to consider when looking for the best split; None: max_features=n_features
    #        max_leaf_nodes=None,  # None: unlimited number of leaf nodes
    #        bootstrap=True,
    #        oob_score=True,  # estimate Out-of-Bag Cross Entropy
    #        n_jobs=multiprocessing.cpu_count() - 4,  # paralellize over all CPU cores but 2
    #        class_weight=None,  # our classes are skewed, but but too skewed
    #        random_state=RANDOM_SEED,
    #        verbose=0,
    #        warm_start=False)
    #rf_model.fit(
    #    X=X_train,
    #    y=y_train)
    #rf_pred_probs = rf_model.predict_proba(X=X_train)
    ##rf_pred_probs = rf_model.predict_log_proba(X=X_train)
    ##result_probs = numpy.hstack((rf_pred_probs,y_train.as_matrix()))
    #result_probs = numpy.column_stack((rf_pred_probs,y_train.as_matrix()))
    ##for item in result_probs:
    ##    print(item)
    #print(metrics.confusion_matrix(y_validation, rf_pred_probs))
    #print(metrics.accuracy_score(y_validation, rf_pred_probs))
    #print(metrics.precision_score(y_validation, rf_pred_probs))
    #print(metrics.f1_score(y_validation, rf_pred_probs))
    #print(metrics.classification_report(y_validation, rf_pred_probs))

    ##############################################################################################################
    # Way two, cross-validation, using KFold spliting the source data set into train and test, repeat k times, the default evaluation
    train_df = pandas.read_csv(train_fullpath,sep=',',na_values='NA',low_memory=False)
    train_df.info()
    #for item in train_df.columns.values:
    #    pandas.to_numeric(train_df[item])
    X_train = train_df[attributes]
    y_train = train_df[target_key]

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
    ##print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch2.best_score_,'\n')
    #print('-----------------------------------------------------------------------------------------------------')
    #param_test2 = {'max_depth': range(2, 16, 2), 'min_samples_split': range(20, 200, 20)}
    #gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=300,
    #                                                         min_samples_leaf=2, max_features='sqrt', oob_score=True,
    #                                                         random_state=19),
    #                        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    #gsearch2.fit(X_train,y_train)
    #for item in gsearch2.grid_scores_:
    #    print(item)
    #print(gsearch2.best_params_)
    #print(gsearch2.best_score_)
    #print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_,'\n')
    ##-----------------------------Find the best parameters' combination of the model------------------------------

    B = 300
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
    test_df = pandas.read_csv(test_fullpath,sep=',',na_values='NA',low_memory=False)
    test_df.info()
    #for item in test_df.columns.values:
    #    pandas.to_numeric(test_df[item])
    X_validation = test_df[attributes]
    y_validation = test_df[target_key]
    model.fit(X_train,y_train)
    print('oob_score: %f' % (model.oob_score_))
    #default evaluation way
    print('-------------------default evaluation----------------------')
    #rf_pred_probs = model.predict_proba(X=X_validation)
    rf_pred_probs = model.predict(X=X_validation)
    result_probs = numpy.column_stack((rf_pred_probs,y_validation.as_matrix()))
    #for item in result_probs:
    #    print(item)
    print(metrics.confusion_matrix(y_validation, rf_pred_probs))
    print(metrics.accuracy_score(y_validation, rf_pred_probs))
    print(metrics.precision_score(y_validation, rf_pred_probs))
    print(metrics.f1_score(y_validation, rf_pred_probs))
    print(metrics.classification_report(y_validation, rf_pred_probs))

    #self-defined evaluation way
    print('-------------------self-defined evaluation----------------------')
    low_prob = 1e-6
    high_prob = 1 - low_prob
    log_low_prob = numpy.log(low_prob)
    g_low_prob = numpy.log(low_prob)
    log_high_prob = numpy.log(high_prob)
    log_prob_thresholds = numpy.linspace(start=log_low_prob,stop=log_high_prob,num=100)
    prob_thresholds = numpy.exp(log_prob_thresholds)
    rf_pred_probs = model.predict_proba(X=X_validation)
    #result_probs = numpy.column_stack((rf_pred_probs,y_validation))
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

    model_oos_performance = bin_classif_eval(rf_pred_probs[:,1],y_validation,pos_cat=1,thresholds=prob_thresholds)
    #print(type(model_oos_performance))
    #for item in model_oos_performance.recall:
    #    print(item)
    recall_threshold = .74
    idx = next(i for i in range(100) if model_oos_performance.recall[i] <= recall_threshold) - 1
    print("idx = %d" % idx)
    selected_prob_threshold = prob_thresholds[idx]
    print("selected_prob_threshold:", selected_prob_threshold)
    print(model_oos_performance.iloc[idx,:])
