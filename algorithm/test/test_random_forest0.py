#!/usr/bin/env python
#-*- coding:utf-8 -*-

import random
import pandas as pd
import pandas as np
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

def data_preprocess(data_path,version=0,file_name=None,sub_directories=None,attributes=None,target_key=None):
    '''
    :param data_path: string, the data's path 
    :param version: int, indicate that what type of data to process:
            0: data isn't split to train and test set, one whole file
            1: data has been split to train and test set, two files
            2: data is split to multiple directories, each directory exists one or multiple files
    :param file_name: string or list of string, 
    :param sub_directories: string or list of string,
    :param attributes: list of string, labels of X
    :param target_key: string, label of target
    :return: 
    '''
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    try:
        if version == 0:
            data_df = pd.read_csv(data_path,sep=',',na_values='NA',low_memory=False)
            #for item in data_df.columns.values:
            #    pd.to_numeric(df[item])
            if attributes is None:
                attributes = list(data_df.columns.values)
            X = data_df[attributes]
            y = data_df[target_key]
            datapreprocessing = DataPreprocessing(pd.concat([X,y],axis=1),attributes,target_key)
            #datapreprocessing.data_summary()
            binary_transform_attrs = ['user_live_address','user_rela_name','user_relation','user_rela_phone','user_high_edu','user_company_name']
            X = datapreprocessing.transform_x_to_binary(binary_transform_attrs)
            X = datapreprocessing.transform_x_dtype(binary_transform_attrs,d_type=[int],uniform_type=True)
            area_attrs = ['user_live_province','user_live_city']
            resource_dir = '../resources'
            X = datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
            X = datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
            X = datapreprocessing.x_dummies_and_fillna()
            #datapreprocessing.data_summary()

    except Exception as e:
        print(e)
    finally:
        return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    #pprint.pprint(sys.path)
    train_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/tmp_train.csv'
    test_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/tmp_test.csv'
    #target_key="user_own_overdue"

    attributes=[]
    target_key="user_own_overdue"
    #target_key="user_own_ninety_overdue_order"
    RANDOM_SEED = 99

    #Gini_DF = pd.concat([X_train,y_train],axis=1)
    ##gini_attrs = Gini_DF.axes[1]
    #gini_attrs = list(X_train.columns.values)
    #gini = GiniIndex(Gini_DF,gini_attrs,target_key,Gini_DF[target_key])
    #gini_index_dict = gini.gini_index()
    #gini_list = sorted(gini_index_dict.items(),key=lambda item:item[1])
    #new_attributes = []
    #new_attribues_num = len(X_train.columns.values)
    #i = 0
    #for item in gini_list:
    #    #print(type(item))
    #    #print(item)
    #    if i < new_attribues_num:
    #        new_attributes.append(str(item[0]))
    #    i = i+1
    #X_train = X_train[new_attributes]

    # Begin: smote
    #new_train_df = pd.concat([X_train,y_train],axis=1)
    #smote_processor = Smote(new_train_df[new_train_df[target_key]==1],N=400,k=5)
    #train_df_sample = smote_processor.over_sampling()
    ##X_sample,y_sample = smote_processor.over_sampling()
    #sample = DataFrame(train_df_sample,columns=new_train_df.columns.values)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pd.concat([X_train,sample[X_train.columns.values]],axis=0)
    #y_train = pd.concat([y_train.to_frame().rename(columns={0:target_key}),sample[target_key].to_frame().rename(columns={0:target_key})],axis=0)[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pd.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    # End: smote
    ## Begin: nearmiss
    #nearmiss_processor = NearMiss(random_state=RANDOM_SEED,n_neighbors=5)
    #X_sample,y_sample = nearmiss_processor.sample(X_train.as_matrix(),y_train.as_matrix())
    #sample = pd.concat([DataFrame(X_sample,columns=X_train.columns.values),Series(y_sample).to_frame().rename(columns={0:target_key})],axis=1)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pd.concat([X_train,DataFrame(X_sample,columns=X_train.columns.values)])
    #y_train = pd.concat([y_train.to_frame(),sample[target_key].to_frame()])[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pd.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    ## End: nearmiss
    ## Begin: smote1
    #smote_processor = Smote(random_seed=RANDOM_SEED,n_neighbors=5,m_neighbors=5)
    #X_sample,y_sample = smote_processor.sample(X_train.as_matrix(),y_train.as_matrix())
    #sample = pd.concat([DataFrame(X_sample,columns=X_train.columns.values),Series(y_sample).to_frame().rename(columns={0:target_key})],axis=1)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pd.concat([X_train,DataFrame(X_sample,columns=X_train.columns.values)])
    #y_train = pd.concat([y_train.to_frame(),sample[target_key].to_frame()])[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pd.concat([X_train,y_train],axis=1),attributes,target_key)
    #merged_train_datapreprocessing.data_summary()
    ## End: smote1
    ## Begin: ensemble
    #ensemble_processor = Ensemble(random_seed=RANDOM_SEED,n_subset=10,n_tree=12)
    #X_sample,y_sample = ensemble_processor.sample(X_train.as_matrix(),y_train.as_matrix())
    #sample = pd.concat([DataFrame(X_sample,columns=X_train.columns.values),Series(y_sample).to_frame().rename(columns={0:target_key})],axis=1)
    ##sample_datapreprocessing = DataPreprocessing(sample,sample.drop(target_key,axis=1,inplace=False).columns.values,target_key)
    ##sample_datapreprocessing.data_summary()
    #X_train = pd.concat([X_train,DataFrame(X_sample,columns=X_train.columns.values)])
    #y_train = pd.concat([y_train.to_frame(),sample[target_key].to_frame()])[target_key]
    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)
    #merged_train_datapreprocessing = DataPreprocessing(pd.concat([X_train,y_train],axis=1),attributes,target_key)
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
    #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch2.best_score_,'\n')
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
    test_df = pd.read_csv(test_fullpath,sep=',',na_values='NA',low_memory=False)
    #for item in test_df.columns.values:
    #    pd.to_numeric(test_df[item])
    X_validation = test_df[attributes]
    y_validation = test_df[target_key]
    validation_datapreprocessing = DataPreprocessing(pd.concat([X_validation,y_validation],axis=1),attributes,target_key)
    #validation_datapreprocessing.data_summary()
    X_validation = validation_datapreprocessing.transform_x_to_binary(binary_transform_attrs)
    X_validation = validation_datapreprocessing.transform_x_dtype(binary_transform_attrs,d_type=[int],uniform_type=True)
    X_validation = validation_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
    X_validation = validation_datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
    X_validation = validation_datapreprocessing.x_dummies_and_fillna(allnull=True,nullvalue=random.randint(0,2))
    #validation_datapreprocessing.data_summary()
    model.fit(X_train,y_train)
    print('oob_score: %f' % (model.oob_score_))
    #default evaluation way
    print('-------------------default evaluation----------------------')
    #X_validation = X_validation[new_attributes]
    rf_pred_probs = model.predict(X=X_validation)
    result_probs = np.column_stack((rf_pred_probs,y_validation.as_matrix()))
    #for item in result_probs:
    #  print(item)
    print("confusion_matrix:\n",metrics.confusion_matrix(y_validation, rf_pred_probs))
    print("accuracy_score:",metrics.accuracy_score(y_validation, rf_pred_probs))
    print("recall_score:",metrics.recall_score(y_validation, rf_pred_probs))
    print("precision_score:",metrics.precision_score(y_validation, rf_pred_probs))
    print("f1_score:",metrics.f1_score(y_validation, rf_pred_probs))
    print("roc_auc_score:",metrics.roc_auc_score(y_validation, rf_pred_probs))
    print("classification_report:\n",metrics.classification_report(y_validation, rf_pred_probs))

    rf_pred_probs = model.predict_proba(X=X_validation)
    result_probs = np.column_stack((rf_pred_probs,y_validation.as_matrix()))
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
    print("ks:",max(tpr-fpr))

    #self-defined evaluation way
    print('-------------------self-defined evaluation----------------------')
    low_prob = 1e-6
    high_prob = 1 - low_prob
    log_low_prob = np.log(low_prob)
    g_low_prob = np.log(low_prob)
    log_high_prob = np.log(high_prob)
    log_prob_thresholds = np.linspace(start=log_low_prob,stop=log_high_prob,num=100)
    prob_thresholds = np.exp(log_prob_thresholds)
    rf_pred_probs = model.predict_proba(X=X_validation)
    #result_probs = np.column_stack((rf_pred_probs,y_validation))
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
