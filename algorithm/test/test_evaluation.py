#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas
import numpy
from preprocessing import DataPreprocessing
from sklearn import model_selection
from gini_index import GiniIndex
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ChicagoBoothML_Helpy.EvaluationMetrics import bin_classif_eval

if __name__ == "__main__":
    #pprint.pprint(sys.path)
    #file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_app_hl_calls_stg1/app_hl_stg1.csv'
    #file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/md_data_stg2.csv'
    file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/md_data_stg2_tmp.csv'
    df = pandas.read_csv(file_fullpath,sep=',',na_values='NA',low_memory=False)
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
        "user_own_ninety_overdue_order",
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
    #print(df['user_income_range'].head(20))
    #print(df['user_last_consume'].head(20))
    df.convert_objects(convert_numeric=True)
    #df['user_last_consume'].str.replace('','')
    #df.info()

    validation_size = 0.20
    seed = 7
    X = df[attributes]
    Y = df[target_key]
    X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
    X_train_datapreprocessing = DataPreprocessing(X_train,attributes,target_key)
    binary_transform_attrs = ['user_live_address','user_rela_name','user_relation','user_rela_phone','user_high_edu','user_company_name']
    X_train = X_train_datapreprocessing.transform_to_binary(binary_transform_attrs)
    X_train = X_train_datapreprocessing.transform_dtype(binary_transform_attrs,d_type=[int],uniform_type=True)
    area_attrs = ['user_live_province','user_live_city']
    resource_dir = '../resources'
    X_train = X_train_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
    X_train = X_train_datapreprocessing.transform_dtype(area_attrs,d_type=[int],uniform_type=True)
    X_train = X_train_datapreprocessing.dummies_and_fillna()
    #X_train.info()
    #print(X_train.head(5))

    #Gini_DF = pandas.concat([X_train,Y_train],axis=1)
    ##gini_attrs = Gini_DF.axes[1]
    #gini_attrs = list(Gini_DF.columns.values)
    #gini = GiniIndex(Gini_DF,gini_attrs,target_key,Gini_DF[target_key])
    #gini_index_dict = gini.gini_index()
    #gini_list = sorted(gini_index_dict.items(),key=lambda item:item[1])
    #for item in gini_list:
    #    print(item)

    scoring = 'accuracy'
    models = []
    models.append(('LR',LogisticRegression()))
    models.append(('CART',DecisionTreeClassifier()))
    #models.append(('LDA',LinearDiscriminantAnalysis()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('NB',GaussianNB()))
    models.append(('RF',RandomForestClassifier()))
    #models.append(('SVM',SVC()))

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
    #lr = LogisticRegression()
    #lr.fit(X_train, Y_train)
    #X_validation_datapreprocessing = DataPreprocessing(X_validation,attributes,target_key)
    #X_validation = X_validation_datapreprocessing.data_pre_process()
    #Y_validation = Y_validation.fillna(value=-1)
    #predictions = lr.predict(X_validation)
    #print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))
    for name,model in models:
        model.fit(X_train,Y_train)
        X_validation_datapreprocessing = DataPreprocessing(X_validation,attributes,target_key)
        X_validation = X_validation_datapreprocessing.transform_to_binary(binary_transform_attrs)
        X_validation = X_validation_datapreprocessing.transform_dtype(binary_transform_attrs,d_type=[int],uniform_type=True)
        X_validation = X_validation_datapreprocessing.china_area_number_mapping(area_attrs,resource_dir)
        X_validation = X_validation_datapreprocessing.transform_dtype(area_attrs,d_type=[int],uniform_type=True)
        X_validation = X_validation_datapreprocessing.dummies_and_fillna()
        #Y_validation = Y_validation.fillna(value=-1)
        #Y_validation = Y_validation.fillna(value=random.randint(0,1))

        #predictions = model.predict(X_validation)
        low_prob = 1e-6
        high_prob = 1 - low_prob
        log_low_prob = numpy.log(low_prob)
        g_low_prob = numpy.log(low_prob)
        log_high_prob = numpy.log(high_prob)
        log_prob_thresholds = numpy.linspace(start=log_low_prob,stop=log_high_prob,num=100)
        prob_thresholds = numpy.exp(log_prob_thresholds)
        pred_probs = model.predict_proba(X=X_validation)
        model_oos_performance = bin_classif_eval(pred_probs[:,1],Y_validation,pos_cat=1,thresholds=prob_thresholds)
        recall_threshold = .75
        # print(type(model_oos_performance.recall))
        # print(model_oos_performance.recall)
        idx = next(i for i in range(100) if model_oos_performance.recall[i] <= recall_threshold) - 1
        print("idx = %d" % idx)
        selected_prob_threshold = prob_thresholds[idx]
        print("selected_prob_threshold:", selected_prob_threshold)
        print(model_oos_performance.iloc[idx, :])
        #print(accuracy_score(Y_validation, predictions))
        #print(confusion_matrix(Y_validation, predictions))
        #print(classification_report(Y_validation, predictions))
