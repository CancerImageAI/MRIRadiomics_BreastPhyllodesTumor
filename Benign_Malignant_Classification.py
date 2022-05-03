# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:46:44 2021

@author: Administrator
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from skfeature.function.similarity_based import reliefF
from sklearn import svm, linear_model
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix,f1_score,matthews_corrcoef
from sklearn.model_selection import LeaveOneOut, KFold
from time import sleep
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold,RFE
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
#from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE, ADASYN
import xlrd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as robj
r = robj.r
from rpy2.robjects.packages import importr
# from xgboost import XGBClassifier
import scipy.stats as stats


def roc_test_r(targets_1, scores_1, targets_2, scores_2, method='delong'):
    # method: “delong”, “bootstrap” or “venkatraman”
    importr('pROC')
    robj.globalenv['targets_1'] = targets_1 = robj.FloatVector(targets_1)
    robj.globalenv['scores_1'] = scores_1 = robj.FloatVector(scores_1)
    robj.globalenv['targets_2'] = targets_2 = robj.FloatVector(targets_2)
    robj.globalenv['scores_2'] = scores_2 = robj.FloatVector(scores_2)

    r('roc_1 <- roc(targets_1, scores_1)')
    r('roc_2 <- roc(targets_2, scores_2)')
    print(r('roc_1'),r('roc_2'))
    r('result = roc.test(roc_1, roc_2, method="%s")' % method)
    p_value = r('p_value = result$p.value')
    return np.array(p_value)[0]

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

def prediction_score(truth, predicted):
    TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    print(TN, FP, FN, TP)
    ACC = (TP+TN)/(TN+FP+FN+TP)
    SEN = TP/(FN+TP)
    SPE = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    print('ACC:',ACC)
    print('Sensitivity:',SEN)
    print('Specifity:',SPE)
    print('PPV:',PPV)
    print('NPV:',NPV)
    OR = (TP*TN)/(FP*FN)
    print('OR:',OR)
    F1_3 = f1_score(truth, predicted)
    print('F1:', F1_3)
    F1_w3 = f1_score(truth, predicted,average='weighted')
    print('F1_weight:',F1_w3)
    MCC3 = matthews_corrcoef(truth, predicted)
    print('MCC:',MCC3)
    
if __name__ == '__main__':
    random.seed(0)
    ## T1-ZQ
    ZQ_File = open('../result/Train_ZQ_Feature.csv')
    ZQ_List = pd.read_csv(ZQ_File)
    ZQ_List = ZQ_List.fillna('0')
    ZQ_Feature = np.array(ZQ_List.values[:,:-2])
    ZQ_Feature_Name = list(ZQ_List.head(0))[:-2]
    ZQ_Feature_Name = ['T1WI-PC-'+i for i in ZQ_Feature_Name]
    idx = np.arange(0, ZQ_Feature.shape[1]) 
    ZQ_Class = np.array(ZQ_List['Class'].tolist())
    ZQ_Class = np.array(ZQ_Class>0, dtype=int)
    
    ZQ_File_Test = open('../result/Test_ZQ_Feature.csv')
    ZQ_List_Test = pd.read_csv(ZQ_File_Test)
    ZQ_List_Test = ZQ_List_Test.fillna('0')
    ZQ_Feature_Test = np.array(ZQ_List_Test.values[:,:-2])
    ZQ_Class_Test = np.array(ZQ_List_Test['Class'].tolist())
    ZQ_Class_Test = np.array(ZQ_Class_Test>0, dtype=int)
    
    min_max_scaler = MinMaxScaler()
    
    ZQ_Feature = min_max_scaler.fit_transform(ZQ_Feature)
    ZQ_Feature_Test = min_max_scaler.transform(ZQ_Feature_Test)
    estimator = Ridge(random_state=0)
    model_ZQ = RFE(estimator=estimator, n_features_to_select=10, step=1).fit(ZQ_Feature, ZQ_Class)
    ZQ_index = idx[model_ZQ.get_support() == True]  #get index positions of kept features
    selected_ZQ = np.array(ZQ_Feature_Name)[ZQ_index]
    print(selected_ZQ)
    
    ZQ_train = model_ZQ.transform(ZQ_Feature)
    ZQ_test = model_ZQ.transform(ZQ_Feature_Test)
    x_train_ZQ, y_train_ZQ = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(ZQ_train, ZQ_Class)

    x_test_ZQ = ZQ_test
    y_test_ZQ = ZQ_Class_Test
    clf_ZQ = SVC(kernel="rbf",  probability=True, random_state=0)
    clf_ZQ.fit(x_train_ZQ, y_train_ZQ)
    test_prob_ZQ = clf_ZQ.predict_proba(x_test_ZQ)[:,1]
    pred_label_ZQ = clf_ZQ.predict(x_test_ZQ)

    fpr_ZQ,tpr_ZQ,threshold_ZQ = roc_curve(y_test_ZQ, np.array(test_prob_ZQ)) ###计算真正率和假正率
    auc_ZQ = auc(fpr_ZQ,tpr_ZQ)
    auc_l_ZQ, auc_h_ZQ, auc_std_ZQ = confindence_interval_compute(np.array(test_prob_ZQ), y_test_ZQ)
    print('T1-ZQ Feature AUC:%.2f+/-%.2f'%(auc_ZQ,auc_std_ZQ),'  95%% CI:[%.2f,%.2f]'%(auc_l_ZQ,auc_h_ZQ))
    print('T1-ZQ Feature ACC:%.4f'%accuracy_score(y_test_ZQ,pred_label_ZQ))
    prediction_score(y_test_ZQ,pred_label_ZQ)
    # train_prob_ZQ = clf_ZQ.predict_proba(x_train_ZQ)[:,1]
    # pred_label_ZQ = clf_ZQ.predict(x_train_ZQ)
    
    # fpr_ZQ,tpr_ZQ,threshold_ZQ = roc_curve(y_train_ZQ, np.array(train_prob_ZQ)) ###计算真正率和假正率
    # auc_ZQ = auc(fpr_ZQ,tpr_ZQ)
    # auc_l_ZQ, auc_h_ZQ, auc_std_ZQ = confindence_interval_compute(np.array(train_prob_ZQ), y_train_ZQ)
    # print('T1-ZQ Feature AUC:%.2f+/-%.2f'%(auc_ZQ,auc_std_ZQ),'  95%% CI:[%.2f,%.2f]'%(auc_l_ZQ,auc_h_ZQ))
    # print('T1-ZQ Feature ACC:%.4f'%accuracy_score(y_train_ZQ,pred_label_ZQ))
    # prediction_score(y_train_ZQ,pred_label_ZQ)   



    ## T1-ZQ1
    ZQ1_File = open('../result/Train_ZQ1_Feature.csv')
    ZQ1_List = pd.read_csv(ZQ1_File)
    ZQ1_List = ZQ1_List.fillna('0')
    ZQ1_Feature = np.array(ZQ1_List.values[:,:-2])
    ZQ1_Feature_Name = list(ZQ1_List.head(0))[:-2]
    ZQ1_Feature_Name = ['T1WI-P1-'+i for i in ZQ1_Feature_Name]
    idx1 = np.arange(0, ZQ1_Feature.shape[1]) 
    ZQ1_Class = np.array(ZQ1_List['Class'].tolist())
    ZQ1_Class = np.array(ZQ1_Class>0, dtype=int)
    
    ZQ1_File_Test = open('../result/Test_ZQ1_Feature.csv')
    ZQ1_List_Test = pd.read_csv(ZQ1_File_Test)
    ZQ1_List_Test = ZQ1_List_Test.fillna('0')
    ZQ1_Feature_Test = np.array(ZQ1_List_Test.values[:,:-2])
    ZQ1_Class_Test = np.array(ZQ1_List_Test['Class'].tolist())
    ZQ1_Class_Test = np.array(ZQ1_Class_Test>0, dtype=int)
    
    min_max_scaler = MinMaxScaler()
    ZQ1_Feature = min_max_scaler.fit_transform(ZQ1_Feature)
    ZQ1_Feature_Test = min_max_scaler.transform(ZQ1_Feature_Test)
    estimator = Ridge(random_state=0)
    model_ZQ1 = RFE(estimator=estimator, n_features_to_select=6, step=1).fit(ZQ1_Feature, ZQ1_Class)
    ZQ1_index = idx1[model_ZQ1.get_support() == True]  #get index positions of kept features
    selected_ZQ1 = np.array(ZQ1_Feature_Name)[ZQ1_index]
    print(selected_ZQ1)
    
    ZQ1_train = model_ZQ1.transform(ZQ1_Feature)
    ZQ1_test = model_ZQ1.transform(ZQ1_Feature_Test)
    x_train_ZQ1, y_train_ZQ1 = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(ZQ1_train, ZQ1_Class)
    x_test_ZQ1 = ZQ1_test
    y_test_ZQ1 = ZQ1_Class_Test
    clf_ZQ1 = SVC(kernel="rbf",  probability=True, random_state=0)
    clf_ZQ1.fit(x_train_ZQ1, y_train_ZQ1)
    test_prob_ZQ1 = clf_ZQ1.predict_proba(x_test_ZQ1)[:,1]
    pred_label_ZQ1 = clf_ZQ1.predict(x_test_ZQ1)

    fpr_ZQ1,tpr_ZQ1,threshold_ZQ1 = roc_curve(y_test_ZQ1, np.array(test_prob_ZQ1)) ###计算真正率和假正率
    auc_ZQ1 = auc(fpr_ZQ1,tpr_ZQ1)
    auc_l_ZQ1, auc_h_ZQ1, auc_std_ZQ1 = confindence_interval_compute(np.array(test_prob_ZQ1), y_test_ZQ1)
    print('T1-ZQ1 Feature AUC:%.2f+/-%.2f'%(auc_ZQ1,auc_std_ZQ1),'  95%% CI:[%.2f,%.2f]'%(auc_l_ZQ1,auc_h_ZQ1))
    print('T1-ZQ1 Feature ACC:%.4f'%accuracy_score(y_test_ZQ1,pred_label_ZQ1))
    prediction_score(y_test_ZQ1,pred_label_ZQ1)
    # train_prob_ZQ1 = clf_ZQ1.predict_proba(x_train_ZQ1)[:,1]
    # pred_label_ZQ1 = clf_ZQ1.predict(x_train_ZQ1)

    # fpr_ZQ1,tpr_ZQ1,threshold_ZQ1 = roc_curve(y_train_ZQ1, np.array(train_prob_ZQ1)) ###计算真正率和假正率
    # auc_ZQ1 = auc(fpr_ZQ1,tpr_ZQ1)
    # auc_l_ZQ1, auc_h_ZQ1, auc_std_ZQ1 = confindence_interval_compute(np.array(train_prob_ZQ1), y_train_ZQ1)
    # print('T1-ZQ1 Feature AUC:%.2f+/-%.2f'%(auc_ZQ1,auc_std_ZQ1),'  95%% CI:[%.2f,%.2f]'%(auc_l_ZQ1,auc_h_ZQ1))
    # print('T1-ZQ1 Feature ACC:%.4f'%accuracy_score(y_train_ZQ1,pred_label_ZQ1))
    # # prediction_score(y_train_ZQ1,pred_label_ZQ1)  
    
    #T1 Fusion
    T1_All_Feature = np.hstack((ZQ_train,ZQ1_train))
    T1_Feature_Test = np.hstack((ZQ_test,ZQ1_test))
    T1_All_Feature_Name = np.hstack((selected_ZQ,selected_ZQ1))
    idx_T1 = np.arange(0, T1_All_Feature.shape[1]) 
    estimator = Ridge(random_state=0)
    model_T1 = RFE(estimator=estimator, n_features_to_select=8, step=1).fit(T1_All_Feature, ZQ_Class)
    T1_train = model_T1.transform(T1_All_Feature)
    T1_test = model_T1.transform(T1_Feature_Test)
    T1_index = idx_T1[model_T1.get_support() == True]  #get index positions of kept features
    selected_T1 = np.array(T1_All_Feature_Name)[T1_index]
    print(selected_T1)
    
    x_train_T1, y_train_T1 = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(T1_train, ZQ_Class)
    x_test_T1 = T1_test
    y_test_T1 = ZQ_Class_Test
    clf_T1 = SVC(kernel="rbf",  probability=True, random_state=0)
    # clf_T1 = MLPClassifier(random_state=0,max_iter=1000)
    clf_T1.fit(x_train_T1, y_train_T1)
    test_prob_T1 = clf_T1.predict_proba(x_test_T1)[:,1]
    pred_label_T1 = clf_T1.predict(x_test_T1)
    
    fpr_T1,tpr_T1,threshold_T1 = roc_curve(y_test_T1, np.array(test_prob_T1)) ###计算真正率和假正率
    auc_T1 = auc(fpr_T1,tpr_T1)
    auc_l_T1, auc_h_T1, auc_std_T1 = confindence_interval_compute(np.array(test_prob_T1), y_test_T1)
    print('T1 All Feature AUC:%.2f+/-%.2f'%(auc_T1,auc_std_T1),'  95%% CI:[%.2f,%.2f]'%(auc_l_T1,auc_h_T1))
    print('T1 All Feature ACC:%.4f'%accuracy_score(y_test_T1,pred_label_T1)) 
    prediction_score(y_test_T1,pred_label_T1)
    
    # train_prob_T1 = clf_T1.predict_proba(x_train_T1)[:,1]
    # pred_label_T1 = clf_T1.predict(x_train_T1)
    # fpr_T1,tpr_T1,threshold_T1 = roc_curve(y_train_T1, np.array(train_prob_T1)) ###计算真正率和假正率
    # auc_T1 = auc(fpr_T1,tpr_T1)
    # auc_l_T1, auc_h_T1, auc_std_T1 = confindence_interval_compute(np.array(train_prob_T1), y_train_T1)
    # print('T1 All Feature AUC:%.2f+/-%.2f'%(auc_T1,auc_std_T1),'  95%% CI:[%.2f,%.2f]'%(auc_l_T1,auc_h_T1))
    # print('T1 All Feature ACC:%.4f'%accuracy_score(y_train_T1,pred_label_T1)) 
    # # prediction_score(y_train_T1,pred_label_T1)
    ## T2
    T2_File = open('../result/Train_T2_Feature.csv')
    T2_List = pd.read_csv(T2_File)
    T2_List = T2_List.fillna('0')
    T2_Feature = np.array(T2_List.values[:,:-2])
    T2_Feature_Name = list(T2_List.head(0))[:-2]
    T2_Feature_Name = ['T2WI-'+i for i in T2_Feature_Name]
    idx_T2 = np.arange(0, T2_Feature.shape[1])
    T2_Class = np.array(T2_List['Class'].tolist())
    T2_Class = np.array(T2_Class>0, dtype=int)
    
    T2_File_Test = open('../result/Test_T2_Feature.csv')
    T2_List_Test = pd.read_csv(T2_File_Test)
    T2_List_Test = T2_List_Test.fillna('0')
    T2_Feature_Test = np.array(T2_List_Test.values[:,:-2])
    T2_Class_Test = np.array(T2_List_Test['Class'].tolist())
    T2_Class_Test = np.array(T2_Class_Test>0, dtype=int)
    
    min_max_scaler = MinMaxScaler()
    T2_Feature = min_max_scaler.fit_transform(T2_Feature)
    T2_Feature_Test = min_max_scaler.transform(T2_Feature_Test)
    # for i in range(3,14):
    #     print('T2 Feature Num:',i)
    # estimator = SVC(kernel="linear",random_state=0)
    estimator = Ridge(random_state=0)
    model_T2 = RFE(estimator=estimator, n_features_to_select=13, step=1).fit(T2_Feature, T2_Class)
    T2_train = model_T2.transform(T2_Feature)
    T2_test = model_T2.transform(T2_Feature_Test)
    T2_index = idx_T2[model_T2.get_support() == True]  #get index positions of kept features
    selected_T2 = np.array(T2_Feature_Name)[T2_index]
    print(selected_T2)
    
    x_train_T2, y_train_T2 = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(T2_train, T2_Class)
    x_test_T2 = T2_test
    y_test_T2 = T2_Class_Test
    clf_T2 = SVC(kernel="rbf",  probability=True, random_state=0)
    # clf_T2 = MLPClassifier(200,learning_rate_init=0.005,random_state=0,max_iter=1000)
    clf_T2.fit(x_train_T2, y_train_T2)
    test_prob_T2 = clf_T2.predict_proba(x_test_T2)[:,1]
    pred_label_T2 = clf_T2.predict(x_test_T2)

    fpr_T2,tpr_T2,threshold_T2 = roc_curve(y_test_T2, np.array(test_prob_T2)) ###计算真正率和假正率
    auc_T2 = auc(fpr_T2,tpr_T2)
    auc_l_T2, auc_h_T2, auc_std_T2 = confindence_interval_compute(np.array(test_prob_T2), y_test_T2)
    print('T2 Feature AUC:%.2f+/-%.2f'%(auc_T2,auc_std_T2),'  95%% CI:[%.2f,%.2f]'%(auc_l_T2,auc_h_T2))
    print('T2 Feature ACC:%.4f'%accuracy_score(y_test_T2,pred_label_T2))
    prediction_score(y_test_T2,pred_label_T2)
    
    # train_prob_T2 = clf_T2.predict_proba(x_train_T2)[:,1]
    # pred_label_T2 = clf_T2.predict(x_train_T2)

    # fpr_T2,tpr_T2,threshold_T2 = roc_curve(y_train_T2, np.array(train_prob_T2)) ###计算真正率和假正率
    # auc_T2 = auc(fpr_T2,tpr_T2)
    # auc_l_T2, auc_h_T2, auc_std_T2 = confindence_interval_compute(np.array(train_prob_T2), y_train_T2)
    # print('T2 Feature AUC:%.2f+/-%.2f'%(auc_T2,auc_std_T2),'  95%% CI:[%.2f,%.2f]'%(auc_l_T2,auc_h_T2))
    # print('T2 Feature ACC:%.4f'%accuracy_score(y_train_T2,pred_label_T2))
    # # prediction_score(y_train_T2,pred_label_T2)       
    
    #T1+T2 Fusion
    All_Feature = np.hstack((T1_train,T2_train))#T1_All_Feature,T2_FeatureZQ_train,ZQ1_train,ZQ2_train
    Feature_Test = np.hstack((T1_test,T2_test))#T1_Feature_Test,T2_Feature_TestZQ_test,ZQ1_test,ZQ2_test
    All_Feature_Name = np.hstack((selected_T1,selected_T2))
    idx_All = np.arange(0, All_Feature.shape[1]) 
    estimator = Ridge(random_state=0)
    model = RFE(estimator= estimator, n_features_to_select=8, step=1).fit(All_Feature, ZQ_Class)
    train = model.transform(All_Feature)
    test = model.transform(Feature_Test)
    All_index = idx_All[model.get_support() == True]  #get index positions of kept features
    selected_All = np.array(All_Feature_Name)[All_index]
    print(selected_All)
    
    x_train, y_train = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train, ZQ_Class)
    x_test = test
    y_test = T2_Class_Test
    clf = SVC(kernel="rbf",  probability=True, random_state=0)
    clf.fit(x_train, y_train)
    test_prob = clf.predict_proba(x_test)[:,1]
    pred_label = clf.predict(x_test)
    
    fpr,tpr,threshold = roc_curve(y_test, np.array(test_prob)) ###计算真正率和假正率
    auc_all = auc(fpr,tpr)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob), y_test)
    print('All Image Feature AUC:%.2f+/-%.2f'%(auc_all,auc_std),'  95%% CI:[%.2f,%.2f]'%(auc_l,auc_h))
    print('All Image Feature ACC:%.4f'%accuracy_score(y_test,pred_label)) 
    prediction_score(y_test,pred_label)
    
    print('All VS T1 P-Value:',roc_test_r(y_test, np.array(test_prob), y_test_T1, np.array(test_prob_T1)))
    print('All VS T2 P-Value:',roc_test_r(y_test, np.array(test_prob), y_test_T2, np.array(test_prob_T2)))
    print('T1 VS T2 P-Value:',roc_test_r(y_test_T2, np.array(test_prob_T2), y_test_T1, np.array(test_prob_T1)))
    # train_prob = clf.predict_proba(x_train)[:,1]
    # pred_label = clf.predict(x_train)
    
    # fpr,tpr,threshold = roc_curve(y_train, np.array(train_prob)) ###计算真正率和假正率
    # auc_all = auc(fpr,tpr)
    # auc_l, auc_h, auc_std = confindence_interval_compute(np.array(train_prob), y_train)
    # print('All Image Feature AUC:%.2f+/-%.2f'%(auc_all,auc_std),'  95%% CI:[%.2f,%.2f]'%(auc_l,auc_h))
    # print('All Image Feature ACC:%.4f'%accuracy_score(y_train,train_prob>0.5)) 
    # # prediction_score(y_train,train_prob>0.5)    
    
    # Clinical
    traindata_path = '../TrainingList.xls'
    train_data = xlrd.open_workbook(traindata_path)
    train_table = train_data.sheets()[0]  
    train_clinical_feature = []
    for i in range(3,10):
        train_clinical_feature.append(train_table.col_values(i)[1:])
    train_clinical_feature = np.array(train_clinical_feature).transpose(1,0)
    train_MR_Findings = []
    for i in range(10,22):
        train_MR_Findings.append(train_table.col_values(i)[1:])
    train_MR_Findings = np.array(train_MR_Findings).transpose(1,0)
    train_MR_Findings[train_MR_Findings=='4A'] = 4.25
    train_MR_Findings[train_MR_Findings=='4a'] = 4.25
    train_MR_Findings[train_MR_Findings=='4B'] = 4.5
    train_MR_Findings[train_MR_Findings=='4b'] = 4.5
    train_MR_Findings[train_MR_Findings=='4C'] = 4.75
    
    testdata_path = '../TestingList.xls'
    test_data = xlrd.open_workbook(testdata_path)
    test_table = test_data.sheets()[0]
    test_clinical_feature = []
    for i in range(3,10):
        test_clinical_feature.append(test_table.col_values(i)[1:])
    test_clinical_feature = np.array(test_clinical_feature).transpose(1,0)
    test_MR_Findings = []
    for i in range(10,22):
        test_MR_Findings.append(test_table.col_values(i)[1:])
    test_MR_Findings = np.array(test_MR_Findings).transpose(1,0)
    test_MR_Findings[test_MR_Findings=='4A'] = 4.25
    test_MR_Findings[test_MR_Findings=='4a'] = 4.25
    test_MR_Findings[test_MR_Findings=='4B'] = 4.5
    test_MR_Findings[test_MR_Findings=='4b'] = 4.5
    test_MR_Findings[test_MR_Findings=='4C'] = 4.75
    
    Clinical_Feature_Name = train_table.row_values(0)[3:10]
    idx_Clinical = np.arange(0, len(Clinical_Feature_Name)) 
    min_max_scaler = MinMaxScaler()
    Clinical_Feature = min_max_scaler.fit_transform(train_clinical_feature)
    Clinical_Feature_Test = min_max_scaler.transform(test_clinical_feature)
    Clinical_Class = T2_Class
    Clinical_Class_Test = T2_Class_Test
    
    estimator = Ridge(random_state=0)
    model_Clinical = RFE(estimator=estimator, n_features_to_select=6, step=1).fit(Clinical_Feature, Clinical_Class)
    Clinical_train = model_Clinical.transform(Clinical_Feature)
    Clinical_test = model_Clinical.transform(Clinical_Feature_Test)
    
    Clinical_index = idx_Clinical[model_Clinical.get_support() == True]  #get index positions of kept features
    selected_Clinical = np.array(Clinical_Feature_Name)[Clinical_index]
    print(selected_Clinical)
    x_train_Clinical, y_train_Clinical = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(Clinical_train, Clinical_Class)
    x_test_Clinical = Clinical_test
    y_test_Clinical = Clinical_Class_Test
    clf_Clinical = SVC(kernel="rbf",  probability=True, random_state=0)
    # clf_Clinical = MLPClassifier(200,learning_rate_init=0.005,random_state=0,max_iter=1000)
    clf_Clinical.fit(x_train_Clinical, y_train_Clinical)
    test_prob_Clinical = clf_Clinical.predict_proba(x_test_Clinical)[:,1]
    pred_label_Clinical = clf_Clinical.predict(x_test_Clinical)

    fpr_Clinical,tpr_Clinical,threshold_Clinical = roc_curve(y_test_Clinical, np.array(test_prob_Clinical)) ###计算真正率和假正率
    auc_Clinical = auc(fpr_Clinical,tpr_Clinical)
    auc_l_Clinical, auc_h_Clinical, auc_std_Clinical = confindence_interval_compute(np.array(test_prob_Clinical), y_test_Clinical)
    print('Clinical Feature AUC:%.2f+/-%.2f'%(auc_Clinical,auc_std_Clinical),'  95%% CI:[%.2f,%.2f]'%(auc_l_Clinical,auc_h_Clinical))
    print('Clinical Feature ACC:%.4f'%accuracy_score(y_test_Clinical,pred_label_Clinical))
    prediction_score(y_test_Clinical,pred_label_Clinical)
   
    # train_prob_Clinical = clf_Clinical.predict_proba(x_train_Clinical)[:,1]
    # pred_label_Clinical = clf_Clinical.predict(x_train_Clinical)

    # fpr_Clinical,tpr_Clinical,threshold_Clinical = roc_curve(y_train_Clinical, np.array(train_prob_Clinical)) ###计算真正率和假正率
    # auc_Clinical = auc(fpr_Clinical,tpr_Clinical)
    # auc_l_Clinical, auc_h_Clinical, auc_std_Clinical = confindence_interval_compute(np.array(train_prob_Clinical), y_train_Clinical)
    # print('Clinical Feature AUC:%.2f+/-%.2f'%(auc_Clinical,auc_std_Clinical),'  95%% CI:[%.2f,%.2f]'%(auc_l_Clinical,auc_h_Clinical))
    # print('Clinical Feature ACC:%.4f'%accuracy_score(y_train_Clinical,pred_label_Clinical))
    # # prediction_score(y_train_Clinical,pred_label_Clinical)
    
    # MR Findings  
    min_max_scaler = MinMaxScaler()
    MRSign_Feature = min_max_scaler.fit_transform(train_MR_Findings)
    MRSign_Feature_Test = min_max_scaler.transform(test_MR_Findings)
    MRSign_Class = T2_Class
    MRSign_Class_Test = T2_Class_Test
    MRSign_Feature_Name = train_table.row_values(0)[10:22]
    idx_MRSign = np.arange(0, len(MRSign_Feature_Name))
    estimator = Ridge(random_state=0)
    model_MRSign = RFE(estimator=estimator, n_features_to_select=12, step=1).fit(MRSign_Feature, MRSign_Class)
    MRSign_train = model_MRSign.transform(MRSign_Feature)
    MRSign_test = model_MRSign.transform(MRSign_Feature_Test)
    MRSign_index = idx_MRSign[model_MRSign.get_support() == True]  #get index positions of kept features
    selected_MRSign = np.array(MRSign_Feature_Name)[MRSign_index]
    print(selected_MRSign)
    x_train_MRSign, y_train_MRSign = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(MRSign_train, MRSign_Class)
    x_test_MRSign = MRSign_test
    y_test_MRSign = MRSign_Class_Test
    clf_MRSign = SVC(kernel="rbf",  probability=True, random_state=0)
    clf_MRSign.fit(x_train_MRSign, y_train_MRSign)
    test_prob_MRSign = clf_MRSign.predict_proba(x_test_MRSign)[:,1]
    pred_label_MRSign = clf_MRSign.predict(x_test_MRSign)

    fpr_MRSign,tpr_MRSign,threshold_MRSign = roc_curve(y_test_MRSign, np.array(test_prob_MRSign)) ###计算真正率和假正率
    auc_MRSign = auc(fpr_MRSign,tpr_MRSign)
    auc_l_MRSign, auc_h_MRSign, auc_std_MRSign = confindence_interval_compute(np.array(test_prob_MRSign), y_test_MRSign)
    print('MRSign Feature AUC:%.2f+/-%.2f'%(auc_MRSign,auc_std_MRSign),'  95%% CI:[%.2f,%.2f]'%(auc_l_MRSign,auc_h_MRSign))
    print('MRSign Feature ACC:%.4f'%accuracy_score(y_test_MRSign,pred_label_MRSign))
    prediction_score(y_test_MRSign,pred_label_MRSign)
    
    # train_prob_MRSign = clf_MRSign.predict_proba(x_train_MRSign)[:,1]
    # pred_label_MRSign = clf_MRSign.predict(x_train_MRSign)

    # fpr_MRSign,tpr_MRSign,threshold_MRSign = roc_curve(y_train_MRSign, np.array(train_prob_MRSign)) ###计算真正率和假正率
    # auc_MRSign = auc(fpr_MRSign,tpr_MRSign)
    # auc_l_MRSign, auc_h_MRSign, auc_std_MRSign = confindence_interval_compute(np.array(train_prob_MRSign), y_train_MRSign)
    # print('MRSign Feature AUC:%.2f+/-%.2f'%(auc_MRSign,auc_std_MRSign),'  95%% CI:[%.2f,%.2f]'%(auc_l_MRSign,auc_h_MRSign))
    # print('MRSign Feature ACC:%.4f'%accuracy_score(y_train_MRSign,pred_label_MRSign))
    # # prediction_score(y_train_MRSign,pred_label_MRSign)
    
    # MR+Clinical
    RCF_All_Feature = np.hstack((train,Clinical_train))#T1_All_Feature,T2_Feature,Clinical_Feature
    RCF_Feature_Test = np.hstack((test,Clinical_test))#T1_Feature_Test,T2_Feature_Test,Clinical_Feature_Test
    RCF_Feature_Name = np.hstack((selected_All,selected_Clinical))
    idx_RCF = np.arange(0, len(RCF_Feature_Name)) 
    estimator = Ridge(random_state=0)
    model_RCF = RFE(estimator=estimator, n_features_to_select=12, step=1).fit(RCF_All_Feature, ZQ_Class)
    RCF_train = model_RCF.transform(RCF_All_Feature)
    RCF_test = model_RCF.transform(RCF_Feature_Test)
    RCF_index = idx_RCF[model_RCF.get_support() == True]  #get index positions of kept features
    selected_RCF = np.array(RCF_Feature_Name)[RCF_index]
    print(selected_RCF)
    x_train_RCF, y_train_RCF = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(RCF_train, ZQ_Class)
    x_test_RCF = RCF_test
    y_test_RCF = ZQ_Class_Test
    clf_RCF = SVC(kernel="rbf",  probability=True, random_state=0)
    # clf_RCF = MLPClassifier(200,learning_rate_init=0.005,random_state=0,max_iter=1000)
    clf_RCF.fit(x_train_RCF, y_train_RCF)
    test_prob_RCF = clf_RCF.predict_proba(x_test_RCF)[:,1]
    pred_label_RCF = clf_RCF.predict(x_test_RCF)
    
    fpr_RCF,tpr_RCF,threshold_RCF = roc_curve(y_test_RCF, np.array(test_prob_RCF)) ###计算真正率和假正率
    auc_RCF = auc(fpr_RCF,tpr_RCF)
    auc_l_RCF, auc_h_RCF, auc_std_RCF = confindence_interval_compute(np.array(test_prob_RCF), y_test_RCF)
    print('RCF All Feature AUC:%.2f+/-%.2f'%(auc_RCF,auc_std_RCF),'  95%% CI:[%.2f,%.2f]'%(auc_l_RCF,auc_h_RCF))
    print('RCF All Feature ACC:%.4f'%accuracy_score(y_test_RCF,test_prob_RCF)) 
    prediction_score(y_test_RCF,test_prob_RCF)
    print('RCF All VS Clinical P-Value:',roc_test_r(y_test_RCF, np.array(test_prob_RCF),y_test_Clinical, np.array(test_prob_Clinical)))
    print('RCF All VS MRSign P-Value:',roc_test_r(y_test_RCF, np.array(test_prob_RCF),y_test_MRSign, np.array(test_prob_MRSign)))
    print('RCF All VS MR P-Value:',roc_test_r(y_test_RCF, np.array(test_prob_RCF),y_test, np.array(test_prob)))
    print('RCF All VS T1 P-Value:',roc_test_r(y_test_RCF, np.array(test_prob_RCF), y_test_T1, np.array(test_prob_T1)))
    print('RCF All VS T2 P-Value:',roc_test_r(y_test_RCF, np.array(test_prob_RCF), y_test_T2, np.array(test_prob_T2)))
    stat_val, p_val = stats.ttest_ind(np.array(test_prob_RCF), np.array(test_prob_MRSign), equal_var=False)
    print('RCF All VS MRSign P-Value:%.5f'%p_val)
    # train_prob_RCF = clf_RCF.predict_proba(x_train_RCF)[:,1]
    # pred_label_RCF = clf_RCF.predict(x_train_RCF)
    
    # fpr_RCF,tpr_RCF,threshold_RCF = roc_curve(y_train_RCF, np.array(train_prob_RCF)) ###计算真正率和假正率
    # auc_RCF = auc(fpr_RCF,tpr_RCF)
    # auc_l_RCF, auc_h_RCF, auc_std_RCF = confindence_interval_compute(np.array(train_prob_RCF), y_train_RCF)
    # print('RCF All Feature AUC:%.2f+/-%.2f'%(auc_RCF,auc_std_RCF),'  95%% CI:[%.2f,%.2f]'%(auc_l_RCF,auc_h_RCF))
    # print('RCF All Feature ACC:%.4f'%accuracy_score(y_train_RCF,pred_label_RCF)) 
    # # prediction_score(y_train_RCF,pred_label_RCF)
    
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)

    lw = 1.5
    plt.figure()
    plt.subplot(221)
    plt.plot(fpr_T1,tpr_T1, color='b',linestyle='-',
              lw=lw, label='T1WI Fusion Model (AUC=%.2f)'%auc_T1)
    
    plt.plot(fpr_ZQ,tpr_ZQ, color='b',linestyle='--',
              lw=lw, label='Pre-contrast T1WI Model (AUC=%.2f)'%auc_ZQ) ###'ROC curve (area = %0.3f)' % auc假正率为横坐标，真正率为纵坐标做曲线fusion_auc

    plt.plot(fpr_ZQ1,tpr_ZQ1, color='b',linestyle='-.',
              lw=lw, label='DCE-T1WI Phase 1 Model(AUC=%.2f)'%auc_ZQ1)
    # plt.plot(fpr_ZQ2,tpr_ZQ2, color='b',linestyle=':',
    #           lw=lw, label='DCE-T1WI Phase 2 Model (AUC=%.2f)'%auc_ZQ2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('LE Image Feature')
    plt.legend(loc="lower right",edgecolor='k',title='T1WI Feature',fontsize=10,fancybox=False)
    ax=plt.gca()
    plt.subplot(222)
    plt.plot(fpr,tpr, color='g',linestyle='-',
              lw=lw, label='T1WI+T2WI Feature Model (AUC=%.2f)'%auc_all)
    
    plt.plot(fpr_T1,tpr_T1, color='b',linestyle='-',
              lw=lw, label='T1WI Fusion Model (AUC=%.2f)'%auc_T1) ###'ROC curve (area = %0.3f)' % auc假正率为横坐标，真正率为纵坐标做曲线fusion_auc

    plt.plot(fpr_T2,tpr_T2, color='gold',linestyle='-',
              lw=lw, label='T2WI Model (AUC=%.2f)'%auc_T2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('HE Image Feature')
    plt.legend(loc="lower right",edgecolor='k',title='MR Radiomics Feature',fontsize=10,fancybox=False)
    ax=plt.gca()
    
    plt.subplot(223)
    # fpr4,tpr4,threshold4 = roc_curve(np.array(real_class_Tumor),Fusion_max)
    plt.plot(fpr_RCF,tpr_RCF, color='r',linestyle='-',
              lw=lw, label='All Radiomics Feature Fusion Model (AUC=%.2f)'%auc_RCF)
    
    plt.plot(fpr,tpr, color='g',linestyle='-',
              lw=lw, label='T1WI+T2WI Feature Model (AUC=%.2f)'%auc_all) ###'ROC curve (area = %0.3f)' % auc假正率为横坐标，真正率为纵坐标做曲线fusion_auc

    plt.plot(fpr_Clinical,tpr_Clinical, color='blueviolet',linestyle='-',
              lw=lw, label='Clinical Feature Model (AUC=%.2f)'%auc_Clinical)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('DES Image Feature')
    plt.legend(loc="lower right",edgecolor='k',title='MRI Radiomics and Clinical Feature',fontsize=10,fancybox=False)
    ax=plt.gca()
    
    plt.subplot(224)
    plt.plot(fpr_RCF,tpr_RCF, color='r',linestyle='-',
              lw=lw, label='Our Proposed Model (AUC=%.2f)'%auc_RCF)
    plt.plot(fpr_MRSign,tpr_MRSign, color='m',linestyle='-',
              lw=lw, label='MRI Findings Model (AUC=%.2f)'%auc_MRSign)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Fusion Image Feature')
    plt.legend(loc="lower right",edgecolor='k',title='Different Model Comprasion',fontsize=10,fancybox=False)
    ax=plt.gca()
    
    
    