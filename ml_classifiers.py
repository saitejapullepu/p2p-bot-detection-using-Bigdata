import csv
import random
import math
import operator
import time
import pandas as pd
import numpy as np
from sklearn import metrics

def getAccuracy(original_test_labels , predictions):
    correct = 0
    for x in range(len(original_test_labels)):
        if original_test_labels [x] == predictions[x]:
            correct += 1
    return (correct/float(len(original_test_labels))) * 100.0

def main():
    
    start= time.time()
    print ("start time ", start)

    original_features = pd.read_csv('/home/stp/Desktop/new dataset.csv')
    original_features = pd.get_dummies(original_features)

    # Use numpy to convert to arrays
    import numpy as np

    # Labels are the values we want to predict
    original_labels = np.array(original_features['target'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    original_features= original_features.drop('target', axis = 1)

    # Saving feature names for later use
    original_feature_list = list(original_features.columns)

    # Convert to numpy array
    original_features = np.array(original_features)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(original_features, original_labels, test_size = 0.3, random_state =0)
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
## NOTE : while using rf part un comment rf block only.
##    # Import the model we are using
##    from sklearn.ensemble import RandomForestRegressor
##    from sklearn.ensemble import RandomForestClassifier
##    print ("Random forest started running please wait...................................................................................")
##    # Instantiate model
##    mid=time.time()
##    rfc = RandomForestClassifier(n_estimators=1000 , random_state=42)
##    #rf = RandomForestRegressor(n_estimators=1000 , random_state=42)
##    
##    # Train the model on training data
##    rfc.fit(original_train_features, original_train_labels);
##    mid1=time.time()
##    print("predictions started .........................................................................................................")
##    # Use the forest's predict method on the test data
##    predictions = rfc.predict(original_test_features)
##    mid2=time.time()
##    from sklearn import metrics
##    #print("Accuracy naive rf :",metrics.accuracy_score(original_test_labels, predictions))
##    accuracy = getAccuracy(original_test_labels, predictions)
##    
##       #comfusion matrix
##    #comfusion matrix
##    from sklearn.metrics import confusion_matrix
##    cmtx = pd.DataFrame(
##        confusion_matrix(original_test_labels, predictions), 
##        index=['true:Bot', 'true:Benign'], 
##        columns=['pred:Bot', 'pred:benign']
##    )
##    print ("confusion matrix of random forest ")
##    print(cmtx)
##    pred = np.array(predictions)
##    tp, fn, fp, tn = confusion_matrix(original_test_labels, predictions).ravel()
##    print("                   ")
##    print("No. of instances classified as Bot :", np.sum(pred==0))
##    print("No. of instances classified as Benign :", np.sum(pred==1))
##    print("                   ")
##    print("Total No. of instances : ",len(predictions))
##    print("--------------classificaton report---------------------------")
##    print("True Positive ",tp)
##    print("False Negative ",fn)
##    print("False Positive ",fp)
##    print("True  Negative ",tn)
##    print("                   ")
##    print('Accuracy of Random Forest model : {0}%'.format(accuracy))
##    prec = float(tp/(tp+fp))
##    rec = float (tp/(tp+fn))
##    f1 = float((2*prec*rec)/(prec+rec))
##    print("precision" , prec)
##    print("Recall", rec)
##    print("f1-measure" ,f1 )
##    print("Training time :",mid1-mid)
##    print("Testing time :",mid2-mid1)
##   
####    
######    from sklearn.model_selection import cross_val_score
####    from sklearn.metrics import classification_report, confusion_matrix
####    print("--------------classificaton report---------------------------")
####    print(classification_report(original_test_labels, predictions))
##    print("--------------cross validation report---------------------------")
##    from sklearn.model_selection import KFold, cross_val_score
##    import statistics
##    k_fold = KFold(len(original_labels), n_splits=3, shuffle=True, random_state=0)
##    k_fold = KFold(n_splits=10)
##    clf =  RandomForestClassifier(n_estimators=1000 , random_state=42)
##    cross_validation = cross_val_score(clf,original_features,  original_labels, cv=k_fold, n_jobs=1)
##    print ("10-fold cross validation :",cross_validation)
##    print ("cross-valadition score:",statistics.mean(cross_validation) )
##    print ("                                                          ")
##----------------------------------------------------------------------------------------------------------------------------------------------------------------------
####NOTE : while using naive bayes  part un comment naive bayes  block only.
##    mid=time.time()    
##    from sklearn.naive_bayes import GaussianNB
##    #Create a Gaussian Classifier
##    gnb = GaussianNB()
##
##    #Train the model using the training sets
##    gnb.fit(original_train_features, original_train_labels)
##    mid1=time.time()
##    #Predict the response for test dataset
##    y_pred = gnb.predict(original_test_features)
##    mid2=time.time()
##    from sklearn import metrics
##
####Model Accuracy, how often is the classifier correct?
##   
##    #comfusion matrix
##    from sklearn.metrics import confusion_matrix
##    cmtx = pd.DataFrame(
##        confusion_matrix(original_test_labels, y_pred), 
##        index=['true:Bot', 'true:Benign'], 
##        columns=['pred:Bot', 'pred:benign']
##    )
##   
##    print ("confusion matrix of naive bayes ")
##    print(cmtx)
##    pred = np.array(y_pred)
##    tp, fn, fp, tn = confusion_matrix(original_test_labels, y_pred).ravel()
##    print("                   ")
##    print("No. of instances classified as Bot :", np.sum(pred==0))
##    print("No. of instances classified as Benign :", np.sum(pred==1))
##    print("                   ")
##    print("Total No. of instances : ",len(y_pred))
##    print("--------------classificaton report---------------------------")
##    print("True Positive ",tp)
##    print("False Negative ",fn)
##    print("False Positive ",fp)
##    print("True  Negative ",tn)
##    print("                   ")
##    print("Accuracy naive bayes :",metrics.accuracy_score(original_test_labels, y_pred))
##    prec = float(tp/(tp+fp))
##    rec = float (tp/(tp+fn))
##    f1 = float((2*prec*rec)/(prec+rec))
##    print("precision" , prec)
##    print("Recall", rec)
##    print("f1-measure" ,f1 )
##    print("Training time :",mid1-mid)
##    print("Testing time :",mid2-mid1)
##    print("--------------cross validation report---------------------------")
##    from sklearn.model_selection import KFold, cross_val_score
##    import statistics
##    k_fold = KFold(len(original_labels), n_splits=3, shuffle=True, random_state=0)
##    k_fold = KFold(n_splits=10)
##    clf = GaussianNB()
##    cross_validation = cross_val_score(clf,original_features,  original_labels, cv=k_fold, n_jobs=1)
##    print ("10-fold cross validation :",cross_validation)
##    print ("cross-valadition score:",statistics.mean(cross_validation) )
##    print ("                                                          ")
    
##    from sklearn.model_selection import cross_val_score
##    from sklearn.metrics import classification_report, confusion_matrix
##    print("--------------classificaton report---------------------------")
##    print(classification_report(original_test_labels, y_pred , output_dict=True))
##----------------------------------------------------------------------------------------------------------------------------------------------------------------------
##  #NOTE : while using knn  part un comment knn  block only.
##    from sklearn.neighbors import KNeighborsClassifier
##    mid=time.time()
##    knn = KNeighborsClassifier(n_neighbors = 3)
##    
##    knn.fit(original_train_features, original_train_labels)
##    mid1=time.time()
##    y_pred = knn.predict(original_test_features)
##    acc = knn.score(original_test_features,original_test_labels)
##    mid2=time.time()
##    
##    #comfusion matrix
##    from sklearn.metrics import confusion_matrix
##    cmtx = pd.DataFrame(
##        confusion_matrix(original_test_labels, y_pred), 
##        index=['true:Bot', 'true:Benign'], 
##        columns=['pred:Bot', 'pred:benign']
##    )
##    print ("confusion matrix of knn  ")
##    print(cmtx)
##    pred = np.array(y_pred)
##    tp, fn, fp, tn = confusion_matrix(original_test_labels, y_pred).ravel()
##    print("                   ")
##    print("No. of instances classified as Bot :", np.sum(pred==0))
##    print("No. of instances classified as Benign :", np.sum(pred==1))
##    print("                   ")
##    print("Total No. of instances : ",len(y_pred))
##    print("--------------classificaton report---------------------------")
##    print("True Positive ",tp)
##    print("False Negative ",fn)
##    print("False Positive ",fp)
##    print("True  Negative ",tn)
##    print("                   ")
##    print("accuracy of knn",acc)
##    prec = float(tp/(tp+fp))
##    rec = float (tp/(tp+fn))
##    f1 = float((2*prec*rec)/(prec+rec))
##    print("precision" , prec)
##    print("Recall", rec)
##    print("f1-measure" ,f1 )
##    print("Training time :",mid1-mid)
##    print("Testing time :",mid2-mid1)
##   
##    from sklearn.model_selection import cross_val_score
##    from sklearn.metrics import classification_report, confusion_matrix
##    print("--------------classificaton report---------------------------")
##    print(classification_report(original_test_labels, y_pred))
##    print("--------------cross validation report---------------------------")
##    from sklearn.model_selection import KFold, cross_val_score
##    import statistics
##    k_fold = KFold(len(original_labels), n_splits=3, shuffle=True, random_state=0)
##    k_fold = KFold(n_splits=10)
##    clf = KNeighborsClassifier(n_neighbors = 3)
##    cross_validation = cross_val_score(clf,original_features,  original_labels, cv=k_fold, n_jobs=1)
##    print ("10-fold cross validation :",cross_validation)
##    print ("cross-valadition score:",statistics.mean(cross_validation) )
##    print ("                                                          ")
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##    from sklearn.tree import DecisionTreeClassifier
##    from sklearn import metrics
##    mid=time.time()
##    classifier = DecisionTreeClassifier()
##    classifier.fit(original_train_features, original_train_labels)
##    mid1=time.time()
##    y_pred = classifier.predict(original_test_features)
##    mid2=time.time()
##        #comfusion matrix
##    from sklearn.metrics import confusion_matrix
##    cmtx = pd.DataFrame(
##        confusion_matrix(original_test_labels, y_pred), 
##        index=['true:Bot', 'true:Benign'], 
##        columns=['pred:Bot', 'pred:benign']
##    )
##    print ("confusion matrix of Decision Tree Classifier ")
##    print(cmtx)
##    pred = np.array(y_pred)
##    tp, fn, fp, tn = confusion_matrix(original_test_labels, y_pred).ravel()
##    print("                   ")
##    print("No. of instances classified as Bot :", np.sum(pred==0))
##    print("No. of instances classified as Benign :", np.sum(pred==1))
##    print("                   ")
##    print("Total No. of instances : ",len(y_pred))
##    print("--------------classificaton report---------------------------")
##    print("True Positive ",tp)
##    print("False Negative ",fn)
##    print("False Positive ",fp)
##    print("True  Negative ",tn)
##    print("                   ")
##    print("Accuracy  of Decision Tree Classifier  :",metrics.accuracy_score(original_test_labels, y_pred))
##    prec = float(tp/(tp+fp))
##    rec = float (tp/(tp+fn))
##    f1 = float((2*prec*rec)/(prec+rec))
##    print("precision" , prec)
##    print("Recall", rec)
##    print("f1-measure" ,f1 )
##    print("Training time :",mid1-mid)
##    print("Testing time :",mid2-mid1)
##    print("--------------cross validation report---------------------------")
##    from sklearn.model_selection import KFold, cross_val_score
##    import statistics
##    k_fold = KFold(len(original_labels), n_splits=3, shuffle=True, random_state=0)
##    k_fold = KFold(n_splits=10)
##    clf = DecisionTreeClassifier()
##    cross_validation = cross_val_score(clf,original_features,  original_labels, cv=k_fold, n_jobs=1)
##    print ("10-fold cross validation :",cross_validation)
##    print ("cross-valadition score:",statistics.mean(cross_validation) )
##    print ("                                                          ")
##----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    from sklearn import svm
    mid=time.time()
    clf = svm.SVC()
    clf.fit(original_train_features, original_train_labels)
    mid1=time.time()
    y_pred = clf.predict(original_test_features)
    mid2=time.time()
    from sklearn.metrics import confusion_matrix
    cmtx = pd.DataFrame(
        confusion_matrix(original_test_labels, y_pred), 
        index=['true:Bot', 'true:Benign'], 
        columns=['pred:Bot', 'pred:benign']
    )
    print ("confusion matrix of SVM ")
    print(cmtx)
    pred = np.array(y_pred)
    tp, fn, fp, tn = confusion_matrix(original_test_labels, y_pred).ravel()
    print("                   ")
    print("No. of instances classified as Bot :", np.sum(pred==0))
    print("No. of instances classified as Benign :", np.sum(pred==1))
    print("                   ")
    print("Total No. of instances : ",len(y_pred))
    print("--------------classificaton report---------------------------")
    print("True Positive ",tp)
    print("False Negative ",fn)
    print("False Positive ",fp)
    print("True  Negative ",tn)
    print("                   ")
    print("Accuracy  of SVM  :",metrics.accuracy_score(original_test_labels, y_pred))
    prec = float(tp/(tp+fp))
    rec = float (tp/(tp+fn))
    f1 = float((2*prec*rec)/(prec+rec))
    print("precision" , prec)
    print("Recall", rec)
    print("f1-measure" ,f1 )
    print("Training time :",mid1-mid)
    print("Testing time :",mid2-mid1)
    print("--------------cross validation report---------------------------")
    from sklearn.model_selection import KFold, cross_val_score
    import statistics
    k_fold = KFold(len(original_labels), n_splits=3, shuffle=True, random_state=0)
    k_fold = KFold(n_splits=10)
    clf = svm.SVC()
    cross_validation = cross_val_score(clf,original_features,  original_labels, cv=k_fold, n_jobs=1)
    print ("10-fold cross validation :",cross_validation)
    print ("cross-valadition score:",statistics.mean(cross_validation) )
    print ("                                                          ")

##    
    

    


    end = time.time()
    print("Total Time elapsed: ", end - start) # CPU seconds elapsed

    
main()
