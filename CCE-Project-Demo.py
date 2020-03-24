# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 06:28:58 2018
CCE-Classification-Project

@author: Laxmidhar 
"""

# ============================== loading libraries ===========================================
import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

# ============================== data preprocessing ===========================================
# define column names for dataset
# Quick Notes on column names
# Preg_Cnt       :Pregnancy_Count
# Plasma_Gls     : Plasma glucose concentration a 2 hours in an oral glucose tolerance test							
# DBP            : Diastolic blood pressure (mm Hg)							
# TSK            : Triceps skin fold thickness (mm)							
# HSI            : 2-Hour serum insulin (mu U/ml)							
# BMI            : Body mass index (weight in kg/(height in m)^2)							
# DPF            : Diabetes pedigree function							
# Class          : Diabetic(1) or Non-Diabetic(0)

# Version of kkn algoritm to be used

# Use default density based KNN
'''
weighted_kkn = 0

#userInput = 0
# Use both version of KNN
#weighted_kkn = 3

#print("Choose option of KNN Algorithm to be used:")
#print("Default KNN - Majority Vote : 1")
#print("Weighted KNN - Use Neighbor distance : 2")

#userInput = int(input("Pick your option : "))

print("User selected input = ", weighted_kkn)

if (userInput > 2 or userInput < 1):
    print("Wrong choice, please rerty with correct option")
    raise SystemExit()

# User input is 2 use weignted KNN
if (userInput == 2 ):
    weighted_kkn = 1
    print("Using Weighted KNN Algorithm : ", weighted_kkn)
else:
    print("Using Default KNN Algorithm : ", weighted_kkn)
'''    
# Predict the classifcation class based on value seen 
# For Diabetic dataset we only have binary class
# 0.0 : Non-Dibetic (Negatice Class)
# 1.0 : Diabetic (Positive Class)

# Test and split.
print("Choose dataset split ratio between training and testing set")
print("Default ratio 0.8, option minimum (0.5) and maximum (0.9)")

# split ratio default
datasetSplitRatio = 0.8
datasetSplitRatio = float(input("Pick dataset split ratio : "))

if(datasetSplitRatio > 0.9 or datasetSplitRatio < 0.5):
    print("Pick correct value of dataset split")
    raise SystemExit()

print("Choose seed value for random generator algoritm")
print("Default value 42, option minimum (0) and maximum (9999)")

# split ratio default
userSeedValue = 42
userSeedValue = int(input("Pick your random generator seed value : "))

if(userSeedValue > 9999 or userSeedValue < 0):
    print("Pick correct value of dataset split")
    raise SystemExit()

'''
********************************************************************
 SET ABSOLUTE PATH FOR DATA SET AS PER PATH ON YOUR ENV
 1: pima-indians-diabetes.csv       : Dataset with 200 datapoint
 2: pima-indians-diabetes.orig.csv  : Full Dataset with 786 datapoint
 3: Enable which ever you want to test against it
********************************************************************
''' 
# loading training data. Use csv file absolute path on your computer
# Full Dataset
#df = pd.read_csv(r"C:\Users\Desktop\pima-indians-diabetes.orig.csv")

# Purned data set
df = pd.read_csv(r"C:\Users\Desktop\pima-indians-diabetes.csv")

print(df.head())
print(df.tail())

# Outcome column name in dataset
columnName = "Class"

# Number of row
print("Number of Row in DataSet : ", df.shape[0])
# Number of column
print("Number of Column in DataSet : ", df.shape[1])
#print(df.shape[1])

# Spliting of dataset
#training_dataSet = df.sample(frac=datasetSplitRatio, random_state=userSeedValue, replace=True)
training_dataSet = df.sample(frac=datasetSplitRatio, random_state=userSeedValue)
#training_dataSet = df.sample(frac=0.8, random_state=1)
print(training_dataSet)
test_dataSet = df.drop(training_dataSet.index)
print("test_dataSet test set")
print(test_dataSet)
    
training_dataSet_feature = training_dataSet.drop([columnName],axis=1)
training_dataSet_label   = training_dataSet[columnName]
print("training_dataSet_feature")
print(training_dataSet_feature)
print("training_dataSet_label")
print(training_dataSet_label)
print("test datataaaaa")
    
test_dataSet_feature = test_dataSet.drop([columnName],axis=1)
print("test_dataSet_feature")
print(test_dataSet_feature)
    
test_dataSet_label = test_dataSet[columnName]
print("test_dataSet_label")
print(test_dataSet_label)
    

def cce_ml_proj_predict(value):
    if value == 0.0:
       print("Non-Diabetic")
    else:
       print("Diabetic")

# Calculated Euclidean Distance between two data point
# If P1(x1,x2,...,xN) and P2(y1,y2,.....,yN) are two points then 
# Euclidean distance between them is calculated as square root of
# squar_root of ((x1-y1)^2 + (x2-y2)^2 + ..... + (xN-yN)^2)
# Notation ^ means power 
      
def cce_ml_proj_euclideanDistance(dataPoint1, dataPoint2, length):
    # Debug print statement to see data
    #print("data1")
    #print(dataPoint1)
    #print("data2 ",dataPoint2)
    #print(dataPoint2)
    #print("length=%d ",length)
    #print(length)
    distance = 0
    for x in range(length):
        # Debug statement to print data using index
        #print("x=%d ",x)
        #print("data2 ", dataPoint2[x])
        #print("data1 ", dataPoint1[x])
        distance += np.square(dataPoint1[x] - dataPoint2[x])
    #print('distance : ', distance)
    return np.sqrt(distance)

def cce_ml_proj_predict_k_Nearest_Nbrs(trainingSet, testInstance, k):
    # Debug Statement
    #print("Entering Funciton : cce_ml_proj_predict_k_Nearest_Nbrs")
    #print(testInstance)
    euclidean_distance = {}
    sort_euclidean_distance = {}
 
    # If Dataset have no header as first now
    # length = testInstance.shape[0]
    # If Dataset have has header as first now
    length = testInstance.shape[1]
    #print("lenght", lenght)
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = cce_ml_proj_euclideanDistance(testInstance, trainingSet.iloc[x], length)
        euclidean_distance[x] = dist[0]
   
    print("Euclidean Distance :") 
    print(euclidean_distance)
    # Sorting euclidean_distance on the basis of distance
    sort_euclidean_distance = sorted(euclidean_distance.items(), key=operator.itemgetter(1))
    print("Sorted Euclidean Distance :")
    print(sort_euclidean_distance)
        
    neighbors = []
    
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sort_euclidean_distance[x][0])
 
    classVotes = {}
    print('Number of neighbors : ', len(neighbors), ' List of Neighbors : ', neighbors)
    #print("len of nbr =",len(neighbors))
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        #response = trainingSet.iloc[neighbors[x]][-1]
        #print("value of x =", x)
        response = training_dataSet_label.iloc[neighbors[x]]
        #print("response", response)
        #print("xaa = ", training_dataSet_label.iloc[x])
                
        #Debug
        print("response", response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
              
    print("Classification found along with number of items belong to each class")
    print(classVotes)
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    print("Sorted classification along each class count")
    print(sortedVotes)
    return(sortedVotes[0][0], neighbors)
    
def cce_ml_proj_find_accuracy_score_with_k_Nearest_Nbrs(trainingSet, testingSet, kvalue):
    rightPrediction = 0
    wrongPrediction = 0
    print('kvalue= ', kvalue)
    for x in range(trainingSet.shape[0]):
        print('value of x = ',x )
        testSet1= [np.array(trainingSet.iloc[x])]
        #print(testSet1)
        testData = pd.DataFrame(testSet1)
        #print("post dataframn")
        print(testData)
        #print(df.head())
        # Modify number of neighbors to odd values between 1 and 11 and check result
        k = kvalue
        # Running KNN model
        predictedResult, neighbors = cce_ml_proj_predict_k_Nearest_Nbrs(trainingSet, testData, k)
        print("predictedResult", predictedResult)
        # Predicted class
        print("List of neighbors found")
        print(neighbors)

        print("Classification Result:")
        cce_ml_proj_predict(predictedResult)
        
        if predictedResult == training_dataSet_label.iloc[x]:
            print("Right Predtiction")
            rightPrediction = rightPrediction + 1
            if (predictedResult == 1):
                training_positive_predicted_positive_score.append(x)
            else:
                training_negative_predicted_negative_score.append(x)
        else:
            print("Wrong Predtiction")
            wrongPrediction = wrongPrediction + 1
            if (predictedResult == 1):
                # Actual value negative (non-dibetic) but predicted as positive(dibetic)
                training_negative_predicted_positive_score.append(x)
            else:
                # Actual Value was positive(dibetic) but predicted was negative (non-dibetic)
                training_positive_predicted_negative_score.append(x)
            
            
    print('rightPrediction = ', rightPrediction)
    print('wrongPrediction = ', wrongPrediction)
    train_accuracy = (rightPrediction / trainingSet.shape[0]) * 100
    print('train_accuracy = ', train_accuracy, 'with k = ', k)   
    
    print("*********** Working with test dataset *************")
    # On test dataset
    rightPrediction = 0
    wrongPrediction = 0
    print('kvalue= ', kvalue)
    for x in range(testingSet.shape[0]):
        print('value of x = ',x )
        testSet1= [np.array(testingSet.iloc[x])]
        #print(testSet1)
        testData = pd.DataFrame(testSet1)
        #print("post dataframn")
        print(testData)
        #print(df.head())
        # Modify number of neighbors to odd values between 1 and 11 and check result
        k = kvalue
        # Running KNN model
        predictedResult, neighbors = cce_ml_proj_predict_k_Nearest_Nbrs(trainingSet, testData, k)
        print("predictedResult", predictedResult)
        # Predicted class
        print("List of neighbors found")
        print(neighbors)

        print("Classification Result:")
        cce_ml_proj_predict(predictedResult)
        
        if predictedResult == test_dataSet_label.iloc[x]:
            print("Right Predtiction")
            rightPrediction = rightPrediction + 1
            if (predictedResult == 1):
                test_positive_predicted_positive_score.append(x)
            else:
                test_negative_predicted_negative_score.append(x)
        else:
            print("Wrong Predtiction")
            wrongPrediction = wrongPrediction + 1
            if (predictedResult == 1):
                # Actual value negative (non-dibetic) but predicted as positive(dibetic)
                test_negative_predicted_positive_score.append(x)
            else:
                # Actual Value was positive(dibetic) but predicted was negative (non-dibetic)
                test_positive_predicted_negative_score.append(x)
         
    print('rightPrediction = ', rightPrediction)
    print('wrongPrediction = ', wrongPrediction)
    test_accuracy = (rightPrediction / testingSet.shape[0]) * 100
    print('test_accuracy = ', test_accuracy, 'with k = ', k)   
    
    return(train_accuracy, test_accuracy)
    
# pp : positive-positive
# pe : positive-negative
# negp being used instead of ne to avoid conflict between numpy arrap variable
# np
# negp : negative-positive
# nn : negative-negative
# graphColors color scheme test/train 
    
def cce_ml_proj_draw_confusion_matrix(pp,pe,negp,nn,graphColors):
    cm = [[pp, pe],[negp, nn]]
    print(cm)
    plt.clf()
    if (graphColors == 'test'):
        plt.title('Diabetic Data Confusion Matrix - Test Data', fontsize='large', fontweight='bold')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.summer)
    else:
        plt.title('Diabetic Data Confusion Matrix - Training Data', fontsize='large', fontweight='bold')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    
    classNames = ['Positive', 'Negative',]
    plt.ylabel('DataSer Label', color='Red', fontsize='large', fontweight='bold')
    plt.xlabel('Predicted Label', color='Blue', fontsize='large', fontweight='bold')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45, fontsize='large', fontweight='bold')
    plt.yticks(tick_marks, classNames, fontsize='large', fontweight='bold')
    s = [['TP','FN'], ['FP', 'TN']]
 
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]),fontsize='large', fontweight='bold', ha='center', va='center')

    if (graphColors == 'test'):
        plt.savefig(r"C:\Users\radyadav.ORADEV\.spyder-py3\confusion-matrix-test.png")
    else:
        plt.savefig(r"C:\Users\radyadav.ORADEV\.spyder-py3\confusion-matrix-trainingest.png")
            
    plt.show()

print("************************************************")
print("***** List of task supported functionality *****")
print("************************************************")
print("1 : Run Prediction on any data point in test dataset.")
#print("2 : Run Prediction on list of data point in test dataset.")
print("2 : Run Prediction on complete test dataset.")
print("3 : Run Prediction on complete training and test dataset.")
print("    Option 3 will for only one value of k specified by user")
print("4 : Run Prediction on complete training and test dataset for k = 1 to user value.")
print("    Option 4 will run for k = 1 to user specified maximum value of k.") 
print("    Execution time of option 3 and 4 depends upon size of dataset and k");
print("    It can take few minutes to hours.")

UserOption = 0
UserOption = int(input("Pick task option for execution : "))

if(UserOption > 4 or UserOption < 1):
    print("Choose correct option from task list ")
    raise SystemExit()

print("Choose number of neighbor to be used in KNN : ")
print("Default value 1, option minimum (1) and maximum (15)")
print("Execution time directly proportional to size of dataset and K).")

UserKNNValue = 1

UserKNNValue = int(input("Pick number of neighbor's : "))

if(UserKNNValue > 15 or UserKNNValue < 1):
    print("Choose correct value of K ")
    raise SystemExit()

print("Number of Neighbor choosen : ", UserKNNValue)

# Use training set data 
useTrainingSetData = 0

# Use it for only option 4 and 5
if(UserOption == 3 or UserOption == 4):
    useTrainingSetData = 1
    
# Training accuracy score
train_acc_score = []

# Test accuracy score
test_acc_score = []

# Confusion matrix for training data
training_positive_predicted_positive_score = []
training_positive_predicted_negative_score = []
training_negative_predicted_negative_score = []
training_negative_predicted_positive_score = []

# Confusion matrix for test data
test_positive_predicted_positive_score = []
test_positive_predicted_negative_score = []
test_negative_predicted_negative_score = []
test_negative_predicted_positive_score = []

# switch case for different options
if (UserOption == 1):
    max_index = test_dataSet_feature.shape[0]
    #print("value : ", max_index)
    max_index = (max_index -1)
    #print("value : ", max_index)
    pick_index = 1
    print("Pick any index from dataset between 0 and ", max_index)
    pick_index = int(input("Pick any index from dataset : "))

    if(pick_index >  (max_index -1) or pick_index < 0):
        print("Wrong value of index choosen")
        raise SystemExit()
    #testSet1= [np.array(df.iloc[24, 0:8])]
    testSet1 = [np.array(test_dataSet_feature.iloc[pick_index])]
    print(testSet1)
    testData1 = pd.DataFrame(testSet1)
    print(testData1)
    #print(df.head())
    # Modify number of neighbors to odd values between 1 and 11 and check result
    # Running KNN model
    result, neighbors = cce_ml_proj_predict_k_Nearest_Nbrs(training_dataSet_feature, testData1, UserKNNValue)

    if result == test_dataSet_label.iloc[pick_index]:
        print("Right Predtiction")
        if (result == 1):
            test_positive_predicted_positive_score.append(pick_index)
        else:
            test_negative_predicted_negative_score.append(pick_index)
    else:
        print("Wrong Predtiction")
        if (result == 1):
            # Actual value negative (non-dibetic) but predicted as positive(dibetic)
            test_negative_predicted_positive_score.append(pick_index)
        else:
            # Actual Value was positive(dibetic) but predicted was negative (non-dibetic)
            test_positive_predicted_negative_score.append(pick_index)
        
    print("Actual value in dataset ", test_dataSet_label.iloc[pick_index])
    # Predicted class
    print("List of neighbors found")
    print(neighbors)
    
    print("Classification Result:")
    cce_ml_proj_predict(result)
    test_positive_positive = len(test_positive_predicted_positive_score)
    test_positive_negative = len(test_positive_predicted_negative_score)
    test_negative_positive = len(test_negative_predicted_positive_score)
    test_negative_negative = len(test_negative_predicted_negative_score)
    
    print("\nTest DataSet : Correct Prediction :")
    print('test_positive_predicted_positive_score :\n TP = ', test_positive_positive, ' TP List : ', test_positive_predicted_positive_score)
    print('test_negative_predicted_negative_score :\n TN = ', test_negative_negative, ' TN List : ', test_negative_predicted_negative_score)

    print("\nTest DataSet : Wrong Prediction : ")
    print('test_positive_predicted_negative_score :\n FP = ', test_positive_negative, ' FP List : ', test_positive_predicted_negative_score)
    print('test_negative_predicted_positive_score :\n FN = ', test_negative_positive, ' FN List : ', test_negative_predicted_positive_score)
    
    cce_ml_proj_draw_confusion_matrix(test_positive_positive, test_positive_negative, \
                                      test_negative_positive, test_negative_negative, "test")

elif ((UserOption == 2) or (UserOption == 3)):
    train_accr, test_accr = cce_ml_proj_find_accuracy_score_with_k_Nearest_Nbrs(training_dataSet_feature, test_dataSet_feature, UserKNNValue)
    print('train_accr = ', train_accr)
    print('test_accr = ', train_accr)
    train_acc_score.append(train_accr)
    test_acc_score.append(test_accr)

    # Only for option 4 we use training set data
    if (UserOption == 3):
        print('\nTraing Accurancy Score :', train_acc_score)
        print('Test Accurancy Score : ',  test_acc_score)

        # Training dataset Confusion matirx 
        train_positive_positive = len(training_positive_predicted_positive_score)
        train_positive_negative = len(training_positive_predicted_negative_score)
        train_negative_positive = len(training_negative_predicted_positive_score)
        train_negative_negative = len(training_negative_predicted_negative_score)

        print("\nTraining DataSet : Correct Prediction :")
        print('training_positive_predicted_positive_score :\n TP = ', train_positive_positive, ' TP List : ', training_positive_predicted_positive_score)
        print('training_negative_predicted_negative_score :\n TN = ', train_negative_negative, ' TN List : ', training_negative_predicted_negative_score)

        print("\nTraining DataSet : Wrong Prediction : ")
        print('training_positive_predicted_negative_score :\n FP = ', train_positive_negative, ' FP List : ', training_positive_predicted_negative_score)
        print('training_negative_predicted_positive_score :\n FN = ', train_negative_positive, ' FN List : ', training_negative_predicted_positive_score)


        cce_ml_proj_draw_confusion_matrix(train_positive_positive, train_positive_negative, \
                                          train_negative_positive, train_negative_negative, "train")

    # Training dataset Confusion matirx 
    test_positive_positive = len(test_positive_predicted_positive_score)
    test_positive_negative = len(test_positive_predicted_negative_score)
    test_negative_positive = len(test_negative_predicted_positive_score)
    test_negative_negative = len(test_negative_predicted_negative_score)
    
    print("\nTest DataSet : Correct Prediction :")
    print('test_positive_predicted_positive_score :\n TP = ', test_positive_positive, ' TP List : ', test_positive_predicted_positive_score)
    print('test_negative_predicted_negative_score :\n TN = ', test_negative_negative, ' TN List : ', test_negative_predicted_negative_score)

    print("\nTest DataSet : Wrong Prediction : ")
    print('test_positive_predicted_negative_score :\n FP = ', test_positive_negative, ' FP List : ', test_positive_predicted_negative_score)
    print('test_negative_predicted_positive_score :\n FN = ', test_negative_positive, ' FN List : ', test_negative_predicted_positive_score)
    
    cce_ml_proj_draw_confusion_matrix(test_positive_positive, test_positive_negative, \
                                      test_negative_positive, test_negative_negative, "test")
elif (UserOption == 4):
    for x in range(1, (UserKNNValue + 1)):
        train_accr, test_accr = cce_ml_proj_find_accuracy_score_with_k_Nearest_Nbrs(training_dataSet_feature, test_dataSet_feature, x)
        print('train_accr = ', train_accr)
        print('test_accr = ', train_accr)
        train_acc_score.append(train_accr)
        test_acc_score.append(test_accr)
   
    print('Traing Accurancy Score :', train_acc_score)
    print('Test Accurancy Score : ',  test_acc_score)
    NumOfNbrs = list(range(1, (UserKNNValue + 1)))
    print(NumOfNbrs)
    plt.plot(NumOfNbrs, train_acc_score)
    plt.plot(NumOfNbrs, test_acc_score)

    plt.xlabel('Number of Neighbors', color='Green', fontsize='large', fontweight='bold')
    plt.ylabel('Accuracy Score (%)', color='Blue', fontsize='large', fontweight='bold')
    plt.title('Plot between accuracy and Neighbors for KNN classifier', color='Red', fontsize='large', fontweight='bold')
    plt.grid(True)
    plt.savefig(r"C:\Users\radyadav.ORADEV\.spyder-py3\performance-test.png")
    plt.show()

