#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the packages and libraries that are required for the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import auc
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[2]:


# Load the datasets
train_data = pd.read_csv('train.csv',index_col=0)

train_data.drop("id",axis=1,inplace=True)

train_data.head()


# In[3]:


# Check for NA values 
train_data.isna().sum()


# In[4]:


# Fill the missing values
train_data["Arrival Delay in Minutes"] = train_data["Arrival Delay in Minutes"].fillna(train_data["Departure Delay in Minutes"])


# In[5]:


# Seperate the data into features and lables 
X_data = train_data.iloc[:,:-1]
y_data = train_data.iloc[:,-1]


# In[6]:


# Count of Labels
sns.countplot(y_data, label="Count")
plt.show()


# In[7]:


# Print the object coloumns 
X_data.select_dtypes(object)


# In[8]:


# Print the class Categories
X_data["Class"].unique()


# In[9]:


# Print the customer type Categories
X_data["Customer Type"].unique()


# In[10]:


# transforming categorical coloumns into integers
X_data["Class"].replace(to_replace=['Eco Plus', 'Business', 'Eco'], value=[1,2,0], inplace=True)
X_data["Customer Type"].replace(to_replace=['Loyal Customer', 'disloyal Customer'], value=[1,-1], inplace=True)


# In[11]:


# Selecting Numerical Coloumns
numerical_cols = list(X_data.select_dtypes(int).columns)


# In[12]:


# Plotting Covarience matrix
fig, axis = plt.subplots(figsize=(20,18))
correlation_matrix = X_data.loc[:,numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt='.2f', ax=axis)
plt.show()


# In[13]:


# histogram plots 
X_data.loc[:,numerical_cols].hist(figsize=(20, 18))
plt.show()


# In[14]:


# One hot encoding 
cat_cols = ["Type of Travel","Gender"]
for col in cat_cols:
    dummies = pd.get_dummies(X_data[col])
    X_data = pd.concat([X_data, dummies], axis=1)
    X_data = X_data.drop([col], axis=1)
X_data.loc[:,X_data.select_dtypes(bool).columns] = X_data.select_dtypes(bool).astype(int)


# In[15]:


# Transforming labels into integers 
y_data.replace(to_replace=['neutral or dissatisfied', 'satisfied'], value=[0,1], inplace=True)


# In[16]:


# # Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=21, stratify=y_data)
for dataset in [X_train, X_test, y_train, y_test]:
    dataset.reset_index(drop=True, inplace=True)


# In[17]:


# Scale the data 
scaler = StandardScaler()
Xs_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
Xs_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have some dataset X and y
# X: Features, y: Labels

# Example of splitting data into train and test
features_train_all, features_test_all, labels_train_all, labels_test_all = train_test_split(X_test,y_test, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
features_train_all_std = scaler.fit_transform(features_train_all)
features_test_all_std = scaler.transform(features_test_all)


# In[19]:


Xs_train.describe()


# In[20]:


# funtion to calculate Metrics
def calc_metrics(confusion_matrix):
    TP, FN = confusion_matrix[0][0], confusion_matrix[0][1]
    FP, TN = confusion_matrix[1][0], confusion_matrix[1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    FNR = FN / (TP + FN)
    Precision = TP / (TP + FP)
    F1_measure = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error_rate = (FP + FN) / (TP + FP + FN + TN)
    BACC = (TPR + TNR) / 2
    TSS = TPR - FPR
    HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    metrics = [TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS]
    return metrics


# In[21]:


# Train the model and return the Metrics
def get_metrics(model, X_train, X_test, y_train, y_test, LSTM_flag):
    metrics = []
    if LSTM_flag == 1:
        Xtrain, Xtest, ytrain, ytest = map(np.array, [X_train, X_test, y_train, y_test])
        shape = Xtrain.shape
        Xtrain_reshaped = Xtrain.reshape(len(Xtrain), shape[1], 1)
        Xtest_reshaped = Xtest.reshape(len(Xtest), shape[1], 1)
        model.fit(Xtrain_reshaped, ytrain, epochs=50,validation_data=(Xtest_reshaped, ytest), verbose=0)
        lstm_scores = model.evaluate(Xtest_reshaped, ytest, verbose=0)
        predict_prob = model.predict(Xtest_reshaped)
        pred_labels = predict_prob > 0.5
        pred_labels_1 = pred_labels.astype(int)
        matrix = confusion_matrix(ytest, pred_labels_1, labels=[1, 0])
        lstm_brier_score = brier_score_loss(ytest, predict_prob)
        lstm_roc_auc = roc_auc_score(ytest, predict_prob)
        metrics.extend(calc_metrics(matrix))
        metrics.extend([lstm_brier_score, lstm_roc_auc, lstm_scores[1]])
    elif LSTM_flag == 0:
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(y_test, predicted, labels=[1, 0])
        model_brier_score = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
        model_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        metrics.extend(calc_metrics(matrix))
        metrics.extend([model_brier_score, model_roc_auc, model.score(X_test, y_test)])
    return metrics


# In[22]:


# Parameter tuning for KNN Algorithm

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Define the KNN model
knn = KNeighborsClassifier()
# Use only a subset  of the training data for hyperparameter tuning beacuse the data is huge and required more time for tuning 
X_train_subset = X_train.sample(frac=0.1, random_state=42)
y_train_subset = y_train[X_train_subset.index]

# Set up the parameter grid with fewer values
param_dist_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree']
}

# Perform randomized search (instead of grid search) with 5 iterations and 10-fold cross-validation
random_search_knn = RandomizedSearchCV(knn, param_dist_knn, n_iter=5, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
random_search_knn.fit(X_train_subset, y_train_subset)

# Print the best parameters and best score
print("Best KNN parameters:", random_search_knn.best_params_)
print("Best KNN score:", random_search_knn.best_score_)


# In[23]:


# Parameter tuning for Random Forest

from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42)
# Use only a subset  of the training data for hyperparameter tuning beacuse the data is huge and required more time for tuning 
X_train_subset = X_train.sample(frac=0.1, random_state=42)
y_train_subset = y_train[X_train_subset.index]

# Set up the parameter grid with fewer values
param_dist_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform randomized search with 5 iterations and 10-fold cross-validation
random_search_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=5, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
random_search_rf.fit(X_train_subset, y_train_subset)

# Print the best parameters and best score
print("Best Random Forest parameters:", random_search_rf.best_params_)
print("Best Random Forest score:", random_search_rf.best_score_)


# In[24]:


# Parameter tuning for SVM

from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assuming your data is already loaded as X_train, y_train

# Scale the data (SVM is sensitive to the scale of data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use only a subset  of the training data for hyperparameter tuning beacuse the data is huge and required more time for tuning 
X_train_subset = X_train.sample(frac=0.1, random_state=42)  
y_train_subset = y_train[X_train_subset.index]

# Initialize LinearSVC for linear classification problems
svm = LinearSVC(C=1.0, max_iter=1000, dual=False)  # Remove n_jobs

# Define a smaller parameter grid for hyperparameter tuning
param_dist = {
    'C': [0.1, 1.0, 10.0],  # Fewer values for C
}

# Use RandomizedSearchCV with fewer iterations to reduce time
random_search = RandomizedSearchCV(svm, param_dist, n_iter=5, cv=10, scoring='accuracy', verbose=1)

# Fit the model
random_search.fit(X_train_subset, y_train_subset)

# Print the best parameters found during the search
print(f"Best Parameters: {random_search.best_params_}")


# In[25]:


from sklearn.model_selection import StratifiedKFold

# Define Stratified K-Fold cross-validator
cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)


# In[26]:


# Compare Classifiers using 10-Fold Stratified Cross-Validation

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_auc_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define Stratified K-Fold cross-validator
cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

# Metric columns
metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision', 
                 'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS', 
                 'Brier_score', 'AUC', 'Acc_by_package_fn']

# Initialize metrics lists for each algorithm
knn_metrics_list, rf_metrics_list, svm_metrics_list, lstm_metrics_list = [], [], [], []

# Set up parameter for SVM
C = 1.0

# Assuming random_search_knn and random_search_rf have been already defined, here is how to use them
# 10 iterations of 10-fold cross-validation
for iter_num, (train_index, test_index) in enumerate(cv_stratified.split(features_train_all_std, labels_train_all), start=1):
   
   # Get KNN best parameters from random search (assuming you already performed RandomizedSearchCV or GridSearchCV)
   knn_params = random_search_knn.best_params_
   
   # KNN Model with correct parameters
   knn_model = KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'],
                                    weights=knn_params['weights'],
                                    algorithm=knn_params['algorithm'])
   
   # Random Forest Model (assuming random_search_rf.best_params_ works similarly)
   rf_params = random_search_rf.best_params_
   rf_model = RandomForestClassifier(min_samples_split=rf_params['min_samples_split'])
   
   # SVM Classifier Model
   svm_model = SVC(C=C, kernel='linear', probability=True)
   
   # LSTM Model
   lstm_model = Sequential()
   lstm_model.add(LSTM(64, activation='relu', input_shape=(8, 1), return_sequences=False))  # Correct input shape
   lstm_model.add(Dense(1, activation='sigmoid'))
   lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   
   # Split data into training and testing sets
   # Convert numpy arrays to pandas DataFrame/Series if they are numpy arrays
   features_train_all_std = pd.DataFrame(features_train_all_std)  # Convert features to DataFrame
   labels_train_all = pd.Series(labels_train_all)  # Convert labels to Series

   features_train, features_test = features_train_all_std.iloc[train_index, :], features_train_all_std.iloc[test_index, :]
   labels_train, labels_test = labels_train_all.iloc[train_index], labels_train_all.iloc[test_index]  # Use iloc for labels
   
   # Get metrics for each algorithm
   knn_metrics = get_metrics(knn_model, features_train, features_test, labels_train, labels_test, 0)
   rf_metrics = get_metrics(rf_model, features_train, features_test, labels_train, labels_test, 0)
   svm_metrics = get_metrics(svm_model, features_train, features_test, labels_train, labels_test, 0)
   lstm_metrics = get_metrics(lstm_model, features_train, features_test, labels_train, labels_test, 1)
   
   # Append metrics to respective lists
   knn_metrics_list.append(knn_metrics)
   rf_metrics_list.append(rf_metrics)
   svm_metrics_list.append(svm_metrics)
   lstm_metrics_list.append(lstm_metrics)
   
   # Create a DataFrame for all metrics in this iteration
   metrics_all_df = pd.DataFrame([knn_metrics, rf_metrics, svm_metrics, lstm_metrics],
                                 columns=metric_columns, index=['KNN', 'RF', 'SVM', 'LSTM'])
   
   # Display metrics for all algorithms in this iteration
   print('\nIteration {}: \n'.format(iter_num))
   print('----- Metrics for all Algorithms in Iteration {} -----\n'.format(iter_num))
   print(metrics_all_df.round(decimals=2).T)
   print('\n')


# In[27]:


# Initialize Metric Index for Iterations

metric_index_df = ['iter1', 'iter2', 'iter3', 'iter4', 'iter5', 'iter6', 'iter7', 'iter8', 'iter9', 'iter10']

knn_metrics_df = pd.DataFrame(knn_metrics_list, columns=metric_columns, index=metric_index_df)
rf_metrics_df = pd.DataFrame(rf_metrics_list, columns=metric_columns, index=metric_index_df)
svm_metrics_df = pd.DataFrame(svm_metrics_list, columns=metric_columns, index=metric_index_df)
lstm_metrics_df = pd.DataFrame(lstm_metrics_list, columns=metric_columns, index=metric_index_df)

for i, metrics_df in enumerate([knn_metrics_df, rf_metrics_df, svm_metrics_df, lstm_metrics_df], start=1):
    print('\nMetrics for Algorithm {}:\n'.format(i))
    print(metrics_df.round(decimals=2).T)
    print('\n')


# In[28]:


# Calculate the average metrics for each algorithm

import pandas as pd

# Define the metric columns
metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision', 
                  'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS', 
                  'Brier_score', 'AUC', 'Acc_by_package_fn']

# Initialize DataFrames to collect metrics for each algorithm
knn_metrics_df = pd.DataFrame(columns=metric_columns)
rf_metrics_df = pd.DataFrame(columns=metric_columns)
svm_metrics_df = pd.DataFrame(columns=metric_columns)
lstm_metrics_df = pd.DataFrame(columns=metric_columns)

# Assuming you already have the cross-validation loop here
for iter_num, (train_index, test_index) in enumerate(cv_stratified.split(features_train_all_std, labels_train_all), start=1):
    
    # Your model training and metric collection code here...
    
    # After getting metrics for each model (knn_metrics, rf_metrics, svm_metrics, lstm_metrics)
    
    # Use pd.concat to append metrics as a new row in the DataFrame
    knn_metrics_df = pd.concat([knn_metrics_df, pd.Series(knn_metrics, index=metric_columns).to_frame().T], ignore_index=True)
    rf_metrics_df = pd.concat([rf_metrics_df, pd.Series(rf_metrics, index=metric_columns).to_frame().T], ignore_index=True)
    svm_metrics_df = pd.concat([svm_metrics_df, pd.Series(svm_metrics, index=metric_columns).to_frame().T], ignore_index=True)
    lstm_metrics_df = pd.concat([lstm_metrics_df, pd.Series(lstm_metrics, index=metric_columns).to_frame().T], ignore_index=True)

# After collecting all the metrics for each model across all iterations
# Calculate the average of each metric for each algorithm
knn_avg_df = knn_metrics_df.mean()
rf_avg_df = rf_metrics_df.mean()
svm_avg_df = svm_metrics_df.mean()
lstm_avg_df = lstm_metrics_df.mean()

# Create a DataFrame with the average performance for each algorithm
avg_performance_df = pd.DataFrame({'KNN': knn_avg_df, 'RF': rf_avg_df, 'SVM': svm_avg_df, 'LSTM': lstm_avg_df}, index=metric_columns)

# Display the average performance for each algorithm
print(avg_performance_df.round(decimals=2))


# In[29]:


#Evaluating the performance of various algorithms by comparing their ROC curves and AUC scores on the test dataset.

# Implementing roc curves and AOC Score for KNN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for KNN
knn_probs = knn_model.predict_proba(features_test)[:, 1]  # Probability for class 1

# Calculate ROC curve for KNN
fpr_knn, tpr_knn, _ = roc_curve(labels_test, knn_probs)

# Calculate AUC for KNN
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curve for KNN
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN (AUC = {:.2f})'.format(roc_auc_knn))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Chance level
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC for KNN
print(f"KNN AUC: {roc_auc_knn:.2f}")


# In[30]:


# Implementing roc curves and AOC Score for Random Forest

# Get predicted probabilities for Random Forest
rf_probs = rf_model.predict_proba(features_test)[:, 1]  # Probability for class 1

# Calculate ROC curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(labels_test, rf_probs)

# Calculate AUC for Random Forest
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curve for Random Forest
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Chance level
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC for Random Forest
print(f"Random Forest AUC: {roc_auc_rf:.2f}")


# In[31]:


# Implementing roc curves and AOC Score for SVM

# Get predicted probabilities for SVM
svm_probs = svm_model.predict_proba(features_test)[:, 1]  # Probability for class 1

# Calculate ROC curve for SVM
fpr_svm, tpr_svm, _ = roc_curve(labels_test, svm_probs)

# Calculate AUC for SVM
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curve for SVM
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM (AUC = {:.2f})'.format(roc_auc_svm))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Chance level
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC for SVM
print(f"SVM AUC: {roc_auc_svm:.2f}")


# In[32]:


# # Implementing roc curves and AOC Score for LSTM

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Split the data once (you can also use train_test_split here, but assuming you're using your pre-split data)
# Use indices or a direct split for training and testing data
train_size = int(0.8 * len(features_train_all_std))  # 80% for training, 20% for testing
features_train = features_train_all_std[:train_size]
labels_train = labels_train_all[:train_size]
features_test = features_train_all_std[train_size:]
labels_test = labels_train_all[train_size:]

# Reshape the data for LSTM (3D format)
features_train_lstm = features_train.values.reshape(features_train.shape[0], features_train.shape[1], 1)
features_test_lstm = features_test.values.reshape(features_test.shape[0], features_test.shape[1], 1)

# Initialize your LSTM model (Ensure it's correctly built as in the previous steps)
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(features_train_lstm.shape[1], 1), return_sequences=False))  # Shape (samples, features, 1)
lstm_model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
lstm_model.fit(features_train_lstm, labels_train, epochs=10, batch_size=32, verbose=1)

# Get predicted probabilities for LSTM (for binary classification, this is the probability for class 1)
lstm_probs = lstm_model.predict(features_test_lstm)

# Calculate ROC curve
fpr_lstm, tpr_lstm, _ = roc_curve(labels_test, lstm_probs)

# Calculate AUC for LSTM
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot ROC curve for LSTM
plt.figure(figsize=(8, 6))
plt.plot(fpr_lstm, tpr_lstm, color='purple', lw=2, label='LSTM (AUC = {:.2f})'.format(roc_auc_lstm))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Chance level
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LSTM ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC for LSTM
print(f"LSTM AUC: {roc_auc_lstm:.2f}")


# In[33]:


# Plotting ROC Curves for All Models 

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize models
knn_model = KNeighborsClassifier(n_neighbors=5)  # Replace with best parameters
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Replace with best parameters
svm_model = SVC(kernel='linear', probability=True, random_state=42)  # SVM with probability output

# Initialize LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(features_train_lstm.shape[1], 1), return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train models
knn_model.fit(features_train, labels_train)
rf_model.fit(features_train, labels_train)
svm_model.fit(features_train, labels_train)
lstm_model.fit(features_train_lstm, labels_train, epochs=10, batch_size=32, verbose=0)

# Get predicted probabilities for each model
knn_probs = knn_model.predict_proba(features_test)[:, 1]  # Probability for class 1
rf_probs = rf_model.predict_proba(features_test)[:, 1]  # Probability for class 1
svm_probs = svm_model.predict_proba(features_test)[:, 1]  # Probability for class 1
lstm_probs = lstm_model.predict(features_test_lstm)  # Probability for class 1

# Calculate ROC curve for each model
fpr_knn, tpr_knn, _ = roc_curve(labels_test, knn_probs)
fpr_rf, tpr_rf, _ = roc_curve(labels_test, rf_probs)
fpr_svm, tpr_svm, _ = roc_curve(labels_test, svm_probs)
fpr_lstm, tpr_lstm, _ = roc_curve(labels_test, lstm_probs)

# Calculate AUC for each model
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot ROC curve for all models
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN (AUC = {:.2f})'.format(roc_auc_knn))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM (AUC = {:.2f})'.format(roc_auc_svm))
plt.plot(fpr_lstm, tpr_lstm, color='purple', lw=2, label='LSTM (AUC = {:.2f})'.format(roc_auc_lstm))

# Plot chance line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

# Formatting the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiple Models')
plt.legend(loc='lower right')
plt.show()


# In[34]:


# Comparing All Models

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize models
knn_model = KNeighborsClassifier(n_neighbors=5)  # Replace with best parameters
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Replace with best parameters
svm_model = SVC(kernel='linear', probability=True, random_state=42)  # SVM with probability output

# Initialize LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(features_train_lstm.shape[1], 1), return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train models
knn_model.fit(features_train, labels_train)
rf_model.fit(features_train, labels_train)
svm_model.fit(features_train, labels_train)
lstm_model.fit(features_train_lstm, labels_train, epochs=10, batch_size=32, verbose=0)

# Get predicted probabilities for each model
knn_probs = knn_model.predict_proba(features_test)[:, 1]  # Probability for class 1
rf_probs = rf_model.predict_proba(features_test)[:, 1]  # Probability for class 1
svm_probs = svm_model.predict_proba(features_test)[:, 1]  # Probability for class 1
lstm_probs = lstm_model.predict(features_test_lstm)  # Probability for class 1

# Calculate ROC curve for each model
fpr_knn, tpr_knn, _ = roc_curve(labels_test, knn_probs)
fpr_rf, tpr_rf, _ = roc_curve(labels_test, rf_probs)
fpr_svm, tpr_svm, _ = roc_curve(labels_test, svm_probs)
fpr_lstm, tpr_lstm, _ = roc_curve(labels_test, lstm_probs)

# Calculate AUC for each model
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot ROC curve for all models
plt.figure(figsize=(8, 6))

# Plot each model's ROC curve with respective AUC score
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN (AUC = {:.2f})'.format(roc_auc_knn))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM (AUC = {:.2f})'.format(roc_auc_svm))
plt.plot(fpr_lstm, tpr_lstm, color='purple', lw=2, label='LSTM (AUC = {:.2f})'.format(roc_auc_lstm))

# Plot chance line (diagonal line)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

# Formatting the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Multiple Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[35]:


# Assuming 'avg_performance_df' has already been calculated from the earlier step

print(avg_performance_df.round(decimals=2))
print('\n')


# In[ ]:




