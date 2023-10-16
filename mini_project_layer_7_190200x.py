# -*- coding: utf-8 -*-
"""Mini_Project_Layer_7_190200X.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QZmAS6UcUoK97Xje-JYqnDM_tfsidw_8

#Layer 7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pip install seaborn

names=[]
# Assign column names to the dataset
for i in range(1,769):
 names.append("feature_"+str(i))
labels=["label_1","label_2","label_3","label_4"]
names+=labels

# Read in the dataset
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L7/train.csv')
valid_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L7/valid.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L7/test.csv')

train_df.shape

train_df.isnull().sum()

valid_df.isnull().sum()

test_df.isnull().sum()
test_df.head()

from sklearn.preprocessing import RobustScaler

x_train ={}
y_train ={}
x_valid ={}
y_valid ={}
x_test = {}



for label in labels:
  scaler = RobustScaler()
  df_t = train_df
  df_v = valid_df
  if label == 'label_2':
    df_t = train_df.dropna()
    df_v = valid_df.dropna()

  x_train[label] = scaler.fit_transform(df_t.drop(labels, axis=1))
  y_train[label] = df_t[label]
  x_valid[label] = scaler.transform(df_v.drop(labels, axis=1))
  y_valid[label] = df_v[label]
  x_test[label] = scaler.transform(test_df.drop(['ID'], axis=1))

"""##Label 1"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=0.97)

# Fit PCA on your training data
x_train_pca = pca.fit_transform(x_train['label_1'])

# Transform your validation and test data using the same PCA model
x_valid_pca = pca.transform(x_valid['label_1'])
x_test_pca = pca.transform(x_test['label_1'])  # If you have a test set

# Now, you can use x_train_pca, x_valid_pca, and x_test_pca as your reduced-dimensional feature vectors for modeling.
x_train_pca.shape

import seaborn as sns
import matplotlib.pyplot as plt


# Calculate the correlation matrix for the original x_train data
corr_matrix = pd.DataFrame(x_train['label_1']).corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix for x_train label_1")
plt.show()

# Calculate the correlation matrix
corr_matrix = pd.DataFrame(x_train_pca).corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix for x_train_pca")
plt.show()

"""###KNN

####Original
"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train['label_1'], y_train['label_1'])

y_pred = classifier.predict(x_valid['label_1'])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""####After PCA"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_pca, y_train['label_1'])

y_pred = classifier.predict(x_valid_pca)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""###SVM

####Original
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint

svm_classifier = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier.fit(x_train['label_1'], y_train['label_1'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid['label_1'])

# Evaluate model performance
print(accuracy_score(y_valid['label_1'], y_pred))

"""####After PCA"""

svm_classifier = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier.fit(x_train_pca, y_train['label_1'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(accuracy_score(y_valid['label_1'], y_pred))

"""####After PCA + HP tuning"""

# svm_classifier = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') #=0.97
svm_classifier = SVC(class_weight='balanced' ,C=100, degree=3,  kernel='rbf') #0.98
svm_classifier.fit(x_train_pca, y_train['label_1'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(classification_report(y_valid['label_1'], y_pred))

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier, x_train_pca, y_train['label_1'], cv=cv, scoring='accuracy')

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define your SVM classifier with the desired hyperparameters
svm_classifier = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') #=0.97 -> 0.95
# svm_classifier = SVC(class_weight='balanced' ,C=100, degree=3,  kernel='rbf') #0.98 -> 0.96

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier, x_train_pca, y_train['label_1'], cv=cv, scoring='accuracy')

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Define the hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1,10,100],  # Adjust the range
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Experiment with different kernels
    'class_weight': ['balanced',None],  # Change class_weight options
    'gamma':['scale', 'auto'],
    'degree': [1,2,3,4]

}

# Create an SVM classifier
svm_classifier = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_classifier, param_grid, scoring='accuracy', cv=5,n_jobs=-1)
grid_search.fit(x_train_pca, y_train['label_1'])

# Get the best model and hyperparameters
best_svm_classifier = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the validation set
y_pred = best_svm_classifier.predict(x_valid_pca)

# Evaluate the model
print("Best Hyperparameters:", best_params)
print("Classification Report on Validation Data:")
print(classification_report(y_valid['label_1'], y_pred))

param_distributions = {
    'C': [0.001,0.1,1,10,100],#uniform(0.001,100),  # Continuous uniform distribution for 'C'
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Experiment with different kernels
    # 'class_weight': ['balanced', None],  # Change class_weight options
    # 'gamma':['scale', 'auto'],
    'degree': [1,2,3,4]
}

# Create an SVM classifier
svm_classifier = SVC()

# Example with Randomized Search
random_search = RandomizedSearchCV(svm_classifier, param_distributions, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(x_train_pca, y_train['label_1'])

# Get the best hyperparameters and model
best_svm_classifier = random_search.best_estimator_
best_params = random_search.best_params_

# Make predictions on the validation data
y_pred = best_svm_classifier.predict(x_valid_pca)

# Evaluate the model
print("Best Hyperparameters:", best_params)
print("Classification Report on Validation Data:")
print(classification_report(y_valid['label_1'], y_pred))

label_1_pred = svm_classifier.predict(x_test_pca)

"""###Random Forest

####Orginal
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train['label_1'], y_train['label_1'])

# Make predictions on the test data
y_pred = model.predict(x_valid['label_1'])

# Evaluate model performance
print(classification_report(y_valid['label_1'], y_pred))

"""###XGBoost"""

pip install xgboost

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize the XGBoost classifier
model = xgb.XGBClassifier()
y_train_shifted = y_train['label_1'] - 1

# Train the model
model.fit(x_train['label_1'],y_train_shifted )

y_pred = model.predict(x_valid['label_1'])

accuracy = accuracy_score(y_valid['label_1'], y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Additional evaluation metrics
print(classification_report(y_valid['label_1'], y_pred))

"""###Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(x_train['label_1'], y_train['label_1'])

y_pred = model.predict(x_valid['label_1'])

accuracy = accuracy_score(y_valid['label_1'], y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Additional evaluation metrics
print(classification_report(y_valid['label_1'], y_pred))

"""##Label 2"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=0.97)

# Fit PCA on your training data
x_train_pca = pca.fit_transform(x_train['label_2'])

# Transform your validation and test data using the same PCA model
x_valid_pca = pca.transform(x_valid['label_2'])
x_test_pca = pca.transform(x_test['label_2'])  # If you have a test set

# Now, you can use x_train_pca, x_valid_pca, and x_test_pca as your reduced-dimensional feature vectors for modeling.
x_train_pca.shape

"""###KNN

####Original
"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train['label_2'], y_train['label_2'])

y_pred = classifier.predict(x_valid['label_2'])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_2'], y_pred))
print(classification_report(y_valid['label_2'], y_pred))

"""####After PCA"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_pca, y_train['label_2'])

y_pred = classifier.predict(x_valid_pca)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_2'], y_pred))
print(classification_report(y_valid['label_2'], y_pred))

"""####KNN with tuning"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

param_grid = {
    'n_neighbors': [1, 2,3,4, 5, 6,7,8, 9,10]  # You can adjust this range as needed
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

grid_search.fit(x_train['label_2'], y_train['label_2'])
best_k = grid_search.best_params_['n_neighbors']

final_classifier = KNeighborsClassifier(n_neighbors=best_k)
final_classifier.fit(x_train['label_2'], y_train['label_2'])

y_pred = final_classifier.predict(x_valid['label_2'])

print('Best k:',best_k)
print(confusion_matrix(y_valid['label_2'], y_pred))
print(classification_report(y_valid['label_2'], y_pred))

"""####PCA+ KNN tuning"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

param_grid = {
    'leaf_size' : list(range(1,50)),
    'n_neighbors' :list(range(1,30)),
    # 'n_neighbors': [1, 2,3,4, 5, 6,7,8, 9,10],  # You can adjust this range as needed
    'p': [1, 2]
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

grid_search.fit(x_train_pca, y_train['label_2'])
best_k = grid_search.best_params_['n_neighbors','leaf_size', 'p']

final_classifier = KNeighborsClassifier(n_neighbors=best_k)
final_classifier.fit(x_train_pca, y_train['label_2'])

y_pred = final_classifier.predict(x_valid_pca)

print('Best k:',best_k)
print(confusion_matrix(y_valid['label_2'], y_pred))
print(classification_report(y_valid['label_2'], y_pred))

print('Best leaf_size:', grid_search.best_estimator_.get_params()['leaf_size'])
print('Best p:', grid_search.best_estimator_.get_params()['p'])
print('Best n_neighbors:', grid_search.best_estimator_.get_params()['n_neighbors'])

"""###SVM

####Original
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint

svm_classifier = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier.fit(x_train['label_2'], y_train['label_2'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid['label_2'])

# Evaluate model performance
print(classification_report(y_valid['label_2'], y_pred))

"""####After PCA"""

svm_classifier = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier.fit(x_train_pca, y_train['label_2'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(classification_report(y_valid['label_2'], y_pred))

"""####After PCA + HP tuning"""

svm_classifier = SVC(class_weight='balanced' ,C=10, degree=1, gamma=0.1, kernel='poly')
svm_classifier.fit(x_train_pca, y_train['label_2'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(classification_report(y_valid['label_2'], y_pred))

# Define the hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1,10,100],  # Adjust the range
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Experiment with different kernels
    'class_weight': ['balanced',None],  # Change class_weight options
    'gamma':['scale', 'auto'],
    'degree': [1,2,3,4]

}

# Create an SVM classifier
svm_classifier = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_classifier, param_grid, scoring='accuracy', cv=5,n_jobs=-1)
grid_search.fit(x_train_pca, y_train['label_1'])

# Get the best model and hyperparameters
best_svm_classifier = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the validation set
y_pred = best_svm_classifier.predict(x_valid_pca)

# Evaluate the model
print("Best Hyperparameters:", best_params)
print("Classification Report on Validation Data:")
print(classification_report(y_valid['label_1'], y_pred))

param_distributions = {
    'C': [0.1,1,10,100],#uniform(0.001,100),  # Continuous uniform distribution for 'C'
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Experiment with different kernels
    # 'class_weight': ['balanced', None],  # Change class_weight options
    # 'gamma':['scale', 'auto'],
    'degree': [1,2,3,4]
}

# Create an SVM classifier
svm_classifier = SVC()

# Example with Randomized Search
random_search = RandomizedSearchCV(svm_classifier, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(x_train_pca, y_train['label_2'])


# Get the best hyperparameters and model
best_svm_classifier = random_search.best_estimator_
best_params = random_search.best_params_

# Make predictions on the validation data
y_pred = best_svm_classifier.predict(x_valid_pca)

# Evaluate the model
print("Best Hyperparameters:", best_params)
print("Classification Report on Validation Data:")
print(classification_report(y_valid['label_2'], y_pred))

label_1_pred = svm_classifier.predict(x_test_pca)

"""###Random Forest

####Orginal
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier  # You can replace this with any other model

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train['label_2'], y_train['label_2'])  # Assuming x_train and y_train are your feature and target data

# Make predictions on the test data
y_pred = model.predict(x_valid['label_2'])

# Evaluate model performance
print(classification_report(y_valid['label_2'], y_pred))

"""##Label 3"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=0.97)

# Fit PCA on your training data
x_train_pca = pca.fit_transform(x_train['label_3'])

# Transform your validation and test data using the same PCA model
x_valid_pca = pca.transform(x_valid['label_3'])
x_test_pca = pca.transform(x_test['label_3'])  # If you have a test set

# Now, you can use x_train_pca, x_valid_pca, and x_test_pca as your reduced-dimensional feature vectors for modeling.
x_train_pca.shape

"""###KNN

####Original
"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train['label_3'], y_train['label_3'])

y_pred = classifier.predict(x_valid['label_3'])

from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
print(confusion_matrix(y_valid['label_3'], y_pred))
print(accuracy_score(y_valid['label_3'], y_pred))

"""####After PCA"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_pca, y_train['label_3'])

y_pred = classifier.predict(x_valid_pca)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_3'], y_pred))
print(accuracy_score(y_valid['label_3'], y_pred))

"""###SVM

####Original
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint

svm_classifier = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier.fit(x_train['label_3'], y_train['label_3'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid['label_3'])

# Evaluate model performance
print(accuracy_score(y_valid['label_3'], y_pred))

"""####After PCA"""

svm_classifier = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier.fit(x_train_pca, y_train['label_3'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(accuracy_score(y_valid['label_3'], y_pred))

from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define your SVM classifier with the desired hyperparameters
svm_classifier = SVC(class_weight='balanced', C=1, kernel='rbf')

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier, x_train['label_3'], y_train['label_3'], cv=cv, scoring='accuracy')

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

"""####After PCA + HP tuning"""

# svm_classifier = SVC(class_weight='balanced' ,C=1, degree=4, gamma='auto', kernel='linear') #0.9918478
svm_classifier = SVC(class_weight='balanced' ,C=50 ,kernel='rbf') #0.998641
#Best Hyperparameters: {'C': 30.46237691733707, 'degree': 1, 'kernel': 'poly'} =0.99
svm_classifier.fit(x_train_pca, y_train['label_3'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(accuracy_score(y_valid['label_3'], y_pred))

# Define the hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1,10,100],  # Adjust the range
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Experiment with different kernels
    'class_weight': ['balanced',None],  # Change class_weight options
    'gamma':['scale', 'auto'],
    'degree': [1,2,3,4]

}

# Create an SVM classifier
svm_classifier = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_classifier, param_grid, scoring='accuracy', cv=5,n_jobs=-1)
grid_search.fit(x_train_pca, y_train['label_1'])

# Get the best model and hyperparameters
best_svm_classifier = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the validation set
y_pred = best_svm_classifier.predict(x_valid_pca)

# Evaluate the model
print("Best Hyperparameters:", best_params)
print("Classification Report on Validation Data:")
print(classification_report(y_valid['label_1'], y_pred))

param_distributions = {
    'C': uniform(0.001,100),  # Continuous uniform distribution for 'C'
    'kernel': ['linear','poly'],  # Experiment with different kernels
    # 'class_weight': ['balanced', None],  # Change class_weight options
    # 'gamma':['scale', 'auto'],
    'degree': [1,2,3,4]
}

# Create an SVM classifier
svm_classifier = SVC()

# Example with Randomized Search
random_search = RandomizedSearchCV(svm_classifier, param_distributions, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(x_train_pca, y_train['label_3'])


# Get the best hyperparameters and model
best_svm_classifier = random_search.best_estimator_
best_params = random_search.best_params_

# Make predictions on the validation data
y_pred = best_svm_classifier.predict(x_valid_pca)

# Evaluate the model
print("Best Hyperparameters:", best_params)
print("Classification Report on Validation Data:")
print(classification_report(y_valid['label_3'], y_pred))

"""##Label 4"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=0.97)

# Fit PCA on your training data
x_train_pca = pca.fit_transform(x_train['label_4'])

# Transform your validation and test data using the same PCA model
x_valid_pca = pca.transform(x_valid['label_4'])
x_test_pca = pca.transform(x_test['label_4'])  # If you have a test set

# Now, you can use x_train_pca, x_valid_pca, and x_test_pca as your reduced-dimensional feature vectors for modeling.
x_train_pca.shape
