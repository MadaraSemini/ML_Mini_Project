# -*- coding: utf-8 -*-
"""Mini_project_L12_Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MOOyLh9xkxGgvFS3Iv3JQ-dQiJfp9ksw

#Layer 12
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

names=[]
# Assign column names to the dataset
for i in range(1,769):
 names.append("feature_"+str(i))
labels=["label_1","label_2","label_3","label_4"]
names+=labels

# Read in the dataset
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/layer_12_train.csv')
valid_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/layer_12_valid.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/layer_12_test.csv')

from sklearn.preprocessing import RobustScaler

x_train ={}
y_train ={}
x_valid ={}
y_valid ={}
x_test = {}

df_t = train_df
df_v = valid_df

for label in labels:
  scaler = RobustScaler()
  df_t = train_df
  df_v = valid_df
  if label == 'label_2':
    print(label)
    df_t = train_df.dropna()
    df_v = valid_df.dropna()

  x_train[label] = scaler.fit_transform(df_t.drop(labels, axis=1))
  y_train[label] = df_t[label]
  x_valid[label] = scaler.transform(df_v.drop(labels, axis=1))
  y_valid[label] = df_v[label]
  x_test[label] = scaler.transform(test_df.drop(labels, axis=1))
  print(label,len(x_valid[label]))

valid_df= pd.DataFrame(x_valid['label_3'])
valid_df.shape

"""#Label 1

##PCA
"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=400)

# Fit PCA on your training data
x_train_pca_l1 = pca.fit_transform(x_train['label_1'])

# Transform your validation and test data using the same PCA model
x_valid_pca_l1 = pca.transform(x_valid['label_1'])
x_test_pca_l1 = pca.transform(x_test['label_1'])  # If you have a test set

# Now, you can use x_train_pca_l1, x_valid_pca_l1, and x_test_pca_l1 as your reduced-dimensional feature vectors for modeling.
x_train_pca_l1.shape

"""##SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_classifier_l1 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') #0.9160589060308556

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l1, x_train_pca_l1, y_train['label_1'], cv=cv, scoring='accuracy')
svm_classifier_l1.fit(x_train['label_1'], y_train['label_1'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_classifier_l1 = SVC(class_weight='balanced',C=100, gamma=0.001, kernel='rbf') #0.9359396914446003

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l1, x_train['label_1'], y_train['label_1'], cv=cv, scoring='accuracy')
svm_classifier_l1.fit(x_train['label_1'], y_train['label_1'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_classifier_l1 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') #0.9160589060308556

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l1, x_train_pca_l1, y_train['label_1'], cv=cv, scoring='accuracy')
svm_classifier_l1.fit(x_train_pca_l1, y_train['label_1'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

svm_classifier_l1 = SVC(class_weight='balanced',C=100, gamma=0.001, kernel='rbf') #
svm_classifier_l1.fit(x_train_pca_l1, y_train['label_1'])

# Make predictions on the test data
y_pred = svm_classifier_l1.predict(x_valid_pca_l1)

# Evaluate model performance
print(accuracy_score(y_valid['label_1'], y_pred))

label_1_pred = svm_classifier_l1.predict(x_valid_pca_l1)

data = {
    'L_12_label_1': label_1_pred
}

feature_df= pd.DataFrame(data)
filename = f'/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/Layer_12_label_1.csv'
feature_df.to_csv(filename,index=False)

"""#Label 2

##PCA
"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=400)

# Fit PCA on your training data
x_train_pca_l2 = pca.fit_transform(x_train['label_2'])

# Transform your validation and test data using the same PCA model
x_valid_pca_l2 = pca.transform(x_valid['label_2'])
x_test_pca_l2 = pca.transform(x_test['label_2'])  # If you have a test set

# Now, you can use x_train_pca_l2, x_valid_pca_l1, and x_test_pca_l1 as your reduced-dimensional feature vectors for modeling.
x_train_pca_l2.shape

"""##SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold

svm_classifier_l2_0 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly')
svm_classifier_l2_0.fit(x_train_pca_l2, y_train['label_2'])

# Make predictions on the test data
y_pred = svm_classifier_l2_0.predict(x_valid_pca_l2)

# Evaluate model performance
print(accuracy_score(y_valid['label_2'], y_pred))

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_classifier_l2 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') # 0.755777460770328

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l2, x_train_pca_l2, y_train['label_2'], cv=cv, scoring='accuracy')
svm_classifier_l2.fit(x_train_pca_l2, y_train['label_2'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# svm_classifier_l2 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') # 0.755777460770328
svm_classifier_l2 = SVC(class_weight='balanced' ,C=100, gamma=0.001, kernel='rbf') #0.8853780313837376

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l2, x_train['label_2'], y_train['label_2'], cv=cv, scoring='accuracy')
svm_classifier_l2.fit(x_train['label_2'], y_train['label_2'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

svm_classifier_l2_1 = SVC(C=100, gamma=0.001, kernel='rbf')
svm_classifier_l2_1.fit(x_train_pca_l2, y_train['label_2'])

# Make predictions on the test data
y_pred = svm_classifier_l2_1.predict(x_valid_pca_l2)

# Evaluate model performance
print(accuracy_score(y_valid['label_2'], y_pred))

label_2_pred = svm_classifier_l2_1.predict(x_valid_pca_l2)

data = {
    'L_12_label_2': label_2_pred
}

feature_df= pd.DataFrame(data)
filename = f'/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/Layer_12_label_2.csv'
feature_df.to_csv(filename,index=False)

"""#Label 3

##PCA
"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=400)

# Fit PCA on your training data
x_train_pca_l3 = pca.fit_transform(x_train['label_3'])

# Transform your validation and test data using the same PCA model
x_valid_pca_l3 = pca.transform(x_valid['label_3'])
x_test_pca_l3 = pca.transform(x_test['label_3'])  # If you have a test set

# Now, you can use x_train_pca_l1, x_valid_pca_l1, and x_test_pca_l1 as your reduced-dimensional feature vectors for modeling.
x_train_pca_l3.shape

"""##SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold

# svm_classifier_l3 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') #=0.97
svm_classifier_l3 = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier_l3.fit(x_train_pca_l3, y_train['label_3'])

# Make predictions on the test data
y_pred = svm_classifier_l3.predict(x_valid_pca_l3)

# Evaluate model performance
print(accuracy_score(y_valid['label_3'], y_pred))

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_classifier_l3 = SVC(class_weight='balanced' ,C=1, kernel='rbf') #0.9926176890156919

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l3, x_train['label_3'], y_train['label_3'], cv=cv, scoring='accuracy')
svm_classifier_l3.fit(x_train['label_3'], y_train['label_3'])

# Print the cross-validation scores and mean accuracy
# print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)

# svm_classifier_l3 = SVC(class_weight='balanced' ,C=1, kernel='rbf') #0.9926176890156919
svm_classifier_l3_1 = SVC(C=100, gamma=0.001, kernel='rbf')

# svm_classifier_l3 = SVC(class_weight='balanced' ,C=10, degree=1, gamma='auto', kernel='poly') #=0.97
svm_classifier_l3.fit(x_train_pca_l3, y_train['label_3'])

# Make predictions on the test data
y_pred = svm_classifier_l3.predict(x_valid_pca_l3)

# Evaluate model performance
print(accuracy_score(y_valid['label_3'], y_pred))

label_3_pred = svm_classifier_l3.predict(x_valid_pca_l3)

data = {
    'L_12_label_3': label_3_pred
}

feature_df= pd.DataFrame(data)
filename = f'/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/Layer_12_label_3.csv'
feature_df.to_csv(filename,index=False)

"""#Label 4

##PCA
"""

from sklearn.decomposition import PCA

# Instantiate PCA with a desired number of components (e.g., n_components=50)
pca = PCA(n_components=400)

# Fit PCA on your training data
x_train_pca_l4 = pca.fit_transform(x_train['label_4'])

# Transform your validation and test data using the same PCA model
x_valid_pca_l4 = pca.transform(x_valid['label_4'])
x_test_pca_l4 = pca.transform(x_test['label_4'])  # If you have a test set

# Now, you can use x_train_pca_l4, x_valid_pca_l1, and x_test_pca_l1 as your reduced-dimensional feature vectors for modeling.
x_train_pca_l4.shape

"""##SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold

svm_classifier_l4 = SVC(class_weight='balanced' ,C=1, kernel='rbf')
svm_classifier_l4.fit(x_train_pca_l4, y_train['label_4'])

# Make predictions on the test data
y_pred = svm_classifier_l4.predict(x_valid_pca_l4)

# Evaluate model performance
print(accuracy_score(y_valid['label_4'], y_pred))

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_classifier_l4 = SVC(class_weight='balanced' ,C=1, kernel='rbf') #0.8104493580599144

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l4, x_train_pca_l4, y_train['label_4'], cv=cv, scoring='accuracy')
svm_classifier_l4.fit(x_train_pca_l4, y_train['label_4'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Define the cross-validation strategy (e.g., StratifiedKFold with 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# svm_classifier_l4 = SVC(class_weight='balanced' ,C=1, kernel='rbf') #0.8104493580599144
svm_classifier_l4 = SVC(class_weight='balanced' ,C=100, gamma=0.001, kernel='rbf') #0.9529243937232525

# Perform cross-validation and calculate accuracy scores
scores = cross_val_score(svm_classifier_l4, x_train['label_4'], y_train['label_4'], cv=cv, scoring='accuracy')
svm_classifier_l4.fit(x_train['label_4'], y_train['label_4'])

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

label_4_pred = svm_classifier_l4.predict(x_valid_pca_l4)

data = {
    'L_12_label_4': label_4_pred
}

feature_df= pd.DataFrame(data)
filename = f'/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/Layer_12_label_4.csv'
feature_df.to_csv(filename,index=False)

"""# CSV Generate"""

data = {
    'ID': np.arange(1, len(label_1_pred) + 1),
    'label_1': label_1_pred,
    'label_2': label_2_pred,
    'label_3': label_3_pred,
    'label_4': label_4_pred
}

feature_df= pd.DataFrame(data)
filename = f'/content/drive/MyDrive/Colab Notebooks/Mini_project/DataSet_L12/190200X_Layer_12.2.csv'
feature_df.to_csv(filename,index=False)