# ML_Mini_Project
The classification model implementations for layer 7 and layer 12 of Wav2Vec transformers are included in this project.

## Layer 7 

### Label 1 
svm_classifier_l1 = SVC(class_weight=\'balanced\',C=100, degree=3, kernel=\'rbf\') #0.98

### Label 2
svm_classifier_l2 = SVC(class_weight=\'balanced\' ,C=1,kernel=\'rbf\') #0.9116847826086957 \
svm_classifier_l2 = SVC(class_weight=\'balanced\' ,C=100, kernel=\'rbf\', degree=4) #0.95

### Label 3
svm_classifier_l3 = SVC(class_weight=\'balanced\' ,C=50,kernel=\'rbf\') #0.998641304347826

### Label 4 
svm_classifier_l4 = SVC(class_weight=\'balanced\' ,C=100,degree=3, gamma=\'auto\', kernel=\'poly\') #0.9714673913043478\
svm_classifier_l4 = SVC(class_weight=\'balanced\' ,C=100, degree=4,kernel=\'rdf\') #0.9741847826086957

## Layer 12 

### Label 1 
svm_classifier_l1 = SVC(class_weight=\'balanced\',C=10, degree=1, gamma=\'auto\', kernel=\'poly\') #0.9160589060308556\
svm_classifier_l2 = SVC(class_weight=\'balanced\' ,C=10,degree=1, gamma=\'auto\', kernel=\'poly\') #0.755777460770328

### Label 3
svm_classifier_l3 = SVC(class_weight=\'balanced\' ,C=1,kernel=\'rbf\') #0.9926176890156919 \
Best Hyperparameters: {\'C\':8.819480326319178, \'class_weight\': None, \'kernel\': \'linear\'}0.981169757489301

### Label 4
svm_classifier_l4 = SVC(class_weight=\'balanced\' ,C=1,kernel=\'rbf\') #0.8104493580599144\
svm_classifier_l4 = SVC(class_weight='balanced' ,C=100, gamma=0.001, kernel='rbf') #0.9529243937232525
