import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

clf2 = SVC(kernel='rbf', C=1.0)
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)

clf3 = SVC(kernel='rbf', C=100.0)
clf3.fit(features_train, labels_train)
pred3 = clf3.predict(features_test)

prettyPicture(clf3, features_test, labels_test)
plt.show()

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data



#### store your predictions in a list named pred





from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc