from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!
    clf = GaussianNB()
    return clf.fit(features_train, labels_train)


def NBAccuracy(features_train, labels_train, features_test, labels_test):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    accuracy = clf.score(features_test, labels_test)
    return accuracy