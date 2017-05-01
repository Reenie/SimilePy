from sklearn import svm, neighbors, tree

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from IPython.display import Image
import pydotplus

import main

class Classifiers:

    classifier_names =  ["", "LDA", "SVM", "SGD", "KN", "DT", "RF", "MLP", "QDA"]

    def __init__(self):
        a=1

    #it returns y_pedict
    def run_classifier(self, x_train, y_train, x_test, classifier=1):
        s = Classifiers
        #y_pred = []
        if classifier==1:
            #print("LDA")
            return s.LDA_classifier(self, x_train, y_train, x_test, solver="lsqr", shrinkage=0.5, store_covariance=True)
        elif classifier == 2:
            #print("SVM")
            return s.SVM_classifier(self, x_train, y_train, x_test)
        elif classifier == 3:
            #print("SGD")
            return s.sgd_classifier(self, x_train, y_train, x_test, loss="hinge", penalty="l2")
        elif classifier == 4:
            #print("KN")
            return s.kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=15)
        elif classifier == 5:
            #print("DT")
            return s. DT_classifier(self, x_train, y_train, x_test)
        elif classifier == 6:
            #print("RF")
            return s.RF_classifier(self, x_train, y_train, x_test)
        elif classifier == 7:
            #print("MLP")
            return s.multiLayerPerceptron_classifier(self, x_train, y_train, x_test)
        elif classifier == 8:
            #print("QDA")
            return s.QDA_classifier(self, x_train, y_train, x_test)
        elif classifier == 9:
            print("NO CLASSIFIER")
        elif classifier == 10:
            print("NO CLASSIFIER")
        #return y_pred


    #8 Quadratic Discriminant Analysis
    def QDA_classifier(self, x_train, y_train, x_test):
        print('QDA')
        qda = QuadraticDiscriminantAnalysis(store_covariances=True)
        return qda.fit(x_train, y_train).predict(x_test)  #y_predict


    #1 Linear Discriminant Analysis
    def LDA_classifier(self, x_train, y_train, x_test, solver="lsqr", shrinkage=0.5, store_covariance=True):
        print('LDA')
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, store_covariance=store_covariance)
        # l = LinearDiscriminantAnalysis(solver='svd', n_components=2)
        # l = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.8, n_components=2)
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)
        return y_pred


    #2
    def SVM_classifier(self, x_train, y_train, x_test):
        print('SVM')
        #kernel = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed'
        #clf = svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto', coef0=0.0, shrinking=True,
        #              probability=False, tol=0.0001, cache_size=300, class_weight=None, verbose=False,
        #             max_iter=-1, decision_function_shape=None, random_state=None)
        #clf = svm.LinearSVC(class_weight='balanced', max_iter=10000)
        clf = svm.LinearSVC()
        #clf = svm.NuSVC()
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #3 Stochastic Gradient Descent
    def sgd_classifier(self, x_train, y_train, x_test, loss="hinge", penalty="l2"):
        print('SGD')
        clf = SGDClassifier(loss=loss, penalty=penalty)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #4
    def kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=15):
        print('KNN')
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #5 Decision Tree
    def DT_classifier(self, x_train, y_train, x_test):
        print('DT')
        #clf = tree.ExtraTreeClassifier(criterion='gini')
        clf = tree.DecisionTreeClassifier(criterion='gini')
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #Print Decision Tree
    def DT_classifier_printTree(self, x_train, y_train, feature_names, target_values, printFor='tenor', pdfName = "DT.pdf"):
        print('Print Decision Tree')
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)  # .transform(x_train)
        target_names = []
        if printFor=='tenor' or printFor=='gender':
            target_names = target_values
        else:  #for simile
            for n in target_values:
                target_names.append(str(int(n)))
        print(feature_names)
        #print(target_names)
        #print(x_train)
        #dot_data = tree.export_graphviz(clf, out_file=None,
        #                 feature_names=feature_names,
        #                 class_names=target_names,
        #                 filled=True, rounded=True,
        #                 special_characters=True)
        #dot_data = tree.export_graphviz(clf, out_file=None)
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(pdfName)
        graph = pydotplus.graph_from_dot_data(dot_data)
        Image(graph.create_png())


    #7
    def multiLayerPerceptron_classifier(self, x_train, y_train, x_test):
        print('MLP')
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1, max_iter=500)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #6 Random forest
    def RF_classifier(self, x_train, y_train, x_test):
        print('RF')
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features='auto', random_state=0)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred