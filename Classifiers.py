from sklearn import svm, neighbors, tree

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from IPython.display import Image
import pydotplus

import main


class Classifiers:

    classifier_names =  ["", "QDA", "LDA", "SVM", "SGD", "KN", "DT", "RF", "MLP"]

    def __init__(self):
        a=1


    #it returns y_pedict
    def run_classifier(self, x_train, y_train, x_test, classifier=1):
        s = Classifiers
        #y_pred = []
        if classifier==1:
            print("QDA")
            return s.QDA_classifier(self, x_train, y_train, x_test)
        elif classifier==2:
            print("LDA")
            return s.LDA_classifier(self, x_train, y_train, x_test, solver="lsqr", shrinkage=0.5, store_covariance=True)
        elif classifier == 3:
            print("SVM")
            return s.SVM_classifier(self, x_train, y_train, x_test)
        elif classifier == 4:
            print("SGD")
            return s.sgd_classifier(self, x_train, y_train, x_test, loss="hinge", penalty="l2")
        elif classifier == 5:
            print("KN")
            return s.kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=15)
        elif classifier == 6:
            print("DT")
            return s. DT_classifier(self, x_train, y_train, x_test)
        elif classifier == 7:
            print("RF")
            return s.RF_classifier(self, x_train, y_train, x_test)
        elif classifier == 8:
            print("MLP")
            return s.multiLayerPerceptron_classifier(self, x_train, y_train, x_test)
        elif classifier == 9:
            print("NO CLASSIFIER")
        elif classifier == 10:
            print("NO CLASSIFIER")

        #return y_pred



    #1 Quadratic Discriminant Analysis
    def QDA_classifier(self, x_train, y_train, x_test):
        qda = QuadraticDiscriminantAnalysis(store_covariances=True)
        return qda.fit(x_train, y_train).predict(x_test)  #y_predict

    #2 Linear Discriminant Analysis
    def LDA_classifier(self, x_train, y_train, x_test, solver="lsqr", shrinkage=0.5, store_covariance=True):
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, store_covariance=store_covariance)
        # l = LinearDiscriminantAnalysis(solver='svd', n_components=2)
        # l = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.8, n_components=2)
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)
        return y_pred

    #3
    def SVM_classifier(self, x_train, y_train, x_test):
       clf = svm.LinearSVC()
       clf.fit(x_train, y_train)  # .transform(x_train)
       y_pred = clf.predict(x_test)
       return y_pred


    #4 Stochastic Gradient Descent
    def sgd_classifier(self, x_train, y_train, x_test, loss="hinge", penalty="l2"):
       clf = SGDClassifier(loss=loss, penalty=penalty)
       clf.fit(x_train, y_train)  # .transform(x_train)
       y_pred = clf.predict(x_test)
       return y_pred

    #5
    def kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=15):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #6 Decision Tree
    def DT_classifier(self, x_train, y_train, x_test):
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred

    #Print Decision Tree
    def DT_classifier_printTree(self, x_train, y_train, feature_names, target_values):
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)  # .transform(x_train)
        target_names = []
        for n in target_values:
            target_names.append(str(int(n)))
        print(feature_names)
        #print(target_names)
        #print(x_train)
        dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=feature_names,
                         class_names=target_names,
                         filled=True, rounded=True,
                         special_characters=True)
        #with open("dt.dot", 'a') as f:
        #    f = tree.export_graphviz(clf, out_file=f)
        #dot_data = tree.export_graphviz(clf, out_file=None)
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("DT.pdf")
        graph = pydotplus.graph_from_dot_data(dot_data)
        Image(graph.create_png())


    #8
    def multiLayerPerceptron_classifier(self, x_train, y_train, x_test):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,), random_state=1)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred


    #7 Random forest
    def RF_classifier(self, x_train, y_train, x_test):
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features='auto', random_state=0)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred