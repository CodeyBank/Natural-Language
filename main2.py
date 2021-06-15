"""
    Program creates a class for analysis of data sets using the Scikit learn Natural Language Processing Tool Kit
    The following classification algorithms are implemented, tested and compared in this project.
    - RandomForest
    - K-neighbours
    - Naive Bayes
    - One-R
    - SVC
    - Decision Trees

    Using pip installer install the following dependencies:
    - Numpy
    - Pandas
    - scikitlearn
    - mlxtend
    - nltk'


    ****************    By Ezeh Stanley, 2021  *******************

"""
# All necessary imports
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from sklearn.datasets import load_files

nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# import mlxtend
from mlxtend.classifier import OneRClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.neural_network import MLPClassifier


class NLToolBox:
    """
    Class instantiation requires not default attributes.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.X_train, self.X_test, self.y_train, self.y_test = self.loadDataSet(self.filepath)
        self.results = []
        
    def loadDataSet(self, filepath):
        """
        To load dataset, provide the absolute filepath as parameter
        File path must be passed as a regular expression
        """
        dataset = load_files(filepath)
        X, y = dataset.data, dataset.target

        documents = []
        stemmer = WordNetLemmatizer()
        for sen in range(0, len(X)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[sen]))

            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)

            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)

            # Converting to Lowercase
            document = document.lower()

            # Lemmatization
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            documents.append(document)

        # Feature vectorization
        vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(documents).toarray()

        # Use TF-IDF algorithm
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X).toarray()

        # split the data set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def saveModel(self, classifier, name):
        filename = '{}_classifier_Model'.format(name)
        with open(filename, 'wb') as picklefile:
            pickle.dump(classifier, picklefile)

        print('{} Model saved in project folder'.format(name))
        print(" \n")

    def randomForest(self):
        """
            Implements the Random Forest classifier
        """
        name = "RF"
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)

        print("*****  Confusion Matrix ******* \n")
        print(confusion_matrix(self.y_test, y_pred))

        print("*****  Classification report ******* \n")
        print(classification_report(self.y_test, y_pred))

        print("*****  Accuracy Score ******* \n")
        print(accuracy_score(self.y_test, y_pred))

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(classifier, self.X_train, self.y_train, cv=10)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        # Save the file
        self.saveModel(classifier, name)
        
        # Append value to the results
        self.results.append((name, cv_scores.mean()))
        return cv_scores

    def KNeighbors(self):
        """
            Implements the KNeighbors classifier
        """
        
        name = "KNeighbors"
        knc = KNeighborsClassifier(n_neighbors=4)
        knc.fit(self.X_train, self.y_train)
        score = knc.score(self.X_train, self.y_train)
        y_pred = knc.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print("Training score: ", score)

        print("*****  Confusion Matrix ******* \n")
        print(confusion_matrix(self.y_test, y_pred))

        print("*****  Classification report ******* \n")
        print(classification_report(self.y_test, y_pred))

        print("*****  Accuracy Score ******* \n")
        print(accuracy_score(self.y_test, y_pred))

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(knc, self.X_train, self.y_train, cv=10)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        # Save the file
        self.saveModel(knc, name)
        
        # Append value to the results
        self.results.append((name, cv_scores.mean()))
        
        return cv_scores

    def naiveBayes(self):
        """
            Implements the naive bayes classifier
        """
        
        # Initialize our classifier
        name = "NaiveBayes"
        gnb = GaussianNB()

        # Train our classifier
        model = gnb.fit(self.X_train, self.y_train)

        # Make predictions
        preds = gnb.predict(self.X_test)

        print("*****  Accuracy Score ******* \n")
        print(accuracy_score(self.y_test, preds))

        report = classification_report(self.y_test, preds)
        print("*****  Classification report ******* \n")
        print(report)

        cmatrix = confusion_matrix(self.y_test, preds)
        print("*****  Confusion Matrix ******* \n")
        print(cmatrix)

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        # Save the file
        self.saveModel(model, name)
        
        # Append value to the results
        self.results.append((name, cv_scores.mean()))
        
        return cv_scores

    def svcClassifier(self):
        """
            Implements the SVC classifier algorithm and performs analysis
        """
        name = 'SVC'
        svc = SVC()
        svc.fit(self.X_train, self.y_train)
        score = svc.score(self.X_train, self.y_train)

        print("*****  Accuracy Score ******* \n")
        print("Score: ", score)

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(svc, self.X_train, self.y_train, cv=10)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        ypredicted = svc.predict(self.X_test)

        cm = confusion_matrix(self.y_test, ypredicted)
        print("*****  Confusion Matrix ******* \n")
        print(cm)

        # Save the model
        self.saveModel(svc, name)
        
        # Append value to the results
        self.results.append((name, cv_scores.mean()))
        
        return cv_scores

    def one_RClassifier(self):
        name = "O_R"
        """
            Implements the OneR classfier
        """
        oner = OneRClassifier()

        oner.fit(self.X_train, self.y_train);
        y_prediction = oner.predict(self.X_train)
        train_acc = np.mean(y_prediction == self.y_train)

        print(f'*****  Training accuracy: ******** \n  {train_acc * 100:.2f}%')
        y_pred = oner.predict(self.X_test)
        test_acc = np.mean(y_pred == self.y_test)
        print(f'Test accuracy {test_acc * 100:.2f}%')

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(oner, self.X_train, self.y_train, cv=10)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        cm = confusion_matrix(self.y_test, y_pred)
        print("*****  Confusion Matrix ******* \n")
        print(cm)
        
        
         # Save the model
        self.saveModel(oner, name)
        
        # Store the values of the results
        self.results.append((name, cv_scores.mean()))
        return name

    def decisionTreeClassifier(self):
        
        """
            Implements the decision tree classifier
        """
        name = "DT"
        model = tree.DecisionTreeClassifier()
        # create a regressor object
        regressor = DecisionTreeRegressor(random_state=0)

        # fit the regressor with X and Y data
        regressor.fit(self.X_train, self.y_train)

        model.fit(self.X_train, self.y_train)

        y_predict = model.predict(self.X_test)
        score = accuracy_score(self.y_test, y_predict)

        print("*****  Accuracy Score ******* \n")
        print("Score: ", score)

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=8)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        cm = confusion_matrix(self.y_test, y_predict)
        print("*****  Confusion Matrix ******* \n")
        print(cm)
        
        # Append value to the results
        self.results.append((name, cv_scores.mean()))
        
        # Save the model
        self.saveModel(model, name)
        
        return 
    
    def neuralNetClassifier(self):
        """
            Artificial Neural Network classifier using multilayer perceptrons
            MLP.
            Implementation solver uses lbfgs solver
            alpha is set to 1e-5, and 6 hidden layers
        """
        
        name = "NN"
        clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=1)

        clf.fit(self.X_train, self.y_train)
        
        y_predict = clf.predict(self.X_test)
        score = accuracy_score(self.y_test, y_predict)

        print("*****  Accuracy Score ******* \n")
        print("Score: ", score)

        ## Cross-validaton  using 10-fold cross validation
        cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=10)
        print("Cross-Validation average score: %.2f \n" % cv_scores.mean())

        cm = confusion_matrix(self.y_test, y_predict)
        print("*****  Confusion Matrix ******* \n")
        print(cm)
        
        # Append value to the results
        self.results.append((name, cv_scores.mean()))
        
        # Save the model
        self.saveModel(clf, name)
        
        return cv_scores
        
        
        
    def testModel(self, modelpath, datapath):
        """
           This method is used for testing the generated models and 
           comparing with training results.
           @params:
            - modelpath: is the absolute file path to the model
        """
        
        with open(modelpath, 'rb') as training_model:
            model = pickle.load(training_model)
            
        X_train, X_test, y_train, y_test = self.loadDataSet(datapath)

        y_pred2 = model.predict(X_test)
        
        print("*****  Confusion Matrix ******* \n")
        print(confusion_matrix(y_test, y_pred2))
        
        print("*****  Classification report ******* \n")
        print(classification_report(y_test, y_pred2))

        print("*****  Accuracy ******* \n")
        print(accuracy_score(y_test, y_pred2))
        
        return
        
    
    def boxPlot(self):
        """
            Create a box plot of all algorthims for testing
        """
        names = []
        results = []
        
        # Save all the names and results in the arrays 
        if len(self.results) > 0:
            for name, result in self.results:
                names.append(name)
                results.append(result)  
        
            # Plot algorithm comparison
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            
            ax = fig.add_axes([0,0,1,1])
            ax.bar(names,results)     
          
            
            plt.show()
        else:
            print("You must generate at least one classifier to plot")
        
if __name__ == '__main__':
    w = NLToolBox(r"TextClass_text")
    # argument = sys.argv[1]
    
    argument = sys.argv[1] if len(sys.argv) > 1 else None
    modelpath = sys.argv[2] if len(sys.argv) > 2 else None
    datapath = sys.argv[3] if len(sys.argv) > 3 else None
    
    # if len(sys.argv) > 3:
    #     modelpath = sys.argv[2]
    #     datapath = sys.argv[3]
    
    try:
        if argument == "test" and modelpath and datapath:
            w.testModel(modelpath, datapath)
   
        if argument == "DT":
            w.decisionTreeClassifier()
        elif argument == "OR":
            w.one_RClassifier()
        elif argument == "SVC":
            w.svcClassifier()
        elif argument == "NN":
            w.neuralNetClassifier()
        elif argument == "NB":
            w.naiveBayes()
        elif argument == "KN":
            w.KNeighbors()
        elif argument == "RF":
            w.randomForest()
        elif argument == "plot":
            w.boxPlot()
    
    except (TypeError, FileNotFoundError):
        error= """
        Please Select a classifier to run \n
        - KN - K-Neighbors \n
        - DT - Decision Tree \n
        - OR - One Rule \n
        - SVC - SVM \n
        - NN - Neural Network \n
        - NB - Naive Bayes  \n    
        - RF - Random Forest \n
        
        To test a model with another dataset use the following
        python main.py test <model_path> <dataset_path> 
        
        Boxplot method can only work on Jupyter Notebook and Spyder
        """
        print(error)
        
        
