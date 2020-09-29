
import json
import pandas as pd
import csv
import sklearn
import dill as pickle
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import math


# def split_into_words(message):
#     from textblob import TextBlob
#     return TextBlob(message).words

def split_into_words(text):
   text = text.upper() 
   # Japanese tinysegmenter
   import tinysegmenter
   tokenizer = tinysegmenter.TinySegmenter()
   words = tokenizer.tokenize(text)
   return words
    
   # Removed stop words
   #stop_words = [x.strip() for x in open('./data/stopwords.txt','r').read().split('\n')]
   #return [word for word in words if word not in stop_words]

# def split_into_words(text):
#     text = text.upper()
#     import MeCab
#     m = MeCab.Tagger(" -d /usr/lib/mecab/dic/mecab-ipadic-neologd/")
#     m.parse("")
#     node = m.parseToNode(text)
#     words = []
#     while node:
#             words.append(node.surface)
#             node = node.next

#     stop_words = [x.strip() for x in open('./data/stopwords.txt','r').read().split('\n')]
#     return [word for word in words if word not in stop_words]

def train():
    # Read data
    df = pd.read_csv('./models/data_all.pkl', sep=':', names=['kata', 'hira', 'partOfSpeech', 'score'])

    #df.loc[df['score'] > 0, 'label'] = 'no'
    #df.loc[df['score'] < 0, 'label'] = 'yes'
    df['message1'] = df['kata'].astype(str)
    df['message2'] = df['hira'].astype(str)
    # or df['message'] = df['kata'].astype(str)
    # file = open("./data/result.txt", "w")
    # for index, row in df.iterrows():
    #   file.write(row['kata']+ '\n')

    # file.close()
    data = df

    tfidf_transformer = TfidfVectorizer(analyzer = split_into_words)

    labels = []
    analysis_data = []
    for index, row in df.iterrows():
      lb = 'positive'
      if(row['score'] < 0):
        lb = 'nagative'
      loop = math.floor(row['score'] * 10)
      if(loop < 0):
        loop = loop*(-1)
      if(loop > 10):
        loop = 10
      for i in range(0,loop):
        labels.append(lb)
        analysis_data.append(row['message1'])
        labels.append(lb)
        analysis_data.append(row['message2'])

    training_data, testing_data, label_train, label_test = train_test_split(analysis_data, labels, test_size=.05, random_state=45)
    print(training_data)

    #Transform data 
    tfidf_transformer = tfidf_transformer.fit(training_data)
    data_train = tfidf_transformer.transform(training_data)
    data_test = tfidf_transformer.transform(testing_data)
    print('train_matrix: {}'.format(data_train.shape))
    print('test_matrix: {}'.format(data_test.shape))

    #Build model
    # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    #         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    #         "Naive Bayes", "QDA"]

    # classifiers = [
    #     KNeighborsClassifier(3),
    #     SVC(kernel="linear", C=0.025),
    #     SVC(gamma=2, C=1),
    #     GaussianProcessClassifier(1.0 * RBF(1.0)),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     MLPClassifier(alpha=1),
    #     AdaBoostClassifier(),
    #     GaussianNB(),
    #     QuadraticDiscriminantAnalysis()]

    # classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(data_train, label_train)
    # classifier = linear_model.LogisticRegression().fit(data_train, label_train)
    # classifier = linear_model.LogisticRegression(penalty='l2',
    #                                             dual=False,
    #                                             tol=0.0001,
    #                                             C=2.0,
    #                                             fit_intercept=True,
    #                                             intercept_scaling=1,
    #                                             class_weight='balanced',
    #                                             random_state=None,
    #                                             solver='lbfgs',
    #                                             max_iter=100,
    #                                             multi_class='ovr',
    #                                             verbose=0,
    #                                             warm_start=False,
    #                                             n_jobs=1).fit(data_train, label_train)

    classifier = svm.LinearSVC().fit(data_train, label_train)
    # classifier = svm.LinearSVC(penalty='l2', 
    #                            loss='squared_hinge', 
    # 				           tol=0.0001, C=1.0, 
    # 				           multi_class='ovr',
    #                            fit_intercept=True, 
    #                            intercept_scaling=1,
    #                            class_weight='balanced', 
    #                            verbose=0, 
    # 				           random_state=None, 
    # 				           max_iter=1000).fit(data_train, label_train)
    # classifier = svm.SVC().fit(data_train, label_train)                                  
    # classifier = svm.SVC(C=1.0, 
    #                      kernel='linear', 
    # 			           degree=3, 
    # 				       gamma='auto',
    #                      coef0=0.0, 
    #                      shrinking=True, 
    # 				     probability=False, 
    # 				     tol=0.001,
    #                      cache_size=200, 
    #                      class_weight='balanced', 
    # 				     verbose=False, max_iter=-1, 
    # 				     decision_function_shape=None, 
    # 				     random_state=None).fit(data_train, label_train)

    #classifier = KNeighborsClassifier(n_jobs=-1).fit(data_train, label_train)

    #from sklearn.tree import DecisionTreeClassifier
    #classifier = DecisionTreeClassifier(random_state=0).fit(data_train.toarray(), label_train)

    #from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(data_train, label_train)

    #from sklearn.neural_network import MLPClassifier
    #classifier = MLPClassifier().fit(data_train, label_train)

    #Testing
    all_predictions = classifier.predict(data_test)
    accuracy = accuracy_score(label_test, all_predictions)
    cm = confusion_matrix(label_test, all_predictions)
    report = classification_report(label_test, all_predictions)

    print(all_predictions)
    print('accuracy: {}'.format(accuracy))
    print('------------------------------------------------------')
    print('confusion matrix: \n{}'.format(cm))
    print('(row=expected, col=predicted)')
    print('------------------------------------------------------')
    print('prediction report: \n{}'.format(report))

    #Visulization
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=None) #cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    #Persist model
    with open('./models/tfidf_transformer.pkl', 'wb') as model:
        pickle.dump(tfidf_transformer, model)

    with open('./models/classifier.pkl', 'wb') as model:
        pickle.dump(classifier, model)

if __name__=='__main__':
    s = time.time()
    print(time.ctime())
    train()
    print(time.ctime())
    e = time.time()
    print("done in {0}sec".format(e-s))