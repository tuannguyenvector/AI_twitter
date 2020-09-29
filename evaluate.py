import pandas as pd
import sklearn
import dill as pickle
import csv
import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


def split_into_words(text): 
    return text.split()


def split_into_words(text):
    text = text.upper()
    import MeCab
    m = MeCab.Tagger("-Owakati")
    m1 = MeCab.Tagger()
    
    node = m.parseToNode(text)
    line = m1.parse(text).splitlines()
    words = []
    #tp = m.parse(text).split()
    while node:
            words.append(node.surface)
            node = node.next
    data = []
    for k in line:
        k.split('\t')
        data.append([k[0],k[len(k)-2]])
        
    #stop_words = [x.strip() for x in open('./data/stopwords.txt','r').read().split('\n')]
    print(data)
    return data

def prediction(text):
    tfidf_transformer = pickle.load(open('./models/tfidf_transformer.pkl', 'rb'))
    classifier = pickle.load(open('./models/classifier.pkl', 'rb'))

    text_tfidf = tfidf_transformer.transform([text])

    predict = classifier.predict(text_tfidf)

    print(split_into_words(text))
    
    for i in split_into_words(text):
        text_tfidf1 = tfidf_transformer.transform([i])

        predict1 = classifier.predict(text_tfidf1)
        print(predict1)
        print(text_tfidf1)
    

    #scores = classifier.predict_proba(text_tfidf)
    #score = max(scores[0])

    #res = {"text": text, "filled_prediction": predict[0]}
    
    return predict[0]

if __name__ == '__main__':
    data = pd.read_csv('./data/getdata.csv', sep=',', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
    analysis_data = data['message'].apply(lambda x: np.str_(x))
    file = open("./data/result.txt", "w")
    
    for f in analysis_data:
        r = prediction(f)
        file.write(r+ '\n')
        
    file.close()
    #while(text != 'quit'):
        #text = input('Enter text: ')
        #prediction = prediction(text)
        #print (prediction(text))