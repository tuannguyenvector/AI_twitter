import pandas as pd
import sklearn
import dill as pickle
import csv
import jaconv
import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# def split_into_words(text): 
#     return text.split()
# def split_into_words_tiny(text):
#    text = text.upper() 
#    # Japanese tinysegmenter
#    import tinysegmenter
#    tokenizer = tinysegmenter.TinySegmenter()
#    words = tokenizer.tokenize(text)
#    #print(words)
#    return words

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
        v = k.split('\t')
        #print(v)
        if len(v) > 3:
            data.append([v[3].split('-')[0]+':'+jaconv.kata2hira(v[1]),v[4].split('-')[0]])
            print([v[3].split('-')[0]+':'+jaconv.kata2hira(v[1]),v[4].split('-')[0]])
        else:
            data.append([v[0]+':'+v[0],''])
    #stop_words = [x.strip() for x in open('./data/stopwords.txt','r').read().split('\n')]
    
    return data

def prediction(text):
    tfidf_transformer = pickle.load(open('./models/tfidf_transformer.pkl', 'rb'))
    classifier = pickle.load(open('./models/classifier.pkl', 'rb'))

    text_tfidf = tfidf_transformer.transform([text])

    predict = classifier.predict(text_tfidf)

    #print(split_into_words(text))
    
    for i in split_into_words(text):
        text_tfidf1 = tfidf_transformer.transform([i])

        predict1 = classifier.predict(text_tfidf1)
        #print(predict1)
        #print(text_tfidf1)
    

    #scores = classifier.predict_proba(text_tfidf)
    #score = max(scores[0])

    #res = {"text": text, "filled_prediction": predict[0]}
    
    return predict[0]
df = pd.read_csv('./models/data_all.pkl', sep=':', names=['kata', 'hira', 'partOfSpeech', 'score'])
analysis_data_push = {'名詞': {}, '動詞': {},'形容詞': {},'副詞': {},'助動詞': {},'助詞': {},'形状詞': {} }
for index, row in df.iterrows():
    if(row['partOfSpeech'] == '形容詞'): 
        row['partOfSpeech'] = '形状詞'
    analysis_data_push[row['partOfSpeech']][row['kata']+':'+row['hira']] =  row['score']
    #analysis_data_push[row['partOfSpeech']][row['hira']] =  row['score']
def check_data(text):
    total = 0
    data_in = []
    #print(split_into_words_tiny(text))
    print(split_into_words(text))
    for i in split_into_words(text):
        if i[1] in analysis_data_push:
            list = analysis_data_push[i[1]]
            if i[0] in list:
            #index = list.index('じゃせつ')
            #if(list.index('じゃせつ'))
                total += list[i[0]]
                data_in.append(i[0] +' '+ ':'+ i[1] + ':'+ str(list[i[0]]))
                #print(i[0] +' '+ str(list[i[0]]))
            else:
                data_in.append(i[0] +':'+ i[1])
        else:
            data_in.append(i[0] +':'+ i[1])
    if(total >= 0):
        #return " , ".join(data_in) + ' '+ str(total)
        return  'positive'
    else:
        #return " , ".join(data_in) + ' '+ str(total)
        return 'negative'

if __name__ == '__main__':
    data = pd.read_csv('./data/getdata.csv', sep=',', quoting=csv.QUOTE_NONE,
                           names=["message"])
    analysis_data = data['message'].apply(lambda x: np.str_(x))
    file = open("./data/result.txt", "w")
    
    for f in analysis_data:
        print(f)
        r = check_data(f)
        file.write(r+ '\n')
        
    file.close()
    #while(text != 'quit'):
        #text = input('Enter text: ')
        #prediction = prediction(text)
        #print (prediction(text))