import pandas as pd
import numpy as np
import re
import os
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from IPython.display import HTML
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers
from sklearn import svm
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words
from nltk.corpus import wordnet
allEnglishWords = words.words() + [w for w in wordnet.words()]
allEnglishWords = np.unique([x.lower() for x in allEnglishWords])

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

path = "/Users/yiyang/Downloads/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]
neg_testFiles = [x for x in os.listdir(path+"test/neg/") if x.endswith(".txt")]
pos_testFiles=[x for x in os.listdir(path+"test/pos/") if x.endswith(".txt")]

positiveReviews, negativeReviews, neg_testReviews, pos_testReviews= [], [], [],[]
for pfile in positiveFiles:
    with io.open(path+"train/pos/"+pfile, encoding="latin1") as f:
        positiveReviews.append(f.read())
for nfile in negativeFiles:
    with io.open(path+"train/neg/"+nfile, encoding="latin1") as f:
        negativeReviews.append(f.read())
for ntfile in neg_testFiles:
    with io.open(path+"test/neg/"+ntfile, encoding="latin1") as f:
        neg_testReviews.append(f.read())
for ptfile in pos_testFiles:
    with io.open(path+"test/pos/"+ptfile, encoding="latin1") as f:
        pos_testReviews.append(f.read())



reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles})
], ignore_index=True).sample(frac=1, random_state=1)
print(reviews.head())
reviews = reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
train = reviews[reviews.label!=-1].sample(frac=1, random_state=1)

test_reviews= pd.concat([
    pd.DataFrame({"review":pos_testReviews, "label":1, "file":pos_testFiles}),
    pd.DataFrame({"review":neg_testReviews, "label":0, "file":neg_testFiles})
], ignore_index=True).sample(frac=1, random_state=1)
test_reviews = test_reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
test = test_reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
print(train.shape)
print(test.shape)


class Preprocessor(object):
    ''' Preprocess data for NLP tasks. '''

    def __init__(self, alpha=True, lower=True, stemmer=True, english=False):
        self.alpha = alpha
        self.lower = lower
        self.stemmer = stemmer
        self.english = english

        self.uniqueWords = None
        self.uniqueStems = None

    def fit(self, texts):
        texts = self._doAlways(texts)

        allwords = pd.DataFrame({"word": np.concatenate(texts.apply(lambda x: x.split()).values)})
        self.uniqueWords = allwords.groupby(["word"]).size().rename("count").reset_index()
        self.uniqueWords = self.uniqueWords[self.uniqueWords["count"] > 1]
        if self.stemmer:
            self.uniqueWords["stem"] = self.uniqueWords.word.apply(lambda x: PorterStemmer().stem(x)).values
            self.uniqueWords.sort_values(["stem", "count"], inplace=True, ascending=False)
            self.uniqueStems = self.uniqueWords.groupby("stem").first()

        # if self.english: self.words["english"] = np.in1d(self.words["mode"], allEnglishWords)
        print("Fitted.")

    def transform(self, texts):
        texts = self._doAlways(texts)
        if self.stemmer:
            allwords = np.concatenate(texts.apply(lambda x: x.split()).values)
            uniqueWords = pd.DataFrame(index=np.unique(allwords))
            uniqueWords["stem"] = pd.Series(uniqueWords.index).apply(lambda x: PorterStemmer().stem(x)).values
            uniqueWords["mode"] = uniqueWords.stem.apply(
                lambda x: self.uniqueStems.loc[x, "word"] if x in self.uniqueStems.index else "")
            texts = texts.apply(lambda x: " ".join([uniqueWords.loc[y, "mode"] for y in x.split()]))
        # if self.english: texts = self.words.apply(lambda x: " ".join([y for y in x.split() if self.words.loc[y,"english"]]))
        print("Transformed.")
        return (texts)

    def fit_transform(self, texts):
        texts = self._doAlways(texts)
        self.fit(texts)
        texts = self.transform(texts)
        return (texts)

    def _doAlways(self, texts):
        # Remove parts between <>'s
        texts = texts.apply(lambda x: re.sub('<.*?>', ' ', x))
        # Keep letters and digits only.
        if self.alpha: texts = texts.apply(lambda x: re.sub('[^a-zA-Z0-9 ]+', ' ', x))
        # Set everything to lower case
        if self.lower: texts = texts.apply(lambda x: x.lower())
        return texts

preprocess = Preprocessor(alpha=True, lower=True, stemmer=True)
trainX = preprocess.fit_transform(train.review)
testX=preprocess.transform(test.review)
print(preprocess.uniqueWords.shape)
print(preprocess.uniqueWords[preprocess.uniqueWords.word.str.contains("disappoint")])
print(preprocess.uniqueStems.shape)
print(preprocess.uniqueStems[preprocess.uniqueStems.word.str.contains("disappoint")])

stop_words = text.ENGLISH_STOP_WORDS.union(["thats","weve","dont","lets","youre","im","thi","ha",
    "wa","st","ask","want","like","thank","know","susan","ryan","say","got","ought","ive","theyre"])
tfidf = TfidfVectorizer(min_df=2, max_features=10000, stop_words=stop_words) #, ngram_range=(1,3)
trainX = tfidf.fit_transform(trainX).toarray()
testX = tfidf.transform(testX).toarray()
trainY = train.label
testY=test.label

print(trainX.shape, trainY.shape)

#correlation feature
getCorrelation = np.vectorize(lambda x: pearsonr(trainX[:,x], trainY)[0])
correlations = getCorrelation(np.arange(trainX.shape[1]))
allIndeces = np.argsort(-correlations)
bestIndeces = allIndeces[np.concatenate([np.arange(1000), np.arange(-1000, 0)])]
vocabulary = np.array(tfidf.get_feature_names())
print(vocabulary[bestIndeces][:10])
print(vocabulary[bestIndeces][-10:])

trainX1 = trainX[:,bestIndeces]
testX1= testX[:,bestIndeces]

#pca feature
trainX2 = trainX
testX2= testX
scaler=StandardScaler()
scaler.fit(trainX2)
trainX2=scaler.transform(trainX2)
testX2=scaler.transform(testX2)
pca=PCA(n_components=2000)
pca.fit(trainX2)
trainX2 = pca.transform(trainX2)
testX2 =pca.transform(testX2)

#random_forest with correlation
# clf = RandomForestClassifier(n_estimators=200, max_depth=25,min_samples_split=100)
# clf = clf.fit(trainX1,trainY)
# y_p=clf.predict(testX1)
# a_s=accuracy_score(trainY,y_p)
# print(a_s)

#random_forest with pca
clf = RandomForestClassifier(n_estimators=200, max_depth=25,min_samples_split=100)
clf = clf.fit(trainX2,trainY)
y_p2=clf.predict(testX2)
a_s2=accuracy_score(trainY,y_p2)
print(a_s2)
