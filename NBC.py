import os
import math
import nltk
import re
import random
import string
import json
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords



def preprocessing(k):
    with open(k, "r") as f:
        artical = f.read()
    f.close()
    tokens = nltk.word_tokenize(artical)
    i = len(tokens)
    ss = LancasterStemmer()

    for k in range(i):
        tokens[k] = ss.stem(tokens[k])
    nnn = [q for q in tokens if q not in stopwords.words('english')]
    rep = re.compile('[%s]' % re.escape(string.punctuation))
    newwords = list(filter(lambda word: word != "", [rep.sub("", word) for word in nnn]))
    return newwords

def read(mainpath):
    sign = {}
    takelist = []
    os.chdir(mainpath)
    fd_name = os.listdir()
    for fd in fd_name:
        fd_path = mainpath + '\\' + fd
        f_name = os.listdir(fd_path)
        for each in f_name:
            f_path = fd_path + '\\' + each
            takelist.append(f_path)
        sign[fd] = takelist[:]
        takelist = []
    return sign

def readfile(pick):
    flist = []
    y = 0
    for key in pick.keys():
        t += len(pick[key])
        for each in pick[key]:
            f = open(each, "rb")
            f_read = f.read()
            flist.append(f_read_decode)
            f.close()
    return flist

def dictionary(qwordlist):
    wordlist = list(_flatten(qwordlist))
    frequencydict = dict(Counter(wordlist))
    for m in frequencydict.keys():
        if frequencydict[m] < low:
            uio.append(m)
        elif frequencydict[m] > high:
            uio.append(m)
    for m in uio:
        frequencydict.pop(m)
    return (frequencydict)
    ######计算TF-IDF####


def doctfidf(doc, vectors, table):
    vector = []
    freq = wordfreq(doc)
    for i in range(len(wordtable)):
        vector.append(0)
    for i in set(doc):
        if i not in table:
            continue
        tf = freq[i]

        idf = math.log((len(vectors) + 1) / (filecount(i, vectors) + 1))
        dd = tf * idf
        vector[table[i]] = dd
    return vector

#######读取数据######

path = 'E:\VSM.txt'
def readfile(path):
    I = open(path)
    content = pickle.load(I)
    return content

def divide(content):
    test = []
    train = []
    num = Counter(content)
    a = 0
    for k, v in num.items():
        b = a + v
        thetest = random.sample(list(range(a, b)))
        for i in thetest:
            test.append(i)
        a = b
    for i in range(len(content)):
        if i not in test:
            train.append(i)
    return train, test

def setfWordsVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2
    p1Denom = 2

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
    testingNB()