import os
from os import listdir,mkdir,path
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import nltk
nltk.download()
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

srcPath='E:/Firstwork/20news-18828'
toPath='E:/Firstwork/chulishuju'
trainPath='E:/Firstwork/trainset'
testPath='E:/Firstwork/testset'

def read_file():
    print( 'Read data:')
    for floder in os.listdir(srcPath):
        if path.exists(toPath+'/'+floder)==False:
            mkdir(toPath+'/'+floder)
        floderPath=srcPath+'/'+floder+'/'
        for file in os.listdir(floderPath):
            filePath = floderPath + file
            tarfilePath=toPath+'/'+floder+'/'+file
            targetfile=open(tarfilePath,'w')
            for line in open(filePath,'rb').readlines():
                line = line.decode("utf-8", errors='ignore')
                text_list=re.sub("[^a-zA-Z]", " ", line).split()
                stoplist = stopwords.words('english')
                for word in text_list:
                    word = PorterStemmer().stem(word.lower())
                    if word not in stoplist:
                        targetfile.write('%s\n' % word)
            targetfile.close()

if __name__ == '__main__':
 read_file()
 print ('success read')



import os
from os import listdir,mkdir,path
import shutil
trainrate=0.8

srcPath='E:/Firstwork/20news-18828'
toPath='E:/Firstwork/chulishuju'
trainPath='E:/Firstwork/trainset'
testPath='E:/Firstwork/testset'


def devide():  #将预处理数据分为训练集和测试集
    for floder in os.listdir(toPath):
        if path.exists(trainPath+'/'+floder)==False:
            mkdir(trainPath+'/'+floder)
        if path.exists(testPath+'/'+floder)==False:
            mkdir(testPath+'/'+floder)
        floderPath=toPath+'/'+floder+'/'
        total=file_count(floderPath)
        count=0
        for file in os.listdir(floderPath):
            count+=1
            if count<=total*trainrate:
                shutil.copy(os.path.join(floderPath,file),trainPath+'/'+floder+'/'+file)
            else:
                shutil.copy(os.path.join(floderPath, file), testPath + '/' + floder + '/' + file)

def file_count(floderPath):
    count=0
    for file in os.listdir(floderPath):
        count+=1
    return count

if __name__ == '__main__':
    devide()
    print('fenlei success')