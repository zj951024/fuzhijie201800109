# -*- coding: UTF-8 -*-
from os import listdir,mkdir,path
import os
from collections import Counter
import math
import json

trainPath='E:/Firstwork/trainset'
testPath='E:/Firstwork/testset'
train_vector_Path='E:/Firstwork/trainVt/train.json'
test_vector_Path='E:/Firstwork/testVt/test.json'
def file_count(setPath):
    n = 0
    floderList = listdir(setPath)
    for i in range(len(floderList)):
        floderPath = setPath + '/' + floderList[i]
        fileList = listdir(floderPath)
        n+=len(fileList)
    return n

def cal_idf():
    files_num= file_count(trainPath)
    word_freq_dic = {}
    word_df_dic = {}
    filterDf_dic = {}
    idf_dic={}
    for floder in os.listdir(trainPath):
        floderPath = trainPath + '/' + floder + '/'
        for file in os.listdir(floderPath):
            filePath = floderPath + file
            words = open(filePath, 'r').readlines()
            tmp_freq_dic = Counter(words)
            for key,value in tmp_freq_dic.items():
                key = key.strip()
                word_freq_dic[key] =word_freq_dic.get(key, 0) + value
                word_df_dic[key] =word_df_dic.get(key, 0) + 1

            for key, value in word_freq_dic.items():
                if value > 8:
                    filterDf_dic[key]=word_df_dic[key]
            sorted_filterDf_dic_dic = sorted(filterDf_dic.items())
            for key,value in sorted_filterDf_dic_dic:
                idf_dic[key] = math.log10(files_num/(value + 1))
    return idf_dic

def createVector(srcPath,tgtPath):
    word_idf_dic = cal_idf()
    for floder in os.listdir(srcPath):
        floderPath = srcPath + '/' + floder + '/'
        for file in os.listdir(floderPath):

            file_vector_list=[]
            file_vector_list.append(floder)
            all_vector_list=[]

            word_tfidf_dic={}
            word_tf_dic={}
            tmp_sort_tfidf_lst=[]
            file_dic = {}
            filePath = floderPath + file
            words = open(filePath, 'rb').readlines()
            for word in words:
                word=word.strip()
                if word in word_idf_dic:
                    if word in word_tf_dic:
                        word_tf_dic[word] += 1
                    else:
                        word_tf_dic[word] = 1

            max_num = word_tf_dic[max(word_tf_dic, key=word_tf_dic.get)]

            for i in word_tf_dic:
                word_tfidf_dic[i]=(word_tf_dic[i]/max_num)*(word_idf_dic[i])

            tmp_sort_tfidf_lst=sorted(word_tfidf_dic.items(),key = lambda item:item[1],reverse=True)
            count= 0
            for key, value in tmp_sort_tfidf_lst:
                if count > 30:
                    break
                count = count + 1
                file_dic[key] = value
            file_vector_list.append(file_dic)
            all_vector_list.append(file_vector_list)
        openw = open(tgtPath, 'w')
        json.dump(file_vector_list, openw, ensure_ascii=False)
        openw.close()

if __name__ == '__main__':
    createVector(trainPath,train_vector_Path)
    print('trainset success')
    createVector(testPath,test_vector_Path)
    print('testset success')