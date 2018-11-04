# -*- coding: UTF-8 -*-
import math
import json
train_vector_Path='E:/Firstwork/trainVt/train.json'
test_vector_Path='E:/Firstwork/testVt/test.json'

def cal_eachVC_len(aVC):
    sum=0
    for word,value in aVC[1].items():
        sum+=value**2
    result=math.sqrt(sum)
    return result
def cal_VClist_len(vcList):
    vc_len_list=[]
    for vc in vcList:
        vc_len_list.append(cal_eachVC_len(vc))
    return vc_len_list

def cos_twoVC(vc1,vc2):
    multi=0
    for key in vc1[1]:
        if key in vc2[1]:
            multi+=vc1[1][key]*vc2[1][key]
    len1=cal_eachVC_len(vc1)
    len2=cal_eachVC_len(vc2)
    cos=multi/(len1*len2)
    return cos

def KNN():
    k=5
    open_tmp = open(train_vector_Path, 'r')
    train_vectors = json.load(open_tmp)
    open_tmp.close()
    open_tmp = open(test_vector_Path, 'r')
    test_vectors = json.load(open_tmp)
    open_tmp.close()

    for testVC in test_vectors:
        all_cos_list=[]
        for trainVC in train_vectors:
            each_cos_list=[]
            each_cos_list.append(trainVC[0])
            each_cos_list.append(cos_twoVC(testVC,trainVC))
            all_cos_list.append(each_cos_list)

        sorted_cos_list=sorted(all_cos_list,key=lambda x:x[1],reverse=True)
        k_cos_dict = {}
        count = 0
        for key, value in sorted_cos_list:
            if count >= k:
                break
            count = count + 1
            k_cos_dict[key] = k_cos_dict.get(key, 0) + 1

        judge_type = ''
        max_type_count = 0

        for key,value in k_cos_dict.items():
            if value > max_type_count:
                max_type_count = value
                judge_type = key
        success = 0
        failure = 0
        if trainVC[0] == judge_type:
            success += 1
        else:
            failure += 1
    successp = (float(success)) / (float(success + failure))
    print((success + failure),success,successp,failure)

if __name__ == '__main__':
    KNN()