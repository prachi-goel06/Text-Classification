"""""Code By Prachi Goel
Student ID-1001234789"""
import time,copy
import os,codecs,string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import math

dir1="C:\\Users\\prach\\Desktop\\New folder\\UTA Ms\\Fall'17\\Machine Learning\\Project_2\\Twenty_newsgroups"
Train_Class=[]
Test_Class=[]
Tokenized_Train_data=[]
Tokenized_Test_data=[]
file_class_train=""
Class_Tokens=[]
Vocabulary=[]

Test_dict={}
''' Train Class's Compressed Data list ,Test Document list, remove end words
for each Train class find token probability,Each class probabilty, Total number of token in each class, total number of token in vocabulary'''

def Splitting_Data_Into_Test_Train(dir1,Train_Class,Test_Class,file_class_train):
    print("Please wait Data is being split into Trianing and Test")
    for folder in os.listdir(dir1):

        iteration=len(Test_Class)
        count =0
        for file in os.listdir(dir1+"\\"+folder):
            with codecs.open(dir1 + "\\" + folder + "\\" + file, 'r',encoding='utf-8',errors='ignore') as readingf:
                if count<=500:
                    file_class_train += readingf.read()
                    count+=1
                if count>500:
                    file_test=readingf.read()
                    Test_Class.append(file_test)
                    count += 1

        print ("Number of documents in test data ",folder," are: ", len(Test_Class)-iteration)
        Train_Class.append(file_class_train)
        print("All The files for Class ", len(Train_Class), "merged")
    print("Data ready for Training")
    return Train_Class,Test_Class


def Tokenization(Data,Tokenized_Data,Vocabulary):
    trans_table = str.maketrans(string.ascii_uppercase + string.digits + string.punctuation, string.ascii_lowercase + ' '*len(string.digits + string.punctuation))
    Data.translate(trans_table)
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for word in Data.split():
        if word not in (stopWords) and len(word)>1:
            # for w in word:
            wordsFiltered.append(word)
            Vocabulary.append(word)
    print( "class",len(Tokenized_Data)+1,"extracted")
    Tokenized_Data.append(wordsFiltered)
    return Tokenized_Data,Vocabulary

def Tokenization_test(Data,Tokenized_Test_data):
    trans_table = str.maketrans(string.ascii_uppercase + string.digits + string.punctuation, string.ascii_lowercase + ' '*len(string.digits + string.punctuation))
    Data.translate(trans_table)
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for word in Data.split():
        if word not in (stopWords) and len(word)>1:
            wordsFiltered.append(word)
    Tokenized_Test_data.append(wordsFiltered)
    print("class", len(Tokenized_Test_data) + 1, "extracted")
    return Tokenized_Test_data

def Token_Probability(Tokenized_Data,Class_Tokens,Vocabulary_Count):
    Different_tokens = Counter(Tokenized_Data)
    total_words = sum(Different_tokens.values())
    for i in Different_tokens:
        Different_tokens[i] = ((Different_tokens[i]+1)/(Vocabulary_Count+total_words))
    (Class_Tokens.append(Different_tokens))
    return Class_Tokens

def Training_Data(Train_Class,Tokenized_Train_Data,Class_Tokens,Vocabulary):
    Number_of_Classes = 20
    Number_of_Total_Articles = 10000
    Number_of_Articles_Each_Class = 500
    Vocabulary_Final=[]
    Prior_Probability_of_Class = (Number_of_Articles_Each_Class / Number_of_Total_Articles)
    for i in range (len(Train_Class)):
        Tokenized_Data=Tokenization(Train_Class[i],Tokenized_Train_data,Vocabulary)
    Vocabulary_Count=(len(set(Tokenized_Data[1])))
    Vocabulary_Final=copy.copy(set(Tokenized_Data[1]))
    for i in range (len(Tokenized_Data[0])):
        print("Please wait finding the probability of token in class: ",i)
        Class_Tokens=Token_Probability(Tokenized_Data[0][i],Class_Tokens,Vocabulary_Count)

    return Class_Tokens,Vocabulary_Final,Prior_Probability_of_Class

def Testing_class(i):
    if i<500:
        Class=0
        return Class
    elif 500<=i<1000:
        Class=1
        return Class
    elif 1000<=i<1500:
        Class=2
        return Class
    elif 1500<=i<2000:
        Class=3
        return Class
    elif 2000<=i<2500:
        Class=4
        return Class
    elif 2500<=i<3000:
        Class=5
        return Class
    elif 3000<=i<3500:
        Class=6
        return Class
    elif 3500<=i<4000:
        Class=7
        return Class
    elif 4000<=i<4500:
        Class=8
        return Class
    elif 4500<=i<5000:
        Class=9
        return Class
    elif 5000<=i<5500:
        Class=10
        return Class
    elif 5500<=i<6000:
        Class=11
        return Class
    elif 6000<=i<6500:
        Class=12
    elif 6500<i<7000:
        Class=13
        return Class
    elif 7000<=7500:
        Class=14
        return Class
    elif 7500<=i<8000:
        Class=15
        return Class
    elif 8000<=i<8500:
        Class=16
        return Class
    elif 8500<=i<9000:
        Class=17
        return Class
    elif 9000<=i<9500:
        Class=18
        return Class
    elif 9500<=i<10000:
        Class=19
        return Class

def Testing_Data(Class_Tokens,Vocabulary,Prior_Probability_of_Class,Test_Class):
    Matched_Class=0
    for i in range(len(Test_Class)):
        Score=0
        Tokenized = Tokenization_test(Test_Class[i],Tokenized_Test_data)
    Class_of_test_Document=[]
    print("Please wait while we calculate the accuracy")
    for text in Tokenized:
        Probability_of_Class_given_document = []
        for c in range(0,20):
            Class_Score=Prior_Probability_of_Class
            for token in text:
                if token in Class_Tokens[c]:
                    Class_Score+=Class_Tokens[c][token]
            Probability_of_Class_given_document.append(Class_Score)
        Class_of_test_Document.append(np.argmax(np.array(Probability_of_Class_given_document)))
    print (Class_of_test_Document)
    for i in range(len(Class_of_test_Document)):
            if Testing_class(i)==Class_of_test_Document[i]:
                Matched_Class+=1
                print (i)
            print (Matched_Class)
    Accuracy=100*Matched_Class/10000
    print ("The Accuracy is: ",Accuracy)



if __name__ == '__main__':
    Stime=time.time()
    Dataset=Splitting_Data_Into_Test_Train(dir1,Train_Class,Test_Class,file_class_train)
    Data_after_Training=Training_Data(Dataset[0],Tokenized_Train_data,Class_Tokens,Vocabulary)
    Testing_Data(Data_after_Training[0],Data_after_Training[1],Data_after_Training[2],Dataset[1])
    Stop_time=time.time()
    Total_time=Stop_time-Stime
    print (Total_time)
