#-*- coding: utf-8 -*-
from numpy import *
import operator
import pdb
from os import listdir
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

                                    
def classify0(inX,dataSet,labels,k):#inX是测试向量，dataSet是样本向量，labels是样本标签，k用于选择最近邻居的数目
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5#以上是求出测试向量到样本向量每一行向量的距离
	sortedDistIndicies = distances.argsort()#对距离进行排序，从小到大
	classCount={}
	for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]  
            classCount[voteIlabel] = classCount.get(voteIlabel,0)+1#得到距离最小的前k个点的分类标签
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)#对classCount字典分解为元素列表，使用itemgetter方法按照第二个元素的次序对元组进行排序，返回频率最高的元素标签
	return sortedClassCount[0][0]

def file2matrix(filename):#将文本记录转换成numpy的解析程序
	fr = open(filename)
	arrayOLines = fr.readlines()#转换成矩阵
	numberOfLines = len(arrayOLines)#得到文件的行数
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
        enumLabelVector = []
	index = 0
	for line in arrayOLines:
            line = line.strip()#截掉所有的回车符
            listFromLine = line.split('\t')#使用tab字符\t将上一步得到的整行数据分割成元素列表
            returnMat[index,:] = listFromLine[0:3]#选取前三个元素存储到特征矩阵中
            classLabelVector.append(listFromLine[-1])#用-1表示最后一列元素
            if cmp(listFromLine[-1],'didntLike')==0:
                enumLabelVector.append(1)
            elif cmp(listFromLine[-1],'smallDoses')==0:
                enumLabelVector.append(2)
            elif cmp(listFromLine[-1],'largeDoses')==0:
                enumLabelVector.append(3)
                index += 1
        return returnMat,enumLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)#获得最小值，（0）是从列中获取最小值，而不是当前行，就是每列都取一个最小值
    maxVals = dataSet.max(0)#获得最大值
    ranges = maxVals - minVals#获得取值范围
    normDataSet = zeros(shape(dataSet))#初始化新矩阵
    m = dataSet.shape[0]#获得列的长度
    normDataSet = dataSet - tile(minVals,(m,1))#特征值是1000×3,而最小值和范围都是1×3,用tile函数将变量内容复制成输入矩阵一样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))#/可能是除法，在numpy中，矩阵除法要用linalg.solve(matA,matB).
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')#读取文件中的数据并归一化
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]#新矩阵列的长度
    numTestVecs = int(m*hoRatio)#代表样本中哪些数据用于测试
    errorCount = 0.0#错误率
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)#前m×hoRatio个数据是测试的，后面的是样本
        print "the calssifier came back with: %d,the real answer is:%d" %(classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))#最后打印出测试错误率

#输入某人的信息，便得出对对方喜欢程度的预测值  
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses'] 
    percentTats = float(raw_input("percentage of time spent playing video games?"))#输入  
    ffMiles = float(raw_input("frequent flier miles earned per year?"))  
    iceCream = float(raw_input("liters of ice cream consumed per year?"))  
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt') #读入样本文件，其实不算是样本，是一个标准文件 
    normMat, ranges, minVals = autoNorm(datingDataMat)#归一化
    inArr = array([ffMiles, percentTats, iceCream])#组成测试向量
#    pdb.set_trace()#可debug
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)#进行分类
#    return test_vec_g,normMat,datingLabels
    print 'You will probably like this person:', resultList[classifierResult - 1]#打印结果

def img2vector(filename):
    returnVect = zeros((1,1024))#初始化一个向量
    fr = open(filename)#打开文件
    for i in range(32):
        lineStr = fr.readline()#读入每行向量
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#把每行的向量分别赋值给初始化向量
    return returnVect#返回向量

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')# 得到目录下所有文件的文件名
    m = len(trainingFileList)#得到目录下文件个数
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]#对文件名进行分解可以得到文件指的数字
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)#把标签添加进list
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)#把所有文件都放在一个矩阵里面
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)#得到一个向量
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)#对向量进行k近邻测试
        print "the classifier came back with: %d the real answer is %d" %(classifierResult,classNumStr)
        if(classifierResult != classNumStr):errorCount += 1.0
    print "\nthe total number of errors is: %d" %errorCount#得到错误率
    print "\nthe total error rate is: %f" %(errorCount/float(mTest))
