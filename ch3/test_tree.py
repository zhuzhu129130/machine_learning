#-*- coding:utf-8 -*-
import tree
mydat,label = tree.createDataSet()
#mydat
#tree.calcShannonEnt(mydat)#得到数据集的熵值

#reload(tree)
#tree.splitDataSet(mydat,0,1)#得到第0个特征值为1的元素list
#tree.splitDataSet(mydat,0,0)

#tree.chooseBestFeatureToSplit(mydat)#得到最佳特征值索引

mytree = tree.createTree(mydat,label)#得到决策树信息
mytree


