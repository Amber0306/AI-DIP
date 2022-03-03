import os
import random

#训练集所占比例
trainval_percent = 0.9
train_percent = 0.9

#标签文件路径
xmlfilepath = 'SSDD\Annotations'
#生成txt目录文件夹所在路径
txtsavepath = 'SSDD\ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)

# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# trainval = random.sample(list, tv)
# train = random.sample(trainval, tr)

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

trainIndex = [2,3,4,5,6,7,8]
valIndex = [0]
testIndex = [1,9]

for i in list:
    name = total_xml[i][:-4] + '\n'
    index = int(name[-2])
    if index not in testIndex:
        ftrainval.write(name)
        if index in trainIndex:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
        

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
