import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
"""
假设我们有如下两个array， y_true由read csv得到，y_score由predict得到
y_true = np.array(
    [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
)
y_score = np.array([
    .9, .8, .7, .6, .55, .54, .53, .52, .51, .505,
    .4, .39, .38, .37, .36, .35, .34, .33, .3, .1
])
"""


def roc(y_true, y_score, pos_label):
    """
    y_true：真实标签
    y_score：模型预测分数
    pos_label：正样本标签，如“1”
    """
    # 统计正样本和负样本的个数
    #num_positive_examples = (y_true == pos_label).sum()
    num_positive_examples = y_true.count(1)
    num_negtive_examples = len(y_true) - num_positive_examples

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1
    
    # 根据排序后的预测分数分别计算fpr和tpr
    for i in np.flip(np.argsort(y_score)):
        # 处理样本预测分数相同的情况
        if y_score[i] != score:
            fpr.append(fp / num_negtive_examples)
            tpr.append(tp / num_positive_examples)
            thresholds.append(score)
            score = y_score[i]
            
        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / num_negtive_examples)
    tpr.append(tp / num_positive_examples)
    thresholds.append(score)

    return fpr, tpr, thresholds

def compute(y_true,y_score,labels):
    #return confusion matrix
    matrix=[0,0,0,0] #TP,FP,FN,TN
    for i in range(len(y_score)):
        #print(y_score[i])
        if labels[i]==1 and y_true[i] == 1:
            matrix[0]+=1
            #print(labels[i],y_true[i])
            #print("TP+1",matrix)
        elif labels[i]==1 and y_true[i] == 0:
            matrix[1]+1
            #print(labels[i],y_true[i])
            #print("FP+1",matrix)
        elif labels[i]==0 and y_true[i]==1:
            matrix[2]+=1
            #print(labels[i],y_true[i])
            #print("FN+1",matrix)
        elif labels[i]==0 and y_true[i] ==0:
            matrix[3]+=1
            #print(labels[i],y_true[i])
            #print("TN+1",matrix)
    if (matrix[0]+matrix[1])==0 or (matrix[0]+matrix[3]) == 0:
        return matrix,0,0
    return matrix, matrix[0]/(matrix[0]+matrix[1]),matrix[0]/(matrix[0]+matrix[3])



predict_csv=sys.argv[1]
target_csv=sys.argv[2]
predict_label=sys.argv[3]
target_label=sys.argv[4]

with open(predict_csv,'r') as csvfile:
    predict_reader = csv.DictReader(csvfile)
    predict_column = [row['yscore'] for row in predict_reader]
with open(predict_csv,'r') as csvfile:
    predict_reader = csv.DictReader(csvfile)
    predict_label = [row[predict_label] for row in predict_reader]
#print(predict_column)

with open(target_csv,'r') as csvfile:
    target_reader = csv.DictReader(csvfile)
    target_column = [row[target_label] for row in target_reader]
#print(target_column)
y_true = []
y_score=[]
predict_labels=[]
for i in predict_column:
    y_score.append(float(i))
for i in target_column:
    if i=="Yes":
        y_true.append(1)
    elif i == "No":
        y_true.append(0)
#print(predict_label)
for i in predict_label:
    if i=="yes":
        predict_labels.append(1)
    else:
        predict_labels.append(0)
#print(y_true)
#print(y_score)
#print(len(predict_labels),len(y_true),len(y_score))

fpr, tpr, thresholds = roc(y_true, y_score, pos_label=1)
matrix, precision, recall = compute(y_true,y_score,predict_labels)
if (precision+recall)==0:
    f1 = 0
else:
    f1 = 2*((precision*recall)/(precision+recall))
#print(f"precision = {precision}, recall = {recall}, f1_score = {f1}")
#print(matrix)

plt.plot(fpr, tpr)
plt.axis("square")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
#plt.show()
plt.savefig(f"path/to/predictions/roc_{target_label}.png")

with open(f"path/to/predictions/f1_{target_label}.txt","w") as f:
    f.write(f"precision = {precision}, recall = {recall}, f1_score = {f1}\n")
    f.write(f"TP={matrix[0]},FP={matrix[1]},FN={matrix[2]},TN={matrix[3]}\n")
