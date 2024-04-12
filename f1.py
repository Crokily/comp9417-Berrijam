#接收数据类型pred：bool， 预测值； actual：bool，实际值
#原有变量matrix=[0,0,0,0], TP,FP,FN,TN
#场景为一张图片输入，生成一对预测值与实际值
def receive_prediciton(matrix,pred,actual):
    if pred == True:
        if actual == True:
            #TP+1
            matrix[0]+=1
        else:
            #FP+1
            matrix[1]+=1
    else:
        if actual == True:
            #FN+1
            matrix[2]+=1
        else:
            #TN+1
            matrix[3]+=1

def compute_f1(matrix):
    #precison = TP/(TP+FP)
    precision = matrix[0]/(matrix[0]+matrix[1])
    #recall = TP/(TP+FN)
    recall = matrix[0]/(matrix[0]+matrix[3])
    #f1 = 2*((precision*recall)/(precision+recall))
    f1 = 2*((precision*recall)/(precision+recall))
    return f1