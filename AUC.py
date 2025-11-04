#手撕AUC
import itertools
def my_auc(labels,preds):
    n_pos = sum(labels)
    n_neg = len(labels)-n_pos
    lst = sorted(zip(labels,preds),key = lambda x: x[1])
    acc_num = 0
    res = 0
    for pred,pairs in itertools.groupby(lst, key =lambda x :x[1]):
        pos_num = 0
        pair_num = 0
        for label,pred in pairs:
            pair_num+=1
            if label ==1:
                pos_num+=1
        res+=pos_num*acc_num + pos_num*(pair_num-pos_num)/2.0
        acc_num+=pair_num-pos_num
    res/=1.0*(n_neg*n_pos)
    return res
labels = [1,0,1,1,1,1,0,0,1]
preds = [0.1,0.4,0.9,0.4,0.4,0.5,0.1,0.8,0.2]
result = my_auc(labels,preds)
print(result)
