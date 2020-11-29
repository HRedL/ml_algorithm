

import numpy as np
import math
import pandas as pd
import sklearn.datasets
def calc_ent(X):
    """
        计算单个属性的熵熵
    """
    headers = X.columns.values
    batch = X.shape[0]
    l = []
    for column in X.columns.tolist()[:-1]:
        l.append(-X[column].value_counts().apply(lambda x: x/batch * math.log2(x/batch)).sum())
    return pd.Series(l,index=headers[:-1])



def calc_condition_ent(X):
    """
        计算条件熵 H(y|x)
    """
    labels = X["label"].value_counts()
    headers = X.columns.values

    batch = X.shape[0]
    l = []
    for column in X.columns.tolist()[:-1]:
        total = 0
        for label in labels.index.tolist():
            temp = X[X["label"] == label]
            num = len(temp)
            total += num/batch * -temp[column].value_counts().apply(lambda x: x / num * math.log2(x / num)).sum()
        l.append(total)

    return pd.Series(l,index=headers[:-1])

def calc_ent_grap(X):
    """
        计算信息增益 grap
    """
    return calc_ent(X) - calc_condition_ent(X)



if __name__=="__main__":
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]

    # (batch,)
    X = pd.DataFrame(dataSet, columns=["length", "circle", "label"])

    result = calc_ent_grap(X)
    print(result[result == result.max()])
    for i in result:
        print(i==result.max())
    print(X['length'].value_counts().index[0])
