
import pandas as pd
from sklearn.model_selection import train_test_split

from data_mining.tree_util import calc_ent_grap


class Tree:
  def __init__(self, node_class, class_basis=None):
    self.node_class = node_class
    self.class_basis = class_basis
    self.children = {}


class RandomForest():
    def __init__(self, n_estimators=10,sigma=0.0,random_state=None):
        """
        随机森林参数
        ----------
        n_estimators:      树数量
        random_state:      随机种子，设置之后每次生成的n_estimators个样本集不会变，确保实验可重复
        """
        self.n_estimators = n_estimators
        self.sigma = sigma
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = dict()

    def fit(self, X, Y):
        """模型训练入口"""
        #随机选取特征

        for i in range(self.n_estimators):
            temp_X = X.sample(axis=1,random_state = self.random_state, replace =False).reset_index(drop=True)
            temp_X['label'] = Y
            self.trees.append(self.create_tree(temp_X))



    def create_tree(self, X):
        features = pd.Series(X.columns.values[:-1])
        # 获取当前结点的类别（数据中心label最多的）
        label_value_count = X['label'].value_counts()
        current_node_value = label_value_count.index[0]
        # 1. 若D中所有实例属于同一类Ck，则T为单节点树，并将类Ck作为该节点的类标记，返回T
        if len(label_value_count) == 1:
            return Tree(current_node_value)

        # 2.若 A =空，则T为单节点树，并将D中实例数最大的类Ck作为该节点的标记，返回T
        if len(features) == 0:
            return Tree(current_node_value)
        # 3.否则，计算各特征对D的信息增益，选择信息增益最大的特征Ag
        #a.计算信息增益
        ents = calc_ent_grap(X)
        Ag = ents.idxmax()
        ent_Ag = ents.max()

        # 4.如果Ag的信息增益小于阈值sigma，则置T为单节点树，并将D中是隶属最大的类Ck作为该节点的类标记，返回T
        if ent_Ag < self.sigma:
            return Tree(current_node_value)

        # 5.否则，对Ag的每一可能取值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子节点
        # 由节点及其子节点构成树T，返回T
        # 6.对第i个子结点，以Di为训练集，以A-{Ag}为特征集，递归调用步1-5，得到子树Ti，返回Ti
        Ag_value_counts = X[Ag].value_counts()
        current_tree = Tree(current_node_value, Ag)
        for value, _ in Ag_value_counts.items():

            child_X = X[X[Ag] == value].drop(Ag, axis=1)

            child = self.create_tree(child_X)

            current_tree.children[value] = child
        return current_tree

    def predict(self, X):
        """输入样本，得到预测值"""
        res = {}
        trees = self.trees
        for tree in trees:
            result = self.get_rs(tree, X)
            if result in res.keys():
                res[result] += 1
            else:
                res[result] = 1

        sorted(res.items(),key=lambda x:x[1],reverse=True)

        return list(res.keys())[0]


    def get_rs(self,tree,X):
        if tree.class_basis == None:
            return tree.node_class
        else:
            return self.get_rs(tree.children[X[tree.class_basis]],X)



if __name__ == '__main__':
    with open("BankChurners.csv", 'r') as f:
        datas = pd.read_csv(f)
    X = datas[['Gender', 'Dependent_count', 'Education_Level',
               'Marital_Status', 'Card_Category']]
    Y = datas['Attrition_Flag']

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.01, random_state=1)


    print("随机森林")
    forest = RandomForest()
    forest.fit(train_X, train_Y)
    predcit_Y_list = []

    for _, X in test_X.iterrows():
        predict_Y = forest.predict(X)
        predcit_Y_list.append(predict_Y)

    count = 0
    for pred_y, real_y in zip(predcit_Y_list, test_Y):
        if pred_y == real_y:
            count += 1
    print("准确率：", count / len(predcit_Y_list))