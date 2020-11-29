import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn
class RidgeRegression(object):

    def __init__(self, n_iterations=5000, learning_rate=0.0001,alpha=0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.w = None

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,1 ))

    def fit(self,X_train, y_train):

        X_b = np.hstack([X_train, np.ones((len(X_train), 1))])

        dim1, dim2 = X_b.shape
        self.initialize_weights(n_features=X_b.shape[1])
        for _ in range(self.n_iterations):
            diff = (np.dot(X_b, self.w) - y_train).repeat(dim2, axis=1)
            grad_w = X_b * diff // 2 // dim1
            grad_w = np.mean(grad_w, axis=0).reshape(-1, 1) + self.alpha * 2 * self.w
            self.w = self.w - self.learning_rate * grad_w


    def predict(self, X):
        X_b = np.hstack([X, np.ones((len(X), 1))])
        y_pred = X_b.dot(self.w)
        return y_pred

if __name__ == "__main__":
    dataset = datasets.load_boston()
    x = dataset.data
    label = dataset.target
    # 把label变为(?, 1)维度，为了使用下面的数据集合分割
    y = np.reshape(label, (len(label), 1))
    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # 构建模型训练
    lr = RidgeRegression()
    lr.fit(x_train, y_train)
    # 预测
    y_pred = lr.predict(x_test)
    # 绘图
    plt.xlim([0, 50])
    plt.plot(range(len(y_test)), y_test, 'r', label='y_test')
    plt.plot(range(len(y_pred)), y_pred, 'g--', label='y_predict')
    plt.title('sklearn: Linear Regression')
    plt.legend()
    plt.show()

    mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
    print(mse)