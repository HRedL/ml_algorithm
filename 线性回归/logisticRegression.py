import numpy as np
import math
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.learning_rate= learning_rate
        self.n_iters = n_iters
        self.w = None

    def sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def J(self, theta, X_b, y):
        y_pred = self.sigmoid(X_b.dot(theta))
        return - np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / len(y)

    def dJ(self, theta, X_b, y):
        return X_b.T.dot(self.sigmoid(X_b.dot(theta)) - y) / len(y)

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X_train, y_train):
        X_b = np.hstack([X_train, np.ones((len(X_train), 1))])
        self.initialize_weights(n_features=X_b.shape[1])
        for _ in range(self.n_iters):
            gradient = self.dJ(self.w, X_b, y_train)

            self.w = self.w - self.learning_rate * gradient

    def predict(self, X_predict):
        X_b = np.hstack([X_predict, np.ones((len(X_predict), 1))])
        result = self.sigmoid(X_b.dot(self.w))
        return np.array(result >= 0.5, dtype='int')



if __name__ =="__main__":
    train_X = np.array([[1, 0, 1], [-1, 1, -2], [2, -6, 1],[1,1,1]])
    train_y = np.array([1, 0, 0, 1])
    model = LogisticRegression()
    model.fit(train_X, train_y)
    print(model.predict(np.array([[1,2,1], [-1,0,-1],[5,10,100],[-5,-3,-2]])))