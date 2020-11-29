import numpy as np
from homework.utils import *


class MyHMM:
    def __init__(self, word2idx=None):
        # 加载word2idx,词表可从外部传入
        if word2idx is None:
            word2idx = load_word_dict()
        self.word2idx = word2idx

        # 加载idx2tag和tag2idx，idx2tag在解码时使用
        self.idx2tag, self.tag2idx = load_tag_dict()

        self.tag_size = len(self.tag2idx)
        self.vocab_size = len(word2idx)

        # 初始化pi transition, emission
        self.transition = np.zeros([self.tag_size,
                                    self.tag_size])

        self.emission = np.zeros([self.vocab_size,
                                  self.tag_size])

        self.pi = np.zeros(self.tag_size)
        self.epsilon = 1e-8

    def fit(self, words_list, tags_list):
        """
        使用极大似然估计，学习HMM的参数
        :param words_list:
        :param tags_list:
        :return:
        """
        tags_list = [[tag if tag in self.tag2idx.keys() else "un" for tag in tags] for tags in tags_list]
        # 估计emission
        for words, tags in zip(words_list, tags_list):
            for word, tag in zip(words, tags):
                if word not in self.word2idx.keys():
                    word = "UNK"
                self.emission[self.word2idx[word], self.tag2idx[tag]] += 1

        self.emission[self.emission == 0] = self.epsilon
        self.emission /= np.sum(self.emission, axis=0, keepdims=True)
        self.emission = np.log(self.emission)

        # 估计pi和transition
        for tags in tags_list:
            for i in range(len(tags) - 1):
                if i == 0:
                    self.pi[self.tag2idx[tags[i]]] += 1
                self.transition[self.tag2idx[tags[i]], self.tag2idx[tags[i + 1]]] += 1

        self.transition[self.transition == 0] = self.epsilon
        self.transition /= np.sum(self.transition, axis=1, keepdims=True)
        self.transition = np.log(self.transition)

        self.pi[self.pi == 0] = self.epsilon
        self.pi /= np.sum(self.pi)
        self.pi = np.log(self.pi)

    def predict(self, text):
        best_tags = self.viterbi_decode(text)
        return best_tags

    def viterbi_decode(self, words):
        """
        矩阵化的维特比算法
        :param words:
        :return:
        """

        seq_len = len(words)

        sigma_table = np.zeros([seq_len, self.tag_size])
        fi_table = np.zeros([seq_len, self.tag_size])
        # 初始化
        sigma_table[0, :] = self.pi + self.get_b(words[0])
        fi_table[0, :] = np.nan
        # 递推
        for i in range(1, seq_len):
            b = self.get_b(words[i])
            sigma_prev = sigma_table[i - 1, :].reshape(self.tag_size, 1)
            sigma_curr = sigma_prev + self.transition + b
            sigma_table[i, :] = np.max(sigma_curr, axis=0)

            fi_curr = sigma_prev + self.transition
            fi_table[i, :] = np.argmax(fi_curr, axis=0)

        # 回溯
        best_tag_id = int(np.argmax(sigma_table[-1, :]))
        best_tags = [best_tag_id, ]
        for i in range(seq_len - 1, 0, -1):
            best_tag_id = int(fi_table[i, best_tag_id])
            best_tags.append(best_tag_id)

        best_tags = list(reversed(best_tags))
        best_tags = [self.idx2tag[idx] for idx in best_tags]
        return best_tags

    def get_b(self, word):
        idx = self.word2idx.get(word, -1)
        if idx == -1:
            return np.log(np.ones(self.tag_size) / self.tag_size)

        return self.emission[idx, :]


if __name__ == "__main__":
    # 加载数据
    words_list, tags_list = load_train_data()
    # 训练模型
    model = MyHMM()
    model.fit(words_list, tags_list)
    # 模型预测
    test_sentences = load_test_data()
    # test_sentences = [["我","是","化红磊"],["化红磊","是","我"]]
    for sentence in test_sentences:
        predict_tags = model.predict(sentence)
        print(sentence)
        print(predict_tags)
