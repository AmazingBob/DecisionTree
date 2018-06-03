import numpy as np
import math
import shelve

DICT = {"Outlook": ["Sunny", "Overcast", "Rain"],
        "Temperature": ["Hot", "Mild", "Cool"],
        "Humidity": ["High", "Normal"],
        "Wind": ["Weak", "Strong"]}
FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]


class Node(object):
    """
    定义树的节点
    """

    def __init__(self):
        self.dict = {}
        self.child = []  # 叶子节点
        self.X = []  # 当前样本
        self.y = []  # 当前标签
        self.label = -1  # 叶子节点的标签：0-No;1-Yes，非叶子节点或无法判断为-1
        self.key = ""  # 父节点中关键特征对应值
        self.features = []  # 当前特征
        self.key_feature = ""  # 关键特征（最大熵增）
        pass

    def __del__(self):
        pass


class decision_tree(object):
    def __init__(self):
        self.X = []  # 样本
        self.y = []  # 标签
        self.root = Node()
        self.features = FEATURES
        self.dict = DICT
        pass

    def fit(self, X, y):
        """
        :param X: 样本，numpy数组
        :param y: 标签，numpy数组
        :return:
        """
        if self.__valid__(X, y):
            self.X = X
            self.y = y
            self.root.X = self.X
            self.root.y = self.y

        else:
            return
        self.root.features = FEATURES
        self.root.dict = DICT
        self.__depth_first_create__(self.root)
        pass

    def predict(self, X_test):
        """
        使用训练过的决策树预测
        :param X_test: 测试样本
        :return: 预测标签
        """
        for X in X_test:
            node = self.root
            last_feature = 0
            while node.label == -1:
                feature_id = self.features.index(node.key_feature)
                id = X[feature_id]
                try:
                    node = node.child[id]
                    last_feature = feature_id
                except:
                    break
            if node.label == 0:
                print("No")
                continue
            if node.label == 1:
                print("Yes")
                continue
            if node.label == -1:
                X_train = self.X
                X_train = np.delete(X_train, last_feature, axis=1)
                X = np.delete(X, last_feature)
                X = np.tile(X, (X_train.shape[0], 1))
                delta = np.sum((X - X_train) ** 2, axis=1)
                id = np.where(delta == 0)[0]
                pre_y = self.y[id]
                if len(np.where(pre_y == 0)[0]) > len(pre_y) / 2:
                    print("No")
                else:
                    print("yes")
        pass

    def dump(self):
        """
        固化模型
        :return:
        """
        try:
            with shelve.open('model') as model:
                model['tree'] = self.root
                model['y'] = self.y
                model['X'] = self.X
                model['features'] = FEATURES
                model['dict'] = DICT
            print("Model dumped")
            return 1
        except:
            print("Error!")
            return 0
        pass

    def restore(self):
        """
        恢复模型
        :return:
        """
        try:
            with shelve.open('model') as model:
                self.root = model['tree']
                self.y = model['y']
                self.X = model['X']
                self.features = model['features']
                self.dict = model['dict']
            print("Model restored!")
            return 1
        except:
            print("Error!")
            return 0
        pass

    def __valid__(self, X, y):
        """
        :param X: 样本
        :param y: 标签
        :return: 判断输入是否合法，并返回对应bool值
        """
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            n_samples, n_Xd = X.shape
            y = y.reshape((-1, 1))
            n_labels, _ = y.shape
            if n_samples == n_labels:
                return True
        print("Invalid input!")
        return False
        pass

    def __depth_first_create__(self, node):
        X = node.X
        y = node.y
        if len(np.unique(y)) == 1:
            node.label = np.unique(y)[0]
            return
        if len(node.features) == 0:
            del node
            return
        feature_ID, entropy_i = self.__max_entropy__(X, y)
        node.key_feature = node.features[feature_ID]

        values = node.dict[node.key_feature]
        n_child = len(np.unique(X[:, feature_ID]))
        child_features = node.features.copy()
        child_features.pop(feature_ID)
        child_dict = node.dict.copy()
        child_dict.pop(node.key_feature)
        child_X = np.delete(X, feature_ID, 1)
        for child_id in range(n_child):
            child = Node()
            rows = np.where(X[:, feature_ID] == child_id)[0]
            child.X = child_X[rows, :]
            child.y = y[rows]
            child.key = values[child_id]
            child.features = child_features
            child.dict = child_dict
            node.child.append(child)
            self.__depth_first_create__(child)
        return
        pass

    def __max_entropy__(self, X, y):
        """
        :param X: 样本
        :param y: 标签
        :return: 最大熵增特征ID，最大熵增
        """
        n_sample, n_features = X.shape  # 样本数 和 特征数
        entropy = self.__get_entropy__(y)
        entropy_i = []
        # 计算样本的熵
        Entropy = self.__get_entropy__(y)
        # 计算熵增益
        for i in range(n_features):
            tmp_x = X[:, i]
            # 计算条件熵
            feature_range = range(len(np.unique(tmp_x)))  # 特征取值范围
            entropy_c = 0  # 条件熵
            for f_value in feature_range:
                id = np.where(X[:, i] == f_value)[0]
                w = len(id) / n_sample
                tmp_y = y[id]
                entropy_c = entropy_c - w * self.__get_entropy__(tmp_y)
            entropy_i.append(entropy - entropy_c)
        entropy_i = np.array(entropy_i)
        id = np.where(entropy_i == np.max(entropy_i))[0][0]
        return id, entropy_i[id]
        pass

    def __get_entropy__(self, y):
        """
        :param y: 标签
        :return: 信息熵
        """
        n_type = len(np.unique(y))
        n_samples = len(y)
        entropy = 0
        for i in range(n_type):
            p = len(np.where(y == i)[0]) / n_samples
            if p != 0:
                entropy = entropy - p * math.log(p, 2)
        return entropy
        pass
