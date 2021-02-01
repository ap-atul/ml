from collections import Counter
import numpy as np

def gini_index(classes):  # # Gini = 1−∑pi2
    ones, size = np.count_nonzero(c for c in classes), len(classes)
    prob_one, prob_zero = ones / size, (size - ones) / size
    return 1 - (prob_zero ** 2 + prob_one ** 2)

def gini_gain(prev_classes, curr_classes): # Gini(A) = Gini(D) − Gini A (D)
    prev_gini, prev_len, curr_gini = gini_index(prev_classes), len(prev_classes), 0
    if len(curr_classes[0]) == 0 or len(curr_classes[1])  == 0:
        return 0

    for cc in curr_classes:
        curr_gini += gini_index(cc) * float(len(cc)) / prev_len

    return prev_gini - curr_gini

def get_most_freq_feature(classes):
    k, v = Counter(classes).most_common(1)[0]
    return k

def partition_classes(x, y, split_att, split_val):
    x_left = x_right = y_left = y_right = list()
    for i in range(len(x)):
        if float(x[i][split_att]) <= split_val:
            x_left.append(x[i])
            y_left.append(y[i])
        else:
            x_right.append(x[i])
            y_right.append(y[i])
    return x_left, x_right, y_left, y_right

class DecisionNode:
    def __init__(self, left, right, dec_func, class_label=None):
        self._left, self._right, self._dec_func, self._class_label = left, right, dec_func, class_label

    def decide(self, feature):  # feature is a list(int)
        if self._class_label is not None:
            return self._class_label
        elif self._dec_func(feature):
            return self._left.decide(feature)
        return self._right.decide(feature)

class DecisionTree:
    def __init__(self, depth_limit=float('inf')):
        self._root, self._depth = None, depth_limit

    def fit(self, features, classes):
        self._root = self._build_tree(features, classes)

    def _build_tree(self, features, classes, depth=0):
        best_gini_gain = best_col_index = best_col_thresh = -1

        if len(classes) == 0:
            return None
        elif len(classes) == 1 or np.all(classes[0] == classes[: ]):  # all are of same class
            return DecisionNode(None, None, None, classes[0])
        elif depth == self._depth:
            return DecisionNode(None, None, None, get_most_freq_feature(classes))
        else: # recursive build

            for col in range(features.shape[1]):
                col_mean = np.mean(features[:, col])
                new_classes = list()
                temp_x_left, temp_x_right, temp_y_left, temp_y_right = partition_classes(features, classes, col, col_mean)
                new_classes += [temp_y_left, temp_y_right]
                col_gini_gain = gini_gain(classes, new_classes)

                if col_gini_gain > best_gini_gain:
                    best_gini_gain, best_col_index, best_col_thresh = col_gini_gain, col, col_mean

            x_left, x_right, y_left, y_right = partition_classes(features, classes, best_col_index, best_col_thresh)
            depth += 1

            left_tree = self._build_tree(np.array(x_left), np.array(y_left), depth)
            right_tree = self._build_tree(np.array(x_right), np.array(y_right), depth)
            return DecisionNode(left_tree, right_tree, lambda feature: feaature[best_col_index] < best_col_thresh)

    def classify(self, features):
        return [self._root.decide(feature) for feature in features]
