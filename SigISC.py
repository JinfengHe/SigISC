import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import combine_pvalues
from sklearn import metrics
import time
import warnings
warnings.filterwarnings("ignore")


def AssignLeaf(node, N):
    pi = [0] * N
    Clusters = []
    Clusters = getLeafObjsIDs(node, Clusters)
    k = 0
    for c in range(len(Clusters)):
        k += 1
        for obj_id in Clusters[c]:
            pi[obj_id - 1] = k
    return pi


def getLeafObjsIDs(node, Clusters):
    if node is None:
        return Clusters
    if node.left is None and node.right is None:
        Clusters.append(node.data)
        return Clusters
    if node.left is not None:
        Clusters = getLeafObjsIDs(node.left, Clusters)
    if node.right is not None:
        Clusters = getLeafObjsIDs(node.right, Clusters)
    return Clusters


class Node:
    def __init__(self, data=None, left=None, right=None, pval=None, category=None):
        self.data = data
        self.left = left
        self.right = right
        self.pval = pval
        self.category = category

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


def createNode(data):
    return Node(data)


def sum_chi(X, m, q):
    # 计算 pi
    pi = (X[:, m] != q).astype(int) + 1

    # 删除第 m 列
    X_drop = np.delete(X, m, axis=1)
    M_drop = X_drop.shape[1]

    dfs = 0
    chis = 0
    pvalues = []

    for m_drop in range(M_drop):
        x = X_drop[:, m_drop]
        df = len(np.unique(x)) - 1
        if df != 0:
            # 计算交叉表
            tbl, chi, expected, dof = crosstab_with_chi(x, pi)

            # 检查表格是否为 2x2
            if tbl.shape[1] == 2:
                n = tbl.values.sum()  # 总样本大小
                expected = np.zeros(tbl.shape)

                for i in range(2):
                    for j in range(2):
                        expected[i, j] = tbl.iloc[i, :].sum() * tbl.iloc[:, j].sum() / n

                # 检查是否有任何期望频率小于 5
                if np.any(expected.flatten() < 5):
                    # 应用 Yates 校正
                    # 获取 tbl 中第1行第1列的值
                    a = tbl.iloc[0, 0]
                    # 获取 tbl 中第1行第2列的值
                    b = tbl.iloc[0, 1]
                    # 获取 tbl 中第2行第1列的值
                    c = tbl.iloc[1, 0]
                    # 获取 tbl 中第2行第2列的值
                    d = tbl.iloc[1, 1]
                    chi = (n * (np.abs(a * d - b * c) - n / 2) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d))

            if chi == 0 and dof == 0:
                pval = np.nan
            else:
                pval = chi2_cdf(chi, dof)
            pvalues.append(pval)

    pvalues1 = [float(val) for val in pvalues]
    l = len(pvalues1)
    if l == 0:
        pval = np.nan
    else:
        res = combine_pvalues(pvalues1, method='stouffer')
        pval = res.pvalue

    return pi, pval, chis

import mpmath

def chi2_cdf(x, df, dps=500):
    mpmath.mp.dps = dps
    half_df = mpmath.mpf(df) / 2
    half_x = mpmath.mpf(x) / 2
    gamma_half_df = mpmath.gamma(half_df)
    gamma_inc_half_df_half_x = mpmath.gammainc(half_df, half_x)
    cdf = gamma_inc_half_df_half_x / gamma_half_df
    return cdf


def crosstab_with_chi(x, pi):
    x_series = pd.Series(x)
    pi_series = pd.Series(pi)

    # 计算交叉表
    tbl = pd.crosstab(x_series, pi_series)

    # 计算 chi 值
    res = chi2_contingency(tbl, correction=False)
    chi2 = res.statistic
    p = res.pvalue
    dof = res.dof
    expected = res.expected_freq

    return tbl, chi2, expected, dof

def find_best_mq(X):
    # 移除 objsID 列
    X = X[:, 1:]
    M = X.shape[1]

    # 初始化列表来存储 pv, pi, 和 discat
    pv = [[] for _ in range(M)]
    pi = [[] for _ in range(M)]
    discat = [[] for _ in range(M)]

    # 计算每个特征的唯一值
    for m in range(M):
        unique_values, inverse_indices = np.unique(X[:, m], return_inverse=True)
        discat[m] = unique_values

    # 对于每个特征，计算 p-values 和分割点
    for m in range(M):
        Q = len(discat[m])
        if Q != 1:
            for q in range(Q):
                pi_current, pv_current, chi_current = sum_chi(X, m, q)
                pi[m].append(pi_current)
                pv[m].append(pv_current)
        else:
            pi[m] = np.ones((X.shape[0], 1))
            pv[m] = [1]

    # 找到最佳特征 m
    min_pval = min(min(pv_i) for pv_i in pv)
    bm = pv.index(min(pv, key=min))
    # 找到最佳类别
    bq = pv[bm].index(min_pval)
    bcat = discat[bm][bq]

    # 找到最佳分割点
    bpi = pi[bm][bq].reshape(-1, 1)

    return min_pval, bpi, bm, bcat


def Binary_divide(X, Q, h):
    node = createNode(X[:, 0])

    if X.shape[0] <= 5:
        return node

    # 找到最佳分割参数
    min_pval, best_pi, best_m, best_cat = find_best_mq(X)
    h = h + 1
    K = 0.01 / Q ** h
    # 如果min_pval大于阈值，则不再分割
    if min_pval > K:
        return node

    idx_left = np.where(best_pi == 1)[0]  # 获取满足条件的索引
    idx_right = np.where(best_pi == 2)[0]
    X_left = X[idx_left]
    X_right = X[idx_right]

    # 如果左右子数组的行数都大于2，则继续分割
    if X_left.shape[0] > 2 and X_right.shape[0] > 2:
        node.pval = min_pval
        node.category = [best_m, best_cat]
        node.left = Binary_divide(X_left, Q, h)
        node.right = Binary_divide(X_right, Q, h)
    return node


def __max_depth__(node):
    if node is None:
        return -1
    else:
        dl = __max_depth__(node.left)
        dr = __max_depth__(node.right)
        return 1 + max(dl, dr)


def __average_leaf_depth__(node):
    total_depth = 0
    leaf_count = 0

    def traverse(node, current_depth):
        nonlocal total_depth, leaf_count
        if node is None:
            return
        if node.is_leaf():
            total_depth += current_depth
            leaf_count += 1
        else:
            traverse(node.left, current_depth + 1)
            traverse(node.right, current_depth + 1)

    traverse(node, 0)

    if leaf_count == 0:
        return 0
    return total_depth / leaf_count

def __count_leaves__(node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return __count_leaves__(node.left) + __count_leaves__(node.right)


def true_label():
    file_path1='C:/dataset/activity.txt'
    true_labels=[]
    file=open(file_path1)
    for line in file:
        data = line.split('	')
        true_labels.append((data[0]))
    return true_labels


import PatternMining
data = PatternMining.binary_dataset
X = np.array(data, dtype=object)
print(len(X))
y = np.array(true_label())
print(len(set(y)))

N, M = X.shape
Q = M
print(Q)
h = 0

# 添加对象ID列
objsID = np.arange(1, N + 1)
# 合并ID列和X
X = np.column_stack((objsID, X))
#开始计时
t_start = time.perf_counter()

Node = Binary_divide(X, Q, h)

# 分配叶节点ID
pi_Node = AssignLeaf(Node, N)
pi_Node = np.array(pi_Node)


def purity(cluster, label):
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0))/len(cluster)

from sklearn.metrics.cluster import normalized_mutual_info_score
nmi = normalized_mutual_info_score(y, pi_Node)

def f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return  f_beta

t_end = time.perf_counter()
print("test time is: ", t_end - t_start)
print("puity:", purity(pi_Node, y))
print("NMI:", nmi)
print("F1-score", f_measure(y, pi_Node))

max_depth = __max_depth__(Node)
num_leaf = __count_leaves__(Node)
avg_leaf_depth = __average_leaf_depth__(Node)
print("Tree depth:", max_depth)
print("num_leaf:", num_leaf)
print("Average leaf depth:", avg_leaf_depth)