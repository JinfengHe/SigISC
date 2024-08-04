from collections import Counter

data = []
file_path='C:/dataset_unlabeled/activity_unlabeled.txt'
with open(file_path) as file_object:
    for line in file_object:
     line = line.replace('\n', '')
     data.append(line.split(' '))
lll = len(data)
ll = lll * 0.5
k = 3
p = 0.3

def extract_kmers(sequence, k):
    n = len(sequence)
    kmers = []
    for i in range(n - k + 1):
        kmers.append(sequence[i:i + k])
    return kmers

def extract_kmers_for_dataset(dataset):
    all_kmers = []
    for sequence in dataset:
        k_mers = extract_kmers(sequence, k)
        all_kmers.extend(k_mers)
    return all_kmers

# 提取所有序列模式
all_kmers = extract_kmers_for_dataset(data)
all_kmer = [tuple(lst) for lst in all_kmers]

count = Counter(all_kmer)
# 排序
sorted_items = sorted(count.items(), key=lambda item: item[1])

# 使用Counter统计每个值出现的次数
value_counts = Counter(count.values())

# 将value_counts转换为列表，并根据值进行排序
sorted_value_counts = sorted(value_counts.items())

# 输出排序后的值及其出现次数
for value, count in sorted_value_counts:
    print(f"支持度阈值为 {value} 的序列模式有 {count} 个.")


def find_min_max_int1(lst, p):
    l = len(lst)
    m = int(l * (1 - p))
    n = l

    min_val = lst[m][0]
    if min_val <= 1:
        min_val = 2
    max_val = lst[n-1][0]

    while True:
        if max_val <= ll:
            break
        m = m - 1
        n = n - 1
        min_val = lst[m][0]
        if min_val <= 1:
            min_val = 2
        max_val = lst[n - 1][0]

    if m <= 0:
        min_val = 2

    return min_val, max_val

min_val, max_val = find_min_max_int1(sorted_value_counts, p)
print(f"最小支持度阈值: {min_val}, 最大支持度阈值: {max_val}")


def extract_tuples_with_int_range(lst,m, n):
    return [list(tup[0]) for tup in lst if m <= tup[1] <= n]
result = extract_tuples_with_int_range(sorted_items,min_val, max_val)
result = result[::-1]
print(result)
print(len(result))


import Levenshtein

def filter_sequences_by_similarity(sequences, min_threshold, max_threshold):
    filtered_sequences = sequences
    i = 0
    while i < len(filtered_sequences):
        # 检查当前序列与所有其他序列的编辑距离
        distances = [Levenshtein.distance(filtered_sequences[i], other) for other in filtered_sequences if
                     filtered_sequences[i] != other]
        if not all(min_threshold <= dist <= max_threshold for dist in distances):
            filtered_sequences.remove(filtered_sequences[i])
        else:
            i += 1
    return filtered_sequences


min_threshold = 2  # 最小编辑距离阈值
max_threshold = k  # 最大编辑距离阈值

# 筛选序列
result = filter_sequences_by_similarity(result, min_threshold, max_threshold)


re = result[:]
def filter_sequence_by_similarity(sequences, max_threshold):
    filtered_sequences = sequences
    i = 0
    while i < len(filtered_sequences):
        # 检查当前序列与所有其他序列的编辑距离
        distances = [Levenshtein.distance(filtered_sequences[i], other) for other in filtered_sequences if
                     filtered_sequences[i] != other]
        if all(dist == max_threshold for dist in distances):
            filtered_sequences.remove(filtered_sequences[i])
        else:
            i += 1
    return filtered_sequences

max_threshold = k  # 最大编辑距离阈值


result1 = filter_sequence_by_similarity(re, max_threshold)
num = len(result1)
if num != 0:
    result = result1
print(result)
print(len(result))

def contains_subsequence(sequence, subsequence):
    """检查sequence是否包含subsequence"""
    n = len(sequence)
    m = len(subsequence)
    j = 0  # 用于遍历subsequence的指针
    for i in range(n):
        if sequence[i] == subsequence[j]:
            j += 1
            if j == m:
                return True
        elif j > 0:
            i -= 1  # 回溯到上一个与subsequence匹配的元素位置
    return False

def convert_to_binary_dataset(dataset, subsequences):
    """将数据集转换为二值数据集，根据是否包含指定的子序列"""
    binary_dataset = []
    for sequence in dataset:
        binary_sequence = [0] * len(subsequences)
        for i, subsequence in enumerate(subsequences):
            if contains_subsequence(sequence, subsequence):
                binary_sequence[i] = 1
        binary_dataset.append(binary_sequence)
    return binary_dataset

binary_dataset = convert_to_binary_dataset(data, result)