import networkx as nx
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from tqdm import tqdm

# 讀取資料集
train_set = pd.read_csv('2024-ntust-practice-of-social-media-analytics-hw2/train.csv')
test_set = pd.read_csv('2024-ntust-practice-of-social-media-analytics-hw2/test.csv')
submission = pd.read_csv('2024-ntust-practice-of-social-media-analytics-hw2/sample_submission.csv')
print(train_set['node1'])
print(type(train_set['node1']))

# # 資料前處理
# connected_node1 = []
# connected_node2 = []
# unconnected_node1 = []
# unconnected_node2 = []

# ## 先將pairs分類成有連接和沒連接
# for index, row in train_set.iterrows():
#     if row['label'] == 1:
#         connected_node1.append(row['node1'])
#         connected_node2.append(row['node2'])
#     elif row['label'] == 0:
#         unconnected_node1.append(row['node1'])
#         unconnected_node2.append(row['node2'])

# ## 對未連接的pairs隨機抽取一定數量 和連接的pairs組合
# connected_dataset = pd.DataFrame()
# connected_dataset["node1"] = connected_node1
# connected_dataset["node2"] = connected_node2
# connected_dataset["label"] = 1

# unconnected_dataset = pd.DataFrame()
# unconnected_dataset["node1"] = unconnected_node1
# unconnected_dataset["node2"] = unconnected_node2
# unconnected_dataset["label"] = 0

# sample_unconnected_dataset_index = np.random.randint(len(unconnected_dataset), size=len(connected_node1))
# sample_unconnected_dataset = unconnected_dataset.iloc[sample_unconnected_dataset_index]

# dataset = pd.concat([connected_dataset, sample_unconnected_dataset], ignore_index=True)

# print('Preprocessing done!')

## Networkx
# edges = []
# lcn = len(connected_node1)
# for i in range(lcn):
#     edges.append((connected_node1[i], connected_node2[i]))
# G = nx.Graph()
# G.add_edges_from(edges)

# print('Graph done!')

# pa = list(nx.preferential_attachment(G))
# # joblib.dump(pa, "Preferential Attachment.joblib")
# # pa = joblib.load("../input/2022-ntust-practice-of-social-media-analytics-hw1/Preferential Attachment.joblib")
# pa_dict = { (a, b) : c for a, b, c in pa }

# pa_list = []
# for index, i in enumerate(tqdm(dataset.values)):
#     if (i[0], i[1]) in pa_dict:
#         pa_list.append(pa_dict[(i[0], i[1])])
#     elif (i[1], i[0]) in pa_dict:
#         pa_list.append(pa_dict[(i[1], i[0])])
#     else:
#         pa_list.append(0)

# print('Networkx done!')

# 模型訓練
n_estimators = 100
max_depth = 10
learning_rate = 0.05

x_train, y_train = np.array(train_set.drop(['label'], axis=1)), np.array(train_set['label'])
x_test = test_set.drop('idx', axis=1)

model = xgb.XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate)
model.fit(x_train, y_train)

predict = model.predict(x_test)

# 產生submission
submission['ans'] = predict
submission.to_csv('submission.csv', index=False)
print('Model training and testing done!')