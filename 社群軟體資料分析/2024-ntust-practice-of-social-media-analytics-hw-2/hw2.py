import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt

from gcn import SimpleGCN

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# 資料前處理(將training data轉成adjacency matrix)
print("total nodes: ", len(trainData))

## Get the unique nodes
unique_nodes = pd.unique(trainData[['Node1', 'Node2']].values.ravel())
num_nodes = len(unique_nodes)
    
## Create a mapping from node to index
node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

## Fill adjacency matrix
adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
for _, row in trainData.iterrows():
    node1, node2 = row['Node1'], row['Node2']
    idx1, idx2 = node_to_index[node1], node_to_index[node2]
    adjacency_matrix[idx1, idx2] = 1
    adjacency_matrix[idx2, idx1] = 1
    
## Convert the adjacency matrix to a PyTorch tensor
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.int32)

print("Adjacency Matrix:", adjacency_matrix.size())

# 生成label(networkx)
## 使用 NetworkX 獲取連通分量
edges = list(trainData.itertuples(index=False, name=None))
G = nx.Graph()
G.add_edges_from(edges)

## 使用 Louvain 社區檢測算法
communities = list(greedy_modularity_communities(G))

## 創建節點到社區的映射
node_to_community = {}
for i, community in enumerate(communities):
    for node in community:
        node_to_community[node] = i

## 創建邊的標籤
labels = []
for node1, node2 in edges:
    if node_to_community[node1] == node_to_community[node2]:
        labels.append(1)
    else:
        labels.append(0)

trainData['Label'] = labels

# 模型訓練
## get training device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("use device:", device)

model = SimpleGCN(adj=adjacency_matrix.float().to(device), num_class=2)
model.to(device)

## preprocess data, adjacency and label
num_nodes = adjacency_matrix.size(0)
adjacency_matrix = adjacency_matrix + torch.eye(num_nodes) # with self-connection
adjacency_matrix = adjacency_matrix / len(trainData)
# unsqueeze(0): add batch dimension (for pytorch training).
# unsqueeze(0): add channels.
datas = torch.ones(num_nodes).unsqueeze(0).unsqueeze(0)

## put data and label to gpu
datas = datas.to(device)
labels = labels.to(device)

print("data size:", datas.size())
print("adjacency_matrix size:", adjacency_matrix.size())

## create optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

MAX_EPOCHS = 500

for epoch in range(MAX_EPOCHS):
    ## training model
    model.train()
    out = model(datas)
    out = out.squeeze(0).transpose(1, 0)
    loss = criterion(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        corrects = torch.max(out, dim=-1)[1].eq(labels).cpu().sum().item()
        train_acc = float(corrects / out.size(0))

        print("epoch:{:.1E}, loss:{}, accuracy:{}".format(
            epoch+1, float(loss), round(100*train_acc, 2)
        ))

# 模型測試
model.eval()      ## 將模型設置為評估模式
predictions = []  ## 準備輸出結果

## 無需計算梯度(關閉梯度計算，這會節省內存並加速運算。)
with torch.no_grad():
    for _, row in testData.iterrows():
        node1, node2 = row['Node1'], row['Node2']
        
        # 獲取這兩個節點的表示（假設使用的模型可以直接輸入節點）
        node1_data = torch.tensor([node1]).to(device)
        node2_data = torch.tensor([node2]).to(device)
        
        # 模型輸出
        out1 = model(node1_data)
        out2 = model(node2_data)
        
        # 計算兩個節點之間的相似性（例如內積）
        similarity = torch.dot(out1, out2)
        
        # 根據相似性決定是否屬於同一個 community
        prediction = 1 if similarity > 0.5 else 0
        predictions.append(prediction)

## 將預測結果加入到 testData 中
testData['Category'] = predictions

## 保存到 sample_submission.csv
testData[['Id', 'Category']].to_csv('sample_submission.csv', index=False)