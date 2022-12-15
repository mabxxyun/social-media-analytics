import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

def LoadData(filedir, filename):
    path = os.path.join(filedir, filename+'.csv')
    data = pd.read_csv(path)
    return data

def save_link(data, delete_same = True):
    link = []
    for i in range(len(data)):
        link.append((data['node1'][i], data['node2'][i]))
    return link

def node_list_gen(data, data1):
    nodes ={}
    for i in range(len(data)):
        nodes[data[i][0]] = 1
        nodes[data[i][1]] = 1
    for i in range(len(data1)):
        nodes[data1[i][0]] = 1
        nodes[data1[i][1]] = 1
    node_list = []
    for i in nodes:
        node_list.append(i)
    node_list = sorted(node_list)
    return node_list

def graph_construct(link, node_num):
    G = nx.DiGraph()
    G.add_nodes_from(range(node_num))
    G.add_edges_from(link)
    nx.draw(G, with_labels=True)
    return G

def unconnected_link_find(G, node_list):
    adj_G = nx.to_numpy_matrix(G, nodelist = node_list)
    #print(adj_G)
    unconnected_link = []
    for i in tqdm(range(adj_G.shape[0])):
      for j in range(adj_G.shape[1]):
          if adj_G[i, j] == 0:
              tmp = True
              if (node_list[i],node_list[j]) not in pred_link:
                  tmp = False
              if tmp == False:
                  unconnected_link.append([node_list[i],node_list[j]])
    return unconnected_link

def df_generate(link, unconnected_link):
    node_1_linked = [link[i][0] for i in range(len(link))]
    node_2_linked = [link[i][1] for i in range(len(link))]
    df = pd.DataFrame()
    df['node1'] = node_1_linked
    df['node2'] = node_2_linked
    df['link'] = 1
    
    node_1_unlinked = [unconnected_link[i][0] for i in range(len(unconnected_link))]
    node_2_unlinked = [unconnected_link[i][1] for i in range(len(unconnected_link))]
    un_df = pd.DataFrame()
    un_df['node1'] = node_1_unlinked
    un_df['node2'] = node_2_unlinked
    un_df['link'] = 0
    df = df.append(un_df[['node1', 'node2', 'link']], ignore_index=True)
    print(df['link'].value_counts())
    return df

if __name__ == "__main__":
    #load data
    filedir = "2022-ntust-practice-of-social-media-analyticshw1"
    train_data = LoadData(filedir, "data_train_edge")
    
    test_data = LoadData(filedir, "predict")

    print("Data loaded")
    link = save_link(train_data)
    pred_link = save_link(test_data)
    
    node_list = node_list_gen(link, pred_link)
    node_num = len(node_list)
    
    #construct graph
    G = graph_construct(link, node_num)

    #find unconnected link
    unconnected_link = unconnected_link_find(G, node_list)
    
    #dataframe = pos + neg
    df = df_generate(link, unconnected_link)
    
    #train node2vec
    node2vec = Node2Vec(G, dimensions=100, walk_length=16, num_walks=50) #d = 100,  16, 50
    model = node2vec.fit(window=7, min_count=1) #win = 7
    x = [(model.wv[str(i)]+model.wv[str(j)]) for i,j in zip(df['node1'], df['node2'])]
    
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), df['link'], test_size = 0.3, random_state = 35)
    
    #train lightGBM model
    train_data = lgbm.Dataset(xtrain, ytrain)
    test_data = lgbm.Dataset(xtest, ytest)
    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'num_threads' : 2,
        'seed' : 76
        }
    
    model1 = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)
    predictions1 = model1.predict(xtest)

    #predict link
    y = []
    for i in range(len(pred_link)):
        y.append(model.wv[str(pred_link[i][0])]+model.wv[str(pred_link[i][1])])
    
    predictions = model1.predict(np.array(y))
    
    prediction = predictions
    result = []
    for i in range(len(pred_link)):
        if prediction[i]>0.5:
            result.append(1)
        else:
            result.append(0)
    
    output_df = pd.DataFrame()
    output_df['predict_nodepair_id'] = range(len(pred_link))
    output_df['ans'] = result
    output_df.to_csv("result.csv", index=False)
    print("File is saved.")
