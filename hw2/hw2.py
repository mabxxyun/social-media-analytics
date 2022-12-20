import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community
#from cdlib import algorithms

def LoadData(filedir, filename):
    path = os.path.join(filedir, filename+'.csv')
    data = pd.read_csv(path)
    return data

def save_link(data, delete_same = True):
    link = []
    for i in range(len(data)):
        link.append((data['Node1'][i], data['Node2'][i]))
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
    G = nx.MultiGraph()
    G.add_nodes_from(range(node_num))
    G.add_edges_from(link)
    #nx.draw(G, with_labels=True)
    return G

if __name__ == "__main__":
    #load data
    filedir = "2022-ntust-practice-of-social-media-analytics-hw2"
    train_data = LoadData(filedir, "train")
    
    test_data = LoadData(filedir, "test")

    print("Data loaded")
    link = save_link(train_data)
    pred_nodes = save_link(test_data)
    
    node_list = node_list_gen(link, pred_nodes)
    node_num = len(node_list)
    print(node_num)
    
    #construct graph
    print("Graph Construct")
    G = graph_construct(link, node_num)
    
    #community
    print("partition")
    partition = community.best_partition(G)
    
    #d = community.generate_dendrogram(G)
    #partition = community.partition_at_level(d, len(d)-1)
    
    #print(set(partition.values()))
    #print(max(partition.values()))
    #print(min(partition.values()))
    #print(len(set(partition.values())))
    

    result = []
    for n1, n2 in pred_nodes:
        if partition[n1] == partition[n2]:
            result.append(1)
        else:
            result.append(0)
    
    output_df = pd.DataFrame()
    output_df['Id'] = range(len(pred_nodes))
    output_df['Category'] = result
    output_df.to_csv("result.csv", index=False)
    print("File is saved.")


