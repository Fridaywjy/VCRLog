import os
import sys
sys.path.append("../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import defaultdict
from Data import data_process as dp
from tqdm import tqdm
last_src_ip = ''
last_type = ''

def create_attribute_graph(data):
    G = nx.Graph()
    data_by_line = defaultdict(list)
    for line in data:
        value,value_type,line_number = line
        data_by_line[int(line_number)].append((value,value_type))
    for line_number, items in data_by_line.items():
        block_id = None
        ip_address = None  
        for item in items:
            val, key = item
            if key == 'blockid':
                block_id = val
                if block_id in G.nodes and 'line' in G.nodes[block_id]:
                    G.nodes[block_id]['line'].append(line_number)
                else:
                    G.add_node(val, type='blockid', line=[line_number])
            elif block_id:  
                if key not in ['srcIP', 'srcPort', 'dstIP', 'dstPort','address']:
                    G.nodes[block_id][key] = val
    
                elif key == 'address':
                    if val in G.nodes and 'line' in G.nodes[val]:
                        G.nodes[val]['line'].append(line_number)
                    else:
                        G.add_node(val, type='address',line = [line_number])
                    G.add_edge(block_id,val)
                elif key == 'srcIP':
                    ip_address = val
                    node_id = f"{val}_src"
                    if node_id in G.nodes and 'line' in G.nodes[node_id]:
                        G.nodes[node_id]['line'].append(line_number)
                    else:
                        G.add_node(node_id, type='srcIP', line=[line_number])
                    G.add_edge(block_id, node_id) 
                elif key == 'srcPort' and ip_address and (line_number in G.nodes[f"{ip_address}_src"].get('line')):
                    if 'srcPort' in G.nodes[f"{ip_address}_src"] and val not in G.nodes[f"{ip_address}_src"][key]:
                        G.nodes[f"{ip_address}_src"][key].append(val)
                    else:
                        G.nodes[f"{ip_address}_src"][key] = [val]
                elif key == 'dstIP':
                    ip_address = val
                    node_id = f"{val}_dst"
                    if node_id in G.nodes and 'line' in G.nodes[node_id]:
                        G.nodes[node_id]['line'].append(line_number)
                    else:
                        G.add_node(node_id, type='dstIP', line=[line_number])
                    G.add_edge(block_id, node_id)
                elif key == 'dstPort' and ip_address and (line_number in G.nodes[f"{ip_address}_dst"].get('line')):
                    if 'dstPort' in G.nodes[f"{ip_address}_dst"] and val not in G.nodes[f"{ip_address}_dst"][key]:
                        G.nodes[f"{ip_address}_dst"][key].append(val)
                    else:
                        G.nodes[f"{ip_address}_dst"][key] = [val]
    return G


if __name__ == '__main__':
    data = dp.extract_variable_from_log('data/test.log')
    G = create_attribute_graph(data)
    # iso_nodes = count_isolated_nodes(G)
    # print(iso_nodes)

    for node in G.nodes(data=True):
        print(node)

    # A = nx.to_numpy_matrix(G)
    # print(A)
    # with open('data/node1.txt','w') as file:
    #     for node in G.nodes(data=True):
    #         file.write(str(node))
    # print_matrix(attr_matrix)
    # 可视化图
    # plt.figure(figsize=(50,50))
    # pos = nx.spring_layout(G) 
    # nx.draw(G, pos ,with_labels=True,node_size = 1,width = 1,font_size = 3)
    # plt.show()