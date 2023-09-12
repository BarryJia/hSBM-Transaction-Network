import pandas as pd
import numpy as np
from sbmtm import sbmtm
import graph_tool.all as gt
import matplotlib.pyplot as plt
# from sbmmultilayer import *
#
# SEED_NUM = 32
#
# input_df_no_outcomes_banks = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/bipartite_adjacency_matrix.csv', index_col='Customer_ID')
# input_df_no_outcomes_cities = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/bipartite_adjacency_matrix_cities.csv', index_col='Customer_ID')
#
# hyperlink_text_hsbm = sbmmultilayer(random_seed=SEED_NUM)
# hyperlink_text_hsbm.make_graph(edited_text, titles, hyperlinks)
# hyperlink_text_hsbm.fit()
# hyperlink_text_hsbm.plot()

input_df_no_outcomes_banks = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/bipartite_adjacency_matrix.csv', index_col='Customer_ID')
input_df_no_outcomes_cities = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/bipartite_adjacency_matrix_cities.csv', index_col='Customer_ID')


transaction_df=pd.DataFrame(np.where(input_df_no_outcomes_banks.eq(1), input_df_no_outcomes_banks.columns, input_df_no_outcomes_banks),
                  index=input_df_no_outcomes_banks.index,
                  columns=input_df_no_outcomes_banks.columns)
features_banks = [[t for t in transaction if t != 0] for transaction in transaction_df.values.tolist()]
transaction_df=pd.DataFrame(np.where(input_df_no_outcomes_cities.eq(1), input_df_no_outcomes_cities.columns, input_df_no_outcomes_cities),
                  index=input_df_no_outcomes_cities.index,
                  columns=input_df_no_outcomes_cities.columns)
features_cities = [[t for t in transaction if t != 0] for transaction in transaction_df.values.tolist()]
id = [h.split()[0] for h in transaction_df.index.values.astype('str')]

df = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/Transfer_Data.csv')
df2 = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/ekko_customers_info.csv')
bank_list = list(set(df.iloc[:, 11]))
user_list = list(set(df.iloc[:, 2]))
city_list = list(set(df2.iloc[:, 3]))

print(len(bank_list))
print(len(user_list))
print(len(city_list))

# Instantiate the graph
g = gt.Graph(directed=False)

# Create property maps for labels and types
label = g.new_vertex_property("string")
type = g.new_vertex_property("int")
color = g.new_vertex_property("string")

# Create dictionaries for banks, users, and cities
banks = {b: g.add_vertex() for b in bank_list}    # replace with your data
users = {u: g.add_vertex() for u in user_list}    # replace with your data
cities = {c: g.add_vertex() for c in city_list}  # replace with your data

# Iterate over each dictionary to set labels and types
for k, v in banks.items():
    label[v] = k
    type[v] = 0

for k, v in users.items():
    label[v] = k
    type[v] = 1

for k, v in cities.items():
    label[v] = k
    type[v] = 2

banks_data = df.iloc[:, 11]
users_data = df.iloc[:, 2]
city_data = df2.iloc[:, 3]
user_city = df2.iloc[:, 0]

edges = []

for user in user_city:
    for u in users_data:
        if user == u:
            edges.append((user, banks_data.loc[list(users_data).index(user)], city_data.loc[list(user_city).index(user)]))

print(len(edges))

# Define color values based on types
color_values = {0: "red", 1: "green", 2: "blue"}  # customize the colors as needed

# Assign colors based on types
for v in g.vertices():
    color[v] = color_values[type[v]]

# Now, add the edges based on your adjacency data
for user_id, bank_id, city_id in edges:  # replace with your data
    g.add_edge(users[user_id], banks[bank_id])
    g.add_edge(users[user_id], cities[city_id])


# Then you can visualize this graph with graph-tool
pos = gt.sfdp_layout(g, eweight=None)
# pos = g.new_vertex_property("vector<double>")

# Arrange vertices
# for i, v in enumerate(banks):
#     pos[v] = (1, i)
# for i, v in enumerate(users):
#     pos[v] = (2, i)
# for i, v in enumerate(cities):
#     pos[v] = (3, i)

gt.graph_draw(g, pos, vertex_text=label, output="multi_bipartite_graph.pdf")

# 运行社区检测算法
state = gt.minimize_nested_blockmodel_dl(g)
state.draw(output="multi_bipartite_graph_new.pdf")

bs = state.get_bs()
bottom_level_blocks = bs[0]

for v in g.vertices():
    block_id = bottom_level_blocks[int(v)]
    print(f'Node {v} is in block {block_id} at the bottom level.')

# 创建一个颜色字典
color_dict = {}

# 创建一个顶点的颜色属性，用于在绘图时给每个顶点着色
vprop_color = g.new_vertex_property('vector<double>')

for level, blocks in enumerate(bs):
    for v in g.vertices():
        block_id = blocks[int(v)]
        if (level, block_id) not in color_dict:
            color = [np.random.rand() for _ in range(3)]  # 使用随机颜色，你也可以使用预定义的颜色方案
            color_dict[(level, block_id)] = color
        vprop_color[v] = color_dict[(level, block_id)]

# 绘制图形
gt.graph_draw(g, vertex_fill_color=vprop_color, vertex_size=20, output_size=(1000, 1000), output="community.pdf")

