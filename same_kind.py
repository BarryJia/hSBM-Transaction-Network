import pandas as pd
import numpy as np
from sbmtm import sbmtm
import graph_tool.all as gt
import matplotlib.pyplot as plt
import time
from sbm_multilayer_new import *
SEED_NUM = 32


def fit_hyperlink_text_hsbm(edited_text, titles, hyperlinks, N_iter):
    """
    Fit N_iter iterations of doc-network sbm on dataset through agglomerative heuristic
    and simulated annealing.
    """
    hyperlink_text_hsbm_post = []

    for _ in range(N_iter):
        print(f"Iteration {_}")
        # Construct 2-layer network hyperlink-text model and fit multilayer SBM.
        hyperlink_text_hsbm = sbmmultilayer(random_seed=SEED_NUM)
        hyperlink_text_hsbm.make_graph(edited_text, titles, hyperlinks)
        hyperlink_text_hsbm.fit()

        # Retrieve state from simulated annealing hSBM
        hyperlink_text_hsbm_post_state = run_multiflip_greedy_hsbm(hyperlink_text_hsbm)

        # Update hSBM model using state from simulated annealing
        updated_hsbm_model = hyperlink_text_hsbm
        updated_hsbm_model.state = hyperlink_text_hsbm_post_state
        updated_hsbm_model.mdl = hyperlink_text_hsbm_post_state.entropy()
        updated_hsbm_model.n_levels = len(hyperlink_text_hsbm_post_state.levels)

        # Save the results
        hyperlink_text_hsbm_post.append(updated_hsbm_model)

    return hyperlink_text_hsbm_post


def run_multiflip_greedy_hsbm(hsbm_model):
    """
    Run greedy merge-split on multilayer SBM.
    Return:
        hsbm_state - State associated to SBM at the end.
    """
    S1 = hsbm_model.mdl
    print(f"Initial entropy is {S1}")

    gt.mcmc_equilibrate(hsbm_model.state, force_niter=40, mcmc_args=dict(beta=np.inf), history=True)

    S2 = hsbm_model.state.entropy()
    print(f"New entropy is {S2}")
    print(f"Improvement after greedy moves {S2 - S1}")
    print(f"The improvement percentage is {((S2 - S1) / S1) * 100}")

    return hsbm_model.state

df = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/merged.csv')
banks_list = df['Bank_of_Receiver'].to_list()
users_list = df['Sender_customer_Id'].to_list()
city_list = df['address.townOrCity'].to_list()

city_bank_user_sbm = sbmmultilayer(random_seed=SEED_NUM, n_init=10)
vprop_color, edge_colors = city_bank_user_sbm.make_graph(banks_list, users_list, city_list)
#
#
# pagerank_values = gt.pagerank(city_bank_user_sbm.g).a
# gt.pagerank(city_bank_user_sbm.g).a = pagerank_values * 10
# vertex_sizes = gt.pagerank(city_bank_user_sbm.g)
# vertex_sizes = city_bank_user_sbm.g.get_total_degrees(city_bank_user_sbm.g.get_vertices())
# degree_prop = city_bank_user_sbm.g.degree_property_map("out")
# degree_values = degree_prop.a
# log_degree = np.log(degree_values + 100)  # Add 1 to avoid log(0)
# log_degree_prop = city_bank_user_sbm.g.new_vertex_property("double", log_degree)
# pos = gt.sfdp_layout(city_bank_user_sbm.g)
# gt.graph_draw(city_bank_user_sbm.g, pos, vertex_size=log_degree_prop, vertex_fill_color=vprop_color, edge_color=edge_colors, output_size=(1000, 1000), output="/Users/jiashichao/Desktop/Edinburgh/valerio_project/figures/multi_bipartite_graph_2.pdf")
#
SEED_NUM = 32
city_bank_user_sbm.basic_fit()
# best_init, best_mdl = city_bank_user_sbm.updated_fit_method()
# # Get the levels of the hierarchy
# levels = city_bank_user_sbm.state.get_levels()
#
# # Print the number of communities at each level
# for i, s in enumerate(levels):
#     print(f"Level {i}: {len(set(s.get_blocks()))} communities")
# #
# # Get the vertex and edge properties
# vertex_names = city_bank_user_sbm.g.vertex_properties["name"]
# #
# for v in city_bank_user_sbm.g.iter_vertices():
#     if vertex_names[v] == "London":
#         communities = [s.get_blocks()[v] for s in levels]
#         print(f"Vertex {v} is in communities {communities}")
#     elif vertex_names[v] == "HSBC BANK PLC":
#         communities = [s.get_blocks()[v] for s in levels]
#         print(f"Vertex {v} is in communities {communities}")

# print(f"best init is: {best_init}, best mdl is: {best_mdl}")

# Print the number of communities
print(f"Number of communities: {len(set(city_bank_user_sbm.state.get_blocks()))}")

