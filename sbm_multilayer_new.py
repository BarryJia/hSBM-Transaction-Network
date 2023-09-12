# -*- coding: utf-8 -*-
#!/usr/bin/python3
'''
Description: 2-layer hierarchical SBM based on doc-word network and hyperlink network.

Author: Chris Hyland and Yuanming Tao
'''


import os,sys
import graph_tool.all as gt
import numpy as np
import pandas as pd
import pickle
from collections import Counter,defaultdict

class sbmmultilayer:
    """
    Topic modelling using Hierarchical Multilayer Stochastic Block Model. The
    model is an implementatino of a 2-layer multilayer SBM where the first layer
    is a bipartite network between ids and word-types based off the TopSBM
    formulation. The second layer is a hyperlink network between the ids.

    Parameters
    ----------
    random_seed : int, default = None
        Controls randomization used in topSBM
    n_init : int
         Number of random initialisations to perform in order to avoid a local
         minimum of MDL. The minimum MDL solution is chosen.

    Attributes
    ----------
    g : graph_tool.Graph
        Multilayered networkf.banks: list
        Word nodes.
    ids: list
        Document nodes.
    state:
        Inferred state from graph_tool.
    groups: dict
        Results of group membership from inference.
        Key is an integer, indicating the level of grouping (starting from 0).
        Value is a dict of information about the grouping which contains:
    mdl: float
        The minimum description length of inferred state.
    n_levels: int
        Number of levels in hierarchy of the inferred state.
    """
    def __init__(self, random_seed = None, n_init = 1):
        self.random_seed = random_seed
        self.n_init = n_init

        self.g = None
        self.banks = []
        self.ids = []
        self.cities = []
        self.state = None
        self.groups = {}
        self.mdl = np.nan
        self.n_levels = np.nan


    def make_graph(self, list_banks, list_id, list_cities):
        """
        Load a corpus and generate the multilayered network where one layer
        is the multigraph word-document bipartite network and another is the document
        hyperlink network.

        Document node will be given be the number 0 and word nodes will be
        given the number 1.

        Parameters
        ----------
        list_banks : type
            Description of parameter `list_banks`.
        list_id : type
            Description of parameter `list_id`.
        list_cities : type
            Description of parameter `list_cities`.

        Returns
        -------
        type
            Description of returned object.

        """
        # Number of IDs
        D = len(list_id)

        # # Colors of Nodes
        # color_dict = {0: [0.1216, 0.4667, 0.7059, 1],  # id，red
        #               1: [0, 255, 0, 1],  # bank，green
        #               2: [0, 0, 255, 1]}  # city，blue
        # # Colors of Edges
        # color_dict_edge = {0: [0, 255, 255, 0.8],  # id-city，yellow
        #               1: [128, 0, 128, 0.8]}  # id-bank，purple

        # Colors of Nodes
        color_dict = {0: [0.6196, 0.3490, 0.7098, 1],  # id，purple
                      1: [1.0, 0.4980, 0.0549, 1],  # bank，orange
                      2: [0.1725, 0.6275, 0.1725, 1]}  # city，green
        # Colors of Edges
        color_dict_edge = {0: [0.5294, 0.8078, 0.9804, 0.8],  # id-city，blue
                           1: [0.8392, 0.1529, 0.1569, 0.8]}  # id-bank，red

        # Initialize a graph to store multilayer graph
        g = gt.Graph(directed=False)

        #### Define node properties ####
        # id - 'id', bank - 'name', city - 'name'
        name = g.vp["name"] = g.new_vp("string")
        # id nodes (0), bank nodes (1), city nodes (2)
        kind = g.vp["kind"] = g.new_vp("int")
        # Specify Vertex Layers: bank node: [0]; id node: [0, 1]; city node: [1]
        vlayers = g.vp["vlayers"] = g.new_vp("vector<int>")
        # Specify Vertex Colors
        vprop_color = g.new_vertex_property('vector<double>')
        # Specify Edge Colors
        edge_colors = g.new_edge_property("vector<double>")

        #### Define edge properties ####
        # Edge multiplicity
        edgeCount = g.ep["edgeCount"] = g.new_ep("int")
        edgeDict = {}

        # Need to specify edgetype to indicate which layer an edge is in
        # id-city edge (1) and id-bank edge (0)
        edgeType = g.ep["edgeType"] = g.new_ep("int")

        # Create dictionary of vertices with key-value pair {name: Vertex}
        id_vertices = defaultdict(lambda: g.add_vertex())
        bank_vertices = defaultdict(lambda: g.add_vertex())
        city_vertices = defaultdict(lambda: g.add_vertex())

        # Initialise document nodes based on name of wikipedia article
        # for id in list_id:
        #     d = id_vertices[id]
        #     vlayers[d] = [0,1]

        #### Construct bipartite id-city graph ####
        # Create edges between ids anf.banks
        for doc_id in range(D):
            id = list_id[doc_id]
            city = list_cities[doc_id]
            d = id_vertices[id]
            name[d] = id
            kind[d] = 0  # label 0 is id node
            vlayers[d] = [0, 1]
            c = city_vertices[city]
            name[c] = city
            kind[c] = 2
            vlayers[c] = [0]
            # e = g.add_edge(d, c)
            if (d, c) in edgeDict:
                e = edgeDict[(d, c)]
                edgeCount[e] += 1
            else:
                e = g.add_edge(d, c)
                edgeCount[e] = 1
                edgeDict[(d, c)] = e
            # edgeCount[e] = 1
            edgeType[e] = 0

        #### Construct bipartite id-bank graph ####
        # Create edges between ids anf.banks
        for doc_id in range(D):
            id = list_id[doc_id]
            bank = list_banks[doc_id]
            d = id_vertices[id]
            name[d] = id
            kind[d] = 0  # label 0 is id node
            vlayers[d] = [0, 1]
            c = bank_vertices[bank]
            name[c] = bank
            kind[c] = 1
            vlayers[c] = [1]
            # e = g.add_edge(d, c)
            if (d, c) in edgeDict:
                e = edgeDict[(d, c)]
                edgeCount[e] += 1
            else:
                e = g.add_edge(d, c)
                edgeCount[e] = 1
                edgeDict[(d, c)] = e
            # edgeCount[e] = 1
            edgeType[e] = 1

        for v in g.vertices():
            vprop_color[v] = color_dict[kind[v]]

        for e in g.edges():
            edge_colors[e] = color_dict_edge[edgeType[e]]

        # Initialisf.banks and ids network to model.
        self.g = g
        self.banks = [ g.vp['name'][v] for v in  g.vertices() if g.vp['kind'][v]==1]
        self.ids = [ g.vp['name'][v] for v in  g.vertices() if g.vp['kind'][v]==0]
        self.cities = [g.vp['name'][v] for v in g.vertices() if g.vp['kind'][v] == 2]

        return vprop_color, edge_colors


    def fit(self):
        """
        Fits the hSBM to the undirected, layered multigraph, where the graph in the doc-word layer is bipartite.
        This uses the independent layer multilayer network where we have a degree-corrected SBM.
        """
        # We need to impose constraints on vertices and edges to keep track which layer are they in.
        state_args = {}
        # Vertices with different label values will not be clustered in the same group
        state_args["pclabel"] = self.g.vp["kind"]
        # Split the network in discrete layers based on edgetype. 0 is for word-doc graph and 1 is for hyperlink graph.
        state_args["ec"] = self.g.ep["edgeType"]
        # Independent layers version of the model (instead of 'edge covariates')
        state_args["layers"] = True
        # Edge multiplicities based on occurrences.
        state_args["eweight"] = self.g.ep.edgeCount

        # self.g.save("foo.gt.gz")
        # Specify parameters for community detection inference
        gt.seed_rng(self.random_seed)
        mdl = np.inf
        # Fit n_init random initializations to avoid local optimum of MDL.
        for _ in range(self.n_init):
            # Enables the use of LayeredBlockState. Use a degree-corrected layered SBM.
            state_temp = gt.minimize_nested_blockmodel_dl(self.g, state_args=dict(base_type=gt.LayeredBlockState,
                                                                                  **state_args))
            mdl_temp = state_temp.entropy()
            if mdl_temp < mdl:
                # We have found a new optimum
                mdl = mdl_temp
                state = state_temp.copy()

        self.state = state
        self.mdl = state.entropy()

        # n_levels  = len(self.state.levels)
        # print(f'total group levels: {n_levels}')
        # # Figure out group levels
        # if n_levels == 2:
        #     # Bipartite network
        #     self.groups = { 0: self.get_groupStats(l=0) }
        #     self.n_levels = len(self.groups)
        # # Omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        # else:
        #     self.groups = { level: self.get_groupStats(l=level) for level in range(n_levels - 2) }
        #     self.n_levels = len(self.groups)

    def basic_fit(self):
        """
        Fits the hSBM to the undirected, layered multigraph, where the graph in the doc-word layer is bipartite.
        This uses the independent layer multilayer network where we have a degree-corrected SBM.
        """
        # We need to impose constraints on vertices and edges to keep track which layer are they in.
        state_args = {}
        # Vertices with different label values will not be clustered in the same group
        state_args["pclabel"] = self.g.vp["kind"]
        # Split the network in discrete layers based on edgetype. 0 is for word-doc graph and 1 is for hyperlink graph.
        state_args["ec"] = self.g.ep["edgeType"]
        # Independent layers version of the model (instead of 'edge covariates')
        state_args["layers"] = True
        # Edge multiplicities based on occurrences.
        state_args["eweight"] = self.g.ep.edgeCount

        # self.g.save("foo.gt.gz")
        # Specify parameters for community detection inference
        gt.seed_rng(self.random_seed)
        mdl = np.inf
        # Fit n_init random initializations to avoid local optimum of MDL.
        for _ in range(self.n_init):
            # Enables the use of LayeredBlockState. Use a degree-corrected layered SBM.
            state_temp = gt.minimize_blockmodel_dl(self.g, state_args=dict(base_type=gt.LayeredBlockState,
                                                                                  **state_args))
            mdl_temp = state_temp.entropy()
            if mdl_temp < mdl:
                # We have found a new optimum
                mdl = mdl_temp
                state = state_temp.copy()

        self.state = state
        self.mdl = state.entropy()

    def basic_fit(self):
        """
        Fits the hSBM to the undirected, layered multigraph, where the graph in the doc-word layer is bipartite.
        This uses the independent layer multilayer network where we have a degree-corrected SBM.
        """
        # We need to impose constraints on vertices and edges to keep track which layer are they in.
        state_args = {}
        # Vertices with different label values will not be clustered in the same group
        state_args["pclabel"] = self.g.vp["kind"]
        # Split the network in discrete layers based on edgetype. 0 is for word-doc graph and 1 is for hyperlink graph.
        state_args["ec"] = self.g.ep["edgeType"]
        # Independent layers version of the model (instead of 'edge covariates')
        state_args["layers"] = True
        # Edge multiplicities based on occurrences.
        state_args["eweight"] = self.g.ep.edgeCount

        # self.g.save("foo.gt.gz")
        # Specify parameters for community detection inference
        gt.seed_rng(self.random_seed)
        mdl = np.inf
        # Fit n_init random initializations to avoid local optimum of MDL.
        for _ in range(self.n_init):
            # Enables the use of LayeredBlockState. Use a degree-corrected layered SBM.
            state_temp = gt.minimize_blockmodel_dl(self.g, state_args=dict(base_type=gt.LayeredBlockState,
                                                                                  **state_args))
            mdl_temp = state_temp.entropy()
            if mdl_temp < mdl:
                # We have found a new optimum
                mdl = mdl_temp
                state = state_temp.copy()

        self.state = state
        self.mdl = state.entropy()
    import graph_tool.all as gt
    import numpy as np

    # Define the best_initialization function again for reference
    def best_initialization(self, g, n_max, random_seed=None, state_args=None):
        """
        Identify the best initialization for SBM based on MDL.

        Parameters:
        - g: The graph object
        - n_max: Maximum number of initializations
        - random_seed: Seed for reproducibility
        - state_args: Dictionary of arguments for the BlockState

        Returns:
        - best_init: The initialization that produces the lowest MDL
        - best_mdl: The lowest MDL value
        """
        best_mdl = np.inf
        best_init = None

        if not state_args:
            state_args = {}

        for init in range(n_max):
            if random_seed:
                gt.seed_rng(random_seed + init)  # Vary seed with each initialization

            # Fit the SBM and compute MDL
            state_temp = gt.minimize_nested_blockmodel_dl(g, state_args=dict(base_type=gt.LayeredBlockState,
                                                                             **state_args))
            mdl_temp = state_temp.entropy()

            # Update best MDL and best initialization if needed
            if mdl_temp < best_mdl:
                best_mdl = mdl_temp
                best_init = init

        return best_init, best_mdl

    def updated_fit_method(self, n_max=10):
        state_args = {}
        state_args["pclabel"] = self.g.vp["kind"]
        state_args["ec"] = self.g.ep["edgeType"]
        state_args["layers"] = True
        state_args["eweight"] = self.g.ep.edgeCount

        best_init, best_mdl = self.best_initialization(g=self.g, n_max=10, random_seed=self.random_seed,
                                                       state_args=state_args)

        # After finding the best initialization, you can refit the model using that specific initialization if desired.
        # For now, just returning the best initialization and MDL value.
        return best_init, best_mdl


    def get_groupStats(self, l=0):
        '''
        Description:
        -----------
            Extract statistics on group membership of nodes form the inferred state.
        Returns:  dict
        -----------
            - B_d, int, number of id-groups
            - B_w, int, number of bank-groups
            - B_n, int, number of city-groups

            - p_td_d, array (B_d, D);
                      id-group membership:
                      # group membership of each doc-node, matrix of ones and zeros, shape B_d x D
                      prob that doc-node d belongs to doc-group td: P(td | d)

            - p_tw_w, array (B_w, V);
                      bank-group membership:
                      # group membership of each word-node, matrix of ones or zeros, shape B_w x V
                      prob that word-node w belongs to word-group tw: P(tw | w)

            - p_tw_d, array (B_w, D);
                      id-topic mixtures:
                      ## Mixture of word-groups into ids P(t_w | d), shape B_w x D
                      prob of word-group tw in doc d P(tw | d)

            - p_w_tw, array (V, B_w);
                      per-topic bank distribution, shape V x B_w
                      prob of word w given topic tw P(w | tw)
        '''

        V = self.get_V() # number of bank nodes
        D = self.get_D() # number of id nodes
        N = self.get_N() # number of city nodes

        g = self.g
        state = self.state

        # Retrieve the number of blocks
        # Project the partition at level l onto the lowest level and return the corresponding state.
        state_l = state.project_level(l).agg_state.copy(overlap=True)
        B = state_l.get_B() # number of blocks

        # Returns an edge property map which contains the block labels pairs for each edge.
        # Note that in the text network, one endpoint will be in doc blocks and other endpoint
        # will be in word type block
        state_l_edges = state_l.get_edge_blocks()

        # Count labeled half-edges, total sum is # of edges
        # Number of half-edges incident on bank-node w and labeled as bank-group tw
        n_wb = np.zeros((V,B)) # will be reduced to (V, B_w)

        # Number of half-edges incident on id-node d and labeled as id-group td
        n_db = np.zeros((D,B)) # will be reduced to (D, B_d)

        # Number of half-edges incident on city-node n and labeled as city-group tn
        n_nb = np.zeros((N,B)) # will be reduced to (D, B_d)

        # Number of half-edges incident on id-node d and labeled as bank-group tw
        n_dbw = np.zeros((D,B))  # will be reduced to (D, B_w)

        # Number of half-edges incident on id-node d and labeled as city-group tn
        n_dbn = np.zeros((D,B))  # will be reduced to (D, B_n)

        # Count labeled half-edges, total sum is # of edges
        for e in g.edges():
            # edges in bank network
            if g.ep.edgeType[e] == 0:
                # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in
                # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
                z1, z2 = state_l_edges[e]
                # v1 ranges from 0, 1, 2, ..., D - 1
                # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
                v1 = int(e.source()) # id node index
                v2 = int(e.target()) # bank type node index
                n_wb[v2-D,z2] += 1 # bank type v2 is in topic z2
                n_db[v1,z1] += 1 # id v1 is in doc cluster z1
                n_dbw[v1,z2] += 1 # id v1 has a word in topic z2
            # edges in city network
            elif g.ep.edgeType[e] == 1:
                # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in
                # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
                z1, z2 = state_l_edges[e]
                # v1 ranges from 0, 1, 2, ..., D - 1
                # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
                v1 = int(e.source()) # id node index
                v2 = int(e.target()) # city type node index
                n_nb[v2-D,z2] += 1 # city type v2 is in topic z2
                n_db[v1,z1] += 1 # id v1 is in id cluster z1
                n_dbn[v1,z2] += 1 # id v1 has a city in topic z2

        # Retrieve the corresponding submatrices
        n_db = n_db[:, np.any(n_db, axis=0)] # (D, B_d)
        n_wb = n_wb[:, np.any(n_wb, axis=0)] # (V, B_w)
        n_nb = n_nb[:, np.any(n_nb, axis=0)] # (N, B_n)
        n_dbw = n_dbw[:, np.any(n_dbw, axis=0)] # (D, B_d)
        n_dbn = n_dbn[:, np.any(n_dbn, axis=0)] # (D, B_d)

        B_d = n_db.shape[1]  # number of document groups
        B_w = n_wb.shape[1] # number of bank groups (topics)
        B_n = n_nb.shape[1] # number of city groups (topics)

        # Group membership of each word-type node in topic, matrix of ones or zeros, shape B_w x V
        # This tells us the probability of topic over word type
        p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

        # Group membership of each doc-node, matrix of ones of zeros, shape B_d x D
        p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

        # Mixture of word-groups into ids P(t_w | d), shape B_d x D
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        # Per-topic word distribution, shape V x B_w
        p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]

        result = {}
        result['Bd'] = B_d # Number of document groups
        result['Bw'] = B_w # Number of word groups
        result['p_tw_w'] = p_tw_w # Group membership of word nodes
        result['p_td_d'] = p_td_d # Group membership of document nodes
        result['p_tw_d'] = p_tw_d # Topic proportions over ids
        result['p_w_tw'] = p_w_tw # Topic distribution ovef.banks
        return result


################################################################################
    def plot(self, filename = None,nedges = 1000):
        '''
        Plot the network and group structure by default.
        optional:
        - filename, str; where to save the plot. if None, will not be saved
        - nedges, int; subsample  to plot (faster, less memory)
        '''
        g = self.g
        self.state.draw(output=filename,
                                 subsample_edges = nedges)
