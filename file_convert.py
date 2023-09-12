import graph_tool.all as gt

# # Load the graph
# g = gt.load_graph("multilayer_network_graph.gt")
#
# # Save as GraphML
# g.save("graph.graphml")

# Load the graph
# g = gt.load_graph("multilayer_network_graph.gt")
#
# # Save as edge list
# g.save("graph_edge_list.txt", fmt="edge_list")

import graph_tool.all as gt

# Load the graph
g = gt.load_graph("multilayer_network_graph.gt")

# Print some basic graph statistics
print(f"Number of vertices: {g.num_vertices()}")
print(f"Number of edges: {g.num_edges()}")

# Get the vertex and edge properties
vertex_types = g.vertex_properties["type"]
edge_types = g.edge_properties["type"]

# Count the number of each type of vertex
id_vertices = sum(1 for v in g.vertices() if vertex_types[v] == "ID")
bank_vertices = sum(1 for v in g.vertices() if vertex_types[v] == "bank")
city_vertices = sum(1 for v in g.vertices() if vertex_types[v] == "city")

# Print the counts
print(f"Number of ID vertices: {id_vertices}")
print(f"Number of bank vertices: {bank_vertices}")
print(f"Number of city vertices: {city_vertices}")

# Count the number of each type of edge
id_bank_edges = sum(1 for e in g.edges() if edge_types[e] == "ID-bank")
id_city_edges = sum(1 for e in g.edges() if edge_types[e] == "ID-city")

# Print the counts
print(f"Number of ID-bank edges: {id_bank_edges}")
print(f"Number of ID-city edges: {id_city_edges}")
