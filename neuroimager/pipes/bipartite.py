import numpy as np
import networkx as nx
from networkx.algorithms import bipartite


def create_bipartite_graph(adj_mat, num_nodes_set1, num_nodes_set2, corr_thresh=0.2):
    # Filter adjacency matrix based on the correlation threshold
    adj_mat = np.where(np.abs(adj_mat) >= corr_thresh, adj_mat, 0)

    # Create an empty bipartite graph
    bipartite_graph = nx.Graph()

    # Add nodes to the graph
    bipartite_graph.add_nodes_from(range(1, num_nodes_set1 + 1), bipartite=0)
    bipartite_graph.add_nodes_from(
        range(num_nodes_set1 + 1, num_nodes_set1 + num_nodes_set2 + 1), bipartite=1
    )

    # Add weighted edges between the two sets
    for node_set1 in range(1, num_nodes_set1 + 1):
        for node_set2 in range(num_nodes_set1 + 1, num_nodes_set1 + num_nodes_set2 + 1):
            if adj_mat[node_set1 - 1, node_set2 - 1] == 0:
                continue
            weight = round(adj_mat[node_set1 - 1, node_set2 - 1], 2)
            bipartite_graph.add_weighted_edges_from([(node_set1, node_set2, weight)])

    return bipartite_graph


def create_full_graph(adj_mat, corr_thresh=0.2):
    # Filter adjacency matrix based on the correlation threshold
    adj_mat = np.where(np.abs(adj_mat) >= corr_thresh, adj_mat, 0)

    # Create an empty graph
    full_graph = nx.Graph()

    # Add nodes to the graph
    num_nodes = adj_mat.shape[0]
    full_graph.add_nodes_from(range(1, num_nodes + 1))

    # Add weighted edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_mat[i, j] == 0:
                continue
            weight = round(adj_mat[i, j], 2)
            full_graph.add_weighted_edges_from([(i + 1, j + 1, weight)])

    return full_graph


# TODO: Define the metrics that are not directly provided
# TODO: Check if the small_worldness are correct
def small_worldness(G):
    # calculate the small worldness of the graph
    L = nx.average_shortest_path_length(G)
    C = nx.average_clustering(G)
    n = len(G.nodes())
    p = nx.density(G)
    random_graph = nx.watts_strogatz_graph(n, 2, p)
    L_r = nx.average_shortest_path_length(random_graph)
    C_r = nx.average_clustering(random_graph)
    return (C / C_r) / (L / L_r)


def compute_bipa_graph_metrics(bipartite_graph, nodes_set1):
    metrics = {
        "clustering": bipartite.clustering(bipartite_graph),
        "clustering_coefficient": bipartite.cluster.clustering(bipartite_graph),
        "betweenness_centrality": bipartite.betweenness_centrality(
            bipartite_graph, nodes_set1
        ),
        "degree_centrality": bipartite.degree_centrality(bipartite_graph, nodes_set1),
        "closeness_centrality": bipartite.closeness_centrality(
            bipartite_graph, nodes_set1
        ),
    }
    return metrics


def compute_full_graph_metrics(full_graph):
    metrics = {
        "clustering": nx.clustering(full_graph),
        "clustering_coefficient": nx.average_clustering(full_graph),
        "betweenness_centrality": nx.betweenness_centrality(full_graph),
        "degree_centrality": nx.degree_centrality(full_graph),
        "closeness_centrality": nx.closeness_centrality(full_graph),
    }
    return metrics
