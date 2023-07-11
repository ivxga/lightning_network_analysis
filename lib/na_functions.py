import os
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from itertools import combinations, permutations

import plotly.express as px
import plotly.graph_objects as go

pd.options.plotting.backend = 'plotly'

def load_gml(filename):
    return nx.read_gml(filename)

def write_gml(G, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    nx.write_gml(G, filename)

def save_graph_plot(G, filename, figsize=[30,30], res=100):
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(G, ax=ax, with_labels=True, font_size=5)
    fig.savefig(filename, dpi=res)

def local_clustering_coeff(G):
    local_clustering_coeff = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        edges_between_neighbors = 0
        possible_edges_between_neighbors = (len(neighbors) * (len(neighbors)-1)) / 2
        
        for a,b in combinations(neighbors, 2):
            if G.has_edge(a,b):
                edges_between_neighbors+=1
        local_clustering_coeff[node] = edges_between_neighbors/possible_edges_between_neighbors if possible_edges_between_neighbors else 0

    return local_clustering_coeff

def get_all_triples(G):
    triples = []
    for t in permutations(G.nodes(), 3):
        if G.has_edge(t[0], t[1]) and G.has_edge(t[1],t[2]) and t[::-1] not in triples:
            triples.append(t)
    return triples

def global_clustering_coeff(G):
    nbr_triples = len(get_all_triples(G))
    nbr_closed_triples = sum(nx.triangles(G).values())
    return nbr_closed_triples/nbr_triples if nbr_triples else 0

def get_subgraph(G, subcomponent):
    subgraph = G.copy()
    subgraph.remove_nodes_from(set(G.nodes()) - subcomponent)
    return subgraph

def get_giant_component(G, is_directed=False):
    undirected_giant_component = get_subgraph(nx.Graph(G), max(nx.connected_components(nx.Graph(G)), key=lambda x: len(x)))
    if not is_directed:
        return undirected_giant_component
    else:
        G = G.copy()
        G.remove_nodes_from(set(G.nodes()) - set(undirected_giant_component.nodes()))
        return G

def get_avg_degree(G, precision=1):
    return round(sum([G.degree[node] for node in G.nodes()]) / nx.number_of_nodes(G), precision)

def get_subcomponents_details(G):
    df = pd.DataFrame({
        'Graph': [get_subgraph(G, comp) for comp in nx.connected_components(G)],
        'Nodes': [nx.number_of_nodes(get_subgraph(G, comp)) for comp in nx.connected_components(G)],
        'Edges': [nx.number_of_edges(get_subgraph(G, comp)) for comp in nx.connected_components(G)],
        'Density': [get_avg_degree(get_subgraph(G, comp))/len(comp) for comp in nx.connected_components(G)],
        'Diameter': [nx.diameter(get_subgraph(G, comp)) for comp in nx.connected_components(G)],
        'Triangles': [int(sum(nx.triangles(get_subgraph(G, comp)).values()) / 3) for comp in nx.connected_components(G)],
        'Avg_Degree': [get_avg_degree(get_subgraph(G, comp)) for comp in nx.connected_components(G)],
        'Avg_Clustering': [nx.average_clustering(get_subgraph(G, comp)) for comp in nx.connected_components(G)],
        'Avg_Shortest_Path': [nx.average_shortest_path_length(get_subgraph(G, comp)) for comp in nx.connected_components(G)],
    }).sort_values(by=['Nodes','Edges'], ascending=False)

    return df

def get_degree_distr(G):
    distribution = Counter([G.degree(node) for node in G])
    return pd.DataFrame({
        'Degree': distribution.keys(),
        'Nbr_of_Nodes': distribution.values(),
        'Probability': map(lambda nodes: nodes/nx.number_of_nodes(G), distribution.values())
    }) 

def random_graph(n, p):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for e in combinations(range(n), 2):
        if random.uniform(0,1) < p:
            G.add_edge(*e)
    return G

def get_top_nodes(G, by, descending=True):
    if by == 'random':
        return random.sample(list(G.nodes()), len(G))
    if by == 'degree':
        return [node for node,_ in sorted(G.degree, key=lambda x: x[1], reverse=descending)]
    if by == 'betweenness':
        return [node for node,_ in sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=descending)]
    if by == 'closeness':
        return [node for node,_ in sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1], reverse=descending)]
    if by == 'pagerank':
        return [node for node,_ in sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=descending)]
    raise ValueError(f'Parameter by={by} not in ["degree", "betweenness", "closeness", "pagerank"]')

def single_removal_attack(G, type, top):
    if type == 'random':
        G.remove_nodes_from(random.sample(G.nodes(), top))
    elif type == 'target':
        G.remove_nodes_from(get_top_nodes(G, by='degree')[:top])
    elif type == 'betweenness':
        G.remove_nodes_from(get_top_nodes(G, by='betweenness')[:top])
    elif type == 'closeness':
        G.remove_nodes_from(get_top_nodes(G, by='closeness')[:top])
    elif type == 'pagerank':
        G.remove_nodes_from(get_top_nodes(G, by='pagerank')[:top])
    else:
        raise ValueError(f'Parameter type={type} not in ["random", "target", "betweenness", "closeness", "pagerank"]')
    return G

def nth_moment(G, n):
    return sum([G.degree[node] ** n for node in G.nodes()]) / len(G)

def get_critical_fraction(G):
    return 1 - 1 / ((nth_moment(G,2) / nth_moment(G,1)) -1)

def graph_stats(G, out_filename):

    undirected_G = nx.Graph(G)
    undirected_giant_component = get_giant_component(undirected_G)

    extension = out_filename.split('.')[-1]
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)

    if extension == 'txt':
        with open(out_filename, 'w') as out:
            out.write('=== Main Graph Stats ===\n')

            out.write(f"\tType: {'Directed Graph' if nx.is_directed(G) else 'Undirected Graph'}\n")
            out.write(f"\tNumber of nodes: {nx.number_of_nodes(G)}\n")
            out.write(f"\tGraph degree (L): {nx.number_of_edges(G)}\n")
            out.write(f"\tAverage degree (<k>): {get_avg_degree(G)}\n")
            out.write(f"\tNetwork density: {get_avg_degree(G) / nx.number_of_nodes(G) if nx.number_of_nodes(G) else 0}\n")
            out.write(f'\tDegree assortativity: {nx.degree_assortativity_coefficient(G) if nx.number_of_edges(G) else 0}\n')
            out.write(f"\tIs the graph connected?: {'Yes' if nx.is_connected(undirected_G) else 'No'}\n")
            out.write(f"\tNumber of isolated nodes: {len(list(nx.isolates(G)))}\n")
            out.write(f"\tNumber of connected components: {len(list(nx.connected_components(undirected_G)))}\n")
            out.write(f'\tNumber of triangles: {int( sum(nx.triangles(undirected_G).values()) / 3 )}\n')
            # out.write(f'\tGlobal Clustering Coefficient: {global_clustering_coeff(undirected_G)}\n')
            out.write(f'\tAverage Clustering Coefficient: {nx.average_clustering(undirected_G)}\n')

            out.write("\n=== Giant Component Stats ===\n")

            out.write(f"\tNumber of nodes: {nx.number_of_nodes(undirected_giant_component)} (~ {round((nx.number_of_nodes(undirected_giant_component)/nx.number_of_nodes(G)) * 100,2) if nx.number_of_nodes(G) and nx.number_of_nodes(undirected_giant_component) else 0}%)\n")
            out.write(f"\tNumber of edges: {nx.number_of_edges(undirected_giant_component)} (~ {round((nx.number_of_edges(undirected_giant_component)/nx.number_of_edges(G)) * 100,2) if nx.number_of_edges(G) and nx.number_of_edges(undirected_giant_component) else 0}%)\n")
            out.write(f"\tDiameter: {nx.diameter(undirected_giant_component)}\n")
    
    elif extension == 'json':
        out = {
            'is_directed': nx.is_directed(G),
            'num_of_nodes': nx.number_of_nodes(G),
            'num_of_edges': nx.number_of_edges(G),
            'average_degree': get_avg_degree(G),
            'density': get_avg_degree(G) / nx.number_of_nodes(G) if nx.number_of_nodes(G) else 0,
            'assortativity': nx.degree_assortativity_coefficient(G) if nx.number_of_edges(G) else 0,
            'is_connected': nx.is_connected(undirected_G),
            'num_of_isolated_nodes': len(list(nx.isolates(G))),
            'num_of_connected_components': len(list(nx.connected_components(undirected_G))),
            'num_of_triangles': int(sum(nx.triangles(undirected_G).values())/3),
            # 'global_clustering_coeff': global_clustering_coeff(undirected_G),
            'average_clustering_coeff': nx.average_clustering(undirected_G),
            'giant_component': {
                'num_of_nodes': nx.number_of_nodes(undirected_giant_component),
                'num_of_nodes_perc': round((nx.number_of_nodes(undirected_giant_component)/nx.number_of_nodes(G)) * 100,2) if nx.number_of_nodes(G) and nx.number_of_nodes(undirected_giant_component) else 0,
                'num_of_edges': nx.number_of_edges(undirected_giant_component),
                'num_of_edges_perc': round((nx.number_of_edges(undirected_giant_component)/nx.number_of_edges(G)) * 100,2) if nx.number_of_edges(G) and nx.number_of_edges(undirected_giant_component) else 0,
                'diameter': nx.diameter(undirected_giant_component),
            }
        }
        json.dump(out, open(out_filename, 'w'), indent=4)
    
    else:
        raise ValueError(f"File extension .{extension} not supported. Choose in ['.json', '.txt']")

def plot_degree_distr(G, out_filename):
    degree_distr = get_degree_distr(G).sort_values(by='Degree')

    fig = px.bar(degree_distr, x='Degree', y='Probability', log_x=True)
    fig.add_traces(go.Scatter(x=degree_distr['Degree'][1:-1], y=degree_distr['Probability'][1:-1], mode='lines', line_width=3))
    fig.update_layout(
        width=900,
        height=600,
        showlegend=False,
        xaxis_title='k',
        yaxis_title='Pk',
    )

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    fig.write_image(out_filename)

def ranking_nodes(G, out_filename, top=10, with_aliases=False):

    hits_hubs, hits_authorities = nx.hits(G, max_iter=3000)
    local_clustering_coefficient = list(filter(lambda x: x[1] > 0, sorted(local_clustering_coeff(G).items(), key=lambda x: x[1], reverse=True)))

    extension = out_filename.split('.')[-1]
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)

    if with_aliases:
        aliases = {}
        for node in G.nodes():
            aliases[node] = G.nodes()[node]['alias']
        
        out = {
            'top': top,
            'degree': [aliases[node] for node in get_top_nodes(G, by='degree')[:top]],
            'pagerank': [aliases[node] for node in get_top_nodes(G, by='pagerank')[:top]],
            'closeness': [aliases[node] for node in get_top_nodes(G, by='closeness')[:top]],
            'betweenness': [aliases[node] for node in get_top_nodes(G, by='betweenness')[:top]],
            'highest_local_clustering': [aliases[node] for node,_ in local_clustering_coefficient[:top]],
            'lowest_local_clustering': [aliases[node] for node,_ in local_clustering_coefficient[-top:]],
            'hits_hubs': [aliases[node] for node,_ in sorted(hits_hubs.items(), key=lambda x: x[1], reverse=True)[:top]],
            'hits_authorities': [aliases[node] for node,_ in sorted(hits_authorities.items(), key=lambda x: x[1], reverse=True)[:top]],
        }
    else:
        out = {
            'top': top,
            'degree': get_top_nodes(G, by='degree')[:top],
            'pagerank': get_top_nodes(G, by='pagerank')[:top],
            'closeness': get_top_nodes(G, by='closeness')[:top],
            'betweenness': get_top_nodes(G, by='betweenness')[:top],
            'highest_local_clustering': [node for node,_ in local_clustering_coefficient[:top]],
            'lowest_local_clustering': [node for node,_ in local_clustering_coefficient[-top:]],
            'hits_hubs': [node for node,_ in sorted(hits_hubs.items(), key=lambda x: x[1], reverse=True)[:top]],
            'hits_authorities': [node for node,_ in sorted(hits_authorities.items(), key=lambda x: x[1], reverse=True)[:top]],
        }

    if extension == 'txt':
        with open(out_filename, 'w') as out:
            out.write(f"=== Ranking Top {top} Nodes ===\n")

            out.write(f"\tTop {top} central nodes according to degree: {out['degree']}\n")
            out.write(f"\tTop {top} central nodes according to pagerank: {out['pagerank']}\n")
            out.write(f"\tTop {top} central nodes according to closeness: {out['closeness']}\n")
            out.write(f"\tTop {top} central nodes according to betweenness: {out['betweenness']}\n")

            out.write(f"\tTop {top} nodes with Highest Local Clustering Coefficient: {out['highest_local_clustering']}\n")
            out.write(f"\tTop {top} nodes with Lowest (but > 0) Local Clustering Coefficient: {out['lowest_local_clustering']}\n")

            out.write(f"\tTop {top} nodes according to HITS hubs: {out['hits_hubs']}\n")
            out.write(f"\tTop {top} nodes according to HITS authorities: {out['hits_authorities']}\n")
    
    elif extension in ['json', 'csv']:
        if extension == 'json':
            json.dump(out, open(out_filename, 'w'), indent=4)
        else: # extension == 'csv'
            out.pop('top')
            pd.DataFrame(out).to_csv(out_filename)
    
    else:
        raise ValueError(f"File extension .{extension} not supported. Choose in ['.json', '.txt', '.csv']")

def removal_attack(G, out_folder, approx=True,
                   fraction = np.linspace(start=0, stop=1, num=21)[:-1],
                   attacks = ['random', 'degree', 'betweenness', 'closeness', 'pagerank']):

    data = pd.DataFrame(columns=['attack_type', 'fraction_removed_nodes', 'diameter', 'giant_comp_size'])
    for attack in attacks:
        top_nodes = get_top_nodes(G, by=attack)

        giant_comp_sizes, diameters = [], []
        for f in fraction:
            attacked_graph = G.copy()
            attacked_graph.remove_nodes_from(top_nodes[:int(f*len(attacked_graph))])
            giant_component = get_giant_component(nx.Graph(attacked_graph))
            if approx:
                diameters.append(nx.approximation.diameter(giant_component))
            else:
                diameters.append(nx.diameter(giant_component))

            giant_comp_sizes.append(nx.number_of_nodes(giant_component))

        data = pd.concat([data, pd.DataFrame(data={'attack_type': attack, 'fraction_removed_nodes': pd.Series(fraction).round(2), 'diameter': diameters, 'giant_comp_size': giant_comp_sizes})])

    os.makedirs(out_folder, exist_ok=True)

    diam = data[['attack_type', 'fraction_removed_nodes', 'diameter']].pivot(index='fraction_removed_nodes', columns='attack_type', values='diameter')
    gcs  = data[['attack_type', 'fraction_removed_nodes', 'giant_comp_size']].pivot(index='fraction_removed_nodes', columns='attack_type', values='giant_comp_size')
    
    diam.to_csv(f'{out_folder}/diameter.csv')
    gcs.to_csv(f'{out_folder}/gc_size.csv')

    diam = diam.reset_index(drop=False)
    gcs = gcs.reset_index(drop=False)

    diam['fraction_removed_nodes'] = (diam['fraction_removed_nodes'] * 100).apply(int)
    diam = diam.set_index('fraction_removed_nodes')
    # diam = diam[['random', 'degree', 'betweenness', 'closeness', 'pagerank']]
    diam.columns = list(map(lambda x: x.capitalize(), diam.columns))
    fig_diam = diam.plot()
    fig_diam.update_layout(
        width=970,
        height=600,
        # showlegend=False,
        legend=dict(title='Measure'),
        xaxis_title='Fraction Removed Nodes (%)',
        yaxis_title='Diameter',
    )
    
    gcs['fraction_removed_nodes'] = (gcs['fraction_removed_nodes'] * 100).apply(int)
    gcs = gcs.set_index('fraction_removed_nodes')
    # diam = diam[['random', 'degree', 'betweenness', 'closeness', 'pagerank']]
    gcs.columns = list(map(lambda x: x.capitalize(), gcs.columns))
    fig_size = gcs.plot()
    fig_size.update_layout(
        width=970,
        height=600,
        # showlegend=False,
        legend=dict(title='Measure'),
        xaxis_title='Fraction Removed Nodes (%)',
        yaxis_title='Giant Component Size',
    )
    fig_size.add_trace(go.Scatter(name='Network Size', x=gcs.index, y=(len(G)*(1-gcs.index/100)).astype(int), mode='markers'))

    fig_diam.write_image(f'{out_folder}/diameter.png')
    fig_size.write_image(f'{out_folder}/gc_size.png')
