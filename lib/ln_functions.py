import os
import json
import random
import numbers
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import lib.charts as charts
import lib.na_functions as na

def convert_to_gml(filename, output_filename=None):
    '''
    Input: json graph filename (path)
    Output: gml graph filename (path)
    '''
    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    gml = '''graph [
        label "Lightning Network - Graph Topology"
        creator "@ivxga (ivangallo99@gmail.com)"
        multigraph 1
        directed 1'''

    node_map = {}
    for i, node in enumerate(data['nodes']):
        node_map[node['pub_key']] = i+1

        node_gml = f'''
        node [
            id {i+1}
            label "{node['pub_key']}"
            alias "{''.join([i for i in node['alias'] if ord(i) < 128]).replace('"','') if node['alias']!='' else i+1}"
        ]'''

        gml += node_gml

    for edge in data['edges']:
        edge_gml = f'''
        edge [
            source {node_map[edge['node1_pub']]}
            target {node_map[edge['node2_pub']]}
            label "{edge['channel_id']}"
            capacity {edge['capacity']}
        ]'''
        gml += edge_gml

    gml += '\n]'

    output_filename = output_filename if output_filename else filename.replace('.json', '.gml')
    with open(output_filename, 'w') as gml_file:
        gml_file.write(gml)

    return output_filename

def get_features(explained=False):
    features = {
        'amp':                      "enables nodes to support Atomic Multi-Path Payments. This feature allows a single payment to be split across multiple paths within the Lightning Network.",
        
        # 'anchor-commitments':       "utilizzo di transazioni di ancoraggio (anchor transactions) per aumentare la sicurezza dei canali in caso di fallimento o di chiusura improvvisa dei partecipanti.",
        'anchor-commitments':       "mechanism for more efficient fee negotiation and channel updates. Nodes can update channel balances without requiring costly on-chain transactions.",
        
        'anchors-zero-fee-htlc-tx': "enhances the anchor commitment mechanism by enabling zero-fee Hash Time-Locked Contract transactions.",
        'data-loss-protect':        "provides protection against data loss during channel closure. Even if a channel's data is lost, nodes can recover and resume their operations without significant disruption.",
        'explicit-commitment-type': "enables nodes to specify an explicit type for channel commitment transactions",
        'gossip-queries':           "indicates support for gossip queries, which are used to gather up-to-date information about the Lightning Network's state, like channel capacities, fee policies, and network topology.",
        'initial-routing-sync':     "improves the process of establishing routing information when a new node joins the Lightning Network.",

        # 'keysend':                  "consente di inviare pagamenti senza dover prima stabilire un canale di pagamento, semplificando il processo di pagamento.",
        'keysend':                  "allows nodes to send payments directly to a destination node without requiring an invoice.",

        'multi-path-payments':      "allows nodes to perform multi-path payments, which involve splitting a payment into multiple routes inside the network.",
        'payment-addr':             "support for payment addresses, which are static alternative addresses used for receiving payments.",
        'scid-alias':               "enables the use of Short Channel ID aliases for channels identification, allowing nodes to assign them human-readable labels",
        'script-enforced-lease':    "introduces lease-based channel opening, where nodes enter into time-limited channel commitments, allowing nodes to close channels automatically after a predefined period.",
        'shutdown-any-segwit':      "adds the ability to execute a graceful channel shutdown using any Segregated Witness (SegWit) output.",
        'static-remote-key':        "involves the use of a static remote key for commitment transactions within the Lightning Network.",
        'tlv-onion':                "refers to the use of Type-Length-Value (TLV) format for constructing onion payloads (encrypted packets used for routing payments anonymously) within the network.",
        'unknown':                  "refers to a not-identified or a not-standard features: could refers also to features that are still in development.",
        
        # 'upfront-shutdown-script':  "consente ai partecipanti di stabilire uno script di chiusura del canale al momento dell'apertura del canale, migliorando la sicurezza del canale.",
        # 'upfront-shutdown-script':  "allows the usage of an upfront shutdown script in channel closure transactions.",
        'upfront-shutdown-script':  "allows the usage of an upfront shutdown script in channel opening transactions, providing additional security and flexibility.",
        
        'wumbo-channels':           "support for wumbo channels, which have larger channel capacity limits than standard channels, overcoming the typically limit of 0.1677 BTC.",
        'zero-conf':                "adds the support for zero-confirmation channels within the Lightning Network, enabling faster channel.",
    }
    return features if explained else list(features.keys())

def get_features_data(data_folder,
                      keep_features=['amp', 'anchors_zero_fee_htlc_tx', 'data_loss_protect', 'explicit_commitment_type', 'gossip_queries', 'keysend', 'multi_path_payments', 'payment_addr', 'scid_alias', 'script_enforced_lease', 'shutdown_any_segwit', 'static_remote_key', 'tlv_onion', 'unknown', 'upfront_shutdown_script', 'wumbo_channels', 'zero_conf'],
                      keep_stats=['num_of_nodes', 'num_of_edges', 'density', 'assortativity', 'average_degree', 'average_clustering_coeff', 'num_of_triangles', 'giant_component_num_of_nodes', 'giant_component_num_of_edges']):

    files = list(filter(lambda x: '.json' in x, [f'{data_folder}/{file}' for file in os.listdir(data_folder)]))

    data = {file.split('/')[-1][:-17]: pd.json_normalize(json.load(open(file, 'r')), sep='_').to_dict(orient='index')[0] for file in files}
    data = pd.DataFrame(data).T.reset_index().sort_values(by='index', ascending=True).set_index('index')

    data = data[keep_stats]
    data = data.drop(index=set(data.index)-set(keep_features))
    
    data['gc_nodes_perc'] = ((data['giant_component_num_of_nodes'] / data['num_of_nodes']) * 100).astype(float).round(2)
    data['gc_edges_perc'] = ((data['giant_component_num_of_edges'] / data['num_of_edges']) * 100).astype(float).round(2)

    return data

def subgraph_by_feature(data, feature):
    subgraph = {}
    subgraph['nodes'] = []
    subgraph['edges'] = []

    for node in data['nodes']:
        if feature in set([feat['name'] for feat in node['features'].values()]):
            subgraph['nodes'].append(node)

    pub_keys = set()
    for node in subgraph['nodes']:
        pub_keys.add(node['pub_key'])
    for edge in data['edges']:
        if edge['node1_pub'] in pub_keys and edge['node2_pub'] in pub_keys:
            subgraph['edges'].append(edge)
    
    json.dump(subgraph, open('tmp.json', 'w'))
    G = na.load_gml(convert_to_gml('tmp.json'))
    os.remove('tmp.json')
    os.remove('tmp.gml')

    return G

def compare_feature_stats(features_data, out_folder, date, main_graph_stats_filename):
    graph_stats = json.load(open(main_graph_stats_filename, 'r'))

    os.makedirs(os.path.dirname(f'{out_folder}/{date}/features/statistics/_'), exist_ok=True)

    fig = charts.double_scale_bar_chart(features_data, ['num_of_nodes', 'num_of_edges'])
    fig.write_image(f'{out_folder}/{date}/features/statistics/multiscale_nodes_edges.png')

    fig = charts.double_scale_bar_chart(features_data, ['average_degree', 'average_clustering_coeff'], hlines=[(graph_stats['average_degree'], "darkblue"), (graph_stats['average_clustering_coeff'], 'darkred')])
    fig.write_image(f'{out_folder}/{date}/features/statistics/multiscale_avgdegree_avgcoeff.png')
    
    fig = charts.double_scale_bar_chart(features_data, ['gc_nodes_perc', 'gc_edges_perc'], hlines=[(graph_stats['giant_component']['num_of_nodes_perc'], "darkblue"), (graph_stats['giant_component']['num_of_edges_perc'], 'darkred')])
    fig.write_image(f'{out_folder}/{date}/features/statistics/multiscale_gcnodes_gcedges.png')

def get_node_aliases(data_folder, dates):
    alias = {}
    for file in [f'{data_folder}/network_graph_{date}.json' for date in dates]:
        for node in json.load(open(file, 'r'))['nodes']:
            alias[node['pub_key']] = node['alias']
    return alias

def generate_synt_graph(G, iteration=5, removal_fraction=0.05, alpha=0.19, beta=0.80, gamma=0.01, delta_in=0.25, delta_out=0, seed=0, out_filename=None):

    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha+beta+gamma must equal 1.")

    def _choose_node(candidates, node_list, delta):
        if delta > 0:
            bias_sum = len(node_list) * delta
            p_delta = bias_sum / (bias_sum + len(candidates))
            if seed.random() < p_delta:
                return seed.choice(node_list)
        return seed.choice(candidates)

    def _add_node(node_list, cursor):
        node_list.append(cursor)
        return cursor, cursor+1

    n = len(G)
    G = G.copy()
    seed = random.Random(seed)

    num_of_components = len(list(nx.connected_components(nx.Graph(G))))

    for _ in range(iteration):
        # remove a percentage of random node
        G.remove_nodes_from(random.sample(list(G.nodes()), int(len(G)*removal_fraction)))
        
        # filter keeping only the giant components
        G = na.get_giant_component(G, is_directed=True)

        # recreating a similar amount of other components
        G.remove_nodes_from(random.sample(list(G.nodes()), num_of_components*3))

        # pre-populate degree states
        vs = sum((count * [idx] for idx, count in G.out_degree()), [])
        ws = sum((count * [idx] for idx, count in G.in_degree()), [])

        # pre-populate node state
        node_list = list(G.nodes())

        # see if there already are number-based nodes
        numeric_nodes = [n for n in node_list if isinstance(n, numbers.Number)]
        if numeric_nodes: cursor = max(int(n.real) for n in numeric_nodes) + 1 # set cursor for new nodes
        else:             cursor = 0                                           # or start at zero

        while len(G) < n:
            r = seed.random() # random choice in alpha, beta, gamma

            if r < alpha:
                v, cursor = _add_node(node_list, cursor)   # add new node v
                w = _choose_node(ws, node_list, delta_in)  # choose w according to in-degree and delta_in
            elif r < alpha + beta:
                v = _choose_node(vs, node_list, delta_out) # choose v according to out-degree and delta_out
                w = _choose_node(ws, node_list, delta_in)  # choose w according to in-degree and delta_in
            else:
                v = _choose_node(vs, node_list, delta_out) # choose v according to out-degree and delta_out
                w, cursor = _add_node(node_list, cursor)   # add new node w
            
            # add edge to graph
            G.add_edge(v, w)

            # update degree states
            vs.append(v)
            ws.append(w)

    if out_filename:
        na.write_gml(G, out_filename)

    return G

def temporal_analysis(dump_dates, data_folder, out_folder):

    ### Analyze Main Graph Statistics

    features = ['num_of_nodes', 'num_of_edges', 'average_degree', 'density', 'assortativity', 'average_clustering_coeff', 'num_of_triangles', 'giant_component_nodes']

    data = []
    for date in dump_dates:
        with open(f'{out_folder}/{date}/statistics/graph_stats.json', 'r') as stats_file:
            json_data = json.loads(stats_file.read())
            json_data['giant_component_nodes'] = json_data['giant_component']['num_of_nodes']
            for k in list(json_data):
                if k not in features:
                    del json_data[k]
        with open(f'{data_folder}/network_graph_{date}.json', 'r') as json_graph:
            graph_data = json.loads(json_graph.read())
            json_data['tot_capacity'] = sum([int(edge['capacity']) for edge in graph_data['edges']])
        json_data['date'] = date
        data.append(json_data)
    
    data = pd.DataFrame(data)
    data['avg_capacity'] = data['tot_capacity'] / data['num_of_edges']

    os.makedirs(os.path.dirname(f'{out_folder}/temporal_analysis/statistics/_'), exist_ok=True)
    data.to_csv(f'{out_folder}/temporal_analysis/statistics/main_graph_statistics.csv', index=None)

    data = data[data['date'].apply(lambda x: int(x[6]) > 3)]

    fig = charts.double_scale_line_chart(data, 'num_of_nodes', 'num_of_edges')
    fig.write_image(f'{out_folder}/temporal_analysis/statistics/num_nodes_and_edges.png')

    fig = charts.double_scale_line_chart(data, 'density', 'assortativity')
    fig.write_image(f'{out_folder}/temporal_analysis/statistics/density_and_assortativity.png')

    fig = charts.double_scale_line_chart(data, 'tot_capacity', 'avg_capacity')
    fig.write_image(f'{out_folder}/temporal_analysis/statistics/tot_and_avg_capacity.png')

    ### Analysis on variation Top Ranking Nodes

    measures = ['degree', 'pagerank', 'closeness', 'betweenness', 'highest_local_clustering', 'lowest_local_clustering', 'hits_hubs', 'hits_authorities']

    for measure in measures:
        data = pd.DataFrame()
        for date in dump_dates:
            json_data = json.load(open(f'{out_folder}/{date}/statistics/ranking_nodes.json', 'r'))
            n_top_ranked = json_data.pop('top')

            df = pd.DataFrame(json_data)[[measure]]
            df['rank'] = df.index + 1
            df['date'] = date

            data = df if data.empty else pd.concat([data, df])

        measure_data = data.pivot(index='date', columns=measure, values='rank')

        plt.figure(figsize=(20, 5))
        axes = charts.bumpchart(measure_data, aliases=get_node_aliases(data_folder, dump_dates), top=n_top_ranked, line_args={"linewidth": 5, "alpha": 0.5}, scatter_args={"s": 100, "alpha": 0.75})
        axes[0].set(xlabel="Dump Dates", ylabel="Rank", title=f"Top {n_top_ranked} nodes ranking variation according to {' '.join(list(map(str.capitalize, measure.split('_'))))}")
        plt.tight_layout()

        os.makedirs(os.path.dirname(f'{out_folder}/temporal_analysis/ranking_nodes/{measure}.png'), exist_ok=True)
        plt.savefig(f'{out_folder}/temporal_analysis/ranking_nodes/{measure}.png')

    ### Analysis on Diameter after attacks

    attacks_data = {}
    for measure in ['random', 'degree', 'closeness', 'pagerank', 'betweenness']:
        attacks_data[measure] = pd.DataFrame(columns=[date for date in dump_dates])
    for date in dump_dates:
        diam_data = pd.read_csv(f'{out_folder}/{date}/attacks/diameter.csv', index_col=0)
        for measure in ['random', 'degree', 'closeness', 'pagerank', 'betweenness']:
            attacks_data[measure][date] = diam_data[measure]

    colors = {
        'random': 'rgb(250, 168, 105)',
        'degree': 'rgb(111, 223, 192)',
        'closeness': 'rgb(236, 98, 72)',
        'pagerank': 'rgb(188, 140, 249)',
        'betweenness': 'rgb(118, 131, 249)',
    }
    
    os.makedirs(os.path.dirname(f'{out_folder}/temporal_analysis/attacks/_.png'), exist_ok=True)
    fig = charts.line_chart_min_max_all(attacks_data, colors, xaxis_title='Fraction of removed nodes', yaxis_title='Diameter')
    fig.write_image(f'{out_folder}/temporal_analysis/attacks/diameter.png')

    fraction = 0.1
    fig = charts.fixed_frac_attack_chart(attacks_data, fraction=fraction, xaxis_title='Dump Dates', yaxis_title='Diameter')
    os.makedirs(os.path.dirname(f'{out_folder}/temporal_analysis/attacks/_.png'), exist_ok=True)
    fig.write_image(f'{out_folder}/temporal_analysis/attacks/diameter_fixed_fraction.png')
