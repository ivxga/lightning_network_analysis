# Imports
import os
import re
import time
import json
import datetime as dt
import networkx as nx

import lib.ln_functions as ln
import lib.na_functions as na

DATA_FOLDER = r'./data'
OUTPUT_FOLDER = r'./output'

# Aux Debugging
def profiling(msg, start_time, out_folder=None, verbose=1):
    '''
    msg (str):
        current activity message
    
    start_time (float):
        previous activity start timestamp (retrieved with `time.time()`), use `None` if is the first activity
    
    out_folder (str):
        output folder path, a file `profiling.txt` could be created inside
    
    verbose (int):
        verbosity level -> {0: nothing | 1: save on file | 2: print on console}
    '''
    if verbose == 0: return
    
    elif verbose == 1:
        os.makedirs(os.path.dirname(f'{out_folder}/profiling.txt'), exist_ok=True)
        with open(f'{out_folder}/profiling.txt', 'a') as profiling_file:
            if start_time: profiling_file.write(f'\t{round(time.time() - start_time, 3)} s\n')
            else:
                profiling_file.truncate(0)
                profiling_file.seek(0)
                profiling_file.write(f'=== Start Profiling [{str(dt.datetime.now())[:-7]}] ===\n\n')
            
            if msg: profiling_file.write(f'=> {msg}\n')
            else:   profiling_file.write(f'\n==== End Profiling [{str(dt.datetime.now())[:-7]}] ====')
    
    elif verbose == 2:
        if start_time: print(f'\t{round(time.time() - start_time, 3)} s')
        else:          print(f'=== Start Profiling [{str(dt.datetime.now())[:-7]}] ===\n\n')
        
        if msg: print(f'=> {msg}')
        else:   print(f'\n==== End Profiling [{str(dt.datetime.now())[:-7]}] ====')
        
    return time.time()

# Single Dump Complete Analysis
def single_snapshot_analysis(dump_date):

    graph_name = 'lightning_network'
    graph_filename = f'{DATA_FOLDER}/network_graph_{dump_date}.json'

    # 1
    start = profiling('Loading JSON graph', None, f'{OUTPUT_FOLDER}/{dump_date}')
    with open(graph_filename, 'r') as json_file:
        json_graph = json.load(json_file)

    # 2
    start = profiling('Generating and Loading GML graph', start, f'{OUTPUT_FOLDER}/{dump_date}')
    gml_graph = na.load_gml(ln.convert_to_gml(graph_filename, output_filename=f'{OUTPUT_FOLDER}/{dump_date}/{graph_name}_{dump_date}.gml'))

    # 3
    start = profiling('Computing Main Statistics of the graph', start, f'{OUTPUT_FOLDER}/{dump_date}')
    na.graph_stats(gml_graph, f'{OUTPUT_FOLDER}/{dump_date}/statistics/graph_stats.json')

    # 4
    start = profiling('Generating the Degree Distribution', start, f'{OUTPUT_FOLDER}/{dump_date}')
    na.plot_degree_distr(gml_graph, f'{OUTPUT_FOLDER}/{dump_date}/statistics/degree_distribution.png')

    # 5
    start = profiling('Ranking Top Nodes', start, f'{OUTPUT_FOLDER}/{dump_date}')
    na.ranking_nodes(gml_graph, f'{OUTPUT_FOLDER}/{dump_date}/statistics/ranking_nodes.json')

    # 6
    start = profiling('Attacking Graph removing Top Nodes', start, f'{OUTPUT_FOLDER}/{dump_date}')
    na.removal_attack(gml_graph, f'{OUTPUT_FOLDER}/{dump_date}/attacks')

    # 7
    start = profiling('Analyzing Features Subgraphs [Statistics & Degree Distribution]', start, f'{OUTPUT_FOLDER}/{dump_date}')
    for feature in ln.get_features():
        subgraph = ln.subgraph_by_feature(json_graph, feature)
        na.write_gml(subgraph, f'{OUTPUT_FOLDER}/{dump_date}/features/graphs/{feature.replace("-","_")}_graph.gml')
        na.graph_stats(subgraph,  f'{OUTPUT_FOLDER}/{dump_date}/features/statistics/{feature.replace("-","_")}_graph_stats.json')
        na.plot_degree_distr(subgraph, f'{OUTPUT_FOLDER}/{dump_date}/features/degree_distr/{feature.replace("-","_")}_degree_distribution.png')

    features_data = ln.get_features_data(data_folder=f'{OUTPUT_FOLDER}/{dump_date}/features/statistics')
    ln.compare_feature_stats(features_data, OUTPUT_FOLDER, dump_date, f'{OUTPUT_FOLDER}/{dump_date}/statistics/graph_stats.json')

    # 8
    start = profiling('Generating Synthetic Graph from dump', start, f'{OUTPUT_FOLDER}/{dump_date}')
    synth_graph = ln.generate_synt_graph(gml_graph, out_filename=f'{OUTPUT_FOLDER}/{dump_date}/synth/synth_network_graph_{dump_date}.gml')

    # 9
    start = profiling('Computing Main Statistics of the Synthetic Graph', start, f'{OUTPUT_FOLDER}/{dump_date}')
    na.graph_stats(synth_graph, f'{OUTPUT_FOLDER}/{dump_date}/synth/synth_graph_stats.json')
    na.plot_degree_distr(synth_graph, f'{OUTPUT_FOLDER}/{dump_date}/synth/synth_degree_distribution.png')

    # 10
    start = profiling('Generating different type of Graphs with similar number of nodes/edges [random / scale_free / barabasi_albert / powerlaw_cluster / watts_strogatz]', start, f'{OUTPUT_FOLDER}/{dump_date}')
    comparison_graphs = {
        'random_graph':           nx.fast_gnp_random_graph(n=len(gml_graph), p=0.00022, directed=True),
        'scale_free_graph':       nx.scale_free_graph(n=len(gml_graph), alpha=0.18, beta=0.77, gamma=0.05),
        'barabasi_albert_graph':  nx.barabasi_albert_graph(n=len(gml_graph), m=int(na.get_avg_degree(gml_graph)/2)),
        'watts_strogatz_graph':   nx.watts_strogatz_graph(n=len(gml_graph), k=int(na.get_avg_degree(gml_graph)), p=0.25),
        'powerlaw_cluster_graph': nx.powerlaw_cluster_graph(n=len(gml_graph), m=int(na.get_avg_degree(gml_graph)/2), p=0.5),
    }

    # 11
    for graph_name, graph in comparison_graphs.items():
        start = profiling(f'Computing Main Statistics of the {" ".join(list(map(str.capitalize, graph_name.split("_"))))}', start, f'{OUTPUT_FOLDER}/{dump_date}')
        na.write_gml(graph, f'{OUTPUT_FOLDER}/{dump_date}/comparison/graphs/{graph_name}.gml')
        na.graph_stats(graph, f'{OUTPUT_FOLDER}/{dump_date}/comparison/statistics/{graph_name}_stats.json')
        na.plot_degree_distr(graph, f'{OUTPUT_FOLDER}/{dump_date}/comparison/degree_distr/{graph_name}_degree_distribution.png')

    _ = profiling(None, start, f'{OUTPUT_FOLDER}/{dump_date}')

if __name__ == '__main__':
    
    dump_dates = set(map(lambda x: x[:-5][14:], filter(lambda x: re.match(r'network_graph_\d{4}_\d{2}_\d{2}.json', x), os.listdir(DATA_FOLDER))))
    dump_dates_analyzed = set(filter(lambda x: re.match(r'\d{4}_\d{2}_\d{2}', x), os.listdir(OUTPUT_FOLDER))) if os.path.exists(OUTPUT_FOLDER) else set()
    dump_dates_to_analyze = dump_dates - dump_dates_analyzed

    for date in sorted(list(dump_dates_to_analyze)):
        print(f'=> {date}')
        single_snapshot_analysis(date)

    ln.temporal_analysis(sorted(list(dump_dates)), data_folder=DATA_FOLDER, out_folder=OUTPUT_FOLDER)
