import os
import numpy as np
import pandas as pd
import argschema as ags
from operator import add
from collections import deque, defaultdict, OrderedDict, Counter
from joblib import dump, load, Parallel, delayed
import copy
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean
import scipy
import navis
import copy
import networkx as nx
import itertools
from uuid import UUID
from pathlib import Path

class ReconnectParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(required=True, description='Input file')
    output_dir = ags.fields.OutputDir(required=True, description='Output directory')
    model_dir = ags.fields.InputDir(required=True, description = 'Model directory')
    pxl_xyz = ags.fields.NumpyArray(dtype=float, required=False, 
                           default=[0.812,0.812,0.704], description='pxl size in um')

def reconnect(infile, swc_outdir, cl, sc, min_nodes = 10, prob_thresh = 0.5, resample=2, query_dis=10, min_collin=.1): 
    if not os.path.isdir(swc_outdir):
        os.mkdir(swc_outdir)
    
    # Split branches 
    morph_split = swc_split_branches(navis.read_swc(infile), os.path.join(swc_outdir, 'segments.swc'), 9)
    
    # Upsample and smooth skeletons
    resample = navis.resample_skeleton(morph_split, resample_to=resample)
    smooth = navis.smooth_skeleton(resample, window=5)
    
    # Extract neuronlist from nodelist
    neurons = extract_neuronlist(smooth, min_nodes)
        
    # Find pairs
    pair_data_iter = find_pairs(neurons, sc, cl, query_dis=query_dis, min_collin=min_collin)
        
    # Merge segment pairs with prob above thresh
    output_file = os.path.join(swc_outdir, 'connect_iter.swc') 
    out_neurons = merge_pairs(neurons, pair_data_iter, prob_thresh)
        
    #output new swc files
    os.makedirs(swc_outdir + 'reconnected_skeletons', exist_ok=True)
    navis.write_swc(out_neurons, swc_outdir + 'reconnected_skeletons')
    return neurons
            
            
def load_swc(filepath):
    "Load swc file as a N X 7 numpy array"
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) == 7:
                    cells = [float(c) for c in cells]
                    swc.append(cells)
                elif len(cells) == 8:
                    cells = [float(c) for c in cells[0:7]]
                    swc.append(cells)                
    return np.array(swc)                     

def save_swc(filepath, swc):
    with open(filepath, 'w') as f:
        f.write('#id,type,x,y,z,r,pid\n')
        for i in range(swc.shape[0]):
            f.write('%.0f %.0f %.0f %.0f %.0f %.3f %d\n' %tuple(swc[i, :].tolist()))
    
def swc_multi_to_single(dirname, fname):
    file_list = [f for f in os.listdir(dirname) if f.endswith('swc')] 
    file_list.sort()
    trace_list = []
    for f in file_list:
        trace = load_swc(os.path.join(dirname, f))
        trace_list.append(trace)
    offset = 0
    for i, trace in enumerate(trace_list):
        select = np.where(trace[:,-1]!=-1)[0]  
        trace_i = np.copy(trace)
        min_id = np.min(trace_i[:,0])
        trace_i[:,0] = trace_i[:,0] + offset - min_id + 1
        trace_i[select,-1] = trace_i[select, -1] + offset - min_id + 1
        offset = np.max(trace_i[:,0])
        if i == 0:
            trace_new = trace_i
        else:
            trace_new = np.concatenate((trace_new, trace_i))
    save_swc(os.path.join(dirname,fname), trace_new)
    

def swc_multi_to_single_subdir(dirname, out_path): #this version pulls all SWCs from subdirectories as well
    files = list(Path(dirname).rglob("*.swc" ))
    file_list = [str(i) for i in files]
    file_list.sort()
    trace_list = []
    for f in file_list:
        trace = load_swc(f)
        trace_list.append(trace)
    offset = 0
    for i, trace in enumerate(trace_list):
        select = np.where(trace[:,-1]!=-1)[0]  
        trace_i = np.copy(trace)
        min_id = np.min(trace_i[:,0])
        trace_i[:,0] = trace_i[:,0] + offset - min_id + 1
        trace_i[select,-1] = trace_i[select, -1] + offset - min_id + 1
        offset = np.max(trace_i[:,0])
        if i == 0:
            trace_new = trace_i
        else:
            trace_new = np.concatenate((trace_new, trace_i))
    save_swc(out_path, trace_new)                    



def swc_prune(morph_in, outfile, pruning_threshold = 30,**kwargs):
    nodes_to_remove = set()
    prune_count = 0
    bifur_nodes = morph_in.branch_points
    
    graph = morph_in.graph
    for i,node in bifur_nodes.iterrows():
        children = morph_in.nodes.loc[morph_in.nodes['parent_id'] == node['node_id']]
        for i, child in children.iterrows():
            child_remove_nodes, child_seg_length = bfs_tree(child['node_id'],graph)
            if child_seg_length < pruning_threshold:
                prune_count+=1
                [nodes_to_remove.add(n) for n in child_remove_nodes]
        
    new_nodes = morph_in.nodes[~morph_in.nodes['node_id'].isin(nodes_to_remove)]
    morph_in.nodes = new_nodes
    
    navis.write_swc(morph_in, outfile)
    return morph_in


def swc_split_branches(morph_in, outfile, node_len):
    # Find branching nodes
    branch_nodes = list(morph_in.branch_points['node_id'])
    nodes_to_remove = []

    # Split branches
    for i in branch_nodes:
        children = list(morph_in.nodes.loc[morph_in.nodes['parent_id'] == i, 'node_id'])
        for child in children:
            morph_in.nodes.loc[morph_in.nodes['node_id'] == child, 'parent_id'] = -1
    
    # Remove segments that are smaller than a given node length
    for tree in morph_in.subtrees:
        if len(tree) < node_len:
            nodes_to_remove.extend(tree)
    morph_in.nodes = morph_in.nodes[~morph_in.nodes['node_id'].isin(nodes_to_remove)]
    
    navis.write_swc(morph_in, outfile)
    return morph_in



def dfs_labeling(st_node, new_starting_id, modifying_dict, graph):
    ct = 0
    queue = deque([st_node])
    while len(queue) > 0:
        ct+=1
        current_node = queue.popleft()
        modifying_dict[current_node] = new_starting_id
        new_starting_id+=1
        for ch_no in list(graph.predecessors(current_node)):
            queue.appendleft(ch_no)
    return ct



def sort_swc(morph_in,outfile):
    #using the root lists, pull the current node order, and pair with an enumerated list as a dictionary
    roots = morph_in.root
    old_ids = np.array([element for nestedlist in morph_in.subtrees for element in nestedlist]).reshape(-1, 1)
    new_ids = np.arange(1, len(old_ids)+1, 1, dtype=int).reshape(-1, 1)
    new_node_ids = dict(np.concatenate((old_ids, new_ids), axis=1))
    new_node_ids[-1] = -1
    
    #alter node ids using dictionary and sort by node id
    morph_in.nodes['parent_id'] = morph_in.nodes.apply(lambda row: new_node_ids[row['parent_id']], axis=1)
    morph_in.nodes['node_id'] = morph_in.nodes.apply(lambda row: new_node_ids[row['node_id']], axis=1)
    morph_in.nodes = morph_in.nodes.sort_values(by=['node_id'])
    
    #save to new file
    out = morph_in.nodes.to_numpy()
    np.savetxt(outfile, out[:,0:-1], fmt='%s') 
    
    return morph_in 



def bfs_tree(st_node,graph):
    "BFS tree traversal, returns nodes in segment and how many"
    queue = deque([st_node])
    nodes_in_segment = []
    seg_len = 0
    while len(queue) > 0:
        seg_len+=1
        current_node = queue.popleft()
        nodes_in_segment.append(current_node)
        for ch_no in list(graph.predecessors(current_node)):
            queue.append(ch_no)

    return nodes_in_segment, len(nodes_in_segment)     



def distance(node1, node2, pxl_xyz):
    node1_coord = np.array((node1['x'], node1['y'], node1['z']))*pxl_xyz
    node2_coord = np.array((node2['x'], node2['y'], node2['z']))*pxl_xyz
    return euclidean(node1_coord, node2_coord)



def get_nodes(morph_in, segment, end_node, n):
    nodes = [end_node]
    segment = list(segment)

    i = 0
    
    if end_node not in segment:
        nodes = nodes
    
    elif int(morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node]['parent_id']) != -1:
        ind = segment.index(end_node)
            
        if ind-n > 0:
            nodes = segment[ind-n:ind+1]
            nodes.reverse()
        else:
            nodes = segment[0:ind+1]
            nodes.reverse()
                        
    else:
        if nodes[0] in segment:
            ind = segment.index(end_node)
            nodes = segment[ind:ind+n+1]
        else:
            nodes = nodes
        
    return nodes



def calculate_vector(coords):
    # TODO replace svd w/ eigh
    _, _, vv = np.linalg.svd(coords - coords.mean(axis=0))
    vector = vv[0]

    # Fix wrong orientation (sign) of vector
    vect_diff = (coords[-1,:] - coords[0,:])/np.linalg.norm(coords[-1,:] - coords[0,:])
    if np.dot(-vector, vect_diff) > np.dot(vector, vect_diff):
        vector*=-1    
    return vector 



def reroot_tree(start_node, tree, morph_in):
    neighbors_dict = {}
    for node in tree:
        node_neighbors = [] 
        parent = int(morph_in.nodes.loc[morph_in.nodes['node_id'] == node, 'parent_id'])

        if parent != -1:
            node_neighbors.append(parent)
        children = morph_in.nodes.loc[morph_in.nodes['parent_id'] == node, 'node_id']
        for ch in children:
            node_neighbors.append(ch)
        neighbors_dict[node] = node_neighbors
   
    start_node_parent = -1
    
    parent_dict = {}
    parent_dict[start_node] = start_node_parent
    queue = deque([start_node])
    while len(queue) > 0:
        current_node = queue.popleft()
        neighbors = neighbors_dict[current_node]

        for node in neighbors:
            if node not in parent_dict:
                parent_dict[node] = current_node
                queue.append(node)
    
    new_tree = []
    for key,val in parent_dict.items():
        node =  morph_in.nodes.loc[morph_in.nodes['node_id'] == key].squeeze()
        node['parent_id'] = val
        new_tree.append(node)
    return new_tree  


def connect_trees(nodes, morph_in):
    tree_list = morph_in.subtrees
    morph_out = copy.deepcopy(morph_in)
    
    n = 0
    for i, tree in enumerate(tree_list):
        if nodes[0] in tree:
            tree1 = list(copy.deepcopy(tree))
            idx1 = i
            n += 1
        if nodes[1] in tree:
            tree2 = list(copy.deepcopy(tree))
            idx2 = i
            n += 1
        if n == 2:
            break
            
    if idx1 != idx2:
        
        new_nodes = copy.deepcopy(morph_out.nodes)
        
        if int(morph_in.nodes.loc[morph_in.nodes['node_id'] == nodes[0]]['parent_id']) == -1:
            leaf_nodes = list(morph_out.leafs['node_id'])
            tree_leaf_nodes = [node for node in tree1 if node in leaf_nodes]
            
            if len(tree_leaf_nodes) == 0:
                print('do not reroot')
            else:
                tree1 = reroot_tree(tree_leaf_nodes[0], tree1, morph_in)
                
                for node in tree1:
                    new_nodes.loc[new_nodes['node_id'] == node[0], 'parent_id'] = node[-2]
                
        if int(morph_in.nodes.loc[morph_in.nodes['node_id'] == nodes[1]]['parent_id']) != -1:
            tree2 = reroot_tree(nodes[1], tree2, morph_in)
        
            for node in tree2:
                new_nodes.loc[new_nodes['node_id'] == node[0], 'parent_id'] = node[-2]
       
        new_nodes.loc[new_nodes['node_id'] == nodes[1], 'parent_id'] = nodes[0]
        morph_out.nodes = new_nodes
    
    else:
        print('loop, do not connect')
        morph_out = morph_in
    
    return morph_out


def calculate_feature(ns, end_node_ids, num_nodes=(5, 50)):
    end_coords = np.vstack([n.nodes[n.nodes.node_id == end_node_id][["x", "y", "z"]].to_numpy() for n, end_node_id in zip(ns, end_node_ids)])
    
    cvect = end_coords[1] - end_coords [0]
    cvect_norm = np.linalg.norm(cvect)
    cvect /= cvect_norm
    
    cf = []

    for num in num_nodes:
        for i, (n, end_node_id) in enumerate(zip(ns, end_node_ids)):
            nodes = get_bfs_neighbor_nodes(n, end_node_id, num)
            neighbor_loc_arr = nodes[["x", "y", "z"]].to_numpy()
            
            if i==0 and end_node_id == nodes["node_id"].to_numpy()[0]:
                neighbor_loc_arr = np.flip(neighbor_loc_arr, axis=0)
                
            if i==1 and end_node_id == nodes["node_id"].to_numpy()[-1]:
                neighbor_loc_arr = np.flip(neighbor_loc_arr, axis=0)
                
            vec = calculate_vector(neighbor_loc_arr)
            cf.append(np.dot(vec, cvect))
            
    return np.array([cvect_norm] + cf)


def collinearity(ns, end_node_ids, num_nodes=(4, 49)):
    end_coords = np.vstack([n.nodes[n.nodes.node_id == end_node_id][["x", "y", "z"]].to_numpy() for n, end_node_id in zip(ns, end_node_ids)])
    cvect = end_coords[1] - end_coords[0]
    cvect_norm = np.linalg.norm(cvect)
    cvect /= cvect_norm

    cf = []

    for num in num_nodes:
        for i, (n, end_node_id) in enumerate(zip(ns, end_node_ids)):
            neighbor_loc_arr = get_bfs_neighbor_nodes(n, end_node_id, num)[["x", "y", "z"]].to_numpy()
            vec = calculate_vector(neighbor_loc_arr)
            cf.append(np.dot(-vec, cvect))
    return cf


def merge_pairs(neuro_list, pair_data, thresh): 
    pair_data = [x for x in pair_data if ((len(np.where(x[1][2:6] < 0.5)[0])==0) and (x[1][0] > thresh))]
    merge_num = 0
    
    #remove duplicates based on highest probability
    pair_data  = pd.DataFrame([list(i[0]) + list(i[1]) for i in pair_data])
    pair_data  = pair_data.sort_values(2, ascending=False).drop_duplicates(0).sort_index()
    pair_data  = pair_data.sort_values(2, ascending=False).drop_duplicates(1).sort_index()

    #create graph object with pairs
    pairs = [(row[0],row[1]) for ind,row in pair_data .iterrows()]
    G = nx.Graph() 
    G.add_edges_from(pairs)
    cc = list(nx.connected_components(G))
    for com in cc:
        merge_num += len(com)-1
        group = []
        for neu in com:
            group.append(neuro_list[neuro_list.id == neu[0]])
            neuro_list = neuro_list[(neuro_list.id != neu[0])]
        new_neu = navis.stitch_skeletons(group, method='LEAFS')
        neuro_list.append(new_neu)
        
    #reset soma to none
    for neu in neuro_list:
        neu.soma = None
        
    print('Pairs Merged: ', merge_num)
    return neuro_list

def find_pairs(neuro_list, sc, cl, query_dis=15, min_collin = 0.1):    
    #find end nodes and just roots nodes
    pts = neuro_list.nodes[neuro_list.nodes['type'].isin(['root', 'end'])]
    roots = list(pts[pts['type'].isin(['root'])]['node_id'])
    ids = dict(zip([i.id for i in neuro_list], list(range(0,len(neuro_list)))))
    
    endpts = np.array(pts[['x','y','z']])
    endpts_neuron_id = np.array(pts['neuron'])
    endpts_node_id = np.array(pts['node_id'])
    endpts_neuron_idx = np.array([ids[x] for x in endpts_neuron_id])
    
    #create kdtree with endpoints and find all pairs with given query distance
    kdt = scipy.spatial.KDTree(endpts, leafsize=2)
    pairs = kdt.query_pairs(query_dis)
    pairs = {frozenset(p) for p in pairs if len({*endpts_neuron_id[[*p]]})>1}

    # calculate features for pairs
    neighbors_considered = 4  
    pair_features = {}
    subfeat = {}
    
    for pq in pairs:
        # initial feature based on condidate pair
        f = tuple()
        p, q = pq
        candidate_neuron_idxs = endpts_neuron_idx[[p, q]]
        candidate_neurons = neuro_list[candidate_neuron_idxs]
        candidate_end_node_ids = endpts_node_id[[p, q]]
    
        try:
            cf = subfeat[str(candidate_neurons.name)]
            
        except:
            cf = tuple(calculate_feature(candidate_neurons, candidate_end_node_ids))
            subfeat[str(candidate_neurons.name)] = cf
            
        #skip pair if collinearity too low
        if len(np.where(np.array(cf[1:5]) < min_collin)[0]) > 0:
            continue
        
        f += cf

        p_neuron, q_neuron = candidate_neurons
        p_end_node_id, q_end_node_id = candidate_end_node_ids
        
        # features from pairing p with next nearest neighbors
        dists, idxs = kdt.query(endpts[p], k=neighbors_considered+2)
        p_knn_idxs = [i for i in idxs if i not in pq][:neighbors_considered]
        p_knn_neurons = [
            (p_neuron, n) for n in neuro_list[endpts_neuron_idx[p_knn_idxs]]
        ]

        p_knn_end_node_ids =[
            (p_end_node_id, end_node_id)
            for end_node_id in endpts_node_id[p_knn_idxs]
        ]
        
        
        for nn_neurons, nn_end_node_ids  in zip(p_knn_neurons, p_knn_end_node_ids):
            try:
                cf = subfeat[str(nn_neurons[0].name) + str(nn_neurons[1].name)]
                
            except:
                cf = tuple(calculate_feature(nn_neurons, nn_end_node_ids))
                subfeat[str(nn_neurons[0].name) + str(nn_neurons[1].name)] = cf
                
            f += cf
            
        # features from pairing q with next nearest neighbors
        dists, idxs = kdt.query(endpts[q], k=neighbors_considered+2)
        q_knn_idxs = [i for i in idxs if i not in pq][:neighbors_considered]
        q_knn_neurons = [
            (q_neuron, n) for n in neuro_list[endpts_neuron_idx[q_knn_idxs]]
        ]
        q_knn_end_node_ids =[
            (q_end_node_id, end_node_id)
            for end_node_id in endpts_node_id[q_knn_idxs]
        ]

        for nn_neurons, nn_end_node_ids  in zip(q_knn_neurons, q_knn_end_node_ids):
            try:
                cf = subfeat[str(nn_neurons[0].name) + str(nn_neurons[1].name)]
                
            except:
                cf = tuple(calculate_feature(nn_neurons, nn_end_node_ids))
                subfeat[str(nn_neurons[0].name) + str(nn_neurons[1].name)] = cf
        
            f += cf
        
        #calculate probability
        f = np.array(f)
        #weird error in reconnect function with na in f array
        f[np.isnan(f)] = 0
        x = sc.transform(f.reshape(1,-1))

        prob = cl.predict_proba(x)[0][1]
        pair_features[ (endpts_neuron_id[p],endpts_node_id[p]), (endpts_neuron_id[q], endpts_node_id[q])] = np.insert(f, 0, prob)


    pair_features = list(pair_features.items())
    return pair_features


def extract_neuronlist(morph_in, node_min):
    def ind_neuron(tree):
        q = morph_in.nodes.query('node_id in @tree')
        neuron = navis.NeuronList(pd.DataFrame(q))
        return neuron
    
    trees = morph_in.subtrees
    res = Parallel(n_jobs=4)(delayed(ind_neuron)(tree) for tree in trees if len(tree) >= node_min)
    neurons = navis.NeuronList(res)
    for neu in neurons:
        neu.name = neu.id
    
    return neurons

def get_bfs_neighbor_nodes(n, node_id, num_nodes):
    node_type = n.nodes[n.nodes.node_id == node_id].type.values[0]
    if node_type == "root":
        node_ids = nx.traversal.bfs_tree(n.graph, node_id, depth_limit=num_nodes-1, reverse=True).nodes
    elif node_type =="end":
        node_ids = nx.traversal.bfs_tree(n.graph, node_id, depth_limit=num_nodes-1).nodes 
    else:
        raise NotImplementedError("only supports root and end nodes")
    return n.nodes[n.nodes.node_id.isin(node_ids)]

def navis_to_morph(morph_in):
    neu = morph_in.nodes.drop(columns=['type'])
    neu = neu.rename(columns={"node_id": "id", "parent_id": "parent"})
    neu = neu.to_dict('records')
    #convert navis object to morphology object
    morph_out = Morphology(neu,
        node_id_cb=lambda node: node['id'],
        parent_id_cb=lambda node: node['parent'] )
    
    return morph_out
    
    
def morph_to_navis(morph_in):
    neu = pd.DataFrame(morph_in.nodes())
    neu = neu.rename(columns={"id": "node_id", "parent": "parent_id"})
    #convert morphology object to navis object
    morph_out = navis.TreeNeuron(neu)

    return morph_out
    

            
class Reconnect(ags.ArgSchemaParser):
        
    def run(self):
        reconnect(self.args['input_file'], self.args['output_dir'], self.args['model_dir'],
        self.args['pxl_xyz'])
   
        
if __name__ == "__main__":
    mod = Reconnect(schema_type=ReconnectParameters)
    mod.run()      