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
from pathlib import Path
import concurrent
from concurrent.futures import ThreadPoolExecutor
import tarfile
import uuid
from io import BytesIO
import copy


class ReconnectParameters(ags.ArgSchema):
    in_skels = ags.fields.InputFile(required=True, description='Input skeletons, as navis objects, swc, or swc.gz')
    cl = ags.fields.InputFile(required=True, description = 'Model File')
    sc = ags.fields.InputFile(required=True, description = 'Scalar File')    
    min_nodes = ags.fields.Int(dtype=int, required=False, default=10, description='Minimum skeleton node length')
    prob_thresh = ags.fields.Float(dtype=float, required=False, default=0.5, description='Minimum probability allowed for merge model prediction')
    resample = ags.fields.Int(dtype=int, required=False, default=2, description='Factor for upsampling skeletons')
    smooth = ags.fields.Int(dtype=int, required=False, default=5, description='Window for smoothing skeletons')
    query_dis = ags.fields.Int(dtype=int, required=False, default=10, description='Maximum query distance for matching end nodes')
    min_collin = ags.fields.Float(dtype=float, required=False, default=.1, description='Minimum collinearity for finding skeleton merge pairs')
            

def reconnect(skels, cl, sc, min_nodes = 10, prob_thresh = 0.5, resample=4, smooth=2, split=True, query_dis=10, min_collin=.1, bound_box=None, dis_end=0):     
    # load skeletons
    if isinstance(skels, navis.core.neuronlist.NeuronList):
      pass
    else:
      if skels.endswith('.gz'):
        skels = read_navis_neurons_tar(skels)
      elif skels.endswith('.swc'):
        skels = navis.read_swc(skels)
    
    if split==True:
      # Split branches 
      skels = swc_split_branches(skels, min_nodes=min_nodes)
    
    # Upsample and smooth skeletons
    if smooth:
      skels = smooth_skeletons(skels, window=smooth)
    if resample:
      skels = navis.resample_skeleton(skels, resample_to=resample, parallel=True, progress=False)

    # Find pairs
    pair_data_iter = find_pairs(neuro_list=skels, sc=sc, cl=cl, query_dis=query_dis, min_collin=min_collin, bound_box=bound_box, dis_end=dis_end)

    try:
      # Merge segment pairs with prob above thresh
      unmerged, merged = merge_pairs(skels, pair_data_iter, prob_thresh)
    except:
      unmerged, merged = skels, navis.NeuronList(None)              
        
    return unmerged, merged
            
            
def load_swc(in_fn):
    "Load swc fn as a N X 7 numpy array"
    swc = []
    with open(in_fn) as f:
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

def save_swc(in_fn, swc):
    with open(in_fn, 'w') as f:
        f.write('#id,type,x,y,z,r,pid\n')
        for i in range(swc.shape[0]):
            f.write('%.0f %.0f %.0f %.0f %.0f %.3f %d\n' %tuple(swc[i, :].tolist()))
    
def swc_multi_to_single(in_dir, fname):
    fn_list = [f for f in os.listdir(in_dir) if f.endswith('swc')] 
    fn_list.sort()
    trace_list = []
    for f in fn_list:
        trace = load_swc(os.path.join(in_dir, f))
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
    save_swc(os.path.join(in_dir,fname), trace_new)
    

def swc_multi_to_single_subdir(in_dir, out_path): #this version pulls all SWCs from subdirectories as well
    fns = list(Path(in_dir).rglob("*.swc" ))
    fn_list = [str(i) for i in fns]
    fn_list.sort()
    trace_list = []
    for f in fn_list:
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



def swc_prune(skels, pruning_threshold = 10):
    skels = navis.prune_twigs(skels, pruning_threshold)

    return skels


def swc_split_branches(morph_in, min_nodes):
    out_sk = []
    for neu in morph_in:
        # Skip if under minimum node length
        if len(neu.nodes) < min_nodes:
            continue

        # Find branching nodes
        branch_nodes = list(neu.branch_points['node_id'])
        if len(branch_nodes) > 0:
            # Split branches
            for i in branch_nodes:
                children = list(neu.nodes.loc[neu.nodes['parent_id'] == i, 'node_id'])
                for child in children:
                    neu.nodes.loc[neu.nodes['node_id'] == child, 'parent_id'] = -1
        
            # Break fragments
            frag = navis.break_fragments(neu)
            for fr in frag:
                if len(fr.nodes) >= min_nodes:
                    fr.id = uuid.uuid1()
                    out_sk.append(fr)
        else:
            out_sk.append(neu)

    out_sk = navis.NeuronList(out_sk)
    return out_sk



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



def sort_swc(morph_in,out_fn):
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
    
    #save to new fn
    out = morph_in.nodes.to_numpy()
    np.savetxt(out_fn, out[:,0:-1], fmt='%s') 
    
    return morph_in 
    
def sort_neurons(skels):
    out_neu = navis.NeuronList(None)
    for sk in skels:
        f=1
        if len(sk.nodes)<2:
            continue
        if len(sk.branch_points)>0:
            out_neu.append(sk)
            continue
        if sk.nodes.iloc[0]['type']!='root' or sk.nodes.iloc[-1]['type']!='end':
            sk.nodes.reset_index(drop=True, inplace=True)
            ind_order = np.zeros(len(sk.nodes)).astype(int)
            #get end node and starting parent id
            end = sk.ends
            end_ind = end.index[0]
            ind_order[end_ind] = 0
            par = list(end['parent_id'])[0]
            while par!=-1:
                nt = sk.nodes[sk.nodes.node_id==par]
                nt_ind = nt.index[0]
                ind_order[nt_ind] = f
                f+=1
                par = list(nt['parent_id'])[0]

            s = pd.Series(ind_order)
            sk.nodes.set_index(s, inplace=True)
            sk.nodes.sort_index(inplace=True)
            out_neu.append(sk)
        else:
            out_neu.append(sk)
    return out_neu

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


def calculate_feature(ns, end_node_ids, num_nodes=(5, 50), dis_end=0):
    alt_end_node_ids = np.copy(end_node_ids)
    if dis_end != 0:
        for ind,n in enumerate(alt_end_node_ids):
            g = list(ns[ind].graph.nodes)
            dif = abs(np.array([g[0],g[-1]])-n)
            if len(g) > dis_end:
              if dif[0] < dif[1]:
                  alt_end_node_ids[ind] = g[0+dis_end]
              else:
                  alt_end_node_ids[ind] = g[-1-dis_end]
                
    end_coords = np.vstack([n.nodes[n.nodes.node_id == alt_end_node_id][["x", "y", "z"]].to_numpy() for n, alt_end_node_id in zip(ns, alt_end_node_ids)])
    cvect = end_coords[1] - end_coords[0]
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

def merge_pairs(neuro_list, pair_data, thresh = None, min_collin = None, dist_check=False):
    merge_num = 0
    merge_list = []
    if min_collin:
        pair_data = [x for x in pair_data if ((len(np.where(x[1][2:6] < min_collin)[0])==0))]
    if thresh:
        pair_data = [x for x in pair_data if (x[1][0] > thresh)]
    else:
        pair_data = [x for x in pair_data]
    
    if dist_check==True:
        pair_data  = pd.DataFrame([list(i[0]) + list(i[1]) for i in pair_data])
        pair_data  = pair_data.sort_values(4, ascending=False).drop_duplicates(0).sort_index()
        pair_data  = pair_data.sort_values(4, ascending=False).drop_duplicates(1).sort_index()
    else:
        pair_data  = pd.DataFrame([list(i[0]) + list(i[1]) for i in pair_data])
        pair_data  = pair_data.sort_values(2, ascending=False).drop_duplicates(0).sort_index()
        pair_data  = pair_data.sort_values(2, ascending=False).drop_duplicates(1).sort_index()
        
    #remove duplicates that are in both pre and post columns
    rem = []
    for ind,row in pair_data.iterrows():
        if row[0] in list(pair_data[1]):
            rem.append(ind)
    pair_data = pair_data.drop(rem)
    
    #create graph object with pairs
    pairs = [(row[0][0],row[1][0]) for ind,row in pair_data.iterrows()]
    G = nx.Graph() 
    G.add_edges_from(pairs)
    cc = list(nx.connected_components(G))
    
    for com in cc:
        merge_num += len(com)-1
        group = []
        for neu in com:
            group.append(neuro_list[neuro_list.id == neu])
            neuro_list = neuro_list[(neuro_list.id != neu)]
        new_neu = navis.stitch_skeletons(group, method='LEAFS')     
        
        #append to merge list
        merge_list.append(new_neu)
        
    #reset soma to none
    for neu in neuro_list:
        neu.soma = None
        
    print('Pairs Merged: ', merge_num)
    return neuro_list, navis.NeuronList(merge_list)

def find_pairs(neuro_list, query_dis=15, min_collin = 0.1, sc = None, cl = None, bound_box=None, min_nodes=None, dis_end=0):
    np.seterr(invalid='ignore')
    #filter if minimum nodes is provided
    if min_nodes:
        neuro_list = navis.NeuronList([x for x in neuro_list if x.n_nodes >= min_nodes])
    
    #filter if bounding box is provided
    if bound_box:
      neuro_list = filter_skeletons(neuro_list,bound_box)
      
    #sort to ensure nodes are in end-to-root order  
    neuro_list = sort_neurons(neuro_list)
    
    #find end and root nodes
    pts = pd.DataFrame(None)
    for sk in neuro_list:
        ends = sk.nodes.iloc[[0, -1]].copy()
        ends['neuron']=sk.id
        pts = pd.concat([pts, ends])
    pts = pts[pts['type'].isin(['root','end'])]
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
            cf = subfeat[str(candidate_neurons.id)]
            
        except:
            cf = tuple(calculate_feature(candidate_neurons, candidate_end_node_ids, dis_end=dis_end))
            subfeat[str(candidate_neurons.id)] = cf
            
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
                cf = subfeat[str(nn_neurons[0].id) + str(nn_neurons[1].id)]
                
            except:
                cf = tuple(calculate_feature(nn_neurons, nn_end_node_ids, dis_end=dis_end))
                subfeat[str(nn_neurons[0].id) + str(nn_neurons[1].id)] = cf
                
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
                cf = subfeat[str(nn_neurons[0].id) + str(nn_neurons[1].id)]
                
            except:
                cf = tuple(calculate_feature(nn_neurons, nn_end_node_ids, dis_end=dis_end))
                subfeat[str(nn_neurons[0].id) + str(nn_neurons[1].id)] = cf
        
            f += cf
        
        f = np.array(f)
        #weird error in reconnect function with na in f array
        f[np.isnan(f)] = 0
        if sc and cl:
            #calculate probability
            x = sc.transform(f.reshape(1,-1))
            prob = cl.predict_proba(x)[0][1]
            pair_features[ (endpts_neuron_id[p],endpts_node_id[p]), (endpts_neuron_id[q], endpts_node_id[q])] = np.insert(f, 0, prob)
        else:
            pair_features[ (endpts_neuron_id[p],endpts_node_id[p]), (endpts_neuron_id[q], endpts_node_id[q])] = np.insert(f, 0, 0)

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
    
    return neurons

def get_bfs_neighbor_nodes(n, node_id, num_nodes):
    node_type = n.nodes[n.nodes.node_id == node_id].type.values[0]
    if node_type == "root":
        node_ids = nx.traversal.bfs_tree(n.graph, node_id, depth_limit=num_nodes-1, reverse=True).nodes
    elif node_type =="end":
        node_ids = nx.traversal.bfs_tree(n.graph, node_id, depth_limit=num_nodes-1).nodes 
    else:
        raise NotImplementedError("only supports root and end nodes, not ",str(node_type))
    return n.nodes[n.nodes.node_id.isin(node_ids)]
    
    
def swap_dimensions(skels, dims=['x','z']):
    out_skels = []
    for sk in skels:
        nodes = sk.nodes
        nodes = nodes.rename(columns={dims[0]: dims[1], dims[1]: dims[0]})
        out_skels.append(navis.NeuronList(nodes))
    out_skels = navis.NeuronList(out_skels)
    out_skels['soma'] = None
    return out_skels

def apply_transform_skeletons(skels, transform=[[1,0,0],[0,1,0],[0,0,1]]):
    transform = np.array(transform)
    #iterate through skeletons and apply transform
    out_skels = []
    for sk in skels:
        xs,ys,zs = [],[],[]
        nodes = sk.nodes
        for index,row in nodes.iterrows():
            x,y,z = np.dot(transform,np.array([row['x'],row['y'],row['z']]))
            xs.append(x),ys.append(y),zs.append(z)
        nodes['x'],nodes['y'],nodes['z'] = xs,ys,zs
        out_skels.append(navis.NeuronList(nodes))
    out_skels = navis.NeuronList(out_skels)
    out_skels['soma'] = None
    return out_skels

def remove_cutout_nodes(skels, bound_box=[0,0,0,0,0,0]):
    out_sk = navis.NeuronList(None)
    x1,x2,y1,y2,z1,z2 = bound_box
    for sk in skels:
        #find nodes within bounding box
        nodes = sk.nodes.copy()
        drop_nodes = list(nodes[nodes['x'].between(x1,x2) & nodes['y'].between(y1,y2) & nodes['z'].between(z1,z2)]['node_id'])
        
        #drop nodes from dataframe
        if len(drop_nodes)>0:
            nodes['parent_id'] = nodes['parent_id'].replace(drop_nodes,-1)
            nodes = nodes[~nodes['node_id'].isin(drop_nodes)].copy()
            if len(nodes) <= 1:
                continue

            nsk = navis.NeuronList(nodes.copy())  
            frag = navis.break_fragments(nsk)
            for fr in frag:
                if len(fr.nodes) > 1:
                    fr.id = uuid.uuid1()
                    out_sk.append(fr)
        else:
            out_sk.append(sk)
    return out_sk
    
def remove_overlap_nodes(skels):
    duplicates = skels.nodes[skels.nodes.duplicated(keep='last')]
    nodes = {}
    for ind,row in duplicates.iterrows():
        try:
            nodes[str(row['neuron'])] += [row['node_id']]
        except:
            nodes[str(row['neuron'])] = [row['node_id']]
    add = navis.NeuronList(None)
    for key,value in nodes.items():
        match = navis.NeuronList([x for x in skels if x.id==uuid.UUID(key)])
        keep,change = match[0].copy(), match[1].copy()
        change.id = uuid.uuid1()
        skels = navis.NeuronList([x for x in skels if x.id!=uuid.UUID(key)])
        add.append(navis.remove_nodes(change, value))
        add.append(keep)
    return navis.NeuronList([skels,add])
    
def translate_nodes(skels, trans=[0,0,0]):
    out_sk = []
    shift_x,shift_y,shift_z = trans
    for sk in skels:  
        nodes = sk.nodes.copy()
        #adjust dimensions
        nodes['x'] = nodes['x'] + shift_x
        nodes['y'] = nodes['y'] + shift_y
        nodes['z'] = nodes['z'] + shift_z
        out_sk.append(navis.TreeNeuron(nodes))  
    out_sk = navis.NeuronList(out_sk)
    return out_sk
    
def create_rectangle_volume(bound_box, name='volume'):
    x1,x2,y1,y2,z1,z2 = bound_box
    vertices = [[x1,y1,z1],[x2,y2,z1],[x1,y2,z1],[x2,y1,z1],[x1,y1,z2],[x2,y2,z2],[x1,y2,z2],[x2,y1,z2]]
    faces = [[0,1,2],[0,1,3],[4,5,6],[4,5,7],[0,4,6],[0,2,6],[3,7,5],[3,1,5],[2,1,5],[2,6,5],[0,3,4],[3,4,7]]
    vol = navis.Volume(vertices, faces=faces, name=name)
    return vol
    
def filter_skeletons(skels,bound_box):
    x1,x2,y1,y2,z1,z2 = bound_box
    sub_neu = navis.NeuronList(None)
    for sk in skels:
        Pass = 0
        for ind,row in sk.nodes.iterrows():
            if row['x'] > x1 and row['x'] < x2:
                if row['y'] > y1 and row['y'] < y2:
                    if row['z'] > z1 and row['z'] < z2:
                        Pass += 1
        if Pass >= 1:
            sub_neu.append(sk)
    return sub_neu
    
def smooth_skeletons(skels, window=5):
    skels = skels.copy()
    out_sk = navis.NeuronList(None)
    for ind,sk in enumerate(skels):
        sk.nodes[['x','y','z']] = sk.nodes[['x','y','z']].rolling(window=window, min_periods=1).mean()
        out_sk.append(sk)
    return out_sk

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
    
    
def read_navis_neurons_tar(tar_fn, concurrency=10, preprocess_func=None):
    preprocess_func = ((lambda x: x) if preprocess_func is None else preprocess_func)
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        futs = []
        with tarfile.open(tar_fn, "r:gz") as t:
            for m in t.getmembers():
                swc_b = t.extractfile(m).read()
                futs.append(e.submit(navis.io.read_swc, swc_b.decode()))
        navis_neurons = navis.NeuronList([
            preprocess_func(fut.result()) for
            fut in concurrent.futures.as_completed(futs)])
    return navis_neurons
    
    
    
def write_kimi_skels_tar(tar_fn, skels, mode='w:gz'):
    with tarfile.open(tar_fn, mode=mode) as t:
        id = 1
        for skel in skels:
            bio = BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            id += 1
            
def write_navis_skels_tar(tar_fn, skels, mode='w:gz'):
    with tarfile.open(tar_fn, mode=mode) as t:
        for sk in skels:
            id = sk.id
            if 'label' not in sk.nodes:
                sk.nodes.insert(1, 'label', list(np.zeros(len(sk.nodes))))
            sk = sk.nodes[['node_id', 'label','x','y','z','radius','parent_id']].values.tolist()
            sk = '\n'.join(str(x)[1:-1] for x in sk).replace(",", "")
            bio = BytesIO(sk.encode())
            info = tarfile.TarInfo(name=f"{id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            
def read_multi_tar(dir_n, n_jobs=1):
    def read_tar(file):
        skels = read_navis_neurons_tar(file)
        return [skels, file]
    files = glob.glob(dir_n + "*.gz")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(read_tar)(file=file) for file in files)
    return results
            
class Reconnect(ags.ArgSchemaParser):
    def run(self):
        reconnect(self.args['in_skels'], self.args['cl'],self.args['sc'],
        self.args['min_nodes'], self.args['prob_thresh'], self.args['resample'], 
        self.args['smooth'], self.args['query_dis'], self.args['min_collin'])
   
        
if __name__ == "__main__":
    mod = Reconnect(schema_type=ReconnectParameters)
    mod.run()       