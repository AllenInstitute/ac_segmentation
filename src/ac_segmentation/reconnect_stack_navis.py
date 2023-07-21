import os
import numpy as np
import pandas as pd
import argschema as ags
from operator import add
from collections import deque, defaultdict, OrderedDict
from joblib import dump, load
import copy
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean
import navis
import copy
import networkx as nx
import itertools

class ReconnectParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(required=True, description='Input file')
    output_dir = ags.fields.OutputDir(required=True, description='Output directory')
    model_dir = ags.fields.InputDir(required=True, description = 'Model directory')
    pxl_xyz = ags.fields.NumpyArray(dtype=float, required=False, 
                           default=[0.812,0.812,0.704], description='pxl size in um')

def reconnect(infile, swc_outdir, modeldir, xyz_pxl): 
    if not os.path.isdir(swc_outdir):
        os.mkdir(swc_outdir)
    
    # Prune short branches 
    swc_prune(infile, os.path.join(swc_outdir, 'pruned.swc'),
                pruning_threshold = 15)
    
    # Split branches and save all segments as a single file
    swc_split_branches(os.path.join(swc_outdir, 'pruned.swc'),
                        os.path.join(swc_outdir, 'segments.swc'))
    sort_swc(os.path.join(swc_outdir, 'segments.swc'),
             os.path.join(swc_outdir, 'segments.swc'))  
    
    # # Upsample swc using vaa3d
    
    # Reconnect segments
    input_file = os.path.join(swc_outdir, 'segments_resampled.swc')
    # Load scaler
    scaler = load(os.path.join(modeldir, 'scaler.joblib'))
    # Load classifier model
    clf = load(os.path.join(modeldir, 'LR_1.joblib')) 
    model_pxl = np.array([0.207,0.207,0.6]) # model pxl
    xyz_pxl = xyz_pxl*np.mean(model_pxl/xyz_pxl)
    max_iter = 3
    thresh_list = [0.5, 0.5, 0.5]
    for num_iter in range(1,max_iter+1):
        threshold = thresh_list[num_iter-1]     
        print('num_iter', num_iter, 'thresh', threshold, 'input_file', input_file)
        
        # Find pairs
        pair_data_iter = find_pairs(input_file, scaler, clf, xyz_pxl, threshold)
                        
        # Remove duplicates
        pair_data_iter = remove_duplicates(pair_data_iter)
        
        # Save pair_data as csv file
        df1 = pd.DataFrame.from_dict(pair_data_iter)
        csv_file = os.path.join(swc_outdir, 'pair_dict_iter%d.csv'%num_iter)
        df1.to_csv(csv_file, index=False)
        
        # Merge segment pairs with prob below thresh
        output_file = os.path.join(swc_outdir, 'connect_iter%d.swc'%num_iter) 
        new_morph = merge_pairs(input_file, output_file, csv_file, threshold)
        input_file = output_file
            
            
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


def remove_duplicates(pair_data): 
    t1 = [pair['tree1'] for pair in pair_data]
    t2 = [pair['tree2'] for pair in pair_data]
    num_tree = max(max(t1), max(t2)) + 1
    dupl_list = []    
    for i in range(num_tree):
        select1 = [pair_data.index(p) for p in pair_data if p['tree1'] == i]
        js = [pair_data[s1]['tree2'] for s1 in select1]
        js_unique = np.unique(js)
        for j in js_unique:
            s2 = np.where(js == j)[0]
            if len(s2) > 1:
                probs = [pair_data[select1[s]]['prob'] for s in s2]
                idx = np.argmax(probs)
                for k, s in enumerate(s2):
                    if k != idx:
                        dupl_list.append(select1[s])
    pair_data1 = [pair_data[i] for i in range(len(pair_data)) if i not in dupl_list]  
    if not pair_data1:
        return
    
    dupl_list = []    
    for i in range(num_tree):
        select1 = [pair_data1.index(p) for p in pair_data1 if p['tree1'] == i]
        for s1 in select1:
            j = pair_data1[s1]['tree2']
            select2 = [pair_data1.index(p) for p in pair_data1 if p['tree1'] == j]
            for s2 in select2:
                k = pair_data1[s2]['tree2']
                if k == i:
                    if pair_data1[s1]['prob'] < pair_data1[s2]['prob']:
                        dupl_list.append(s1)
                    elif pair_data1[s1]['prob'] > pair_data1[s2]['prob']:
                        dupl_list.append(s2)
                    else:    
                        dupl_list.append(max(s1,s2))
    pair_data1 = [pair_data1[i] for i in range(len(pair_data1)) if i not in dupl_list]
    if not pair_data1:
        return
    
    n1 = [pair['nid1'] for pair in pair_data1]
    n2 = [pair['nid2'] for pair in pair_data1]
    num_nid = max(max(n1), max(n2)) + 1
    
    dupl_list = []
    for i in range(num_nid):
        select = []
        select1 = [pair_data1.index(p) for p in pair_data1 if p['nid1'] == i]
        if len(select1) > 0:
            for s1 in select1:
                select.append(s1)
        select2 = [pair_data1.index(p) for p in pair_data1 if p['nid2'] == i]
        if len(select2) > 0:
            for s2 in select2:
                select.append(s2)
        if len(select) > 1:
            probs = [pair_data1[s]['prob'] for s in select]
            idx = np.argmax(probs)
            for k, s in enumerate(select):
                if k != idx:
                    dupl_list.append(s)
    pair_data1 = [pair_data1[i] for i in range(len(pair_data1)) if i not in dupl_list]
    return pair_data1    


def swc_prune(morph_in, outfile, pruning_threshold = 30,**kwargs):
    nodes_to_remove = set()
    prune_count = 0
    bifur_nodes = morph_in.branch_points
    
    graph = morph_in.get_graph_nx()
    for i,node in bifur_nodes.iterrows():
        children = morph_in.nodes.loc[morph_in.nodes['parent_id'] == node['node_id']]
        for i, child in children.iterrows():
            child_remove_nodes, child_seg_length = bfs_tree(child['node_id'],graph)
            if child_seg_length < pruning_threshold:
                prune_count+=1
                [nodes_to_remove.add(n) for n in child_remove_nodes]
                
    rem = [x - 1 for x in nodes_to_remove]   
    new_nodes = morph_in.nodes.drop(rem)
    morph_in.nodes = new_nodes
    
    navis.write_swc(morph_in, outfile)  


def swc_split_branches(morph_in, outfile, node_len):
    # Find branching nodes
    branch_nodes = list(morph_in.branch_points['node_id'])

    # Split branches
    for i in branch_nodes:
        children = list(morph_in.nodes.loc[morph_in.nodes['parent_id'] == i, 'node_id'])
        for child in children:
            morph_in.nodes.loc[morph_in.nodes['node_id'] == child, 'parent_id'] = -1
    
    # Find all trees and remove short ones
    tree_list = morph_in.segments
    node_remove = []
    
    for tree in tree_list:
        if len(tree) <= node_len:
            node_remove += list(tree)
            
    node_remove = [x - 1 for x in node_remove]

    # Drop nodes from node list and replace
    morph_in.nodes = morph_in.nodes.drop(node_remove)
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
    #extract the new node order using depth first search
    graph = morph_in.get_graph_nx()
    new_node_ids = {}
    start_label = 1
    for root in morph_in.root:
        seg_len = dfs_labeling(root,start_label,new_node_ids,graph)
        start_label += seg_len 
    
    #edit the node and parent ids
    new_output_dict = {}
    for i,row in morph_in.nodes.iterrows():
        new_id = new_node_ids[row['node_id']]
        old_parent = row['parent_id']
        if old_parent == -1:
            new_parent = -1
        else:
            new_parent = new_node_ids[old_parent]

        new_line = row.astype(str).values.flatten().tolist()
        new_line[0] = str(new_id)
        new_line[-2] = str(new_parent)
        new_line = new_line[:-1]
        new_line[-1] += '\n'
        new_line = " ".join(new_line)
        new_output_dict[new_id] = new_line
    
    #write to new file
    with open(outfile,"w") as f2:
        for k in sorted(list(new_output_dict.keys())):
            new_write_line = new_output_dict[k]
            f2.write(new_write_line) 



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
    i = 0
    if int(morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node]['parent_id']) != -1:
        while i < n:
            next_node = [node for node in segment if node==int(morph_in.nodes.loc[morph_in.nodes['node_id'] == nodes[-1]]['parent_id'])]
            if next_node==[]:
                i = n
            else:
                nodes.append(next_node[0])
            i+=1
                         
    else:
        while i < n:
            next_node = [node for node in segment if int(morph_in.nodes.loc[morph_in.nodes['node_id'] == node]['parent_id'])==nodes[-1]] 
            if next_node==[]:
                i = n
            elif len(next_node) > 1: 
                i = n
            else:
                nodes.append(next_node[0])
            i+=1
    return nodes 



def calculate_vector(nodes, pxl_xyz):
    nodes = nodes.values.tolist()
    nodes_coord = np.vstack([np.array((node[2], node[3], node[4]))*pxl_xyz for node in nodes])
    nodes_coord_mean = nodes_coord.mean(axis=0)
    _, _, vv = np.linalg.svd(nodes_coord - nodes_coord_mean)
    vector = vv[0]
    
    vect_diff = (nodes_coord[-1,:] - nodes_coord[0,:])/np.linalg.norm(nodes_coord[-1,:] - nodes_coord[0,:])
    if np.dot(-vector, vect_diff) > np.dot(vector, vect_diff):
        vector*=-1
        
    return vector 



def reroot_tree(start_node, tree, morph_in):
    neighbors_dict = {}
    for node in tree:
        node_neighbors = [] 
        parent = int(morph_in.nodes.loc[morph_in.nodes['node_id'] == node]['parent_id'])

        if parent != -1:
            node_neighbors.append(parent)
        children = morph_in.nodes.loc[morph_in.nodes['parent_id'] == node]['node_id']
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


def calculate_features(morph, tree1, end1, tree2, end2, pxl_xyz):
    dist = distance(end1, end2, pxl_xyz)
    treelist = [tree1, tree2]
    endlist = [end1, end2]
    cfactors = collinearity(morph, treelist, endlist, pxl_xyz)
    return np.array([dist] + cfactors)


def collinearity(morph_in, segments, end_nodes, pxl_xyz, num_nodes = [4,49]):
    end_coord = [np.array((node['x'], node['y'], node['z']))*pxl_xyz for node in end_nodes]
    cvect = end_coord[1] - end_coord[0]
    cvect_norm = np.linalg.norm(cvect)
    
    if cvect_norm != 0:
        cvect = cvect/cvect_norm
        cf = []
        for num in num_nodes:
            vectors = []
            for n in range(len(end_nodes)):
                nodes = get_nodes(morph_in, segments[n], end_nodes[n]['node_id'], num)
                df = morph_in.nodes.loc[morph_in.nodes['node_id'].isin(nodes)]
                vect = calculate_vector(df.reindex([x - 1 for x in nodes]), pxl_xyz)
                vectors.append(vect)
                
            cf.append(np.dot(-vectors[0], cvect))
            cf.append(np.dot(vectors[1], cvect))
    else:
        cf = [np.NaN, np.NaN]
    return cf  


def merge_pairs(morph, outfile, pair_data, thresh):
    tree_list = morph.subtrees
    pair_idx = np.arange(0,len(tree_list))
    idx_list = []
    for i in pair_idx:
        select1 = [pair_data.index(p) for p in pair_data if p['tree1'] == i]
        select2 = [pair_data.index(p) for p in pair_data if p['tree2'] == i]
        
        for s1 in select1:
            if pair_data[s1]['prob'] > thresh:
                if s1 not in idx_list:
                    cfactors = np.array([pair_data[s1]['cf0_near0'], pair_data[s1]['cf1_near0'],
                                        pair_data[s1]['cf0_far0'], pair_data[s1]['cf1_far0']])
                    
                    if len(np.where(cfactors < 0.5)[0]):
                        print('idx %d'%(pair_data[s1]['idx']), cfactors)
                    else:
                        end_nodes = [pair_data[s1]['nid1'], pair_data[s1]['nid2']]
                        morph = connect_trees(end_nodes, morph)
                        idx_list.append(s1)
                               
        for s2 in select2:
            if pair_data[s2]['prob'] > thresh:
                if s2 not in idx_list:
                    cfactors = np.array([pair_data[s2]['cf0_near0'], pair_data[s2]['cf1_near0'],
                                        pair_data[s2]['cf0_far0'], pair_data[s2]['cf1_far0']])
                    
                    if len(np.where(cfactors < 0.5)[0]):
                        print('idx %d'%(pair_data[s2]['idx']), cfactors)
                    else:
                        end_nodes = [pair_data[s2]['nid2'], pair_data[s2]['nid1']]
                        morph = connect_trees(end_nodes, morph)
                        idx_list.append(s2)


    navis.write_swc(morph, outfile)
    return morph 



def find_pairs(morph_in, sc, cl, pxl_xyz, thresh, radius=range(0,15,5)):
    leaf_nodes = list(morph_in.leafs['node_id'])
    root_nodes = list(morph_in.root)
    tree_list = morph_in.subtrees
    
    end_node_list = []
    idx_list = []
    
    for i, tree in enumerate(tree_list):
        root_node = [node for node in tree if node in root_nodes]
        if int(morph_in.nodes['label'].loc[morph_in.nodes['node_id'] == root_node[0]]) != 1:
            idx_list.append(i)
            end_node_list.append(root_node[0])
            
        tree_leaf_nodes = [i for i in tree if i in leaf_nodes]
        
        for node in tree_leaf_nodes:
            idx_list.append(i)
            end_node_list.append(node)
    
    idx_arr = np.array(idx_list)
    end_node_coord = np.array([np.array((float(morph_in.nodes['x'].loc[morph_in.nodes['node_id'] == node])*pxl_xyz[0], float(morph_in.nodes['y'].loc[morph_in.nodes['node_id'] == node])*pxl_xyz[1],
                            float(morph_in.nodes['z'].loc[morph_in.nodes['node_id'] == node])*pxl_xyz[2])) for node in end_node_list])
    end_node_coord = end_node_coord.round(decimals=2, out=None)
    
    kdtree_list = []
    for idx1, tree in enumerate(tree_list):
        s1 = np.where(idx_arr == idx1)[0]
        s2 = np.where(idx_arr != idx1)[0]
        kdtree_list.append(KDTree(end_node_coord[s2], leaf_size=2))
    
    farray = np.zeros((len(end_node_list),len(end_node_list),5))
    n = 0
    num_ends = 5
    pair_data = []
    
    
    for idx1, tree in enumerate(tree_list):
        
        s1 = np.where(idx_arr == idx1)[0]
        s2 = np.where(idx_arr != idx1)[0]
        kdtree = kdtree_list[idx1]
        for s in s1:
            min_dist_idx, dist = kdtree.query_radius(end_node_coord[s].reshape(1,3), radius[-1],
                                              return_distance=True, sort_results=True)
            select = []
            for i in range(len(radius)-1):
                select.append(np.where((dist[0]>radius[i])&(dist[0]<=radius[i+1])))
                
            for l in select:
                p_arr = s2[min_dist_idx[0][l]]
                idx2_arr = idx_arr[p_arr]
                
                if len(p_arr) > 0:
                    probs = np.zeros((len(p_arr),))
                    nodes = np.empty((len(p_arr),), dtype=object)
                    features = np.empty((len(p_arr),), dtype=object)
                    for i, p in enumerate(p_arr):
                        if farray[s,p,0] == 0:
                            f = calculate_features(morph_in, tree_list[idx1],  morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[s]].squeeze(),
                                                  tree_list[idx2_arr[i]], morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[p]].squeeze(), pxl_xyz)
                            farray[s,p,:] = f
                        else:
                            f = farray[s,p,:]
                            
                        dist1, min_dist_idx1 = kdtree.query(end_node_coord[s].reshape(1,3), k=num_ends)
                        q_arr = s2[min_dist_idx1[0]]
                        s3 = np.where(q_arr != p)[0][0:num_ends-1]
                        q_arr = q_arr[s3]
                        q_arr = q_arr[:num_ends-1]
                        dist1 = dist1[:,s3]
                        idx3_arr = idx_arr[q_arr]
                        for j, q in enumerate(q_arr):
                            if farray[s,q,0] == 0:
                                f1 = calculate_features(morph_in, tree_list[idx1], morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[s]].squeeze(),
                                              tree_list[idx3_arr[j]], morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[q]].squeeze(), pxl_xyz)
                                farray[s,q,:] = f1
                            else:
                                f1 = farray[s,q,:]
                            f = np.concatenate((f,f1))

                        s3 = np.where(idx_arr != idx2_arr[i])[0]
                        kdtree1 = kdtree_list[idx2_arr[i]]
                        dist1, min_dist_idx1 = kdtree1.query(end_node_coord[p].reshape(1,3), k=num_ends)
                        q_arr = s3[min_dist_idx1[0]]
                        s4 = np.where(q_arr != s)[0][0:num_ends-1]
                        q_arr = q_arr[s4]
                        q_arr = q_arr[:num_ends-1]
                        dist1 = dist1[:,s4]
                        idx3_arr = idx_arr[q_arr]
                        for j, q in enumerate(q_arr):
                            if farray[p,q,0] == 0:
                                f1 = calculate_features(morph_in, tree_list[idx2_arr[i]], morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[p]].squeeze(),
                                              tree_list[idx3_arr[j]], morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[q]].squeeze(), pxl_xyz)
                                farray[p,q,:] = f1
                            else:
                                f1 = farray[p,q,:]
                            f = np.concatenate((f,f1))
                        features[i] = f
                        x = sc.transform(f.reshape(1,-1))
                        probs[i] = cl.predict_proba(x)[0][1]
                        
                    probs_max = np.max(probs)
                    k_max = np.argmax(probs)
                    if probs_max >= thresh:
                        pair_dict = {}
                        pair_dict['idx'] = n
                        pair_dict['tree1'] = idx1
                        pair_dict['tree2'] = idx2_arr[k_max]
                        pair_dict['nid1'] = int(morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[s]]['node_id'])
                        pair_dict['pid1'] = int(morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[s]]['parent_id'])
                        pair_dict['nid2'] = int(morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[p_arr[k_max]]]['node_id'])
                        pair_dict['pid2'] = int(morph_in.nodes.loc[morph_in.nodes['node_id'] == end_node_list[p_arr[k_max]]]['parent_id'])
                        pair_dict['prob'] = probs_max
                        for i in range(num_ends*2 -1):
                            pair_dict['distance%d'%i] = features[k_max][i*5]
                            pair_dict['cf0_near%d'%i] = features[k_max][i*5+1]
                            pair_dict['cf1_near%d'%i] = features[k_max][i*5+2]
                            pair_dict['cf0_far%d'%i] = features[k_max][i*5+3]
                            pair_dict['cf1_far%d'%i] = features[k_max][i*5+4]
                        pair_data.append(pair_dict)
                        n += 1
                        break
                        
    return pair_data
    

            
class Reconnect(ags.ArgSchemaParser):
        
    def run(self):
        reconnect(self.args['input_file'], self.args['output_dir'], self.args['model_dir'],
        self.args['pxl_xyz'])
   
        
if __name__ == "__main__":
    mod = Reconnect(schema_type=ReconnectParameters)
    mod.run()      