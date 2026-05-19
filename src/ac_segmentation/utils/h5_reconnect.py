#updated 01/28/2026

# Standard library
import os
import io
from io import BytesIO
import uuid
from uuid import UUID
import copy
import math
import colorsys
import tarfile
from pathlib import Path
import glob
import itertools
from operator import add
from collections import deque, defaultdict, OrderedDict, Counter
import random
import concurrent
from concurrent.futures import ThreadPoolExecutor

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
import navis
import networkx as nx
import requests
import scipy
from sklearn.neighbors import KDTree
from joblib import dump, load, Parallel, delayed, parallel_config
from natsort import natsorted
import argschema as ags
from concurrent.futures import ProcessPoolExecutor, as_completed
from cloudvolume import Skeleton
from tqdm import tqdm

    

def calculate_vector(coords):
    # TODO replace svd w/ eigh
    _, _, vv = np.linalg.svd(coords - coords.mean(axis=0))
    vector = vv[0]

    # Fix wrong orientation (sign) of vector
    vect_diff = (coords[-1,:] - coords[0,:])/np.linalg.norm(coords[-1,:] - coords[0,:])
    if np.dot(-vector, vect_diff) > np.dot(vector, vect_diff):
        vector*=-1    
    return vector 
    
  
def merge_continuous(a1, a2):
    """Merge two arrays of 3D points so they connect smoothly."""
    if np.allclose(a1[-1], a2[0]):
        # a1 end connects to a2 start
        return np.vstack((a1, a2[1:]))
    elif np.allclose(a1[-1], a2[-1]):
        # a1 end connects to a2 end — flip a2
        return np.vstack((a1, np.flipud(a2[:-1])))
    elif np.allclose(a1[0], a2[-1]):
        # a1 start connects to a2 end — prepend a2
        return np.vstack((a2, a1[1:]))
    elif np.allclose(a1[0], a2[0]):
        # a1 start connects to a2 start — flip a2 then prepend
        return np.vstack((np.flipud(a2), a1[1:]))
    else:
        # No connection — just stack with a gap
        return np.vstack((a1, a2))


def closest_points(skel, node_id, rank=1):
    """Merge a list of vertex arrays into one continuous array."""
    vertices = skel.vertices
    array_list = skel.paths()
    merged = array_list[0]
    rank = min(rank, len(vertices))
    for arr in array_list[1:]:
        merged = merge_continuous(merged, arr)

    node_vertex = vertices[node_id]
    closest_ranked_points = merged[0:rank+1]
    closest_ranked_node_ids = []
    if list(node_vertex) == list(merged[-1]):
        closest_ranked_points =  merged[-1-rank:][::-1]

    for vertex in closest_ranked_points:
        closest_ranked_node_ids.append(np.where((vertices == vertex).all(axis=1))[0][0])

    return np.array(closest_ranked_points), closest_ranked_node_ids

def end_nodes_pre(skel):
    paths = skel.paths()
    ends_arr = np.array([paths[0][0], paths[0][-1]])
    nodes = [np.where((skel.vertices == row).all(axis=1))[0][0] for row in ends_arr]
    return nodes
    
    
    
def calculate_features(ns, end_node_ids, mode="inference", num_nodes=(5, 50)):

    if mode == "inference":
        alt_end_node_ids = np.copy(end_node_ids)
        end_coords = np.vstack([
            n.vertices[alt_end_node_id]
            for n, alt_end_node_id in zip(ns, alt_end_node_ids)
        ])
        cvect = end_coords[1] - end_coords[0]
        cvect_norm = np.linalg.norm(cvect)
        cvect /= cvect_norm
    
        cf = []
        for num in num_nodes:
            for i, (n, end_node_id) in enumerate(zip(ns, end_node_ids)):
                neighbor_loc_arr, nodes = closest_points(n, end_node_id, num)
    
                vec = calculate_vector(neighbor_loc_arr)
                cf.append(np.dot(vec, cvect))
                
        return np.array([cvect_norm] + cf)


    elif mode == "dist":
        if isinstance(num_nodes, tuple):
            num = num_nodes[0]
        else:
            num = num_nodes

        
        cf = []
        vecs = []
        for i, (n, end_node_id) in enumerate(zip(ns, end_node_ids)):
            neighbor_loc_arr, nodes = closest_points(n,end_node_id, num)
            vecs.append(calculate_vector(neighbor_loc_arr)) 
        
        cf = -np.dot(vecs[0], vecs[1])
        return cf

    else:
        raise ValueError("mode must be 'inference' or 'dist'")


 
def merge_pairs_inference_og(neuro_list, pair_data, prob_thresh = None, min_collin = None):
    if prob_thresh:
        pair_data = [x for x in pair_data if (x[1][0] > prob_thresh)]
        pair_data  = pd.DataFrame([list(i[0]) + list(i[1]) for i in pair_data])
        pair_data  = pair_data.sort_values(2, ascending=False).drop_duplicates(0).sort_index()
        pair_data  = pair_data.sort_values(2, ascending=False).drop_duplicates(1).sort_index()
    else:
        pair_data = [x for x in pair_data]
        pair_data  = pd.DataFrame([list(i[0]) + list(i[1]) for i in pair_data])
        pair_data  = pair_data.sort_values(4, ascending=False).drop_duplicates(0).sort_index()
        pair_data  = pair_data.sort_values(4, ascending=False).drop_duplicates(1).sort_index()                
        
    #remove duplicates that are in both pre and post columns
    rem = []
    for ind,row in pair_data.iterrows():
        if row[0] in list(pair_data[1]):
            rem.append(ind)
    pair_data = pair_data.drop(rem)

    return pair_data

 
def deduplicate(pair_data, threshold=None):
    #remove duplicates that are in both pre and post columns
    if threshold:
        pair_data = [x for x in pair_data if (x[4] > threshold)]
    pair_data  = pd.DataFrame([list(x) for x in pair_data], columns=["id1", "node1", "id2", "node2", "metric"])
    pair_data  = pair_data.sort_values("metric", ascending=False).drop_duplicates("id1").sort_index()
    pair_data  = pair_data.sort_values("metric", ascending=False).drop_duplicates("id2").sort_index()

    return pair_data      
    
def merge_pairs(neuro_list, pair_data,  prob_thresh = None, min_collin = None):
    
    
    #remove duplicates that are in both pre and post columns
    pair_data = deduplicate(pair_data)
    
    
    #declare lists
    neuro_list = copy.deepcopy(neuro_list)
    merge_num = 0
    neuro_ids = {n.id: i for i, n in enumerate(neuro_list)}

    used_nodes = set()
    merge_list = []
    merge_ids_pairs = []
    merge_ids_single = []
    unmerge_list = []

    #keep track of vertices
    merge_vertices = {}

    
    for _, row in pair_data.iterrows():
        id1, node1, id2, node2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])      
        
        # skip if either neuron-node was already used
        if (id1, node1) in used_nodes or (id2, node2) in used_nodes:
            continue

        # retrieve neurons
        m1 = neuro_list[neuro_ids[id1]]
        m2 = neuro_list[neuro_ids[id2]]

        merge_vertices[str(m1.vertices[0])] = id1
        merge_vertices[str(m2.vertices[0])] = id2 

        # --- identical merge logic ---
        end = m2.vertices[node2]
        m1.vertices = np.append(m1.vertices, [end], axis=0)
        m1.edges = np.append(m1.edges, [[node1, len(m1.vertices) - 1]], axis=0)
        m1.radius = np.append(m1.radius, 1)
        m1.vertex_types = np.append(m1.vertex_types, 0)
        merge_list += [m1, m2]
        # --------------------------------

        # mark both endpoints as used
        used_nodes.update([(id1, node1), (id2, node2)])
        merge_ids_pairs.append([id2,id1])
        merge_ids_single.append(id2)
        merge_ids_single.append(id1)
        merge_num += 1

    for sk in neuro_list:
        if sk.id not in merge_ids_single:
            unmerge_list.append(sk)
    
    merge_list = Skeleton.simple_merge(merge_list).consolidate().components()

    #reapply old id
    for sk in merge_list:
        sk.id = (merge_vertices[str(sk.vertices[0])])

    print(f"Pairs merged: {merge_num}")
    return merge_list, unmerge_list, merge_ids_pairs
    

def _extract_endpoints_batch(neuron_batch, min_nodes):
    """
    neuron_batch: list of (neuron_index, neuron_object)
    """
    results = []

    for idx, sk in neuron_batch:
        if len(sk.branches()) != 0:
            continue
            
        if min_nodes:
            if len(sk.vertices) < min_nodes:
                continue

        node_ids = end_nodes_pre(sk)
        for e in node_ids:
            results.append((
                idx,                # neuron index (CRITICAL)
                sk.id,              # neuron id
                e,                  # node id
                sk.vertices[e],     # endpoint coord
            ))

    return results


def chunked_enumerate(iterable, size):
    it = enumerate(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def extract_endpoints_parallel(neuro_list, n_jobs=10, batch_size=50, min_nodes=None):
    batches = list(chunked_enumerate(neuro_list, batch_size))

    all_results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(_extract_endpoints_batch, batch, min_nodes)
            for batch in batches
        ]

        # completion order does NOT matter
        for fut in as_completed(futures):
            all_results.extend(fut.result())

    # now flatten deterministically
    endpts = []
    endpts_node_id = []
    endpts_neuron_id = []
    endpts_neuron_idx = []

    for neuron_idx, neuron_id, node_id, coord in all_results:
        endpts.append(coord)
        endpts_node_id.append(node_id)
        endpts_neuron_id.append(neuron_id)
        endpts_neuron_idx.append(neuron_idx)

    return (
        np.asarray(endpts),
        np.asarray(endpts_node_id),
        np.asarray(endpts_neuron_id),
        np.asarray(endpts_neuron_idx),
    )

    

def _navis_to_cloudvol_batch(indexed_skels):
    """
    indexed_skels: list of (index, navis_skeleton)
    """
    results = []

    for idx, sk in indexed_skels:
        # Ensure 'label' column exists and is integer
        if 'label' not in sk.nodes:
            sk.nodes.insert(1, 'label', [0] * len(sk.nodes))
        else:
            sk.nodes['label'] = sk.nodes['label'].astype(int)

        # Reformat skeleton node table to SWC
        skt = navis.TreeNeuron(sk.nodes.copy()).nodes

        # If label column has strings, convert to 0
        if isinstance(skt['label'].iloc[0], str):
            skt = skt.copy()
            skt.loc[:, 'label'] = 0

        # SWC-relevant columns
        sk0 = skt[['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']].copy()

        # Safe conversion to scalar integers
        for col in ['node_id', 'label', 'parent_id']:
            sk0[col] = sk0[col].apply(
                lambda x: int(np.array(x).item())
                if isinstance(x, (np.ndarray, list))
                else int(x)
            )

        # Convert to SWC string
        sk0_list = [list(x) for x in zip(*(sk0[c].values.tolist() for c in sk0.columns))]
        swc_str = '\n'.join(str(x)[1:-1].replace(",", "") for x in sk0_list)

        # Create CloudVolume Skeleton
        skel = Skeleton.from_swc(swc_str)
        skel.id = str(sk.id)
        skel = skel.consolidate()

        results.append((idx, skel))

    return results

def navis_to_cloudvol(skels,n_jobs=10,batch_size=20):
    if n_jobs is None:
        n_jobs = os.cpu_count()

    batches = list(chunked_enumerate(skels, batch_size))

    results = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(_navis_to_cloudvol_batch, batch)
            for batch in batches
        ]

        for fut in as_completed(futures):
            for idx, skel in fut.result():
                results[idx] = skel

    # Reconstruct in original order
    return [results[i] for i in range(len(skels))]


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def _process_pair_batch(
    pair_batch,
    kdt,
    neuro_list,
    endpts_neuron_id,
    endpts_node_id,
    endpts_neuron_idx,
    endpts,
    sc,
    cl,
    neighbors_considered=4,
    min_collin = .1
):
    subfeat = {}  # shared across the batch
    results = []
    

    for ind, pq in enumerate(pair_batch):
        p, q = tuple(pq)

        candidate_neuron_idxs = endpts_neuron_idx[[p, q]]
        candidate_neurons = [neuro_list[idx] for idx in candidate_neuron_idxs]
        candidate_end_node_ids = endpts_node_id[[p, q]]

        key = (candidate_neurons[0].id, candidate_neurons[1].id)
        if key not in subfeat:
            subfeat[key] = tuple(
                calculate_features(
                    candidate_neurons,
                    candidate_end_node_ids,
                    mode="inference",
                )
            )

        cf = subfeat[key]
        if np.any(np.array(cf[1:5]) < min_collin):
            continue

        f = list(cf)

        # p neighbors
        _, idxs = kdt.query(endpts[p], k=neighbors_considered + 2)
        for knn_idx in [i for i in idxs if i not in (p, q)][:neighbors_considered]:
            nn = (candidate_neurons[0], neuro_list[endpts_neuron_idx[knn_idx]])
            nn_ids = (candidate_end_node_ids[0], endpts_node_id[knn_idx])
            key = (nn[0].id, nn[1].id)
            if key not in subfeat:
                subfeat[key] = tuple(
                    calculate_features(nn, nn_ids, mode="inference")
                )
            f.extend(subfeat[key])

        # q neighbors
        _, idxs = kdt.query(endpts[q], k=neighbors_considered + 2)
        for knn_idx in [i for i in idxs if i not in (p, q)][:neighbors_considered]:
            nn = (candidate_neurons[1], neuro_list[endpts_neuron_idx[knn_idx]])
            nn_ids = (candidate_end_node_ids[1], endpts_node_id[knn_idx])
            key = (nn[0].id, nn[1].id)
            if key not in subfeat:
                subfeat[key] = tuple(
                    calculate_features(nn, nn_ids, mode="inference")
                )
            f.extend(subfeat[key])

        f = np.asarray(f)
        f[np.isnan(f)] = 0

        x = sc.transform(f.reshape(1, -1))
        prob = cl.predict_proba(x)[0][1]

        results.append([endpts_neuron_id[p], endpts_node_id[p],
            endpts_neuron_id[q], endpts_node_id[q], prob])

    return results
    

def find_pairs_inference_parallel(
    kdt,
    neuro_list,
    pairs,
    endpts_neuron_id,
    endpts_node_id,
    endpts_neuron_idx,
    endpts,
    sc=None,
    cl=None,
    n_jobs=None,
    batch_size=25,
    min_collin=.1
):
    if n_jobs is None:
        n_jobs = os.cpu_count()

    pair_features = []

    pair_batches = list(chunked(pairs, batch_size))
    
    print("Pair Batches: ", len(pair_batches))

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(
                _process_pair_batch,
                batch,
                kdt,
                neuro_list,
                endpts_neuron_id,
                endpts_node_id,
                endpts_neuron_idx,
                endpts,
                sc,
                cl,
                4,
                min_collin
            )
            for batch in pair_batches
        ]

        for fut in tqdm(futures, total=len(futures)):
            for res in fut.result():
                pair_features.append(res)

    return pair_features



def find_pairs_dist(kdt, neuro_list, pairs, endpts_neuron_id, endpts_node_id, endpts_neuron_idx, endpts, query_dis=5, bound_box=None, min_collin=.9, min_nodes=None):
    # calculate features for pairs
    pair_features = []
    for ind,pq in enumerate(pairs):
        # initial feature based on condidate pair
        f = tuple()
        p, q = pq

        candidate_neuron_idxs = endpts_neuron_idx[[p, q]] 
        candidate_neurons = []
        for idx in candidate_neuron_idxs:
            candidate_neurons.append(neuro_list[idx])
        
        candidate_end_node_ids = endpts_node_id[[p, q]]
        cf = calculate_features(candidate_neurons, candidate_end_node_ids, mode="dist")
            
        if cf and cf > min_collin:
            pair_features.append([endpts_neuron_id[p],endpts_node_id[p], endpts_neuron_id[q], endpts_node_id[q], cf])

    return pair_features



def find_pairs(
    neuro_list,
    query_dis=10,
    sc=None,
    cl=None,
    bound_box=None,
    min_nodes=None,
    min_collin=0.9,
    n_jobs=10,
    batch_size=1000,
):
    np.seterr(invalid="ignore")

    # --- parallel endpoint extraction ---
    (
        endpts,
        endpts_node_id,
        endpts_neuron_id,
        endpts_neuron_idx,
    ) = extract_endpoints_parallel(
        neuro_list,
        n_jobs=n_jobs,
        batch_size=batch_size,
        min_nodes=min_nodes
    )

    # --- KDTree + pair search ---
    kdt = scipy.spatial.KDTree(endpts, leafsize=2)
    pairs = kdt.query_pairs(query_dis)

    # remove same-neuron pairs
    pairs = {
        frozenset(p)
        for p in pairs
        if len({*endpts_neuron_id[list(p)]}) > 1
    }

    # --- downstream logic unchanged ---
    if sc is not None and cl is not None:
        pairs =  find_pairs_inference_parallel(
            kdt,
            neuro_list,
            pairs,
            endpts_neuron_id,
            endpts_node_id,
            endpts_neuron_idx,
            endpts,
            sc=sc,
            cl=cl,
            n_jobs=n_jobs,
            batch_size=batch_size,
            min_collin=min_collin
        )
    else:
    
        pairs =  find_pairs_dist(
            kdt,
            neuro_list,
            pairs,
            endpts_neuron_id,
            endpts_node_id,
            endpts_neuron_idx,
            endpts,
            query_dis=query_dis,
            bound_box=bound_box,
            min_collin=min_collin,
            min_nodes=min_nodes,
        )
    
    
    dtype = np.dtype([
    ("id1", np.int64),
    ("count1", np.int32),
    ("id2", np.int64),
    ("count2", np.int32),
    ("score", np.float32),])

    pairs = np.array([tuple(row) for row in pairs], dtype=dtype)
    return pairs


def break_branches(skeletons, min_nodes=0):
    """
    Break skeletons at branches.
    
    skeletons: list or dict of Skeleton objects (must have .vertices, .edges, .components(), .remove_disconnected_vertices())
    min_nodes: minimum number of vertices for a fragment to be kept
    """
    if isinstance(skeletons, list):        
        skeletons = {item.id: item for item in skeletons}
    
    max_id = max(skeletons.keys())
    new_skels = {}
    split_num = 0

    for sk_id in skeletons.keys():
        sk = skeletons[sk_id]
        branch_nodes = list(sk.branches())

        if branch_nodes:
            # remove edges connected to branch nodes
            connected = sk.edges[np.isin(sk.edges, branch_nodes).any(1)]
            pre, post = [], []
            for x, y in connected:
                if x in branch_nodes:
                    pre.append(list(sk.vertices[y]))
                    post.append(list(sk.vertices[x]))
                else:
                    pre.append(list(sk.vertices[x]))
                    post.append(list(sk.vertices[y]))

            pre = np.array(pre)
            post = np.array(post)

            sk.edges = sk.edges[~np.isin(sk.edges, branch_nodes).any(1)]
            sk = sk.remove_disconnected_vertices()

            # get connected components
            split_skels = sk.components()
            # sort by size descending to ensure largest fragment is first
            split_skels = sorted(split_skels, key=lambda s: len(s.vertices), reverse=True)

            for ind, split in enumerate(split_skels):
                if len(split.vertices) >= min_nodes:
                    if ind == 0:
                        # largest fragment keeps original ID
                        split.id = sk_id
                        split.parent_id = sk_id
                        new_skels[sk_id] = split
                    else:
                        # new ID for smaller fragments
                        max_id += 1
                        split.id = max_id
                        split.parent_id = sk_id
                        new_skels[max_id] = split
                    split_num += 1

                    # update vertices positions based on branch adjustments
                    for idx, vert in enumerate(split.vertices):
                        a = np.all(pre == vert, axis=1)
                        true_idx = np.where(a)[0]
                        if len(true_idx) > 0:
                            split.vertices[idx] = ((post[true_idx] * 0.8) + (split.vertices[idx] * 0.2))
        else:
            # skeleton with no branches is unchanged
            sk.parent_id = sk_id
            new_skels[sk_id] = sk

    #print(f'Number of splits created: {split_num}')
    return new_skels

def cloudvol_to_navis(skels):
    out_sk = navis.NeuronList(None)
    try:
        for sk in skels:
            out_sk.append(navis.NeuronList(sk.to_swc()))
    except:
        out_sk.append(navis.NeuronList(skels.to_swc()))
    return out_sk
    
def prune_to_furthest_end_path(skels):
    """
    Prune skeleton to the shortest path between the two terminal nodes
    that are furthest apart in Euclidean distance (loops removed).

    Parameters
    ----------
    edges : list of (int, int)
        Skeleton edges
    skel : object
        Skeleton with skel.vertices (x, y) or (x, y, z)

    Returns
    -------
    pruned_edges : list of (int, int)
        Edge list of the backbone path
    """
    
    for ind,sk in enumerate(skels):
        if len(sk.branches()) > 0:
            # Build graph
            edges = sk.edges
            G = nx.Graph()
            G.add_edges_from(edges)
            if G.number_of_nodes() == 0:
                return []
        
            # Node coordinates
            coords = {i: np.array(v) for i, v in enumerate(sk.vertices)}
        
            # Remove loops: build forest via spanning trees
            forest = nx.Graph()
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                mst = nx.minimum_spanning_tree(subgraph)
                forest.add_edges_from(mst.edges)
        
            # Terminal nodes (degree 1)
            leaves = [n for n in forest.nodes if forest.degree[n] == 1]
            if len(leaves) < 2:
                return edges  # nothing to prune
        
            # Find two leaves furthest apart (Euclidean)
            max_dist = -1
            end_a, end_b = leaves[0], leaves[1]
            for i, u in enumerate(leaves):
                for v in leaves[i+1:]:
                    d = np.linalg.norm(coords[u] - coords[v])
                    if d > max_dist:
                        max_dist = d
                        end_a, end_b = u, v
        
            # Shortest path between furthest ends (on loop-free graph)
            backbone_path = nx.shortest_path(forest, end_a, end_b)
        
            # Keep only edges on this path
            keep = set(backbone_path)
            pruned_edges = [(a, b) for a, b in edges if a in keep and b in keep]
            sk.edges = np.array(pruned_edges)
            sk = sk.remove_disconnected_vertices()
            skels[ind] = sk
            
    return skels