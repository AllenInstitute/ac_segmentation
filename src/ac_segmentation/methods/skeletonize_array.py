import os
import argschema
import skimage
import kimimaro
import cloudvolume
from cloudvolume import Skeleton
from scipy.spatial import KDTree
from joblib import Parallel, delayed, parallel_config, dump, load
import itertools
import scipy
import tarfile
import navis
import tarfile
from io import BytesIO
import concurrent.futures
import itertools
import numpy as np
from kimimaro.intake import merge
from matplotlib import pyplot as plt
import cc3d
import ast
from collections import defaultdict, deque
import networkx as nx
import uuid
from cloudvolume import CloudVolume, Skeleton, paths
from pathlib import Path
     
from ac_segmentation.utils.tensorstore import open_tensor, create_kvstore, AWS_Parameters
from ac_segmentation.utils.io import write_cv_skels_tar
from ac_segmentation.utils.h5_skeletons import *
from ac_segmentation.utils.h5_reconnect import *



def label_binary_array(binary_arr, size_threshold=20):
    
    labeled_arr, num_features = cc3d.connected_components(binary_arr, connectivity=6, return_N=True)
    if num_features > 1:
        labeled_arr = skimage.morphology.remove_small_objects(
            labeled_arr, min_size=size_threshold, connectivity=3, out=labeled_arr)
        
    return labeled_arr, len(np.unique(labeled_arr))

def threshold_binarize_array(arr, threshold=0.2):
    return (arr >= threshold)

def skeletonize(out_arr, probability_threshold=0.2, label_size_threshold=50, scale=10, constant=10, 
                fill_holes=False, parallel=1, dust_threshold=10, max_paths=None):
    # binarize volume, label, and skeletonize
    binary_arr = threshold_binarize_array(out_arr, threshold=probability_threshold)
    labeled_arr, num_feat = label_binary_array(binary_arr, size_threshold=label_size_threshold)
    skels = kimimaro.skeletonize(
        labeled_arr,
        teasar_params={
            "scale": scale, 
            "const": constant, # influences the finger branches allowed
            "pdrf_scale": 10000,
            "pdrf_exponent": 1,
            "soma_acceptance_threshold": 3500, # physical units
            "soma_detection_threshold": 750, # physical units
            "soma_invalidation_const": 300, # physical units
            "soma_invalidation_scale": 2,
            "max_paths": None, # default None
        },
        dust_threshold=dust_threshold, # skip connected components with fewer than this many voxels
        anisotropy=(1,1,1), # default True #influences the dimension scale
        fix_branching=True, # default True
        fix_borders=True, # default True
        fill_holes=fill_holes, # default False
        fix_avocados=False, # default False
        progress=False, # default False, show progress bar
        parallel=parallel, # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100, # how many skeletons to process before updating progress bar
    )

    return skels


# In[3]:


###NEW EDIT
def kimi_to_navis(skels, tag=None):
    out_sk = navis.NeuronList(None)
    try:
        for sk in skels:
            sk = navis.TreeNeuron(sk.to_swc())
            if tag:
                sk.name = tag
            out_sk.append(sk)
    except:
        out_sk.append(navis.NeuronList(skels.to_swc()))

    return out_sk


# In[4]:


###Edited
def create_chunked_dims(arr_shape, chunk_size, overlap=0):
    # Ensure chunk_size is appropriate for the arr_shape length
    if len(arr_shape) != len(chunk_size):
        raise ValueError("arr_shape and chunk_size must have the same number of dimensions")

    dx, dy, dz = arr_shape[-3:]
    xch, ych, zch = chunk_size

    starts_x = list(range(0, dx, xch))
    starts_y = list(range(0, dy, ych))
    starts_z = list(range(0, dz, zch))

    comb1, comb2 = [], []

    for sx, sy, sz in itertools.product(starts_x, starts_y, starts_z):
        ex, ey, ez = min(sx + xch, dx), min(sy + ych, dy), min(sz + zch, dz)

        # Expand by overlap pixels, within bounds
        sx, sy, sz = max(0, sx - overlap), max(0, sy - overlap), max(0, sz - overlap)
        ex, ey, ez = min(dx, ex + overlap), min(dy, ey + overlap), min(dz, ez + overlap)

        comb1.append((sx, sy, sz))
        comb2.append((ex, ey, ez))
        
    return comb1,comb2


def break_branches(skeletons, min_nodes=4):
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
            if len(sk.vertices) >0:
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
                        #for idx, vert in enumerate(split.vertices):
                            #a = np.all(pre == vert, axis=1)
                            #true_idx = np.where(a)[0]
                            #if len(true_idx) > 0:
                                #split.vertices[idx] = ((post[true_idx] * 0.8) + (split.vertices[idx] * 0.2))
        else:
            # skeleton with no branches is unchanged
            sk.parent_id = sk_id
            new_skels[sk_id] = sk

    return new_skels


def TS_skeletonize_volume(seg_arr, chunk_size=[1000, 1000, 1000], cutout=None, n_jobs=4, prob_thresh=0.2, label_size_threshold=20, overlap=4):
    def skel_chunk(start, end):
        arr = seg_arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        skels = skeletonize(
            np.array(arr),
            probability_threshold=prob_thresh,
            label_size_threshold=label_size_threshold
        )

        if len(skels) != 0:
            skels = list(skels.values())
                                            
            #adjust skeleton vertices
            for sk in skels:
                sk.vertices += start                         
                
            return skels

    if len(seg_arr.shape) == 5:
        seg_arr = seg_arr[0,0,:,:,:]
    start,end = create_chunked_dims(seg_arr.shape, chunk_size=chunk_size, overlap=overlap)

    if cutout:
        start_new, end_new = [], []
        x1, x2, y1, y2, z1, z2 = cutout
        seg_shape = np.array(seg_arr.shape)
        sx,sy,sz = seg_shape[-3:]
        for s, e in zip(start, end):
            offset = np.array([x1, y1, z1])
            s = np.array(s[-3:]) + offset
            e = np.array(e[-3:]) + offset
            if (s[0] <= x2 and s[1] <= y2 and s[2] <= z2):
                if (s[0] <= sx and s[1] <= sy and s[2] <= sz):
                    # Clip end to not exceed array shape
                    e = np.minimum(e, seg_shape)
                    start_new.append(tuple(s))
                    end_new.append(tuple(e))
        start, end = start_new, end_new

    with parallel_config(backend="loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=n_jobs)(
            delayed(skel_chunk)(s, e) for s, e in zip(start, end)
        )


    results = [x for x in results]
    final = []
    for x in results:
        if x:
            final += x
    return final




class SkeletonizeProbabilitiesParameters(argschema.ArgSchema):
    input_path = argschema.fields.String(required=True)
    skeleton_output = argschema.fields.String(required=True)
    probability_threshold = argschema.fields.Float(
        required=False, default=20)
    label_size_threshold = argschema.fields.Int(required=False, default=80)
    n_jobs = argschema.fields.Int(required=False, default=15)
    cutout = argschema.fields.String(required=False, default=None, allow_none=True)
    output_json = argschema.fields.OutputFile(required=False, allow_none=True)
    skel_h5 = argschema.fields.Boolean(required=False, dump_default=True)
    
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
    
class SkeletonizeProbabilitiesModule(argschema.ArgSchemaParser):
    default_schema = SkeletonizeProbabilitiesParameters

    def run(self):
    
        #Convert all "None" strings to actual None 
        for key, value in self.args.items():
            if value == "None":
                self.args[key] = None
                
                                           
        if not self.args['endpoint']:
            endpoint=None
        
        kvstore_in = None
        if 's3://' in self.args['input_path']:
            if self.args['profile']:
                AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                kvstore_in = create_kvstore(fpath=str(self.args['input_path']), store='s3', AWS_param=AWS_param)              
                                        
            if self.args['AWS_key']:
                AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                kvstore_in = create_kvstore(fpath=str(self.args['input_path']), store='s3', AWS_param=AWS_param)           
        
        
        input_arr = open_tensor(self.args['input_path'], kvstore=kvstore_in, bytes_limit= 100_000_000, driver='zarr')
        base = 'skeletons_raw.swcs.tar.gz'
        if self.args["cutout"]:
            base = '{0}_skeletons_raw.swcs.tar.gz'.format(str(self.args["cutout"]))
            self.args["cutout"] = [int(s.replace("'", "")) for s in self.args["cutout"].split(',')]                          
            
        skels = TS_skeletonize_volume(input_arr, chunk_size=[100,100,100], n_jobs=self.args["n_jobs"], prob_thresh=self.args["probability_threshold"], label_size_threshold=self.args["label_size_threshold"], overlap=4, cutout=self.args["cutout"])      
        skels = [x for x in skels if x is not None]
        
        if len(skels) > 0 :       
            for ind in range(len(skels)):
                skels[ind].id = int(uuid.uuid4().int % 1e14)  
            print("Completed skeletonization, # of skels", len(skels))              
            
            #break branches
            skels = list(break_branches(skels).values())
                                                                                                                       
            #merge overlap skeletons
            fused = Skeleton.simple_merge(skels).consolidate().components()         
                    
            #remove twigs
            fused= prune_to_furthest_end_path(fused)
                   
            for ind in range(len(fused)):
                fused[ind].id = int(uuid.uuid4().int % 1e14)
                if len(fused[ind].vertices) < 10:
                     fused[ind] = None                 
            fused = [x for x in fused if x]
            
                                                  
            #write raw swc       
            last_2 = Path(*Path(self.args["input_path"]).parents[0].parts[-2:])
            os.makedirs(os.path.join(self.args["skeleton_output"], last_2), exist_ok=True)        
            skels_outpath = os.path.join(self.args["skeleton_output"], last_2, base)                                    
            write_cv_skels_tar(str(skels_outpath), fused, mode='w:gz')
                     
            #write h5
            if self.args["skel_h5"]: 
                out_skels_dic = {}
                for i in fused:     
                    out_skels_dic[i.id] = i
                global_index = shard_and_write_skeletons(out_skels_dic , os.path.join(self.args["skeleton_output"], last_2, "skeleton_shards"), max_skeletons_per_shard=10000, n_workers=10, label=str(self.args["cutout"]))
                
            

if __name__ == "__main__":
    mod = SkeletonizeProbabilitiesModule()
    mod.run()


__all__ = [
    "SkeletonizeProbabilitiesModule",
    "SkeletonizeProbabilitiesParameters"
]
