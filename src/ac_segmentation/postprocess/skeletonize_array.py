import os
import argschema
import skimage
import kimimaro
from cloudvolume import Skeleton
from joblib import Parallel, delayed, parallel_config
import itertools
import numpy as np
import navis
import cc3d
from collections import defaultdict
import time
from ac_segmentation.utils.tensorstore import open_tensor, create_kvstore, AWS_Parameters
from ac_segmentation.utils.io import write_kimi_skels_tar

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
            "max_paths": max_paths, # default None
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



def TS_skeletonize_volume(seg_arr, chunk_size=[1000, 1000, 1000], cutout=None, n_jobs=4, prob_thresh=0.2, label_size_threshold=20, overlap=4):
    def skel_chunk(start, end):
        skels = None
        try:
            arr = seg_arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            
            skels = skeletonize(
                np.array(arr),
                probability_threshold=prob_thresh,
                label_size_threshold=label_size_threshold
            )
        
        except:
            pass

        if skels:
            skels = [s for _, s in skels.items()]
            skels = Skeleton.simple_merge(skels).consolidate()
            skels.vertices += start  # shift back to global coords
            #skels = kimi_to_navis(skels.components(), tag=str([start[0],end[0], start[1],end[1], start[2],end[2]]))
            return skels 

    if len(seg_arr.shape) == 5:
        seg_arr = seg_arr[0,0,:,:,:]
    start,end = create_chunked_dims(seg_arr.shape, chunk_size=chunk_size, overlap=overlap)

    if cutout:
        start_new, end_new = [], []
        x1, x2, y1, y2, z1, z2 = cutout
        seg_shape = np.array(seg_arr.shape)
        for s, e in zip(start, end):
            offset = np.array([x1, y1, z1])
            s = np.array(s[-3:]) + offset
            e = np.array(e[-3:]) + offset
            if (s[0] <= x2 and s[1] <= y2 and s[2] <= z2):
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
    return results


class SkeletonizeProbabilitiesParameters(argschema.ArgSchema):
    input_zarr = argschema.fields.String(required=True)
    skeleton_output = argschema.fields.String(required=True)
    probability_threshold = argschema.fields.Float(
        required=False, default=0.05)
    label_size_threshold = argschema.fields.Int(required=False, default=80)
    n_jobs = argschema.fields.Int(required=False, default=10)
    cutout = argschema.fields.Raw(required=False, allow_none=True, missing=None)
    output_json = argschema.fields.OutputFile(required=False, allow_none=True)
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
    
class SkeletonizeProbabilitiesModule(argschema.ArgSchemaParser):
    default_schema = SkeletonizeProbabilitiesParameters

    def output(self, d):
        out_json = self.args.get("output_json")
        if out_json:
            pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(f, d)
                
    def run(self):
        
         # --- Convert all "None" strings to actual None ---
        for key, value in self.args.items():
            if value == "None":
                self.args[key] = None

        # --- Convert bound_box from string to list if present ---
        if self.args['cutout'] and type(self.args['cutout'])==str:
            self.args['cutout'] = [int(x.strip("'")) for x in self.args["cutout"].split(',')]
            

        kvstore_in = None
        if 's3' in self.args['input_zarr']:
            if self.args['profile']:
                AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                kvstore_in = create_kvstore(fpath=str(self.args['input_zarr']), store='s3', AWS_param=AWS_param)              
                                        
            if self.args['AWS_key']:
                AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                kvstore_in = create_kvstore(fpath=str(self.args['input_zarr']), store='s3', AWS_param=AWS_param)
                
        array = open_tensor(fpath=self.args["input_zarr"], driver='zarr3', kvstore=kvstore_in)
        
        os.makedirs(self.args["skeleton_output"], exist_ok=True)
        skels_outpath = os.path.join(self.args["skeleton_output"], 'skeletons.swcs.tar.gz')
        
        if self.args["cutout"]:
            skels_outpath = os.path.join(self.args["skeleton_output"], str(self.args["cutout"]) + '.swcs.tar.gz')
            
        start = time.time()
            
        skels = TS_skeletonize_volume(array, chunk_size=[100,100,100], n_jobs=self.args["n_jobs"], prob_thresh=self.args["probability_threshold"], label_size_threshold=self.args["label_size_threshold"], overlap=4, cutout=self.args["cutout"])
        print("Completed skeletonization")
        
        skels = [x for x in skels if x is not None]
        
        if len(skels) != 0:
            fused = Skeleton.simple_merge(skels).consolidate().components()
            write_kimi_skels_tar(skels_outpath, fused, mode='w:gz')

        end = time.time()
        print(end - start)
        

            

if __name__ == "__main__":
    mod = SkeletonizeProbabilitiesModule()
    mod.run()


__all__ = [
    "SkeletonizeProbabilitiesModule",
    "SkeletonizeProbabilitiesParameters"
]

    
    
    
    

