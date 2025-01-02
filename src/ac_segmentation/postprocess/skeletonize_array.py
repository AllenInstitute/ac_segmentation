import joblib
import numpy
import scipy.spatial
import skimage.morphology
import cc3d
import argschema

import cloudvolume
import kimimaro

from ac_segmentation.utils.tensorstore import open_tensor, create_tensor, create_kvstore
from ac_segmentation.utils.io import write_cv_skels_iter_tar, create_chunked_dims
import fastremap

np = numpy

def label_binary_array(binary_arr, size_threshold=20, chunk_offset=0):
    
    labeled_arr, num_features = cc3d.connected_components(binary_arr, connectivity=6, return_N=True)
    if num_features > 1:
        labeled_arr = skimage.morphology.remove_small_objects(
            labeled_arr, min_size=size_threshold, connectivity=3, out=labeled_arr)
    if chunk_offset != 0:
        imax = np.max(labeled_arr)+1
        mappings = dict(zip(list(range(1,imax)), list(range(chunk_offset,chunk_offset+imax))))
        labeled_arr = fastremap.remap(labeled_arr, mappings, preserve_missing_labels=True)
    return labeled_arr, len(np.unique(labeled_arr))

def threshold_binarize_array(arr, threshold=0.2):
    return (arr >= threshold)

    
def skeletonize_labeled_array(out_arr, probability_threshold=0.2, label_size_threshold=50, scale=2, constant=5, 
                fill_holes=False, parallel=1, dust_threshold=10, chunk_offset=0):
    # binarize volume, label, and skeletonize
    binary_arr = threshold_binarize_array(out_arr, threshold=probability_threshold)
    labeled_arr, num_feat = label_binary_array(binary_arr, size_threshold=label_size_threshold, chunk_offset=chunk_offset)
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
            "max_paths": 50, # default None
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
    return skels, labeled_arr


def join_components(skels, radius=2):
    # log which chunks the skeletons came from
    total_skels = []
    chunk_ind = []
    for ind, sk in enumerate(skels):
        temp_skels = sk.components()
        total_skels += temp_skels
        chunk_ind += len(temp_skels) * [ind]
        
    # extract root node coords
    root_vts = []  # node coordinates
    root_ind = []  # skeleton number from total_skels list
    root_node = []  # node associated with root_vt
    for ind, sk in enumerate(total_skels):  
        t_ids = sk.terminals()
        root_node += list(t_ids)
        ends = [sk.vertices[i] for i in t_ids]
        for end in ends:
            root_vts.append(list(end))
            root_ind.append(ind)
    
    # create kdtree and find all end nodes within given radius
    tree = scipy.spatial.KDTree(root_vts, leafsize=2)
    pairs = tree.query_pairs(radius)
    
    # get merge pair indices
    merge_pairs = []
    merge_pairs_vts = []
    for p1, p2 in pairs:
        merge_pairs.append([root_ind[p1], root_ind[p2]])
        merge_pairs_vts.append([root_vts[p1], root_vts[p2]])
    
    # check if skeletons in same chunk, if not merge
    fused_skels = []
    for ind, (m1, m2) in enumerate(merge_pairs):
        if chunk_ind[m1] == chunk_ind[m2]:
            continue
        else:
            try:
                fused = total_skels[m1].merge(total_skels[m2])
                v1, v2 = merge_pairs_vts[ind]
                n1, n2 = (fused.vertices.tolist().index(v1),
                          fused.vertices.tolist().index(v2))
                fused.edges = np.append(
                    fused.edges,
                    np.array([n1, n2]).reshape((1, 2)),
                    axis=0)
                total_skels[m1] = None
                total_skels[m2] = None
                fused_skels.append(fused)
            except Exception:
                pass
                
    out_skels = ([i for i in total_skels if i is not None] +
                 [i for i in fused_skels if i is not None])
    return out_skels


def skeletonize_labeled_array_concurrent(
        in_arr, chunk_size=[1000, 1000, 1000],
        n_jobs=4,  probability_threshold=0.05, label_size_threshold=80,  scale=2, constant=5, out_file=None):
    def skel_chunk(start, end, chunk_offset=0):
        skels, labeled_arr = skeletonize_labeled_array(
            out_arr=np.array(in_arr[
                start[0]:end[0],
                start[1]:end[1],
                start[2]:end[2]]), probability_threshold=probability_threshold, label_size_threshold=label_size_threshold,  
            scale=scale, constant=constant, chunk_offset=chunk_offset)
        if len(skels) != 0:
            for key in skels.keys():
                skels[key].vertices += start
            skels = list(skels.values())
            write = out_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]].write(labeled_arr)
            return [skels, write]
        else:
            return []

    #create chunk offsets
    xch, ych, zch = chunk_size
    comb1,comb2 = create_chunked_dims(in_arr.shape, chunk_size=chunk_size)
    chunk_offsets = []
    for i in range(len(comb1)):
        chunk_offsets.append(int((i*xch*ych*zch)/20))
         
    #create labelled array
    if out_file and 'zarr' in out_file.lower():
        out_arr = create_tensor(fpath=out_file, arr_shape=in_arr.shape, dtype = 'uint32', fill_value=-np.inf, driver='zarr3')
    else:
        out_arr = ts.array(np.zeros(in_arr.shape).astype('uint32'))
    
    #skeletonize
    skels = []
    if len(comb1) == 1:
        res = [skel_chunk(comb1[0], comb2[0])]
    else:
        with joblib.parallel_config(backend='threading'):
            res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(skel_chunk)(x, y, z) for x, y, z in zip(comb1,comb2, chunk_offsets))
    for chunk in res:
        skels += chunk[0]
        chunk[1].result()

    if out_file and 'npy' in out_file.lower():
        with gzip.open(out_file, 'wb') as f:
            np.save(f, out_arr)
    if not out_file:
        out_arr = out_arr.read().result()
        
    return skels, out_arr
    


def run(input_zarr, skeleton_output_path, probability_threshold=0.05,
        label_size_threshold=80, n_jobs=10, scale=2, constant=5):
                           
    # Load segmentation
    try:
        prob_map = open_tensor(input_zarr, driver='zarr')
    except:
        prob_map = open_tensor(input_zarr, driver='zarr3')

    # skels = skeletonize_labeled_array(labeled_arr, **skeletonize_options)
    skels = skeletonize_labeled_array_concurrent(prob_map,
        probability_threshold=probability_threshold, 
        label_size_threshold=label_size_threshold,
        n_jobs=n_jobs, scale=scale, constant=constant)

    # write skeletons to swc zip
    write_cv_skels_iter_tar(skeleton_output_path, skels)


class SkeletonizeZarrParameters(argschema.ArgSchema):
    input_zarr = argschema.fields.InputDir(required=True)
    skeleton_output = argschema.fields.OutputFile(required=True)
    probability_threshold = argschema.fields.Float(
        required=False, default=0.05)
    label_size_threshold = argschema.fields.Int(required=False, default=80)
    n_jobs = argschema.fields.Int(required=False, default=10)


class SkeletonizeZarrModule(argschema.ArgSchemaParser):
    default_schema = SkeletonizeZarrParameters 

    def run(self):
        run(self.args["input_zarr"],
            self.args["skeleton_output"],
            self.args["probability_threshold"],
            self.args["label_size_threshold"],
            self.args["n_jobs"])


if __name__ == "__main__":
    mod = SkeletonizeZarrModule()
    mod.run()

__all__ = [
    "SkeletonizeZarrModule",
    "SkeletonizeZarrParameters"]
    
