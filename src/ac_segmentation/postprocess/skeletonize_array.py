import itertools

import joblib
import numpy
import scipy.spatial
import skimage.morphology
import cc3d
import argschema

import cloudvolume
import kimimaro

from ac_segmentation.utils.tensorstore import open_tensor
from ac_segmentation.utils.io import (write_cv_skels_iter_tar)

np = numpy


def label_binary_array(binary_arr, size_threshold=200):
    
    labeled_arr, num_features = cc3d.connected_components(binary_arr, connectivity=6, return_N=True)
    if num_features > 1:
        labeled_arr = skimage.morphology.remove_small_objects(
            labeled_arr, min_size=size_threshold, connectivity=3, out=labeled_arr)
        
    return labeled_arr, len(np.unique(labeled_arr))


def threshold_binarize_array(arr, threshold=0.05):
    # dumb method
    return (arr >= threshold)


def skeletonize_labeled_array(
        labeled_arr, scale=2, constant=5,
        fill_holes=False, parallel=5,
        dust_threshold=10, **kwargs):

    skels = kimimaro.skeletonize(
        labeled_arr,
        teasar_params={
            "scale": scale, 
            "const": constant,  # influences the finger branches allowed
            "pdrf_scale": 10000,
            "pdrf_exponent": 1,
            "soma_acceptance_threshold": 3500,  # physical units
            "soma_detection_threshold": 750,  # physical units
            "soma_invalidation_const": 300,  # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 50,  # default None
        },
        dust_threshold=dust_threshold,  # skip connected components with fewer than this many voxels
        anisotropy=(1, 1, 1),  # default True #influences the dimension scale
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=fill_holes,  # default False
        fix_avocados=False,  # default False
        progress=False,  # default False, show progress bar
        parallel=parallel,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )
    return skels


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
        labeled_array, chunk_size=[1000, 1000, 1000],
        n_jobs=4, skel_search_radius=50, **kwargs):
    def skel_chunk(start, end):
        skels = skeletonize_labeled_array(
            np.array(labeled_array[
                start[0]:end[0],
                start[1]:end[1],
                start[2]:end[2]])
        )

        if len(skels) != 0:
            skels = [skel for skid, skel in skels.items()]
            skels = cloudvolume.Skeleton.simple_merge(skels).consolidate()
            skels.vertices += start
            return skels

    dx, dy, dz = labeled_array.shape
    xch, ych, zch = chunk_size
    sind_x, sind_y, sind_z = (
        list(range(0, dx, xch)),
        list(range(0, dy, ych)),
        list(range(0, dz, zch))
    )
    eind_x, eind_y, eind_z = (
        [x + xch for x in sind_x],
        [x + ych for x in sind_y],
        [x + zch for x in sind_z]
    )
    eind_x, eind_y, eind_z = (
        [dx if ele > dx else ele for ele in eind_x],
        [dy if ele > dy else ele for ele in eind_y],
        [dz if ele > dz else ele for ele in eind_z]
    )
    comb1 = list(itertools.product(sind_x, sind_y, sind_z))
    comb2 = list(itertools.product(eind_x, eind_y, eind_z))
    del sind_x, sind_y, sind_z, eind_x, eind_y, eind_z

    #skeletonize
    if len(comb1) == 1:
        res = skel_chunk(comb1[0], comb2[0])
    else:
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(skel_chunk)(x, y) for x, y in zip(comb1,comb2))

    #extract individual skeletons
    out_skels = []
    if res==None:
        pass
    elif isinstance(res,cloudvolume.skeleton.Skeleton):
        out_skels = res.components()
    else:
        res = [i for i in res if i is not None]
        for sk in res:
                out_skels += sk.components()

    return out_skels


def run(input_zarr, skeleton_output_path, probability_threshold=0.05,
        label_size_threshold=80, skeletonize_options=None):
    skeletonize_options = ({} if skeletonize_options is None
                           else skeletonize_options)
                           
    # Load segmentation
    prob_map = open_tensor(input_zarr).read().result()

    # binarize volume, label, and skeletonize
    binary_arr = threshold_binarize_array(
        prob_map, threshold=probability_threshold)
    labeled_arr, _ = label_binary_array(
        binary_arr, size_threshold=label_size_threshold)

    # skels = skeletonize_labeled_array(labeled_arr, **skeletonize_options)
    skels = skeletonize_labeled_array_concurrent(
        labeled_arr, **skeletonize_options
    )

    # write skeletons to swc zip
    # write_kimi_skels_tar(skeleton_output_path, skels)
    write_cv_skels_iter_tar(skeleton_output_path, skels)


class SkeletonizationOptions(argschema.schemas.DefaultSchema):
    n_jobs = argschema.fields.Int(required=False, allow_none=True)


class SkeletonizeZarrParameters(argschema.ArgSchema):
    input_zarr = argschema.fields.InputDir(required=True)
    skeleton_output = argschema.fields.OutputFile(required=True)
    probability_threshold = argschema.fields.Float(
        required=False, default=0.05)
    label_size_threshold = argschema.fields.Int(required=False, default=80)
    skeletonize_options = argschema.fields.Nested(
        SkeletonizationOptions, required=False, default=None, allow_none=True)


class SkeletonizeZarrModule(argschema.ArgSchemaParser):
    default_schema = SkeletonizeZarrParameters

    @property
    def skeletonize_options(self):
        return self.args["skeletonize_options"]    

    def run(self):
        run(self.args["input_zarr"],
            self.args["skeleton_output"],
            self.args["probability_threshold"],
            self.args["label_size_threshold"],
            self.skeletonize_options)


if __name__ == "__main__":
    mod = SkeletonizeZarrModule()
    mod.run()

__all__ = [
    "SkeletonizeZarrModule",
    "SkeletonizeZarrParameters"]
