import os
import numpy as np
import pandas as pd
import argschema as ags
import json
from operator import add
import itertools
from collections import deque, defaultdict
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, skeletonize_3d
from sklearn.neighbors import KDTree
import tifffile as tif
from ac_segmentation.deprecated.swc_morphology.classes import Morphology
from ac_segmentation.deprecated.swc_morphology.readwrite_swc import morphology_from_swc, morphology_to_swc

class AddRadiusParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(required=True, description='Input file')
    output_file = ags.fields.OutputFile(required=True, description='Output file')
    csv_file = ags.fields.InputFile(required=True, description='Segmentation file')
    pxl_xy = ags.fields.Int(dtype='int', required=False, default=812, 
                                desctiption='pxl xy size in nm') 

def process(infile, outfile, csvfile, pxl_xy):
    df = pd.read_csv(csvfile)
    arr = df.values
    morph_in = morphology_from_swc(infile)
    dict_out = add_radius_to_morph_10x(morph_in, arr, pxl_xy)
    morphology_to_swc(dict_out['morph'], outfile)              

def add_radius_to_morph_10x(morph, non_specific_segmentation, pxl_xy, 
                            wiggle_room = 50, **kwargs):
    """
    This function will iterate through each node in swc file and get 10 nodes up and 10 nodes down.
    With this segment a bounding box is created. Wiggle room is added to x-y dimensions because the 
    20 node segment is only a skeleton structure wheras the segmentation.csv will be much wider. 
    Wiggle_room is the padding for this bounding box, 80 pixels left and 80 pixels right of the 
    max and min x/y values. Only one z-slice is taken as we are only considering x-y for radius calculations

    """
    # Modified version to calculate radius only for every 10th node
    mod_morph = morph.clone()
    numnodes = len(morph.nodes())
    print("     {} Nodes to add. Estimated time to complete = {} minutes".format(numnodes,numnodes/1000))
    empty_ct = 0
    node_ct = 0
    for node in [n for n in mod_morph.nodes() if n['type'] != 1]:
        if node_ct % 10 == 0:
            node_coord = (node['x'],node['y'])
            coords_up_and_down = n_nodes_up_and_down(node,10,mod_morph)
            xyz_coords = np.asarray(list(coords_up_and_down))

            min_bb = [int(min(xyz_coords[:,j])) for j in range(0,3)]
            max_bb = [int(max(xyz_coords[:,j])) for j in range(0,3)] 
            inside_bbox = bounding_box(non_specific_segmentation, 
                          min_x = min_bb[0] - wiggle_room , max_x = max_bb[0] + wiggle_room,
                          min_y = min_bb[1] - wiggle_room , max_y = max_bb[1] + wiggle_room,
                          min_z = min_bb[2] - 1 , max_z = max_bb[2] + 1)
            segmented_local_xyz_array = non_specific_segmentation[inside_bbox]
            segmented_local_xy_array = segmented_local_xyz_array[:,0:2]

            if segmented_local_xy_array.size != 0:
                local_segmentation_lookup_tree_raw = KDTree(segmented_local_xy_array,leaf_size=40)
                dist_stepper=0
                condition=0
                explored = []
                while condition != 1:
                    dist_stepper+=1
                    movement_vectors = [p for p in itertools.product([n for n in range(1,dist_stepper+1)]+[-n for n in range(0,dist_stepper+1)], repeat=2) if p != (0,0)]
                    x = {v:((v[0]**2)+(v[1]**2))**0.5 for v in movement_vectors}
                    ordered_dict = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
                    offsets=[]
                    for offset in ordered_dict.keys():
                        node_to_check = np.array([sum(x) for x in zip(offset,node_coord)])
                        explored.append(node_to_check)
                        dist, _ = local_segmentation_lookup_tree_raw.query(node_to_check.reshape(1,2), k=1)
                        dist = dist[0][0]
                        offsets.append(offset)
                        if dist !=0:
                            distance = ((offset[0]**2)+(offset[1]**2))**0.5
                            condition = 1
                            break
                node['radius'] = distance*pxl_xy/1000 #nm->um

            else:
                empty_ct+=1
                node['radius'] = 0.1
            r = node['radius']    
        else:
            node['radius'] = r
        node_ct += 1
        
    add_missing_radius_vals(mod_morph)
    
    result_dict = {}
    result_dict['morph'] = mod_morph
    return result_dict

def n_nodes_up_and_down(st_node,n,morph):
    return n_nodes_up(st_node,n,morph).union(n_nodes_down(st_node,n,morph))

def n_nodes_up(st_node,n,morph):
    
    cur_node = st_node
    ct=0
    nodes_up = set()
    nodes_up.add((int(st_node['x']),int(st_node['y']),int(st_node['z'])))
    while ct != n:
        
        parent_id = cur_node['parent']
        if parent_id == -1:
            return nodes_up
        else:
            next_node = morph.node_by_id(parent_id)
            nodes_up.add((int(next_node['x']),int(next_node['y']),int(next_node['z'])))
            cur_node = next_node
        ct+=1
    return nodes_up

def n_nodes_down(node,n,morph):
    ct = 0
    nodes_down = set()
    while n != ct:
        
        next_node = morph.get_children(node)
        if next_node !=[]:
            for node in next_node:
                nodes_down.add((int(node['x']),int(node['y']),int(node['z'])))
        else:
            return nodes_down
        ct+=1

    return nodes_down

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter

def add_missing_radius_vals(radius_morph):
    """
    Because some nodes were added during postprocessing (i.e connection algorithm)
    They may not be in the segmentation.csv This script finds the nearest node up
    or down stream that has a radius value calculated. 
    Defaults to tree averages if the above fails
    """
    for missing_rad_node in [n for n in radius_morph.nodes() if n['radius'] == 0.1]:
        try:
            curr_node = missing_rad_node
            up_down_dict = defaultdict(dict)
            upstep =0
            while curr_node['radius'] == 0.1:
                upstep+=1
                curr_node = radius_morph.node_by_id(curr_node['parent'])
            up_down_dict['up']['steps'] = upstep
            up_down_dict['up']['radius'] = curr_node['radius']


            queue = deque([missing_rad_node['id']])
            curr_node_down_id = missing_rad_node['id']
            downstep=0
            while radius_morph.node_by_id(curr_node_down_id)['radius'] == 0.1:
                downstep+=1
                curr_node_down_id = queue.popleft()
                for ch_no in radius_morph.get_children(radius_morph.node_by_id(curr_node_down_id)):
                    queue.append(ch_no['id'])

            up_down_dict['down']['steps'] = downstep
            up_down_dict['down']['radius'] = radius_morph.node_by_id(curr_node_down_id)['radius']
            closest_direction = [k for k,v in up_down_dict.items() if v['steps'] == min([s['steps'] for s in up_down_dict.values() ])][0]

            missing_rad_node['radius'] = up_down_dict[closest_direction]['radius']
        except:
            missing_rad_node['radius'] = np.mean([n['radius'] for n in radius_morph.nodes() if n['type']!=1])
    
def add_radius_to_morph(morph, non_specific_segmentation, pxl_xy,
                        wiggle_room = 50,**kwargs):
    """
    This function will iterate through each node in swc file and get 10 nodes up and 10 nodes down.
    With this segment a bounding box is created. Wiggle room is added to x-y dimensions because the 
    20 node segment is only a skeleton structure wheras the segmentation.csv will be much wider. 
    Wiggle_room is the padding for this bounding box, 80 pixels left and 80 pixels right of the 
    max and min x/y values. Only one z-slice is taken as we are only considering x-y for radius calculations

    """
    mod_morph = morph.clone()
    numnodes = len(morph.nodes())
    print("     {} Nodes to add. Estimated time to complete = {} minutes".format(numnodes,numnodes/1000))
    empty_ct = 0
    for node in [n for n in mod_morph.nodes() if n['type'] != 1]:
        node_coord = (node['x'],node['y'])
        coords_up_and_down = n_nodes_up_and_down(node,10,mod_morph)
        xyz_coords = np.asarray(list(coords_up_and_down))
        
        min_bb = [int(min(xyz_coords[:,j])) for j in range(0,3)]
        max_bb = [int(max(xyz_coords[:,j])) for j in range(0,3)] 
        inside_bbox = bounding_box(non_specific_segmentation, 
                      min_x = min_bb[0] - wiggle_room , max_x = max_bb[0] + wiggle_room,
                      min_y = min_bb[1] - wiggle_room , max_y = max_bb[1] + wiggle_room,
                      min_z = min_bb[2] - 1 , max_z = max_bb[2] + 1)
        segmented_local_xyz_array = non_specific_segmentation[inside_bbox]
        segmented_local_xy_array = segmented_local_xyz_array[:,0:2]

        if segmented_local_xy_array.size != 0:
            local_segmentation_lookup_tree_raw = KDTree(segmented_local_xy_array,leaf_size=40)
            dist_stepper=0
            condition=0
            explored = []
            while condition != 1:
                dist_stepper+=1
                movement_vectors = [p for p in itertools.product([n for n in range(1,dist_stepper+1)]+[-n for n in range(0,dist_stepper+1)], repeat=2) if p != (0,0)]
                x = {v:((v[0]**2)+(v[1]**2))**0.5 for v in movement_vectors}
                ordered_dict = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
                offsets=[]
                for offset in ordered_dict.keys():
                    node_to_check = np.array([sum(x) for x in zip(offset,node_coord)])
                    explored.append(node_to_check)
                    dist, _ = local_segmentation_lookup_tree_raw.query(node_to_check.reshape(1,2), k=1)
                    dist = dist[0][0]
                    offsets.append(offset)
                    if dist !=0:
                        distance = ((offset[0]**2)+(offset[1]**2))**0.5
                        condition = 1
                        break
            node['radius'] = distance*pxl_xy/1000 #nm->um

        else:
            empty_ct+=1
            node['radius'] = 0.1
    add_missing_radius_vals(mod_morph)
    
    result_dict = {}
    result_dict['morph'] = mod_morph
    return result_dict
    
def n_nodes_up_and_down(st_node,n,morph):
    return n_nodes_up(st_node,n,morph).union(n_nodes_down(st_node,n,morph))

def n_nodes_up(st_node,n,morph):
    
    cur_node = st_node
    ct=0
    nodes_up = set()
    nodes_up.add((int(st_node['x']),int(st_node['y']),int(st_node['z'])))
    while ct != n:
        
        parent_id = cur_node['parent']
        if parent_id == -1:
            return nodes_up
        else:
            next_node = morph.node_by_id(parent_id)
            nodes_up.add((int(next_node['x']),int(next_node['y']),int(next_node['z'])))
            cur_node = next_node
        ct+=1
    return nodes_up

def n_nodes_down(node,n,morph):
    ct = 0
    nodes_down = set()
    while n != ct:
        
        next_node = morph.get_children(node)
        if next_node !=[]:
            for node in next_node:
                nodes_down.add((int(node['x']),int(node['y']),int(node['z'])))
        else:
            return nodes_down
        ct+=1

    return nodes_down

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter
    
class AddRadius(ags.ArgSchemaParser):
        
    def run(self):
        process(self.args['input_file'], self.args['output_file'], self.args['csv_file'],
                self.args['pxl_xy'])
   
        
if __name__ == "__main__":
    mod = AddRadius(schema_type=AddRadiusParameters)
    mod.run()      