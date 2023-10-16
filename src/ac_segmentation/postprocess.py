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
import tifffile as tif
import kimimaro
import zarr
import navis
from pathlib import Path
import colorsys
from ac_segmentation.reconnect_stack_navis import reconnect, swc_multi_to_single_subdir

class PostprocessParameters(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(required=True, description='Output directory')
    chunk_size = ags.fields.Int(dtype='int', required=False, default=416)
    overlap = ags.fields.Int(dtype='int', required=False, default=16)

def postprocess(outdir, chunk_size, overlap, threshold=0.2, size_threshold=2000):
    num_chunks = len([f for f in os.listdir(os.path.join(outdir, 'Segmentation')) 
                            if f.startswith('chunk')])
    
    savedir = os.path.join(outdir, 'swc_files')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    df = pd.read_csv(os.path.join(outdir, 'inputs','bbox_deskewed.csv'))
    bb = df.bounding_box.values
    
    for n in range(num_chunks):
        stack = load_stack(os.path.join(outdir, 'Segmentation', 'chunk%02d'%n))
        
        # Crop stack to original size 
        new_stack_size = bb[5], bb[4], bb[3]  #zyx
        stack = stack[0:new_stack_size[0],0:new_stack_size[1],0:new_stack_size[2]]
        stack_size = stack.shape

        # Zero values below threshold
        stack[stack <= int(np.round(255*threshold))] = 0
        
        # Save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(outdir, 'Segmentation', 'chunk%02d.csv'%n), 
                   np.stack((x,y,z,I), axis=1), fmt='%u', delimiter=',', header='x,y,z,I')
        
        # Binarize stack based on threshold
        stack = (stack > int(np.round(255*threshold))).astype(np.uint8)

        # Label connected components
        s = ndi.generate_binary_structure(3,3)
        stack = ndi.label(stack,structure=s)[0].astype(np.uint16)
        num_cc = np.max(stack)

        if num_cc != 0:
            # Remove components smaller than size_threshold 
            stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)
            unique_labels, counts = np.unique(stack,return_counts=True)

            # Convert all connected component labels to 1
            stack = (stack > 0).astype(np.uint8)

            # Skeletonize stack
            stack = skeletonize_3d(stack)   
            
            # Label connected components
            s = ndi.generate_binary_structure(3,3)
            stack = ndi.label(stack,structure=s)[0].astype(np.uint16)
            num_cc = np.max(stack)
            
            # Create dict with xyz coordinates for each cc
            cc_dict = {}
            cc_range = range(1,num_cc+1)
            for cc in cc_range:
                cc_dict[cc] = {'X':[],'Y':[],'Z':[]}

            for j in range(stack_size[0]):
                img = stack[j,:,:]
                unique_labels = np.unique(img) #return indices and ignore 0 
                for l in unique_labels:
                    if l != 0:
                        idx = np.where(img==l)
                        [cc_dict[l]['Y'].append(coord) for coord in idx[0]]
                        [cc_dict[l]['X'].append(coord) for coord in idx[1]]
                        [cc_dict[l]['Z'].append(j) for coord in idx[0]]   

            chunkdir = os.path.join(savedir, 'chunk%02d'%n)
            if not os.path.isdir(chunkdir):
                os.mkdir(chunkdir)

            # Create swc file for each cc in cc_dict
            cc2swc(cc_dict, chunkdir) 

            # Add radius to swc file
            file_list = [f for f in os.listdir(chunkdir) if f.endswith('swc')] 
            file_list.sort()
            for f in file_list:
                try:
                    morph_in = morphology_from_swc(os.path.join(chunkdir, f))
                    dict_out = add_radius_to_morph(morph_in, np.stack((x,y,z,I), axis=1))
                    morphology_to_swc(dict_out['morph'], os.path.join(chunkdir, f))
                except:
                    print('error')


def postprocess_kimi_stack(outdir, bound_box, overlap, threshold=0.2, size_threshold=2000, check_rad=False, **kwargs):
    chunks = [name for name in os.listdir(os.path.join(outdir, 'Segmentation')) if name.startswith('chunk') == True]
    
    #set default keyword arguments
    defaultKwargs = {'scale': 2, 'constant': 5, 'fill_holes' : False, 'parallel': 1, 'dust_threshold' : 10}
    kwargs = { **defaultKwargs, **kwargs }
    
    savedir = os.path.join(outdir, 'swc_files_KIMI')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    for folder in chunks:
        stack = load_stack(os.path.join(outdir, 'Segmentation', folder))
        
        # Crop stack to original size 
        new_stack_size = bound_box[2], bound_box[1], bound_box[0]  #zyx
        stack = stack[0:new_stack_size[0],0:new_stack_size[1],0:new_stack_size[2]]
        stack_size = stack.shape

        # Zero values below threshold
        stack[stack <= int(np.round(255*threshold))] = 0
        
        # Save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(outdir, 'Segmentation', 'nonzero_pix.csv'), 
                   np.stack((x,y,z,I), axis=1), fmt='%u', delimiter=',', header='x,y,z,I')
        
        # Binarize stack based on threshold
        stack = (stack > int(np.round(255*threshold))).astype(np.uint8)

        # Label connected components
        s = ndi.generate_binary_structure(3,3)
        stack = ndi.label(stack,structure=s)[0].astype(np.uint16)
        num_cc = np.max(stack)

        if num_cc != 0:
            # Remove components smaller than size_threshold 
            stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)
            unique_labels, counts = np.unique(stack,return_counts=True)
            
            skels = kimimaro.skeletonize(
              stack, 
              teasar_params={
                "scale": kwargs['scale'], 
                "const": kwargs['constant'], # influences the finger branches allowed
                "pdrf_scale": 10000,
                "pdrf_exponent": 1,
                "soma_acceptance_threshold": 3500, # physical units
                "soma_detection_threshold": 750, # physical units
                "soma_invalidation_const": 300, # physical units
                "soma_invalidation_scale": 2,
                "max_paths": 50, # default None
              },
              dust_threshold=kwargs['dust_threshold'], # skip connected components with fewer than this many voxels
              anisotropy=(1,1,1), # default True #influences the dimension scale
              fix_branching=True, # default True
              fix_borders=True, # default True
              fill_holes=kwargs['fill_holes'], # default False
              fix_avocados=False, # default False
              progress=True, # default False, show progress bar
              parallel=kwargs['parallel'], # <= 0 all cpu, 1 single process, 2+ multiprocess
              parallel_chunk_size=1, # how many skeletons to process before updating progress bar
            )
            
        for key in skels.keys():
            
            skels[key].vertices = skels[key].vertices[:, [2, 1,0]]
            with open(outdir + '/swc_files_KIMI/' + str(skels[key].id).zfill(4) + '.swc', 'wt') as f:
                f.write(skels[key].to_swc())  
                
def postprocess_kimi_array(outdir, stack, bound_box, overlap, threshold=0.2, size_threshold=2000, check_rad=False, **kwargs):
    
    #set default keyword arguments
    defaultKwargs = {'scale': 2, 'constant': 5, 'fill_holes' : False, 'parallel': 1, 'dust_threshold' : 10}
    kwargs = { **defaultKwargs, **kwargs }
    
    savedir = os.path.join(outdir, 'swc_files_KIMI')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        
    # Crop stack to original size 
    new_stack_size = bound_box[2], bound_box[1], bound_box[0]  #zyx
    stack = stack[0:new_stack_size[0],0:new_stack_size[1],0:new_stack_size[2]]
    stack_size = stack.shape

    # Zero values below threshold
    stack[stack <= int(np.round(255*threshold))] = 0
        
    # Binarize stack based on threshold
    stack = (stack > int(np.round(255*threshold))).astype(np.uint8)

    # Label connected components
    s = ndi.generate_binary_structure(3,3)
    stack = ndi.label(stack,structure=s)[0].astype(np.uint16)
    num_cc = np.max(stack)

    if num_cc != 0:
        # Remove components smaller than size_threshold 
        stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)
        unique_labels, counts = np.unique(stack,return_counts=True)
            
        skels = kimimaro.skeletonize(
            stack, 
            teasar_params={
            "scale": kwargs['scale'], 
            "const": kwargs['constant'], # influences the finger branches allowed
            "pdrf_scale": 10000,
            "pdrf_exponent": 1,
            "soma_acceptance_threshold": 3500, # physical units
            "soma_detection_threshold": 750, # physical units
            "soma_invalidation_const": 300, # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 50, # default None
            },
            dust_threshold=kwargs['dust_threshold'], # skip connected components with fewer than this many voxels
            anisotropy=(1,1,1), # default True #influences the dimension scale
            fix_branching=True, # default True
            fix_borders=True, # default True
            fill_holes=kwargs['fill_holes'], # default False
            fix_avocados=False, # default False
            progress=True, # default False, show progress bar
            parallel=kwargs['parallel'], # <= 0 all cpu, 1 single process, 2+ multiprocess
            parallel_chunk_size=1, # how many skeletons to process before updating progress bar
        )
            
    for key in skels.keys(): 
        skels[key].vertices = skels[key].vertices[:, [2, 1,0]]
        with open(outdir + '/swc_files_KIMI/' + str(skels[key].id).zfill(4) + '.swc', 'wt') as f:
            f.write(skels[key].to_swc())  
            
def postprocess_kimi_zarr_strips(in_dir, outdir, sc, cl, strip_range, bound_box, chunk_size = 1024, 
                            iter_thresh = [0.1,0.1,0.1,0.1,0.1], match_query_dis = 20, min_collin=0.1, size_thresh = 500, thresh = 0.05):
    
    for strip in range(strip_range[0], strip_range[1]+1):
        pos_dir = in_dir + 'Pos' + str(strip) + "/"
        seg_data = zarr.open(pos_dir + 'Pos' + str(strip) + '_Segmented.zarr')
        
        #chunk image and skeletonize
        z_start = list(range(0,seg_data.shape[2],chunk_size))
        
        for start in z_start:
            os.makedirs(pos_dir + 'chunk' + str(start), exist_ok=True)
            #index the zarr
            test_arr = seg_data[:,:,start:start+chunk_size]
            try:
                postprocess_kimi_array(outdir = pos_dir + 'chunk' + str(start), stack = test_arr, bound_box = [bound_box[2], bound_box[1], bound_box[0]], chunk_size = [512, 512, 64], overlap = [512, 512, 64], threshold=thresh, size_threshold=size_thresh, check_rad=True)
            except:
                print('chunk' + str(start) + " Has no skeletons")
                
            #adjust z coordinates to reflect original places in volume, transpose to original xyz space
            skel_dir = pos_dir + 'chunk' + str(start) + "/swc_files_KIMI/"
            skels = os.listdir(skel_dir)
            
            for sk_name in skels:
                s = navis.read_swc(skel_dir + sk_name)
                s.nodes['x'] = s.nodes['x'] + float(start)
                s.nodes = s.nodes.rename(columns={"x": "z", "z": "x"})
                navis.write_swc(s, skel_dir + sk_name)
                os.rename(skel_dir + sk_name, skel_dir + 'chunk' + str(start) + '_' + sk_name)

        #Convert all SWCs to a single SWC
        all_skel = navis.read_swc(pos_dir, include_subdirs=True)
        swc_multi_to_single_subdir(pos_dir, pos_dir + 'consolidated.swc' )
        
        #Break and reconnect skeletons
        os.makedirs(pos_dir + "Reconnected/", exist_ok=True)
        skels_rec = reconnect(infile = pos_dir + 'consolidated.swc', \
                                    swc_outdir = pos_dir + "Reconnected/", cl = cl, sc = sc, xyz_pxl=[1.0,1.0,1.0], \
                                    min_nodes = 10, iter_thresh = iter_thresh, query_dis = match_query_dis, min_collin=min_collin)
        
        #Convert all SWCs to a single SWC
        os.makedirs(out_dir + "Skeletons/", exist_ok=True)
        swc_multi_to_single_subdir(pos_dir + "Reconnected/reconnected_skeletons/",\
                                   outdir + "Skeletons/Pos" + str(strip) + "_Skels.swc" ) 
        
        print("Position " + str(strip) + " Complete!")
        
        
def get_skeleton_orient(swc_path, min_node):
    skels = navis.read_swc(swc_path)
    skels = extract_neuronlist(skels, min_node) 
    for skel in skels:
        skel.name = None
    
    #extract orientation from dotprop and add as column attribute to neuronlist
    new_nl = []
    for ex_sk in skels:
        ex_sm_sk = ex_sk.copy()
        ex_sm_dp = navis.make_dotprops(ex_sk)
        ex_sm_sk.nodes["hsv"] = [colorsys.rgb_to_hsv(*(vec))[0] for vec in ex_sm_dp.vect+1]
        new_nl.append(ex_sm_sk)
    new_nl = navis.NeuronList(new_nl)
    return new_nl
            
def load_stack(dirname):
    # Load image stack filenames
    filelist = [f for f in os.listdir(dirname) if f.endswith('.tif')] 
    filelist.sort()
    
    # Calculate stack size
    filename = os.path.join(dirname, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = len(filelist), img.shape[0], img.shape[1]
        
    stack = np.zeros(cell_stack_size, dtype=img.dtype)
    for i, f in enumerate(filelist):
        filename = os.path.join(dirname, f)
        img = tif.imread(filename)
        stack[i,:,:] = img
        
    return stack         

def cc2swc(cc_dict, dirname):
    # Create swc for each cc in cc_dict 
    cc_range = range(1,len(cc_dict)+1)
    for cc in cc_range:
        try:
            coord_values = cc_dict[cc]
            component_coordinates = np.array([coord_values['X'],coord_values['Y'],coord_values['Z']]).T

            # Make a node dictionary for this con comp so we can lookup in the 26 node check step
            node_dict = {}
            count=0
            for c in component_coordinates:
                count+=1
                node_dict[tuple(c)] = count

            # 26 nodes to check in defining neighbors dict
            movement_vectors = [p for p in itertools.product([0,1,-1], repeat=3)]
            movement_vectors.remove((0,0,0))

            # Create neighbors dict and find root nodes (nodes with only one neighbor)
            root_node_dict = {}
            neighbors_dict = {}
            count = 0
            for node in component_coordinates:
                count+=1
                node_neighbors = [] 
                num_neighbors = 0
                for vect in movement_vectors:
                    node_to_check = tuple(list(map(add,tuple(node),vect)))
                    if node_to_check in node_dict.keys():
                        node_neighbors.append(node_to_check)
                if len(node_neighbors) == 1:
                    root_node_dict[tuple(node)] = count
                neighbors_dict[tuple(node)] = node_neighbors   

            # Set start node    
            start_node = min(root_node_dict, key=root_node_dict.get)
            start_nodes_parent = -1
            # Assign parent-child relation
            parent_dict = {}
            parent_dict[start_node] = start_nodes_parent

            queue = deque([start_node])
            while len(queue) > 0:
                current_node = queue.popleft()
                my_connections = neighbors_dict[current_node]
                for node in my_connections:
                    if node not in parent_dict:
                        parent_dict[node] = current_node
                        queue.append(node)
                    else:
                        p = 'Initial start node' if parent_dict[node] == start_nodes_parent else str([parent_dict[node]])

            # Number each node for swc
            ct=0
            big_node_dict = {}
            for j in parent_dict.keys():
                ct+=1
                big_node_dict[tuple(j)] = ct

            # Make swc list for swc file writing        
            node_type = 2 #axon
            swc_list = []
            for k,v in parent_dict.items():
                # id,type,x,y,z,r,pid
                if v == -1:
                    parent = -1
                else:
                    parent = big_node_dict[v]
                swc_line = [big_node_dict[k]] + [node_type] + list(k) + [1] + [parent]

                swc_list.append(swc_line)

            # Write swc file
            swc_file = os.path.join(dirname, '%05d.swc'%cc)
            with open(swc_file, 'w') as f:
                f.write('#id,type,x,y,z,r,pid\n')
                for i in range(len(swc_list)):
                    f.write('%d %d %d %d %d %.1f %d\n'%tuple(swc_list[i]))
        except:
            print('error')    

def add_radius_to_morph(morph,non_specific_segmentation,wiggle_room = 80,**kwargs):
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
            node['radius'] = distance*0.406

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

def add_missing_radius_vals(radius_morph):
    """
    Because some nodes were added during postprocessing (i.e connectino algorithm)
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
    
class Postprocess(ags.ArgSchemaParser):
        
    def run(self):
        postprocess(self.args['output_dir'], chunk_size=self.args['chunk_size'], 
        overlap = self.args['overlap'])   
        
if __name__ == "__main__":
    mod = Postprocess(schema_type=PostprocessParameters)
    mod.run()       