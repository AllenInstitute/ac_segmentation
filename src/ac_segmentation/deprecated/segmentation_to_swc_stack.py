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

class PostprocessParameters(ags.ArgSchema):
    input_dir = ags.fields.InputDir(required=True, description='Input directory')
    output_file = ags.fields.OutputFile(required=True, description='Output file')
    # output_dir = ags.fields.OutputDir(required=True, description='Output directory')
    csv_file = ags.fields.OutputFile(required=True, description='Segmentation file')
    crop_size =  ags.fields.List(ags.fields.Int, required=False, default=[0,0,0],
                                    description='stack size zyx')
    threshold = ags.fields.Int(dtype='int', required=False, default=20, 
                                desctiption='percent value')                         

def postprocess(indir, outfile, csvfile, crop_size, threshold, size_threshold=2000):
    threshold = threshold/100
    stack = load_stack(indir)
    stack_size = stack.shape
    
    if crop_size != [0,0,0]:
        # Crop stack to original size 
        stack = stack[0:crop_size[0],0:crop_size[1],0:crop_size[2]]
        stack_size = stack.shape
    
    # Binarize stack based on threshold
    stack = (stack >= int(np.round(255*threshold))).astype(np.uint8)

    # Label connected components
    s = ndi.generate_binary_structure(3,3)
    # stack = ndi.label(stack,structure=s)[0].astype(np.uint16)
    stack = ndi.label(stack,structure=s)[0].astype(np.uint32)
    num_cc = np.max(stack)

    if num_cc != 0:
        # Remove components smaller than size_threshold 
        stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)
        unique_labels, counts = np.unique(stack,return_counts=True)

        # Convert all connected component labels to 1
        stack = (stack > 0).astype(np.uint8)

        # Save nonzero pixels as csv file (x,y,z)
        z,y,x = np.nonzero(stack)
        np.savetxt(csvfile, np.stack((x,y,z), axis=1), fmt='%u', delimiter=',', 
                    header='x,y,z')
           
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
        
        # Create one swc file for all ccs in cc_dict
        cc2swc_single(cc_dict, outfile)

        # # Create swc file for each cc in cc_dict
        # cc2swc(cc_dict, outdir) 
            
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

def cc2swc_single(cc_dict, fname):
    # Create swc for each cc in cc_dict 
    trace_list = []
    for cc in cc_dict.keys():
        try:
            coord_values = cc_dict[cc]
            component_coordinates = np.array([coord_values['X'], coord_values['Y'],
                                              coord_values['Z']]).T

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
            node_type = 2 # axon
            node_list = []
            for k,v in parent_dict.items():
                # id,type,x,y,z,r,pid
                if v == -1:
                    parent = -1
                else:
                    parent = big_node_dict[v]
                node_line = [big_node_dict[k]] + [node_type] + list(k) + [1] + [parent]

                node_list.append(node_line)
                
            trace_list.append(np.array(node_list))
    
        except:
            print('error') 
            
    # Save all ccs as a single swc file
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
    save_swc(fname, trace_new)            
    
    
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

def save_swc(filepath, swc):
    with open(filepath, 'w') as f:
        f.write('#id,type,x,y,z,r,pid\n')
        for i in range(swc.shape[0]):
            f.write('%.0f %.0f %.3f %.3f %.3f %.3f %d\n' %tuple(swc[i, :].tolist()))
                
class Postprocess(ags.ArgSchemaParser):
        
    def run(self):
        postprocess(self.args['input_dir'], self.args['output_file'],
        self.args['csv_file'], self.args['crop_size'], self.args['threshold'])
   
        
if __name__ == "__main__":
    mod = Postprocess(schema_type=PostprocessParameters)
    mod.run()      