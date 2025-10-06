import os
import numpy as np
import pandas as pd
import argschema as ags
import json
import pathlib
import posixpath
from scipy.ndimage import affine_transform
import tifffile as tif

class PreprocessParameters(ags.ArgSchema):
    input_dir = ags.fields.InputDir(required=True, description='Input directory')
    output_dir = ags.fields.OutputDir(required=True, description='Output directory')
    chunk_size = ags.fields.Int(dtype='int', required=False, default=416)
    overlap = ags.fields.Int(dtype='int', required=False, default=16)

def chunk_volume(indir, outdir, chunk_size, overlap, mip=0):
    dir_list = ['chunks', 'deskewed', 'inputs']
    for d in dir_list:
        savedir = os.path.join(outdir, d)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)    
    fullres_ds = miplevel_from_group_dir(indir, mip=mip)
    
    # Get volume size from metadata
    mfile = os.path.join(indir, 's%d'%mip, 'attributes.json')
    with open(mfile,'r') as f:
        data = json.load(f)
    volume_size = data['dimensions']
    volume_size.reverse() # xyz -> zyx
    idx1 = np.arange(0, volume_size[0], chunk_size - overlap)
    idx2 = idx1 + chunk_size
    num_chunks = np.min(np.where(volume_size[0] <= idx2)[0]) + 1
    idx1 = idx1[:num_chunks]
    idx2 = idx2[:num_chunks]
    
    for i in range(num_chunks):
        stack = fullres_ds[idx1[i]:idx2[i], ...]
        if idx2[i] > volume_size[0]:
            stack = np.pad(stack, ((0, idx2[i] - volume_size[0]),(0, 0), (0, 0)), 'edge')
        mip_xy = np.max(stack,0) # save mip
        tif.imwrite(os.path.join(outdir, dir_list[0], 'MAX_xy_chunk%02d.tif'%i), mip_xy)     
#         chunk_dir = os.path.join(outdir, dir_list[0], 'chunk%02d'%i)
#         if not os.path.isdir(chunk_dir):
#             os.mkdir(chunk_dir)
#         for j in range(stack.shape[0]):
#             tif.imwrite(os.path.join(chunk_dir, '%03d.tif'%j), stack[j,:,:])
        
        # Deskew stack
        chunk_dir = os.path.join(outdir, dir_list[1], 'chunk%02d'%i)
        if not os.path.isdir(chunk_dir):
            os.mkdir(chunk_dir)        
        stack = deskew_stack(stack, chunk_dir)
        mip_xy = np.max(stack,0) # save mip
        tif.imwrite(os.path.join(outdir, dir_list[1], 'MAX_xy_chunk%02d.tif'%i), mip_xy) 
        
        # Create input 
        file_name = os.path.join(outdir, dir_list[2], 'chunk%02d.tif'%i)
        stack = create_input(stack, file_name, Imax=32640)
        mip_xy = np.max(stack,0) # save mip
        tif.imwrite(os.path.join(outdir, dir_list[2], 'MAX_xy_chunk%02d.tif'%i), mip_xy)

        # Create bounding box for segmentation
        if i == 0:
            stack_size = stack.shape
            bounding_box = [0,0,0, stack_size[2], stack_size[1], stack_size[0]]
            file_name = os.path.join(outdir, dir_list[2], 'bbox.csv')
            pd.DataFrame({'bounding_box':bounding_box}).to_csv(file_name)

def deskew_stack(stack_in, chunkdir=None, pxl_xy=0.406, angle=32.1, step_stage=2, shift=-1):
    # shift=1 (right to left), shift=-1 (left to right)
    h_xz = int(np.round(shift*np.cos(angle*np.pi/180.0)*step_stage/pxl_xy))
    stack_in_size = stack_in.shape

    # Shear transform
    shear_affine = np.eye(4)
    shear_affine[2,0] = h_xz
    if h_xz < 0:
        dx = int(np.ceil(-h_xz*(stack_in_size[0]-1))) # add translation for negative h_xz
        shear_affine[2,3] = dx
   
    # Calculate output stack shape
    nz, ny, nx = stack_in_size
    bb_in = [(0, 0, 0, 1), (nz-1, 0, 0, 1), (0, ny-1, 0, 1), (0, 0, nx-1, 1),
            (nz-1, ny-1, 0, 1), (nz-1, 0, nx-1, 1), (0, ny-1, nx-1, 1), 
             (nz-1, ny-1, nx-1, 1)]
    bb_out = list(map(lambda c: shear_affine @ np.array(c), bb_in))
    bb_out = np.concatenate(bb_out).reshape((-1, 4))
    output_shape = np.max(bb_out, axis=0) - np.min(bb_out, axis=0) + 1
    base = 4
    output_shape = (base * np.ceil(output_shape/base)).astype(np.int64)[:3]
    
    # Apply shear transform (order=1-linear, 3-cubic)
    stack_out = affine_transform(stack_in, np.linalg.inv(shear_affine), 
                                 output_shape=output_shape,order=1) 
    stack_out_size = stack_out.shape
    
    if stack_out_size[0] > stack_in_size[0]:
        # remove last (extra) z plane
        stack_out = stack_out[:-1,:,:]
        stack_out_size = stack_out.shape
    
    # Save new stack as multiple tif files
    if chunkdir:
        for j in range(stack_out_size[0]):
            tif.imwrite(os.path.join(chunkdir, '%03d.tif'%j), stack_out[j,:,:])        
    return(stack_out)

def create_input(stack_in, fname, Imax=65535):
   # Convert to 8bit, pad to make xyz size a multiple of 64,64,16 for segmentation
    cx, cy, cz = 64, 64, 16
    stack_in_size = stack_in.shape

    # Convert to 8bit
    coef = int(Imax/255) # linear conversion
    stack_in[stack_in>Imax]=Imax
    stack_in = np.uint8(np.round(stack_in/coef))

    #Pad to make xyz size a multiple of 64,64,16
    stack_out_size = (int(cz*np.ceil(stack_in_size[0]/cz)), 
                      int(cy*np.ceil(stack_in_size[1]/cy)), 
                      int(cx*np.ceil(stack_in_size[2]/cx)))

    stack_out = np.pad(stack_in, ((0, stack_out_size[0] - stack_in_size[0]),
                               (0, stack_out_size[1] - stack_in_size[1]), 
                               (0, stack_out_size[2] - stack_in_size[2])), 'edge')

    # Save stack as single tif file
    tif.imwrite(fname, stack_out) 
    return stack_out   
    

# General purpose helper methods
def ldel(string_to_edit, del_string):
    """delete a leading string from another string"""
    slcstart = (
        len(del_string) if string_to_edit.startswith(del_string)
        else 0)
    return string_to_edit[slcstart:]


def iter_parents(path):
    """iterate over the nearest parent paths of a given path"""
    path = pathlib.Path(path)
    yield path.parent
    yield from iter_parents(path.parent)


# General n5 or bdv.n5 utility methods
def get_n5_dir(group_or_dataset_dir):
    """
    get the root directory of the n5 structure
        containing a group or dataset directory
    """
    def is_n5_path(p):
        with (p / "attributes.json").open() as f:
            j = json.load(f)
        if "n5" in j:
            return True
        return False
    group_or_dataset_path = pathlib.Path(group_or_dataset_dir)
    for parent in iter_parents(group_or_dataset_path):
        if is_n5_path(parent):
            return str(parent)


    
class Preprocess(ags.ArgSchemaParser):
        
    def run(self):
        chunk_volume(self.args['input_dir'], self.args['output_dir'],
            chunk_size=self.args['chunk_size'], overlap = self.args['overlap'])
   
        
if __name__ == "__main__":
    mod = Preprocess(schema_type=PreprocessParameters)
    mod.run()      