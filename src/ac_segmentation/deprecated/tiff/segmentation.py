import os
import numpy as np
import pandas as pd
import argschema as ags
import json
import tifffile as tif

import ac_segmentation.neurotorch.datasets.datatypes
import ac_segmentation.neurotorch.datasets.dataset
import ac_segmentation.neurotorch.nets.RSUNet
import ac_segmentation.neurotorch.core.predictor
import ac_segmentation.neurotorch.datasets.filetypes

Predictor = ac_segmentation.neurotorch.core.predictor.Predictor
Vector = ac_segmentation.neurotorch.datasets.datatypes.Vector
BoundingBox = ac_segmentation.neurotorch.datasets.datatypes.BoundingBox
Array = ac_segmentation.neurotorch.datasets.dataset.Array
RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet
TiffVolume = ac_segmentation.neurotorch.datasets.filetypes.TiffVolume



class SegmentationParameters(ags.ArgSchema):
    ckpt = ags.fields.InputFile(required=True, description='Model file')
    output_dir = ags.fields.OutputDir(required=True, description='Output directory')
    chunk_size = ags.fields.Int(dtype='int', required=False, default=416)
    overlap = ags.fields.Int(dtype='int', required=False, default=16)

def predict(checkpoint, outdir, mip=0, **kwargs):
    
    #set default keyword arguments
    defaultKwargs = {'iter_size': BoundingBox(Vector(0, 0, 0), Vector(128, 128, 32)) , 'stride': Vector(64, 64, 16),
                    'gpu_device': 0, 'batch_size':20}
    kwargs = { **defaultKwargs, **kwargs }
    
    #set number of chunks and load/set bounding box
    num_chunks = len([f for f in os.listdir(os.path.join(outdir, 'inputs')) 
                        if f.startswith('chunk')])
    df = pd.read_csv(os.path.join(outdir, 'inputs','bbox.csv'))
    bb = df.bounding_box.values
    bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], bb[5]))
        
    # Initialize the U-Net architecture
    net = RSUNet()
       
    for n in range(num_chunks):
        filename = os.path.join(outdir, 'inputs', 'chunk%02d.tif'%n)
        print('chunk%02d.tif'%n) 
        
        with TiffVolume(filename, bbn, kwargs['iter_size'], kwargs['stride']) as inputs:              
            # Predict
            predictor = Predictor(net, checkpoint, gpu_device=kwargs['gpu_device']) 
            output_volume = Array(-np.inf*np.ones(inputs.getBoundingBox().getNumpyDim(), dtype=np.float32))
            predictor.run(inputs, output_volume, batch_size=kwargs['batch_size'])
                            
            convert_prob_map(outdir, output_volume, n)
            
    return num_chunks
    
def predict_array(checkpoint, outdir, inarr, mip=0, **kwargs):
    
    #set default keyword arguments
    defaultKwargs = {'output_type': 'volume', 'iter_size': BoundingBox(Vector(0, 0, 0), Vector(128, 128, 32)), 'stride': Vector(64, 64, 16),
                    'gpu_device': None, 'batch_size':20}
    kwargs = { **defaultKwargs, **kwargs }
    
    #set array object
    inarr = Array(inarr, iteration_size= kwargs['iter_size'], stride= kwargs['stride'])
        
    # Initialize the U-Net architecture
    net = RSUNet()
    
    # Predict
    predictor = Predictor(net, checkpoint, gpu_device=kwargs['gpu_device']) 
    output_volume = Array(-np.inf*np.ones(inarr.getBoundingBox().getNumpyDim(), dtype=np.float32))
    predictor.run(inarr, output_volume, batch_size=kwargs['batch_size'])
    
    if kwargs['output_type'] == 'volume':
        return output_volume
    
    else:
        convert_prob_map(outdir, output_volume, 1)
        

def convert_prob_map(outdir, output_volume, chunk_n):
    savedir = os.path.join(outdir, 'Segmentation')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    mips_dir = os.path.join(outdir, 'Segmentation', 'mips')   
    if not os.path.isdir(mips_dir):
        os.mkdir(mips_dir)
            
    # Convert to probability map and save
    probability_map = 1/(1+np.exp(-output_volume.getArray()))
    chunkdir = os.path.join(savedir, 'chunk%02d'%chunk_n)

    if not os.path.isdir(chunkdir):
        os.mkdir(chunkdir)
    for i in range(probability_map.shape[0]):# save 8-bit version as multiple tif files
        tif.imsave(os.path.join(chunkdir,'%03d.tif'%i), np.uint8(255*probability_map[i,:,:]))          
    mip_xy = np.max(probability_map,0) # save mip
    tif.imsave(os.path.join(mips_dir, 'MAX_xy_chunk%02d.tif'%chunk_n), np.uint8(255*mip_xy)) 
    
def load_stack_select(dirname, idx1, idx2):
    # Load image stack filenames
    filelist = [f for f in os.listdir(dirname) if f.endswith('.tif')] 
    filelist.sort()
    
    # Calculate stack size
    filename = os.path.join(dirname, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = idx2 - idx1 + 1, img.shape[0], img.shape[1]    
    stack = np.zeros(cell_stack_size, dtype=img.dtype)
    for i, idx in enumerate(range(idx1, idx2 + 1)):
        filename = os.path.join(dirname, filelist[idx])
        img = tif.imread(filename)
        stack[i,:,:] = img
        
    return stack                
            
def blend_chunks(outdir, num_chunks, chunk_size, overlap):
    savedir = os.path.join(outdir, 'Segmentation')
    for n1 in range(num_chunks - 1):
        n2 = n1 + 1
        # Load overlapping tif files
        indir1 = os.path.join(savedir, 'chunk%02d'%n1)
        indir2 = os.path.join(savedir, 'chunk%02d'%n2)
        idx11 = chunk_size - overlap
        idx12 = chunk_size - 1
        stack1 = load_stack_select(indir1, idx11, idx12)
        idx21 = 0
        idx22 = overlap - 1
        stack2 = load_stack_select(indir2, idx21, idx22)
        # Blend overlapping images
        shift_x = (chunk_size - overlap)*4
        stack2_shift = np.zeros((stack1.shape), dtype=stack1.dtype)
        stack2_shift[:,:,0:stack1.shape[2] - shift_x] = stack2[:,:,shift_x:]
        stack_blended = np.maximum(stack1, stack2_shift)
        # Save blended images
        for i, idx in enumerate(range(idx11, idx12 + 1)):
            tif.imsave(os.path.join(indir1, '%03d.tif'%idx), stack_blended[i,:,:])
        stack_blended_shift = np.zeros((stack1.shape), dtype=stack1.dtype)
        stack_blended_shift[:,:,shift_x:] = stack_blended[:,:,0:stack1.shape[2] - shift_x] 
        for i, idx in enumerate(range(idx21, idx22 + 1)):
            tif.imsave(os.path.join(indir2, '%03d.tif'%idx), stack_blended_shift[i,:,:])
            
class Segmentation(ags.ArgSchemaParser):
        
    def run(self):
        num_chunks = predict(self.args['ckpt'], self.args['output_dir'],
            chunk_size=self.args['chunk_size'], overlap = self.args['overlap']) 
        blend_chunks(self.args['output_dir'], num_chunks, 
            chunk_size=self.args['chunk_size'], overlap = self.args['overlap'])     
        
if __name__ == "__main__":
    mod = Segmentation(schema_type=SegmentationParameters)
    mod.run()      