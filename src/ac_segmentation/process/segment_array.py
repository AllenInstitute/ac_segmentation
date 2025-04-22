import numpy
import torch
import argschema
from pathlib import Path

try:
    import zarr
except ImportError:
    pass
    
from ac_segmentation.utils.io import (
    gzip_array, write_cv_skels_iter_tar, upload_to_ceph
)

import ac_segmentation
import ac_segmentation.neurotorch.datasets.datatypes
import ac_segmentation.neurotorch.datasets.dataset
import ac_segmentation.neurotorch.nets.RSUNet
import ac_segmentation.neurotorch.core.predictor
from ac_segmentation.utils.tensorstore import open_ZarrTensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.preprocess import lut_preprocess_array


Predictor = ac_segmentation.neurotorch.core.predictor.Predictor
Vector = ac_segmentation.neurotorch.datasets.datatypes.Vector
BoundingBox = ac_segmentation.neurotorch.datasets.datatypes.BoundingBox
TSArray = ac_segmentation.neurotorch.datasets.dataset.TSArray
Array = ac_segmentation.neurotorch.datasets.dataset.Array
RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet
np = numpy

ONE_GiB = 1_000_000_000


# hack to set number of threads w/ OMP environment variable for old torch
import os
if os.getenv("OMP_NUM_THREADS"):
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS")))
    

def predict_array(
        weights_file, arr,
        iter_size=BoundingBox(Vector(0, 0, 0), Vector(64, 64, 64)),
        stride=Vector(32, 32, 32),
        batch_size=80, gpu_device=None):
    inarr = Array(arr, iteration_size=iter_size, stride=stride)
    outarr = Array(
        -np.inf * np.ones(
            inarr.getBoundingBox().getNumpyDim(), dtype=np.float32),
        iteration_size=iter_size, stride=stride)
    net = ac_segmentation.neurotorch.nets.RSUNet.RSUNet()
    predictor = Predictor(net, weights_file, gpu_device=gpu_device)
    predictor.run(inarr, outarr, batch_size=batch_size)

    # prob_map = 1/(1+np.exp(-outarr.getArray()))
    prob_map = torch.special.expit(
        torch.from_numpy(outarr.getArray())
    ).numpy()
    return prob_map


# TODO predict_arr_chunked function


def predict_zarr(zarr_loc, weights_file, level=0,
                 max_intensity=30000, **kwargs):
    z = zarr.load(zarr_loc)
    ds = z[level]
    data = numpy.transpose(ds[0, 0, ...])
    data = lut_preprocess_array(data, max_int=max_intensity)

    prob_arr = predict_array(weights_file, data, **kwargs)
    return numpy.transpose(prob_arr)


def predict_zarr_ts(zarr_loc, out_fn, weights_file, level=0,
                    max_intensity=30000, bytes_limit=(5 * ONE_GiB),
                    iter_size=BoundingBox(Vector(0, 0, 0), Vector(64, 64, 64)),
                    stride=Vector(32, 32, 32),
                    batch_size=80, gpu_device=None, bound_box=None, AWS_param=None, out_same=False): 
                    
    if 's3' in zarr_loc:          
        try:
            kvstore = create_kvstore(fpath=f"{zarr_loc}/{level}", store='s3', AWS_param=AWS_param) 
            in_ts = open_ZarrTensor(f"{zarr_loc}/{level}", bytes_limit=bytes_limit, kvstore=kvstore)
        except:
            kvstore = create_kvstore(fpath=zarr_loc, store='s3', AWS_param=AWS_param) 
            in_ts = open_ZarrTensor(zarr_loc, bytes_limit=bytes_limit, kvstore=kvstore)  
                     
    else:
        try:
            kvstore = create_kvstore(fpath=f"{zarr_loc}/{level}", store='file') 
            in_ts = open_ZarrTensor(f"{zarr_loc}/{level}", kvstore=kvstore)
        except:
            kvstore = create_kvstore(fpath=zarr_loc, store='file') 
            in_ts = open_ZarrTensor(zarr_loc, kvstore=kvstore)
    
    in_ts = in_ts[0,0,...].transpose()
    if bound_box:
        x1,x2,y1,y2,z1,z2 = bound_box
        in_ts = in_ts[z1:z2,y1:y2,x1:x2]
                    
    in_arr = TSArray(in_ts, iteration_size=iter_size, stride=stride)
    net = ac_segmentation.neurotorch.nets.RSUNet.RSUNet()
    predictor = Predictor(net, weights_file, gpu_device=gpu_device)
    
    
    if 'zarr' in out_fn:
        if 's3' in out_fn:
            kvstore = create_kvstore(fpath=out_fn, store='s3', AWS_param=AWS_param)
        else:
            kvstore = create_kvstore(fpath=out_fn, store='file')
            
        try:
            out_arr = open_ZarrTensor(kvstore=kvstore, driver='zarr3')
        except:
            out_arr = create_tensor(arr_shape=in_ts.shape, kvstore=kvstore, driver='zarr3', dtype='float16', fill_value=-np.inf)
        
        out_arr = TSArray(out_arr, iteration_size=iter_size, stride=stride, prob_map=True, out_same=out_same)
        predictor.run(
            in_arr, out_arr, batch_size=batch_size, max_pix=max_intensity)
    
    elif 'npy' in out_fn:
        out_arr = Array(
            -np.inf * np.ones(in_ts.shape,
                              dtype=np.float32),
            iteration_size=iter_size, stride=stride)
        predictor.run(
            in_arr, out_arr, batch_size=batch_size, max_pix=max_intensity)
    
        prob_map = torch.special.expit(
            torch.from_numpy(
                out_arr.getArray())
        ).numpy()
        return prob_map.transpose()
    
    
def run(weights_file, input_zarr, probability_output_path,
        zarr_level=0, filter_max_intensity=30000,
        predict_options={'gpu_device':None, 'batch_size':80, 'bound_box':None}, AWS_param=None, out_same=False):
    
    # predict and return as probability
    prob_map = predict_zarr_ts(
        input_zarr, probability_output_path, weights_file, level=zarr_level,
        max_intensity=filter_max_intensity, gpu_device=predict_options['gpu_device'], 
        bound_box=predict_options['bound_box'], batch_size=predict_options['batch_size'], AWS_param=AWS_param, out_same=out_same)
    
    if 'npy' in probability_output_path:    
        # write out uint8 representation of probabilities
        uint8_prob_map = (prob_map * 255).astype(numpy.uint8)
        if 's3' in probability_output_path:
            if AWS_param.credential_file:
                file_path = Path(AWS_param.credentials_file_path)
                with file_path.open('r') as file:
                    file_contents = file.readlines()
                AWS_key = file_contents[1].split('= ')[1][0:-1]
                AWS_sec_key = file_contents[2].split('= ')[1][0:-1]
                
                upload_to_ceph(arr=uint8_prob_map, out_file=probability_output_path, aws_access_key=AWS_key, aws_secret_key=AWS_sec_key, endpoint=AWS_param.endpoint_url)
            else:
                upload_to_ceph(arr=uint8_prob_map, out_file=probability_output_path, profile=AWS_param.profile, endpoint=AWS_param.endpoint_url)
        else:
            gzip_array(probability_output_path, uint8_prob_map)
        del uint8_prob_map
        

class CloudOptions(argschema.schemas.DefaultSchema):
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-east-1')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
     
class PredictOptions(argschema.schemas.DefaultSchema):
    gpu_device = argschema.fields.String(required=False, allow_none=True, default=None)
    batch_size = argschema.fields.Int(required=False, allow_none=True, default=80)
    bound_box = argschema.fields.List(argschema.fields.Int(),required=False, default=None, allow_none=True)

class SegmentZarrParameters(argschema.ArgSchema):
    input_zarr = argschema.fields.String(required=True)
    weights_file = argschema.fields.InputFile(required=True)
    probability_output = argschema.fields.String(required=True)      

    zarr_level = argschema.fields.Int(required=False, default=0)
    filter_max_intensity = argschema.fields.Int(required=False, default=30000)
    predict_options = argschema.fields.Nested(
        PredictOptions, required=False, default={'gpu_device':None, 'batch_size':80, 'bound_box':None})
    cloud_options = argschema.fields.Nested(
        CloudOptions, required=False, default={})
    output_json = argschema.fields.OutputFile(required=False, allow_none=True)
    
class SegmentZarrModule(argschema.ArgSchemaParser):
    default_schema = SegmentZarrParameters

    def output(self, d):
        out_json = self.args.get("output_json")
        if out_json:
            pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(f, d)
    
    @property
    def predict_options(self):
        if self.args["predict_options"]==None:
            return {'gpu_device':None,'bound_box':None}
        else:
            return self.args["predict_options"]
            
    @property
    def cloud_options(self):
        return self.args["cloud_options"]


    def run(self):
        AWS_param=None
        cloud = self.cloud_options
        if cloud['profile']:
            AWS_param = AWS_Parameters(profile=cloud['profile'], region=cloud['region'], endpoint_url=cloud['endpoint'])                    
                            
        if cloud['AWS_key']:
            AWS_param = AWS_Parameters(region=cloud['region'], endpoint_url=cloud['endpoint'])
            AWS_param.add_credentials(access_key_id=cloud['AWS_key'], secret_access_key=cloud['AWS_sec_key'])
                                                              
        if AWS_param:
            run(self.args["weights_file"], self.args["input_zarr"],
                        self.args["probability_output"],
                        self.args["zarr_level"],
                        self.args["filter_max_intensity"],
                        self.predict_options, AWS_param, out_same=True)          
                                                               
        else:
            run(self.args["weights_file"], self.args["input_zarr"],
              self.args["probability_output"],
              self.args["zarr_level"],
              self.args["filter_max_intensity"],
              self.predict_options, out_same=True)

        self.output(self.args)

if __name__ == "__main__":
    mod = SegmentZarrModule()
    mod.run()

__all__ = [
    "SegmentZarrModule",
    "SegmentZarrParameters"]
