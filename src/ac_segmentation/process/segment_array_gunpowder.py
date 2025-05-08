import ac_segmentation.neurotorch.nets.RSUNet
from ac_segmentation.utils.process.gunpowder_nodes import TensorStoreSource, ApplyModel, ContrastAdjust
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.process.utils.io import create_chunked_dims, create_overlap_chunks

import tensorstore as ts
import gunpowder as gp
import torch


RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet


def segment_gunpowder(input_path, output_path, checkpoint, iter_size=(64,64,64), batch_size=3, cutout=None, gpu_device=None, cpus=20, preprocess={'method':'percentile','values':[96,97]}):
    
    if int(cpus) > int(os.cpu_count()):
        cpus = os.cpu_count()
    torch.set_num_threads(cpus) 
    
    # Set-up model
    device = torch.device("cuda:{}" .format(gpu_device) if gpu_device is not None else "cpu")
    model = RSUNet()
    model = model.to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    #get input array size
    if isinstance(input_path, str):
        input_arr = open_tensor(input_path, bytes_limit= 100_000_000, driver='zarr')
    else:
        input_arr = input_path

    start_req = (0,0,0)
    if len(input_arr.shape) == 5:
        iter_size = (1,1,) + iter_size
        start_req = (0,0,0,0,0)
    chunk_size = batch_size*np.array(iter_size)
    
    #define the pipeline
    raw = gp.ArrayKey('RAW')
    source = TensorStoreSource(
        {
            raw: input_arr
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        })
    
    #create tensorstore
    if isinstance(output_path, str):
        if os.path.isdir(output_path):
            output_arr = open_tensor(output_path, bytes_limit= 100_000_000, driver='zarr3')
        else:
            output_arr = create_tensor(fpath=output_path, arr_shape=input_arr.shape[-3:], dtype = 'float32', fill_value=-np.inf, driver='zarr3')
    else:
        output_arr = output_path

    # Define the chunk size
    iter_coord = gp.Coordinate(iter_size) 
    
    # Define the scan request with overlap
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi(start_req, iter_coord)
    scan = gp.Scan(scan_request)
    
    # Create the ApplyModel instance
    apply_model = ApplyModel(model, raw, output_arr, device)
    
    # Build the pipeline with Scan
    method, values = preprocess['method'], preprocess['values']
    pipeline = (
        source +
        ContrastAdjust(raw,raw,values,version=method) +
        apply_model +
        scan
    )
    
    stime = datetime.now()
    #create chunks
    start, end = create_chunked_dims(arr_shape=input_arr.shape, chunk_size=chunk_size)
    start_new, end_new = [], []

    if cutout:
        x1, x2, y1, y2, z1, z2 = cutout
        for i, (s, e) in enumerate(zip(start, end)):
            offset = np.array([x1,y1,z1])
            s, e = np.array(s[-3:])+offset, np.array(e[-3:])+offset
            if (e[0] <= x2+chunk_size[-3] and e[1] <= y2+chunk_size[-2] and e[2] <= z2+chunk_size[-1]):
                start_new.append(s)
                end_new.append(e)
        start, end = start_new, end_new
        start_new, end_new = [], []
        
    for i in range(len(start)):
        sover, eover = create_overlap_chunks(start[i][-3:], end[i][-3:], overlap=int(iter_size[-1]/2))
        start_new += sover
        end_new += eover
    
    save = np.array(start_new[0])
    for i in range(len(start_new)):
        if len(input_arr.shape) == 5:
            start_new[i] = [0,0]+start_new[i]
            end_new[i] = [1,1]+end_new[i]
        arr1, arr2 = np.array(start_new[i]), np.array(end_new[i])
        if np.any(arr2 - arr1 < np.array(iter_size)) == True:
            start_new[i] = list(np.minimum(arr1, save))
        else:
            save = arr1

    #run pipeline    
    with gp.build(pipeline):
        for i in range(len(start)):
            end[i] = np.minimum(end[i], np.array(input_arr.shape))
            arr = np.array(end[i])-np.array(start[i])
            total_roi = gp.Roi(start[i], arr)

            iter_new = np.minimum(arr, np.array(iter_size))
            scan_request[raw] = gp.Roi(start_req, iter_coord)
        
            # Create a request for the entire volume
            request = gp.BatchRequest()
            request[raw] = total_roi

            # Request the batch
            batch = pipeline.request_batch(request)
    
    # Retrieve the list of write objects and write them
    write_objects = apply_model.get_write_objects()
    for write in write_objects:
        write.result()
    apply_model.clear_write_objects()
    etime = datetime.now()
    print(etime-stime)
    
    #output seg card
    compute = {'GPUs': gpu_device} if gpu_device else {'CPUs': cpus}
    seg_card = {'date':datetime.today().strftime('%Y-%m-%d'),
                'paths':{'inpath':input_path,'outpath':output_path}, 
                'preprocessing':{'method':method,'values':values}, 
                'compute':compute,
                'time_lapse':etime-stime}

    with open(os.path.join(output_path, "seg_card.txt"), 'w') as f:
        f.write(str(seg_card))
    
    return seg_card