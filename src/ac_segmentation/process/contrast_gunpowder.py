from ac_segmentation.utils.process.gunpowder_nodes import TensorStoreSource, ContrastAdjustWrite
from ac_segmentation.utils.process.utils.io import create_chunked_dims, create_overlap_chunks
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor

import gunpowder as gp



def adjust_contrast_gunpowder(input_arr, output_arr, iter_size=(64,64,64), batch_size=5, cutout=None, preprocess={'method':'percentile','values':[96,97]}):
    
    raw = gp.ArrayKey('RAW')
    source = TensorStoreSource(
        {
            raw: input_arr
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        })
    
    iter_coord = gp.Coordinate(iter_size)
    start_req = (0,0,0)
    if len(input_arr.shape) == 5:
        iter_size = (1,1,) + iter_size
        start_req = (0,0,0,0,0)
    chunk_size = batch_size*np.array(iter_size)
    iter_coord = gp.Coordinate(iter_size)
    
    # Define the scan request with overlap
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi(start_req, iter_coord)
    scan = gp.Scan(scan_request)
    
    
    #create chunks
    start, end = create_chunked_dims(arr_shape=input_arr.shape, chunk_size=chunk_size)
    start_new, end_new = [], []
    for i in range(len(start)):
        sover, eover = create_overlap_chunks(start[i][-3:], end[i][-3:], overlap=32)
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
                      
    if cutout:
        cutout[0], cutout[1], cutout[2] = cutout[0]-chunk_size[-3], cutout[1]+chunk_size[-3] , cutout[2]-chunk_size[-2]
        cutout[3], cutout[4], cutout[5] = cutout[3]+chunk_size[-2] , cutout[4]-chunk_size[-1], cutout[5]+chunk_size[-1]
        x1, x2, y1, y2, z1, z2 = cutout
        start = []
        end = []
        for i, (s, e) in enumerate(zip(start_new, end_new)):
            s, e = s[-3:], e[-3:]
            if (s[0] >= x1 and s[1] >= y1 and s[2] >= z1
                and e[0] <= x2 and e[1] <= y2 and e[2] <= z2):
                start.append(start_new[i])
                end.append(end_new[i])
    else:
        start = start_new
        end = end_new
        
    if len(start) == 0:
        print('Batch_size needs to be lowered to accommodate the cutout size.')
        return

    
    method, values = preprocess['method'], preprocess['values']
    contrast = ContrastAdjustWrite(raw,raw,input_arr,output_arr,int_range=values,version=method)
        
    # Build the pipeline with Scan
    pipeline = (
            source +
            contrast +
            scan)
    
    with gp.build(pipeline):
        for i in range(len(start)):
            end[i] = np.minimum(end[i], np.array(input_arr.shape))
            arr = np.array(end[i])-np.array(start[i])
            total_roi = gp.Roi(start[i], arr)
            
            # Create a request for the entire volume
            request = gp.BatchRequest()
            request[raw] = total_roi
    
            # Request the batch
            batch = pipeline.request_batch(request)
        
            # Retrieve the list of write objects and write them
            write_objects = contrast.get_write_objects()
            for write in write_objects:
                write.result()
            contrast.clear_write_objects()