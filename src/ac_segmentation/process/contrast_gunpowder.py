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
    scan = gp.Scan(scan_request, num_workers=1)
    
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
    
    method, values = preprocess['method'], preprocess['values']
    contrast = ContrastAdjustWrite(raw,raw,input_arr,output_arr,int_range=values,version=method)
        
    # Build the pipeline with Scan
    pipeline = (
            source +
            contrast +
            scan)

    btime = datetime.now()
    with gp.build(pipeline):
        for i in range(len(start_new)):
            stime = datetime.now()
            end_new[i] = np.minimum(end_new[i], np.array(input_arr.shape))
            arr = np.array(end_new[i])-np.array(start_new[i])
            total_roi = gp.Roi(start_new[i], arr)
                
            # Create a request for the entire volume
            request = gp.BatchRequest()
            request[raw] = total_roi
        
            # Request the batch
            batch = pipeline.request_batch(request)
            
            # Retrieve the list of write objects and write them
            write_objects = contrast.get_write_objects()
            if len(write_objects) > 2000:
                for write in write_objects:
                    write.result()
                contrast.clear_write_objects()

            print(datetime.now()-btime)
            etime = datetime.now()
            print(etime-stime)