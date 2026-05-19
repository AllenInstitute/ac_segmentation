from ac_segmentation.utils.tensorstore import *

def zarr_to_CATMAID_project(zarr_path, out_path, container_id, AWS_Key, AWS_Secret_Key, project='NewProject', 
                            stack='NewStack', store='s3', chunks=(64,64,64), translation=(0,0,0), cutout=None, tile_dim=[128,128], ortho=False):

    #convert zarr to precomputed
    zarr_to_precomputed(zarr_path=zarr_path, AWS_Key=AWS_Key, AWS_Secret_Key=AWS_Secret_Key, 
                           out_path=out_path, store=store, chunks=(64,64,64), cutout=cutout)
    
    #extract url 
    split = out_path.split("/")
    bucket = split[0]
    key = out_path.replace(bucket+"/", '')
    url = os.path.join("https://", bucket+".s3-us-west-2.amazonaws.com", key, "%SCALE_DATASET%/")
    
    #get resolution and shape
    shape = open_tensor(zarr_path + "0/")[0,0,:,:,:].shape
    if cutout!=None:
        x1,x2,y1,y2,z1,z2 = cutout
        x,y,z = x2-x1,y2-y1,z2-z1
        shape = [x,y,z]
    r_path = os.path.join(os.path.dirname(os.path.dirname(zarr_path + "0/")), ".zattrs")
    res = json.loads(open(r_path, "r").read())['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][2:]
    
    #create project data json
    if ortho==True:
      stack_file = [{
        "project": {
          "title": project,
          "stacks": [{
            "title": stack+"xy",
            "dimension": str(tuple(shape)),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "0_1_2"
            }],
            "resolution": str(tuple(res)),
            "translation": str(translation),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 0,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "channel"}]
          },
          {
            "title": stack+"xz",
            "dimension": str(tuple([shape[0],shape[2],shape[1]])),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "0_2_1"
            }],
            "resolution": str(tuple([res[0],res[2],res[1]])),
            "translation": str(tuple([translation[0],translation[2],translation[1]])),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 1,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "view"}]
          },
          {
            "title": stack+"zy",
            "dimension": str(tuple([shape[2],shape[1],shape[0]])),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "2_1_0"
            }],
            "resolution": str(tuple([res[2],res[1],res[0]])),
            "translation": str(tuple([translation[2],translation[1],translation[0]])),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 2,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "view"}]
          }]
        }
      }]
      
    else:
      stack_file = [{
        "project": {
          "title": project,
          "stacks": [{
            "title": stack+"xy",
            "dimension": str(tuple(shape)),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "0_1_2"
            }],
            "resolution": str(tuple(res)),
            "translation": str(translation),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 0,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "channel"}]
          }]
        }
      }]
                                                              
    #save json to local                    
    json_fpath = os.path.join("./", "data.json")       
    with open(json_fpath, 'w') as f:
        json.dump(stack_file, f)
    
    #open docker container and import json
    client = docker.from_env()
    container = client.containers.get(container_id)
    copy_string = json_fpath + " " + container_id + ":" + "/home/django/projects/data.json"
    os.system("docker cp " + str(copy_string))
    container.exec_run('python3 manage.py catmaid_import_projects --input data.json --permission user:admin:can_import user:admin:can_annotate')

    #clean-up
    container.exec_run('rm input data.json')
    client.close()
    os.remove(json_fpath)

    return stack_file