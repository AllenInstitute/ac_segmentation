import glob
import os
import pytest

def get_submod(directory, filetype):
    submods = []
    for file in glob.glob(f"{directory}/**/{filetype}", recursive=True):
        submods.append(file)
    return submods                 

#get submodules and reformat
path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + os.sep + os.pardir)+"/src/ac_segmentation/"
matches = ['init','neurotorch','test']
submods = get_submod(path, '*.py')
submods = [x.split('src/')[1].replace('/','.').replace('.py','') for x in submods]
submods = [ x for x in submods if any(y in x for y in matches)==False]

@pytest.mark.parametrize("mod", submods)
def test_import(mod):
    imp = __import__(mod)
      