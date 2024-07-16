import pytest
import os
import glob
from ac_segmentation.process.segment_and_skeletonize_strip import run


def test_seg_skel(zarr_file):
    path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + os.sep + os.pardir)
    weights = glob.glob(f"{path}/**/{'best.ckpt'}", recursive=True)[0]
    run(weights_file=weights, 
        input_zarr=zarr_file+"test_zarr.zarr", 
        probability_output_path=os.path.join(zarr_file, "output_seg.npy.gz"),
            skeleton_output_path=os.path.join(zarr_file, "output_skel.swcs.tar.gz"), zarr_level=0, probability_threshold=0.05,
            label_size_threshold=80, filter_max_intensity=30000)

