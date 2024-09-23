#!/usr/bin/env python

# segment_and_skeletonize_strip.py updated with new features

import json
import pathlib

import argschema
import numpy

from ac_segmentation.postprocess.skeletonize_array import (
    label_binary_array, threshold_binarize_array,
    skeletonize_labeled_array_concurrent
)

from ac_segmentation.utils.io import (
    gzip_array, write_cv_skels_iter_tar
)
from ac_segmentation.process.segment_array import (
    predict_zarr_ts
)


import torch

# hack to set number of threads w/ OMP environment variable for old torch
import os
if os.getenv("OMP_NUM_THREADS"):
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS")))

        
def run(weights_file, input_zarr, probability_output_path,
        skeleton_output_path, zarr_level=0, probability_threshold=0.05,
        label_size_threshold=80, filter_max_intensity=30000,
        predict_options=None, skeletonize_options=None):
        
    # predict and return as probability
    prob_map = predict_zarr_ts(
        input_zarr, weights_file, level=zarr_level,
        max_intensity=filter_max_intensity, gpu_device=predict_options['gpu_device'], 
        bound_box=predict_options['bound_box'])

    # write out uint8 representation of probabilities
    uint8_prob_map = (prob_map * 255).astype(numpy.uint8)
    gzip_array(probability_output_path, uint8_prob_map)
    del uint8_prob_map

    # binarize volume, label, and skeletonize
    binary_arr = threshold_binarize_array(
        prob_map, threshold=probability_threshold)
    labeled_arr, _ = label_binary_array(
        binary_arr, size_threshold=label_size_threshold)

    # skels = skeletonize_labeled_array(labeled_arr, **skeletonize_options)
    skels = skeletonize_labeled_array_concurrent(
        labeled_arr, n_jobs=skeletonize_options['n_jobs'])

    # write skeletons to swc zip
    # write_kimi_skels_tar(skeleton_output_path, skels)
    write_cv_skels_iter_tar(skeleton_output_path, skels)


class SegmentationPredictOptions(argschema.schemas.DefaultSchema):
    gpu_device = argschema.fields.Int(required=False, allow_none=True, default=None)
    batch_size = argschema.fields.Int(required=False, allow_none=True, default=80)
    bound_box = argschema.fields.List(argschema.fields.Int(),required=False, default='', allow_none=True)

class SkeletonizationOptions(argschema.schemas.DefaultSchema):
    n_jobs = argschema.fields.Int(required=False, allow_none=True)


class SegmentSkeletonizeZarrParameters(argschema.ArgSchema):
    input_zarr = argschema.fields.InputDir(required=True)
    weights_file = argschema.fields.InputFile(required=True)
    probability_output = argschema.fields.OutputFile(required=True)
    skeleton_output = argschema.fields.OutputFile(required=True)

    zarr_level = argschema.fields.Int(required=False, default=1)
    probability_threshold = argschema.fields.Float(
        required=False, default=0.05)
    filter_max_intensity = argschema.fields.Int(required=False, default=30000)
    label_size_threshold = argschema.fields.Int(required=False, default=80)
    predict_options = argschema.fields.Nested(
        SegmentationPredictOptions, required=False,
        default=None, allow_none=True)
    skeletonize_options = argschema.fields.Nested(
        SkeletonizationOptions, required=False, default=None, allow_none=True)

    output_json = argschema.fields.OutputFile(required=False, allow_none=True)


class SegmentSkeletonizeZarrModule(argschema.ArgSchemaParser):
    default_schema = SegmentSkeletonizeZarrParameters

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
    def skeletonize_options(self):
        if self.args["skeletonize_options"]==None:
            return {'n_jobs':10}
        else:
            return self.args["skeletonize_options"] 

    def run(self):
        run(self.args["weights_file"], self.args["input_zarr"],
            self.args["probability_output"], self.args["skeleton_output"],
            self.args["zarr_level"],
            self.args["probability_threshold"],
            self.args["label_size_threshold"],
            self.args["filter_max_intensity"],
            self.predict_options,
            self.skeletonize_options)
        self.output(self.args)


if __name__ == "__main__":
    mod = SegmentSkeletonizeZarrModule()
    mod.run()

__all__ = [
    "SegmentSkeletonizeZarrModule",
    "SegmentSkeletonizeZarrParameters"]
