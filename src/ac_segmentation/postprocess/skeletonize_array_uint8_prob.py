#!/usr/bin/env python

import json
import pathlib

import argschema
import numpy

from ac_segmentation.postprocess.skeletonize_array import (
    label_binary_array, threshold_binarize_array,
    skeletonize_labeled_array_concurrent
)
from ac_segmentation.utils.io import (
    read_gzip_array, write_cv_skels_iter_tar
)


def run(probability_input_path, skeleton_output_path,
        probability_threshold=0.05, label_size_threshold=80,
        skeletonize_options=None):
    skeletonize_options = (
        {} if skeletonize_options is None
        else skeletonize_options
    )

    # explicitly uint8 to float32....
    prob_map = read_gzip_array(
        probability_input_path,
        preprocess_func=lambda x: (x.astype(numpy.float32) / 255.))

    binary_arr = threshold_binarize_array(
        prob_map, threshold=probability_threshold)
    labeled_arr, _ = label_binary_array(
        binary_arr, size_threshold=label_size_threshold)

    skels = skeletonize_labeled_array_concurrent(
        labeled_arr, **skeletonize_options
    )

    write_cv_skels_iter_tar(skeleton_output_path, skels)


class SkeletonizationOptions(argschema.schemas.DefaultSchema):
    n_jobs = argschema.fields.Int(required=False, allow_none=True)


class SkeletonizeProbabilitiesParameters(argschema.ArgSchema):
    probability_input = argschema.fields.InputFile(required=True)
    skeleton_output = argschema.fields.OutputFile(required=True)
    probability_threshold = argschema.fields.Float(
        required=False, default=0.05)
    label_size_threshold = argschema.fields.Int(required=False, default=80)
    skeletonize_options = argschema.fields.Nested(
        SkeletonizationOptions,
        required=False, default=None, allow_none=True)

    output_json = argschema.fields.OutputFile(required=False, allow_none=True)


class SkeletonizeProbabilitiesModule(argschema.ArgSchemaParser):
    default_schema = SkeletonizeProbabilitiesParameters

    def output(self, d):
        out_json = self.args.get("output_json")
        if out_json:
            pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(f, d)

    @property
    def skeletonize_options(self):
        return self.args["skeletonize_options"]    

    def run(self):
        run(
            self.args["probability_input"],
            self.args["skeleton_output"],
            probability_threshold=self.args["probability_threshold"],
            label_size_threshold=self.args["label_size_threshold"],
            skeletonize_options=self.skeletonize_options
        )
        self.output(self.args)


if __name__ == "__main__":
    mod = SkeletonizeProbabilitiesModule()
    mod.run()


__all__ = [
    "SkeletonizeProbabilitiesModule",
    "SkeletonizeProbabilitiesParameters"
]
