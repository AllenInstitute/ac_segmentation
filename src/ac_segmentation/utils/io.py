import concurrent.futures
import gzip
import io
import tarfile

import numpy

import navis


def gzip_array(fn, arr):
    with gzip.open(fn, "wb") as f:
        numpy.save(f, arr)


def read_gzip_array(fn):
    with gzip.open(fn, "rb") as f:
        a = numpy.load(f)
    return a


def write_kimi_skels_tar(tar_fn, skels):
    with tarfile.open(tar_fn, mode="w:gz") as t:
        for skid, skel in skels.items():
            bio = io.BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{skid}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)


# FIXME CL code does not preserve ids
def write_cv_skels_iter_tar(tar_fn, skels):
    with tarfile.open(tar_fn, mode="w:gz") as t:
        for skid, skel in enumerate(skels):
            bio = io.BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{skid}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)


def read_navis_neurons_tar(tar_fn, concurrency=10, preprocess_func=None):
    preprocess_func = ((lambda x: x) if preprocess_func is None else preprocess_func)
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        futs = []
        with tarfile.open(tar_fn, "r:gz") as t:
            for m in t.getmembers():
                swc_b = t.extractfile(m).read()
                futs.append(e.submit(navis.io.read_swc, swc_b.decode()))
        navis_neurons = navis.NeuronList([
            preprocess_func(fut.result()) for
            fut in concurrent.futures.as_completed(futs)])
    return navis_neurons