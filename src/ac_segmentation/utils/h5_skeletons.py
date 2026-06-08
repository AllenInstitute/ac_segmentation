import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import io
import glob
from io import BytesIO
import h5py
import uuid
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from cloudvolume import Skeleton
import json
import navis
import numpy as np
import glob
import navis
import time



def create_random_skeleton(segid, n_vertices=10, space_shape=(100, 100, 100)):
    vertices = np.random.rand(n_vertices, 3) * np.array(space_shape)
    edges = [[i, i + 1] for i in range(n_vertices - 1)]
    skel = Skeleton(
        vertices=vertices,
        edges=edges,
        radii=np.random.uniform(0.5, 2.0, size=n_vertices).astype(np.float32),
        vertex_types=np.zeros(n_vertices, dtype=np.uint8),
        segid=segid,
    )
    return skel



def kimi_to_navis(skels):
    out_sk = navis.NeuronList(None)
    try:
        for sk in skels:
            out_sk.append(navis.NeuronList(sk.to_swc()))
    except Exception:
        out_sk.append(navis.NeuronList(skels.to_swc()))
    return out_sk




def write_single_shard(shard_skeleton_items, output_dir):
    """
    Write a shard using flat-buffer layout for maximum read/write speed.
    Automatically handles small shards by capping chunk sizes.
    """
    shard_dict = dict(shard_skeleton_items)
    shard_id = int(uuid.uuid4().int % 1e14)
    shard_name = f"{shard_id:03d}.h5"
    shard_path = os.path.join(output_dir, shard_name)

    n_skel = len(shard_dict)
    total_verts = sum(len(s.vertices) for s in shard_dict.values())
    total_edges = sum(len(s.edges) for s in shard_dict.values())

    shard_xmin = shard_ymin = shard_zmin = np.inf
    shard_xmax = shard_ymax = shard_zmax = -np.inf

    vert_offsets = np.zeros((n_skel, 2), dtype=np.int64)
    edge_offsets = np.zeros((n_skel, 2), dtype=np.int64)
    segids = np.zeros(n_skel, dtype=np.int64)
    index_ds = np.zeros((n_skel, 8), dtype=np.float32)

    def cap_chunk(size, max_chunk=65536):
        return min(size, max_chunk)

    with h5py.File(shard_path, "w") as f:
        verts_ds = f.create_dataset(
            "vertices", (total_verts, 3), dtype="float32",
            chunks=(cap_chunk(total_verts), 3),
        )
        edges_ds = f.create_dataset(
            "edges", (total_edges, 2), dtype="int32",
            chunks=(cap_chunk(total_edges), 2),
        )
        rad_ds = f.create_dataset(
            "radius", (total_verts,), dtype="float32",
            chunks=(cap_chunk(total_verts),),
        )
        types_ds = f.create_dataset(
            "types", (total_verts,), dtype="uint32",
            chunks=(cap_chunk(total_verts),),
        )

        vert_cursor = 0
        edge_cursor = 0

        for i, (segid, skel) in enumerate(shard_dict.items()):
            nv = len(skel.vertices)
            ne = len(skel.edges)

            verts_ds[vert_cursor:vert_cursor + nv] = skel.vertices
            edges_ds[edge_cursor:edge_cursor + ne] = skel.edges
            rad_ds[vert_cursor:vert_cursor + nv] = skel.radius
            types_ds[vert_cursor:vert_cursor + nv] = skel.vertex_types

            vert_offsets[i] = (vert_cursor, nv)
            edge_offsets[i] = (edge_cursor, ne)

            xmin, ymin, zmin = skel.vertices.min(axis=0)
            xmax, ymax, zmax = skel.vertices.max(axis=0)
            shard_xmin = min(shard_xmin, xmin)
            shard_ymin = min(shard_ymin, ymin)
            shard_zmin = min(shard_zmin, zmin)
            shard_xmax = max(shard_xmax, xmax)
            shard_ymax = max(shard_ymax, ymax)
            shard_zmax = max(shard_zmax, zmax)

            index_ds[i] = [segid, xmin, xmax, ymin, ymax, zmin, zmax, nv]
            segids[i] = segid

            vert_cursor += nv
            edge_cursor += ne

        f.create_dataset("vert_offsets", data=vert_offsets, dtype="int64")
        f.create_dataset("edge_offsets", data=edge_offsets, dtype="uint64")
        f.create_dataset("index", data=index_ds, dtype="float32")
        f.create_dataset("segids", data=segids, dtype="int64")

    id_to_shard = {int(segid): shard_name for segid in shard_dict.keys()}

    return shard_name, [
        int(shard_xmin), int(shard_ymin), int(shard_zmin),
        int(shard_xmax), int(shard_ymax), int(shard_zmax),
    ], id_to_shard




def read_shard(shard_full_path, segid_filter=None, bbox=None):
    """
    Ultra-fast shard reader using flat-buffer layout with batch reads.
    Returns a list of Skeleton objects, or [] on error.
    """
    if segid_filter is not None and not isinstance(segid_filter, set):
        segid_filter = set(segid_filter)

    try:
        with h5py.File(shard_full_path, "r") as f:
            segids      = f["segids"][:]
            vert_offsets = f["vert_offsets"][:]
            edge_offsets = f["edge_offsets"][:]
            index_ds    = f["index"][:]

            verts_ds  = f["vertices"]
            edges_ds  = f["edges"]
            rad_ds    = f["radius"]
            types_ds  = f["types"]

            # ---- Vectorized filtering ----
            valid_mask = segids > 0
            if segid_filter is not None:
                valid_mask &= np.isin(segids, list(segid_filter))

            if bbox is not None:
                qxmin, qymin, qzmin, qxmax, qymax, qzmax = bbox
                xmin, xmax = index_ds[:, 1], index_ds[:, 2]
                ymin, ymax = index_ds[:, 3], index_ds[:, 4]
                zmin, zmax = index_ds[:, 5], index_ds[:, 6]
                valid_mask &= ~(
                    (xmax < qxmin) | (xmin > qxmax) |
                    (ymax < qymin) | (ymin > qymax) |
                    (zmax < qzmin) | (zmin > qzmax)
                )

            indices = np.nonzero(valid_mask)[0]
            if len(indices) == 0:
                return []

            # ---- Compute bulk vertex/edge ranges ----
            v_start = vert_offsets[indices, 0].min()
            v_end   = (vert_offsets[indices, 0] + vert_offsets[indices, 1]).max()
            e_start = edge_offsets[indices, 0].min()
            e_end   = (edge_offsets[indices, 0] + edge_offsets[indices, 1]).max()

            verts_block = verts_ds[v_start:v_end]
            edges_block = edges_ds[e_start:e_end]
            rad_block   = rad_ds[v_start:v_end]
            types_block = types_ds[v_start:v_end]

            skels = [Skeleton() for _ in range(len(indices))]
            for j, i in enumerate(indices):
                skel = skels[j]
                skel.id = int(segids[i])

                vs, nv = vert_offsets[i]
                es, ne = edge_offsets[i]

                skel.vertices     = verts_block[vs - v_start: vs - v_start + nv]
                skel.edges        = edges_block[es - e_start: es - e_start + ne]
                skel.radius       = rad_block[vs - v_start: vs - v_start + nv]
                skel.vertex_types = types_block[vs - v_start: vs - v_start + nv]

            return skels

    except Exception as e:
        print(f"[read_shard] ERROR reading {shard_full_path}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return []


def _read_matching_ids_from_shard(args):
    shard_dir, shard_name, segids = args
    shard_path = os.path.join(shard_dir, shard_name)
    segids = set(segids)

    try:
        with h5py.File(shard_path, "r") as f:
            segid_col    = f["segids"][:]
            mask         = np.isin(segid_col, list(segids))
            idxs         = np.where(mask)[0]

            if len(idxs) == 0:
                return shard_name, []

            verts_ds     = f["vertices"]
            edges_ds     = f["edges"]
            rad_ds       = f["radius"]
            types_ds     = f["types"]
            vert_offsets = f["vert_offsets"][:]
            edge_offsets = f["edge_offsets"][:]

            skels = []
            for i in idxs:
                segid = int(segid_col[i])
                if segid <= 0:
                    continue

                skel = Skeleton()
                skel.id = segid

                v_start, nv = vert_offsets[i]
                e_start, ne = edge_offsets[i]

                skel.vertices     = verts_ds[v_start:v_start + nv]
                skel.edges        = edges_ds[e_start:e_start + ne]
                skel.radius       = rad_ds[v_start:v_start + nv]
                skel.vertex_types = types_ds[v_start:v_start + nv]

                skels.append(skel)

        return shard_name, skels

    except Exception as e:
        print(f"[_read_matching_ids_from_shard] ERROR reading {shard_path}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return shard_name, []




def _load_all_json_matching(shard_dir, pattern):
    """Load and merge all JSON files matching pattern. Returns combined dict."""
    merged = {}
    for path in glob.glob(os.path.join(shard_dir, pattern)):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                merged.update({str(k): v for k, v in data.items()})
        except Exception as e:
            print(f"[_load_all_json_matching] Skipping {path}: {e}")
    return merged


def load_global_index(shard_dir):
    """Load and merge all global shard index JSON files in a directory."""
    global_index = _load_all_json_matching(shard_dir, "*global*index.json")
    if not global_index:
        raise FileNotFoundError(f"No global index files found in {shard_dir}")
    return global_index


def save_id_to_shard_h5(id_to_shard, output_path):
    """Save a large id_to_shard mapping to HDF5."""
    shard_paths  = sorted(set(id_to_shard.values()))
    shard_to_idx = {p: i for i, p in enumerate(shard_paths)}

    segids        = np.array(list(id_to_shard.keys()), dtype=np.int64)
    shard_indices = np.array([shard_to_idx[id_to_shard[k]] for k in segids], dtype=np.uint32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("segids", data=segids, compression="gzip")
        f.create_dataset("shard_indices", data=shard_indices, compression="gzip")
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("shard_paths", data=np.array(shard_paths, dtype=dt))


def load_id_to_shard_h5(h5_path):
    """Load ID -> shard mapping from HDF5. Returns dict[int, str]."""
    with h5py.File(h5_path, "r") as f:
        segids        = f["segids"][:]
        shard_indices = f["shard_indices"][:]
        shard_paths   = f["shard_paths"][:]

    id_to_shard = {
        int(segid): (shard_paths[idx].decode() if isinstance(shard_paths[idx], bytes) else shard_paths[idx])
        for segid, idx in zip(segids, shard_indices)
    }
    return id_to_shard



def shard_and_write_skeletons(
    skeletons,
    output_dir,
    max_skeletons_per_shard=10000,
    label="",
    n_workers=4,
):
    os.makedirs(output_dir, exist_ok=True)
    global_shard_index = {}
    id_to_shard_index  = {}

    if isinstance(skeletons, dict):
        shard_items = list(skeletons.items())
    else:
        dic = {}
        for sk in skeletons:
            dic[sk.id] = sk
        shard_items = list(dic.items())

    shards = [
        shard_items[i:i + max_skeletons_per_shard]
        for i in range(0, len(shard_items), max_skeletons_per_shard)
    ]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(write_single_shard, chunk, output_dir)
                for chunk in shards
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Writing shards"):
                try:
                    shard_name, bbox, idmap = f.result()
                    global_shard_index[shard_name] = bbox
                    id_to_shard_index.update(idmap)
                except Exception as e:
                    print(f"[shard_and_write_skeletons] Worker error: {type(e).__name__}: {e}")
                    traceback.print_exc()
    else:
        for chunk in shards:
            shard_name, bbox, idmap = write_single_shard(chunk, output_dir)
            global_shard_index[shard_name] = bbox
            id_to_shard_index.update(idmap)

    with open(os.path.join(output_dir, f"{label}_global_shard_index.json"), "w") as f:
        json.dump(global_shard_index, f, indent=2)

    return global_shard_index




def load_all_skeletons(shard_dir, n_workers=1):
    global_index = load_global_index(shard_dir)
    shard_names  = list(global_index.keys())

    if not shard_names:
        return [], {}

    all_skeletons = []
    shard_to_ids  = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(read_shard, os.path.join(shard_dir, name)): name
                for name in shard_names
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc="Loading all shards"):
                shard_name = futures[f]
                try:
                    skels = f.result()
                    all_skeletons.extend(skels)
                    shard_to_ids[shard_name] = [s.id for s in skels]
                except Exception as e:
                    print(f"[load_all_skeletons] Worker error on {shard_name}: {type(e).__name__}: {e}")
                    traceback.print_exc()
    else:
        for name in tqdm(shard_names, desc="Loading all shards"):
            skels = read_shard(os.path.join(shard_dir, name))
            all_skeletons.extend(skels)
            shard_to_ids[name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids



def query_skeletons_by_bb(query_bbox, shard_dir, n_workers=1):
    global_index = load_global_index(shard_dir)

    qxmin, qymin, qzmin, qxmax, qymax, qzmax = query_bbox
    candidates = [
        name for name, bbox in global_index.items()
        if not (
            bbox[3] < qxmin or bbox[0] > qxmax or
            bbox[4] < qymin or bbox[1] > qymax or
            bbox[5] < qzmin or bbox[2] > qzmax
        )
    ]

    if not candidates:
        return [], {}

    all_skeletons = []
    shard_to_ids  = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    read_shard,
                    os.path.join(shard_dir, name),
                    bbox=query_bbox,
                ): name
                for name in candidates
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc="Loading shards"):
                shard_name = futures[f]
                try:
                    skels = f.result()
                    all_skeletons.extend(skels)
                    shard_to_ids[shard_name] = [s.id for s in skels]
                except Exception as e:
                    print(f"[query_skeletons_by_bb] Worker error on {shard_name}: {type(e).__name__}: {e}")
                    traceback.print_exc()
    else:
        for name in tqdm(candidates, desc="Loading shards"):
            skels = read_shard(os.path.join(shard_dir, name), bbox=query_bbox)
            all_skeletons.extend(skels)
            shard_to_ids[name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids




def query_skeletons_by_id(segids, shard_dir, n_workers=1):
    segids = set(segids if isinstance(segids, (list, set)) else [segids])

    global_index = load_global_index(shard_dir)
    shard_names  = list(global_index.keys())

    if not shard_names or not segids:
        return [], {}

    all_skeletons = []
    shard_to_ids  = {}
    tasks = [(shard_dir, name, segids) for name in shard_names]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_read_matching_ids_from_shard, task)
                for task in tasks
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Querying shards by ID"):
                try:
                    shard_name, skels = f.result()
                    if skels:
                        all_skeletons.extend(skels)
                        shard_to_ids[shard_name] = [s.id for s in skels]
                except Exception as e:
                    print(f"[query_skeletons_by_id] Worker error: {type(e).__name__}: {e}")
                    traceback.print_exc()
    else:
        for task in tqdm(tasks, desc="Querying shards by ID"):
            shard_name, skels = _read_matching_ids_from_shard(task)
            if skels:
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids




def delete_skeletons_in_shard(shard_path, skeleton_ids_set, retries=8, base_delay=0.25):
    for attempt in range(retries):
        try:
            with h5py.File(shard_path, "r+") as f:
                segids_ds = f["segids"]
                segid_col = segids_ds[:]

                mask = np.isin(segid_col, list(skeleton_ids_set))
                indices_to_delete = np.where(mask)[0]

                if len(indices_to_delete) == 0:
                    return

                segids_ds[indices_to_delete] = 0
                f["index"][indices_to_delete, 0] = 0
            return  # success

        except BlockingIOError:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def delete_skeletons_parallel(shard_dir, skeleton_ids, n_workers=4):
    """Delete specified skeleton IDs across all shards in parallel."""
    global_index      = load_global_index(shard_dir)
    shard_names       = list(global_index.keys())
    skeleton_ids_set  = set(skeleton_ids)
    shard_paths       = [os.path.join(shard_dir, name) for name in shard_names]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(delete_skeletons_in_shard, path, skeleton_ids_set)
            for path in shard_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deleting skeletons"):
            try:
                future.result()
            except Exception as e:
                print(f"[delete_skeletons_parallel] Worker error: {type(e).__name__}: {e}")
                traceback.print_exc()




def overwrite_skeletons(
    updated_skeletons,
    shard_dir,
    n_workers=4,
    max_skeletons_per_shard=10000,
):
    """Delete old versions of skeletons and write updated ones back."""
    skel_ids = [sk.id for sk in updated_skeletons]

    delete_skeletons_parallel(
        shard_dir=shard_dir,
        skeleton_ids=skel_ids,
        n_workers=n_workers,
    )

    label = f"update_{uuid.uuid4().hex}"

    shard_and_write_skeletons(
        skeletons=updated_skeletons,
        output_dir=shard_dir,
        max_skeletons_per_shard=max_skeletons_per_shard,
        label=label,
        n_workers=n_workers,
    )

    return label