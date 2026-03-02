import os
import numpy as np
import io
import glob
from io import BytesIO
import h5py
from uuid import uuid4
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from cloudvolume import Skeleton
import json


def write_single_shard(shard_skeleton_items, output_dir):
    """
    Write a shard using flat-buffer layout for maximum read/write speed.
    Automatically handles small shards by capping chunk sizes.
    """
    shard_dict = dict(shard_skeleton_items)
    shard_id = random.randint(10**13, 10**14 - 1)
    shard_name = f"{shard_id:03d}.h5"
    shard_path = os.path.join(output_dir, shard_name)

    # Count skeletons and total geometry
    n_skel = len(shard_dict)
    total_verts = sum(len(s.vertices) for s in shard_dict.values())
    total_edges = sum(len(s.edges) for s in shard_dict.values())

    # Preallocate bounding box tracking
    shard_xmin = shard_ymin = shard_zmin = np.inf
    shard_xmax = shard_ymax = shard_zmax = -np.inf

    # Allocate arrays for offsets and index
    vert_offsets = np.zeros((n_skel, 2), dtype=np.uint64)  # start, count
    edge_offsets = np.zeros((n_skel, 2), dtype=np.uint64)
    segids = np.zeros(n_skel, dtype=np.uint64)
    index_ds = np.zeros((n_skel, 8), dtype=np.float32)

    # Helper to cap chunks to dataset size
    def cap_chunk(size, max_chunk=65536):
        return min(size, max_chunk)

    with h5py.File(shard_path, "w") as f:
        # Create flat datasets
        verts_ds = f.create_dataset(
            "vertices", (total_verts, 3), dtype="float32",
            chunks=(cap_chunk(total_verts), 3)
        )
        edges_ds = f.create_dataset(
            "edges", (total_edges, 2), dtype="int32",
            chunks=(cap_chunk(total_edges), 2)
        )
        rad_ds = f.create_dataset(
            "radius", (total_verts,), dtype="float32",
            chunks=(cap_chunk(total_verts),)
        )
        types_ds = f.create_dataset(
            "types", (total_verts,), dtype="uint32",
            chunks=(cap_chunk(total_verts),)
        )

        vert_cursor = 0
        edge_cursor = 0

        for i, (segid, skel) in enumerate(shard_dict.items()):
            nv = len(skel.vertices)
            ne = len(skel.edges)

            # Write geometry
            verts_ds[vert_cursor:vert_cursor+nv] = skel.vertices
            edges_ds[edge_cursor:edge_cursor+ne] = skel.edges
            rad_ds[vert_cursor:vert_cursor+nv] = skel.radius
            types_ds[vert_cursor:vert_cursor+nv] = skel.vertex_types

            # Record offsets
            vert_offsets[i] = (vert_cursor, nv)
            edge_offsets[i] = (edge_cursor, ne)

            # Update bounding box
            xmin, ymin, zmin = skel.vertices.min(axis=0)
            xmax, ymax, zmax = skel.vertices.max(axis=0)
            shard_xmin = min(shard_xmin, xmin)
            shard_ymin = min(shard_ymin, ymin)
            shard_zmin = min(shard_zmin, zmin)
            shard_xmax = max(shard_xmax, xmax)
            shard_ymax = max(shard_ymax, ymax)
            shard_zmax = max(shard_zmax, zmax)

            # Fill index dataset
            index_ds[i] = [segid, xmin, xmax, ymin, ymax, zmin, zmax, nv]
            segids[i] = segid

            vert_cursor += nv
            edge_cursor += ne

        # Save offsets & index
        f.create_dataset("vert_offsets", data=vert_offsets, dtype="uint64")
        f.create_dataset("edge_offsets", data=edge_offsets, dtype="uint64")
        f.create_dataset("index", data=index_ds, dtype="float32")
        f.create_dataset("segids", data=segids, dtype="uint64")

    # Build mapping for this shard
    id_to_shard = {int(segid): shard_name for segid in shard_dict.keys()}

    return shard_name, [
        int(shard_xmin), int(shard_ymin), int(shard_zmin),
        int(shard_xmax), int(shard_ymax), int(shard_zmax)
    ], id_to_shard



def read_shard(shard_full_path, segid_filter=None, bbox=None):
    """
    Ultra-fast shard reader using flat-buffer layout with batch reads.
    Returns a list of Skeleton objects.
    """
    if segid_filter is not None and not isinstance(segid_filter, set):
        segid_filter = set(segid_filter)

    with h5py.File(shard_full_path, "r") as f:
        # Load index-level arrays once
        segids = f["segids"][:]
        vert_offsets = f["vert_offsets"][:]
        edge_offsets = f["edge_offsets"][:]
        index_ds = f["index"][:]

        verts_ds = f["vertices"]
        edges_ds = f["edges"]
        rad_ds = f["radius"]
        types_ds = f["types"]

        # ---- Vectorized filtering ----
        valid_mask = segids >= 0
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
        n_skel = len(indices)
        if n_skel == 0:
            return []

        # ---- Compute bulk vertex/edge ranges ----
        v_start = vert_offsets[indices, 0].min()
        v_end = (vert_offsets[indices, 0] + vert_offsets[indices, 1]).max()
        e_start = edge_offsets[indices, 0].min()
        e_end = (edge_offsets[indices, 0] + edge_offsets[indices, 1]).max()

        # ---- Bulk read slices ----
        verts_block = verts_ds[v_start:v_end]
        edges_block = edges_ds[e_start:e_end]
        rad_block = rad_ds[v_start:v_end]
        types_block = types_ds[v_start:v_end]

        # ---- Build Skeleton objects using views ----
        skels = [Skeleton() for _ in range(n_skel)]
        for j, i in enumerate(indices):
            skel = skels[j]
            skel.id = int(segids[i])

            vs, nv = vert_offsets[i]
            es, ne = edge_offsets[i]

            # Use views into the bulk blocks
            skel.vertices = verts_block[vs - v_start : vs - v_start + nv]
            skel.edges = edges_block[es - e_start : es - e_start + ne]
            skel.radius = rad_block[vs - v_start : vs - v_start + nv]
            skel.vertex_types = types_block[vs - v_start : vs - v_start + nv]

        return skels



def _load_all_json_matching(shard_dir, pattern):
    """
    Internal helper: load and merge all JSON files matching pattern.
    Returns a combined dict.
    """
    merged = {}
    for path in glob.glob(os.path.join(shard_dir, pattern)):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                merged.update({str(k): v for k, v in data.items()})
        except Exception as e:
            print(f"?? Skipping {path}: {e}")
    return merged


# ---------------------------------------------------------------------
#  GLOBAL + ID INDEX LOADERS
# ---------------------------------------------------------------------

def load_global_index(shard_dir):
    """
    Load and merge *all* global shard index JSON files in a directory.
    (Matches '*global*index.json')
    """
    global_index = _load_all_json_matching(shard_dir, "*global*index.json")
    if not global_index:
        raise FileNotFoundError(f"No global index files found in {shard_dir}")
    return global_index


def overwrite_skeletons(
    updated_skeletons,
    shard_dir,
    n_workers=4,
    max_skeletons_per_shard=10000
):
    """
    Overwrite skeletons by:
    1) deleting old ones
    2) writing updated versions into the SAME directory
       with a unique global index label
    """

    # 1. Collect IDs to delete
    skel_ids = [sk.id for sk in updated_skeletons]

    # 2. Tombstone old skeletons
    delete_skeletons_parallel(
        shard_dir=shard_dir,
        skeleton_ids=skel_ids,
        n_workers=n_workers
    )

    # 3. Write updated skeletons back into same dir
    label = f"update_{uuid4().hex}"

    shard_and_write_skeletons(
        skeletons=updated_skeletons,
        output_dir=shard_dir,
        max_skeletons_per_shard=max_skeletons_per_shard,
        label=label,
        n_workers=n_workers
    )

    return label




def save_id_to_shard_h5(id_to_shard, output_path):
    """
    Save a large id_to_shard mapping to HDF5.

    id_to_shard: dict[int, str]  # segid -> shard path
    output_path: str
    """
    # Get unique shard paths
    shard_paths = sorted(set(id_to_shard.values()))
    shard_to_idx = {p: i for i, p in enumerate(shard_paths)}

    segids = np.array(list(id_to_shard.keys()), dtype=np.uint64)
    shard_indices = np.array([shard_to_idx[id_to_shard[k]] for k in segids], dtype=np.uint32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("segids", data=segids, compression="gzip")
        f.create_dataset("shard_indices", data=shard_indices, compression="gzip")
        # Store shard paths as variable-length strings
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("shard_paths", data=np.array(shard_paths, dtype=dt))


def load_id_to_shard_h5(h5_path):
    """
    Load ID -> shard mapping from HDF5.

    Returns:
        id_to_shard : dict[int, str]
    """
    with h5py.File(h5_path, "r") as f:
        segids = f["segids"][:]
        shard_indices = f["shard_indices"][:]
        shard_paths = f["shard_paths"][:]

    # Build mapping
    id_to_shard = {int(segid): shard_paths[idx].decode() if isinstance(shard_paths[idx], bytes) else shard_paths[idx]
                   for segid, idx in zip(segids, shard_indices)}
    return id_to_shard




def shard_and_write_skeletons(skeletons, output_dir,
                              max_skeletons_per_shard=10000,
                              label="", n_workers=4):

    os.makedirs(output_dir, exist_ok=True)
    global_shard_index = {}
    id_to_shard_index = {}

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
                executor.submit(write_single_shard,shard_items_chunk, output_dir)
                for i, shard_items_chunk in enumerate(shards)
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Writing shards"):
                shard_name, bbox, idmap = f.result()
                global_shard_index[shard_name] = bbox
                id_to_shard_index.update(idmap)
    else:
        # Serial fallback
        for i, shard_items_chunk in enumerate(shards):
            shard_name, bbox, idmap = write_single_shard(shard_items_chunk, output_dir)
            global_shard_index[shard_name] = bbox
            id_to_shard_index.update(idmap)

    # Save the global index
    with open(os.path.join(output_dir, f"{label}_global_shard_index.json"), "w") as f:
        json.dump(global_shard_index, f, indent=2)

    return global_shard_index



def load_all_skeletons(shard_dir, n_workers=1):
    global_index = load_global_index(shard_dir)
    shard_names = list(global_index.keys())

    if not shard_names:
        return [], {}

    all_skeletons = []
    shard_to_ids = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    read_shard,
                    os.path.join(shard_dir, shard_name)
                ): shard_name
                for shard_name in shard_names
            }

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading all shards"
            ):
                shard_name = futures[f]
                skels = f.result()
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]
    else:
        for shard_name in tqdm(shard_names, desc="Loading all shards"):
            shard_path = os.path.join(shard_dir, shard_name)
            skels = read_shard(shard_path)
            all_skeletons.extend(skels)
            shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids


def query_skeletons_by_bb(query_bbox, shard_dir, n_workers=1):
    global_index = load_global_index(shard_dir)

    qxmin, qymin, qzmin, qxmax, qymax, qzmax = query_bbox
    candidates = []

    for shard_name, bbox in global_index.items():
        sxmin, symin, szmin, sxmax, symax, szmax = bbox
        if not (sxmax < qxmin or sxmin > qxmax or
                symax < qymin or symin > qymax or
                szmax < qzmin or szmin > qzmax):
            candidates.append(shard_name)

    if not candidates:
        return [], {}

    all_skeletons = []
    shard_to_ids = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    read_shard,
                    os.path.join(shard_dir, shard_name),
                    bbox=query_bbox
                ): shard_name
                for shard_name in candidates
            }

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading shards"
            ):
                shard_name = futures[f]
                skels = f.result()
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]
    else:
        for shard_name in tqdm(candidates, desc="Loading shards"):
            shard_path = os.path.join(shard_dir, shard_name)
            skels = read_shard(shard_path, bbox=query_bbox)
            all_skeletons.extend(skels)
            shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids

def _read_matching_ids_from_shard(args):
    shard_dir, shard_name, segids = args
    shard_path = os.path.join(shard_dir, shard_name)
    segids = set(segids)

    with h5py.File(shard_path, "r") as f:
        index_ds = f["index"]
        segid_col = index_ds[:, 0].astype(int)

        mask = np.isin(segid_col, list(segids))
        idxs = np.where(mask)[0]

        if len(idxs) == 0:
            return shard_name, []

        verts_ds = f["vertices"]
        edges_ds = f["edges"]
        rad_ds   = f["radius"]
        types_ds = f["types"]
        vert_offsets = f["vert_offsets"][:]
        edge_offsets = f["edge_offsets"][:]

        skels = []
        for i in idxs:
            segid = segid_col[i]
            if segid < 0:
                continue

            skel = Skeleton()
            skel.id = segid

            v_start, nv = vert_offsets[i]
            e_start, ne = edge_offsets[i]

            skel.vertices = verts_ds[v_start:v_start+nv]
            skel.edges = edges_ds[e_start:e_start+ne]
            skel.radius = rad_ds[v_start:v_start+nv]
            skel.vertex_types = types_ds[v_start:v_start+nv]

            skels.append(skel)

        return shard_name, skels



def query_skeletons_by_id(segids, shard_dir, n_workers=1):
    segids = set(segids if isinstance(segids, (list, set)) else [segids])

    global_index = load_global_index(shard_dir)
    shard_names = list(global_index.keys())

    if not shard_names or not segids:
        return [], {}

    all_skeletons = []
    shard_to_ids = {}

    tasks = [(shard_dir, shard_name, segids) for shard_name in shard_names]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_read_matching_ids_from_shard, task)
                for task in tasks
            ]

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Querying shards by ID"
            ):
                shard_name, skels = f.result()
                if skels:
                    all_skeletons.extend(skels)
                    shard_to_ids[shard_name] = [s.id for s in skels]
    else:
        for task in tqdm(tasks, desc="Querying shards by ID"):
            shard_name, skels = _read_matching_ids_from_shard(task)
            if skels:
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids



def delete_skeletons_in_shard(shard_path, skeleton_ids_set):
    """
    Load a shard and mark specified skeleton IDs as deleted.
    """
    with h5py.File(shard_path, "r+") as f:
        index_ds = f["index"]
        segids = index_ds[:, 0].astype(int)
        mask = np.isin(segids, list(skeleton_ids_set))
        indices_to_delete = np.where(mask)[0]
        if len(indices_to_delete) == 0:
            return None  # No matching skeletons
        
        # Mark as deleted
        index_ds[indices_to_delete, 0] = -1
        # Optionally, zero out associated datasets here

def delete_skeletons_parallel(shard_dir, skeleton_ids, n_workers=4):
    """
    Load global index, get shard names, and delete specified skeleton IDs in parallel.
    """
    # Load global index to get shard names
    global_index = load_global_index(shard_dir)
    shard_names = list(global_index.keys())

    # Convert skeleton IDs to set for faster lookup
    skeleton_ids_set = set(skeleton_ids)

    # Create full paths for each shard
    shard_paths = [os.path.join(shard_dir, shard_name) for shard_name in shard_names]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(delete_skeletons_in_shard, shard_path, skeleton_ids_set)
            for shard_path in shard_paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Deleting skeletons"):
            future.result()  # To catch exceptions if any