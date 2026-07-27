import os

import numpy as np
import torch

# Same dataset paths/hash sensorium/training/train.py uses to build dataloaders for these sessions.
DATA_HASH = "94c6ff995dac583098847cfecd43e7b6"
BASE_CANDIDATES = [
    ("/srv/user/polina/sensorium/sensorium/notebooks/data", "-GrayImageNet-"),
    ("/user/turishcheva/more_data_like_sensorium_2022", "-"),
]

RADIUS = 4  # max lateral (x, y) drift allowed between adjacent z-planes
CORR_THRESHOLD = 0.8


def resolve_session_dir(session_id):
    for base, link in BASE_CANDIDATES:
        candidate = os.path.join(base, f"static{session_id}{link}{DATA_HASH}")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find a data directory for session {session_id}")


def load_coords(session_id):
    return np.load(os.path.join(resolve_session_dir(session_id), "meta", "neurons", "cell_motor_coordinates.npy"))


def load_responses(session_id):
    path = os.path.join(resolve_session_dir(session_id), "data", "responses")
    n = len(os.listdir(path))
    return np.vstack([np.load(os.path.join(path, f"{j}.npy")) for j in range(n)]).T  # (neurons, images)


def dedup_animal(coords_a, responses_a):
    """Groups neurons that are the same physical cell imaged across adjacent z-planes.

    Returns (unique_groups, individual_neurons): unique_groups is a list of sets of neuron
    indices (size >= 2) that are duplicates of each other; individual_neurons is a sorted list
    of neuron indices that have no duplicate.
    """
    n = len(coords_a)

    def corr(x, y):
        x = x - x.mean()
        y = y - y.mean()
        return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))

    def walk_z(planes, z_order, target_resp, prev_xy):
        copies = []
        for z in z_order:
            gidx, x, y = min(planes[z], key=lambda t: abs(t[1] - prev_xy[0]) + abs(t[2] - prev_xy[1]))
            if abs(x - prev_xy[0]) > RADIUS or abs(y - prev_xy[1]) > RADIUS:
                break
            if corr(target_resp, responses_a[gidx]) <= CORR_THRESHOLD:
                break
            copies.append(gidx)
            prev_xy = (x, y)
        return copies

    planes = {}
    for g in range(n):
        x, y, z = coords_a[g]
        planes.setdefault(z, []).append((g, x, y))
    sorted_z = sorted(planes)

    copy_dict = {}
    for g in range(n):
        x, y, z = coords_a[g]
        target = responses_a[g]
        up = [zz for zz in sorted_z if zz > z]
        down = [zz for zz in sorted_z if zz < z][::-1]
        copies = walk_z(planes, up, target, (x, y)) + walk_z(planes, down, target, (x, y))
        if copies:
            copy_dict[g] = copies

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for neuron, copies in copy_dict.items():
        for c in copies:
            union(neuron, c)

    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    unique_groups = list(groups.values())
    grouped = set().union(*unique_groups) if unique_groups else set()
    individual_neurons = sorted(set(range(n)) - grouped)
    return unique_groups, individual_neurons


def compute_dedup_groups(session_id):
    """List of index-lists covering every neuron in `session_id` exactly once: singleton lists
    for non-duplicated neurons, longer lists for duplicate-cell groups."""
    coords_a = load_coords(session_id)
    responses_a = load_responses(session_id)
    unique_groups, individual_neurons = dedup_animal(coords_a, responses_a)
    return [[n] for n in individual_neurons] + [sorted(g) for g in unique_groups]


def build_dedup_tensors(session_ids, device):
    """Precompute, once per session, the tensors needed to vectorize dedup-aware pooling of a
    session's per-neuron feature matrix during training (see `pool_features` in trainers.py).

    Returns {session_id: {"group_id", "n_groups", "group_sizes", "group_members_padded"}}.
    """
    dedup_info = {}
    for session_id in session_ids:
        groups = compute_dedup_groups(session_id)
        n_neurons = sum(len(g) for g in groups)
        n_groups = len(groups)
        max_size = max(len(g) for g in groups)

        group_id = torch.empty(n_neurons, dtype=torch.long)
        group_members_padded = torch.zeros(n_groups, max_size, dtype=torch.long)
        group_sizes = torch.tensor([len(g) for g in groups], dtype=torch.long)

        for gi, members in enumerate(groups):
            for m in members:
                group_id[m] = gi
            reps = max_size // len(members) + 1
            group_members_padded[gi] = torch.tensor((members * reps)[:max_size], dtype=torch.long)

        dedup_info[session_id] = {
            "group_id": group_id.to(device),
            "n_groups": n_groups,
            "group_sizes": group_sizes.to(device),
            "group_members_padded": group_members_padded.to(device),
        }
    return dedup_info
