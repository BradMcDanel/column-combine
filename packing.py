import math

import numpy as np
import torch

import util

def gen_mat(p, n=10, minval=4):
    t = np.zeros((n, len(p)))

    for i in range(n):
        if i == 0:
            num_nonzero = minval
        else:
            last_row = int(np.sum(t[i-1]))
            num_nonzero = np.random.randint(last_row, last_row+2)
        t[i,  np.random.choice(len(p), num_nonzero, replace=False, p=p)] = 1
    return t

def overlap(u, v):
    s = u + v
    s = np.clip(s - 1, 0, 1e10)
    return np.sum(s)

def overlap_dist(mat):
    mat = torch.clamp(mat.unsqueeze(0) + mat.unsqueeze(1) - 1, 0, 1e10).sum(dim=2)
    return mat

def density_dist(mat):
    mat = (mat.unsqueeze(0) + mat.unsqueeze(1) == 0).float().sum(dim=2)
    return mat

def path_dist(mat):
    mat = torch.from_numpy(mat.astype('f')).cuda()
    mat = (mat.unsqueeze(0) + mat.unsqueeze(1)).float().sum(dim=2)
    return mat

def density(u, v):
    return (u+v==0).sum()


def densify_(mat, idxs):
    new_mat = np.zeros((mat.shape[0], len(idxs)))
    for i, cols in enumerate(idxs):
        new_mat[:, i] = mat[:, cols].sum(axis=1)

    return new_mat

def densify(mat, col_idxs, row_idxs):
    mat = densify_(mat, col_idxs)
    mat = densify_(mat.T, row_idxs).T
    return mat

def gen_control_bits(model):
    mats = []
    for i, layer in enumerate(util.get_conv_layers(model)):
        mat = layer.weight.data
        B, C, W, H = mat.shape
        mat = mat.view(B, C*W*H)
        mat = mat.cpu().numpy()
        _, col_idxs = model.packed_layer_idxs[i]

        new_mat = np.zeros((mat.shape[0], len(col_idxs)))
        for i, cols in enumerate(col_idxs):
            zero_idxs = mat[:, cols].sum(axis=1) == 0
            c = np.array(cols).reshape(-1, 1)
            vals = np.dot(mat[:, cols] != 0, c).flatten()
            vals[zero_idxs] = cols[0]
            new_mat[:, i] = vals
        mats.append(new_mat.astype('i'))
    
    return mats

def fill_diagonal(mat, value):
    w, h = mat.shape
    mat[torch.arange(w).long(), torch.arange(h).long()] = value

def compute_dists(d_mat, row_idxs, col_idxs, max_paths, axis=0):
    if axis == 1:
        idxs = col_idxs
    else:
        idxs = row_idxs

    num_paths = np.array([len(idx) for idx in idxs])
    dists = overlap_dist(d_mat)
    fill_diagonal(dists, 1e10)
    num_streams = path_dist(num_paths.reshape(-1,1))
    stream_idxs = (num_streams > max_paths).long().nonzero()
    if len(stream_idxs) > 0:
        dists[stream_idxs[:,0], stream_idxs[:,1]] = 1e10

    return dists


def compute_metric(d_mat, row_idxs, col_idxs, dists, max_overlap, axis=0):
    dists = dists.clone()

    idxs = (dists > max_overlap).long().nonzero()
    if len(idxs) > 0:
        dists[idxs[:,0], idxs[:,1]] = 1e10

    idxs = (dists <= max_overlap).long().nonzero()
    if len(idxs) > 0:
        dists[idxs[:,0], idxs[:,1]] = 0

    num_full = density_dist(d_mat)
    dists += num_full

    return dists

def pack_matrix(mat, max_paths, max_overlap, metric='min'):
    mat = np.copy(mat)
    mat[mat!=0] = 1
    col_idxs = []
    # only columns with nonzeros
    for i in range(mat.shape[1]):
        if np.sum(mat[:, i]) > 0:
            col_idxs.append([i])

    # only rows with nonzeros
    row_idxs = []
    for i in range(mat.shape[0]):
        if np.sum(mat[i]) > 0:
            row_idxs.append([i])

    while True:
        d_mat = densify(mat, col_idxs, row_idxs)
        d_mat = torch.from_numpy(d_mat.astype('f')).cuda()
        d_mat = d_mat.transpose(0, 1).contiguous()

        col_dists = compute_dists(d_mat, row_idxs, col_idxs, max_paths, axis=1)
        col_metric = compute_metric(d_mat, row_idxs, col_idxs, col_dists, max_overlap, axis=1)

        if col_dists.min() > max_overlap:
            break

        vals, idxs = col_metric.min(0)
        c1 = (vals < 1e9).nonzero()[0][0].item()
        c2 = idxs[c1].item()
        col_idxs[c1].extend(col_idxs[c2])
        col_idxs.pop(c2)

    return row_idxs, col_idxs

def compute_flat_idxs(row_idxs, col_idxs, num_cols):
    idxs = []
    for row in row_idxs:
        for col in col_idxs:
            sublist = []
            for rid in row:
                for cid in col:
                    sublist.append(rid*num_cols + cid)
            idxs.append(sublist)

    return idxs

def fix_mask(layer, row_idxs, col_idxs):
    B, C, W, H = layer.weight.shape
    flat_idxs = compute_flat_idxs(row_idxs, col_idxs, C*W*H)
    weight = layer.weight.data.view(-1).cpu().numpy()
    for flat_idx in flat_idxs:
        sorted_idxs = np.argsort(np.abs(weight[flat_idx]))
        flat_idx = np.array(flat_idx)
        if len(flat_idx) <= 1: continue
        idxs = flat_idx[sorted_idxs[:-1]].tolist()
        layer._mask[idxs] = 0

def count_nonzeros(model_path):
    model = torch.load(model_path)
    model.eval()
    model.cuda()
    rows, cols = [], []
    for i, layer in enumerate(util.get_conv_layers(model)):
        param = layer.weight.data
        B, C, W, H = param.size()
        param = param.view(B, C*W*H).cpu().numpy()
        mat = np.copy(param)
        mat[mat!=0] = 1
        rows.append((mat.sum(axis=1) > 0).sum())
        cols.append((mat.sum(axis=0) > 0).sum())
    
    return rows, cols
 

def pack_model(model, overlap_pct, metric='min', verbose=False, section_size=256):
    model.eval()
    model.cuda()
    combined_rows, combined_cols = [], []
    total_row_idxs, total_col_idxs = [], []
    for i, layer in enumerate(util.get_conv_layers(model)):
        param = layer.weight.data
        B, C, W, H = param.shape
        overlap = int(overlap_pct*B)
        param = param.view(B, C*W*H).cpu().numpy()
        num_sections = math.ceil(param.shape[1] / section_size)
        col_idxs = []
        for j in range(num_sections):
            if verbose: 
                print(j)
            start_idx = section_size*j
            end_idx = section_size*(j+1)
            row_idxs, cols_j = pack_matrix(param[:, start_idx:end_idx],
                                           max_paths=layer.groups,
                                           max_overlap=overlap, metric=metric)
            cols_j = [[cj + start_idx for cj in ci] for ci in cols_j]
            col_idxs.extend(cols_j)

        total_row_idxs.append(row_idxs)
        total_col_idxs.append(col_idxs)

        if verbose:
            print('Layer {}:\t({}, {})'.format(i, len(row_idxs), len(col_idxs)))
        combined_rows.append(len(row_idxs))
        combined_cols.append(len(col_idxs))
        fix_mask(layer, row_idxs, col_idxs)

    model.packed_layer_size = list(zip(combined_rows, combined_cols))
    model.packed_layer_idxs = list(zip(total_row_idxs, total_col_idxs))
    model.overlap_pct = overlap_pct

    return model
