import torch
import numpy as np

def farthest_point_sample(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C], type:Tensor[]
        npoint: number of samples, type:int
    Return:
        centroids: sampled pointcloud index, [B, npoint], type:Tensor[]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  
    distance = torch.ones(B, N).to(device) * 1e10  
    distance = distance.float()  

    if RAN:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device) 
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)  
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) 
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.float()  
        mask = dist < distance  
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C],type:Tensor[]
        idx: sample index data, [B, S],type:Tensor[]
    Return:
        new_points:, indexed points data, [B, S, C],type:Tensor[]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
