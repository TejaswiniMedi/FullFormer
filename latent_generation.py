import models.Global as model   
import models.data.voxelized_data_shapenet as voxelized_data
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from evaluation import *
from fps import farthest_point_sample, index_points
import open3d as o3d
from sample_transformer import GPT_Generator
import torch                                    
import os
import trimesh
import numpy as np
from tqdm import tqdm

net = model.NDF()
input_res = 32
num_points = 2048

data_dir = "/home/Tejaswini/PycharmProjects/NDF/Data/03001627"
split_file = "/home/Tejaswini/PycharmProjects/NDF/Data/chairs.npz"

batch_size = 10
num_sample_points_training = 4000 
sample_ratio = [0.01, 0.49, 0.5]
sample_std_dev = [0.08, 0.02, 0.003]
num_sample_points_generation = 4000 
exp_name = "cars"    #change cars
device = torch.device("cuda")
print(torch.cuda.is_available())
net = model.NDF()
dataset = voxelized_data.VoxelizedDataset('test',
                                          res= input_res,
                                          pointcloud_samples=num_points,
                                          data_path=data_dir,
                                          split_file=split_file,
                                          batch_size=1,
                                          num_sample_points=num_sample_points_generation,
                                          num_workers=30,
                                          sample_distribution=sample_ratio,
                                          sample_sigmas=sample_std_dev)


gen = GPT_Generator()

out_path = 'experiments/chair_int_rec_01/'

def offfile_to_plyfile(offfile, plyfile):
    with open(offfile, 'r') as f:
        lines = f.readlines()
    with open(plyfile, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(lines) - 3))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for line in lines[3:]:
            f.write(line)
            
def normalize_point_clouds(pcs, mode):  # As we use normalized data, we dont use this function
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs

def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)
    list_pc = []
    list_ref = []
    for i, data in tqdm(enumerate(loader)):
        print(i)
        print(data['path'][0] + "***")
        voxel_path = data["voxel_path"][0]
        #print(voxel_path)
        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue
        else:
            os.makedirs(export_path)

        for num_steps in [7]:
            point_cloud, duration = gen.generate_point_cloud(data, num_steps)
            print('num_steps', num_steps, 'duration', duration)
            trimesh.Trimesh(vertices=point_cloud, faces=[]).export(
                export_path + 'dense_point_cloud_{}.off'.format(num_steps))
            path = export_path + 'dense_point_cloud_{}.off'.format(num_steps)
            filename = os.path.basename(export_path + 'dense_point_cloud_{}.off'.format(num_steps))
            offfile_to_plyfile(path,export_path+'dense_point_cloud_7.ply')
            mesh = o3d.io.read_triangle_mesh(export_path + 'dense_point_cloud_{}.ply'.format(num_steps))#
            pcd = o3d.io.read_point_cloud(export_path + 'dense_point_cloud_{}.ply'.format(num_steps))
            R = mesh.get_rotation_matrix_from_xyz((0,-0.5 * np.pi,0))
            pc = mesh.rotate(R, center=(0, 0, 0))
            pc = mesh.vertices
            pc = np.asarray(pc)
            g_path = data['path'][0] + "/model_normalized_scaled.off"
            gt_mesh = trimesh.load(g_path, process=False)
            pointcloud_gt, idx = gt_mesh.sample(50000, return_index=True)
            pointcloud_gt = pointcloud_gt.astype(np.float32)
            ref = pointcloud_gt
            p1 = torch.from_numpy(pc)
            p1 = torch.unsqueeze(p1, 0).cuda()
            centroids_p1 = farthest_point_sample(p1, 2048)
            p1_new = index_points(p1, centroids_p1)
            p2 = torch.from_numpy(ref)
            p2 = torch.unsqueeze(p2, 0).cuda()
            centroids_p2 = farthest_point_sample(p2, 2048)
            p2_new = index_points(p2, centroids_p2)
            list_pc.append(p1_new)
            list_ref.append(p2_new)

    gen_pcs = torch.cat(list_pc, dim=0)
    ref_pcs = torch.cat(list_ref,dim=0)
    gen = gen_pcs.cpu()
    ref = ref_pcs.cpu()
    np.save( 'out_chair_rec_01.npy', gen.numpy())
    np.save( 'ref_chair_rec_01.npy', ref.numpy())

