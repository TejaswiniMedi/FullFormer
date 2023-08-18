import models.Global as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import torch

torch.cuda.empty_cache()
net = model.NDF()
## input_res represents the desired voxel resolution for sampled points.
input_res = 32
# num_points -- number of points sampled from each ground truth shape.
num_points = 5000
# data_dir -- path to Input directory
data_dir = "/home/Tejaswini/PycharmProjects/NDF/Data/00_good"
# split_file -- path to split file
split_file = "/home/Tejaswini/PycharmProjects/NDF/Data/Int_cars.npz"
batch_size = 8
num_sample_points_training = 10000
# sample_ratio --Ratio of standard deviations for samples used for training
sample_ratio = [0.01, 0.49, 0.5]
# sample_std_dev --Standard deviations of guassian samples.
sample_std_dev = [0.08, 0.02, 0.003]
exp_name = "cars"

train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=input_res,
                                          pointcloud_samples=num_points,
                                          data_path=data_dir,
                                          split_file=split_file,
                                          batch_size=batch_size,
                                          num_sample_points=num_sample_points_training,
                                          num_workers=30,
                                          sample_distribution=sample_ratio,
                                          sample_sigmas=sample_std_dev)
val_dataset = voxelized_data.VoxelizedDataset('val',
                                          res=input_res,
                                          pointcloud_samples=num_points,
                                          data_path=data_dir,
                                          split_file=split_file,
                                          batch_size=batch_size,
                                          num_sample_points=num_sample_points_training,
                                          num_workers=30,
                                          sample_distribution=sample_ratio,
                                          sample_sigmas=sample_std_dev)

trainer = training.Trainer(net,
                           torch.device("cuda"),
                           train_dataset,
                           val_dataset,
                           exp_name,
                           optimizer="Adam",
                           lr=1e-6)
trainer.train_model(20000)
