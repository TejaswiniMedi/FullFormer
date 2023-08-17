from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.boundary_sampling import boundary_sampling
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np

# path ---> Input data path to .obj mesh files
path = /../

# num_chunks ---> To distribute preprocessing on multiple machines, Input files are split into num_chunks.
num_chunks = 3

# num_cpus ---> Number of cpu cores required to run the script (-1 is using all the available cpus).
num_cpus = -1

# sample_std_dev --> Standard deviations of guassian samples.
sample_std_dev = [0.08, 0.02, 0.003]

print('Finding raw files for preprocessing.')
path = glob(path)
paths = sorted(path)

#Not needed 
#files = os.listdir(path)
#paths_list = []
#for i in path:
#	os.chdir(i)
#	for file in glob("*.off"):
#		path_i = os.path.join(i,file)
#		paths_list.append(path_i)
#paths_list = sorted(paths_list)

chunks = np.array_split(paths,num_chunks)
for paths in chunks:
	if num_cpus == -1:
		num_cpus = mp.cpu_count()
	else:
		num_cpus = num_cpus
	def multiprocess(func):
		p = Pool(num_cpus)
		p.map(func, paths)
		p.close()
		p.join()
	print('Start scaling.')
	multiprocess(to_off)
	print('Start distance field sampling.')
	for sigma in sample_std_dev:
		print(f'Start distance field sampling with sigma: {sigma}.')
		# multiprocess(partial(boundary_sampling, sigma = sigma))
		# this process is multi-processed for each path: IGL parallelizes the distance field computation of multiple points.
		for path in paths:
			boundary_sampling(path, sigma)
	print('Start voxelized pointcloud sampling.')
	voxelized_pointcloud_sampling.init()
	multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling)
