import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import time
from transformer import Transformer

class GPT_Generator(object):
    def __init__(self,threshold = 0.1, device = torch.device("cuda")):
        self.device = device
        self.threshold = threshold
        self.transformer = Transformer().to(device="cuda")  # Trans
        self.transformer.load_state_dict(torch.load(os.path.join("Transformer", "chair_870.pt")))   
        print("loaded ....sucessfully")
        self.transformer.eval()
        self.sos_token = 0

    def generate_point_cloud(self, data, num_steps = 10, num_points = 25000*4, filter_val = 0.009):
        start = time.time()
        inputs = data['inputs'].to(self.device)
        for param in self.transformer.parameters():
            param.requires_grad = False
        sample_num = 5000
        samples_cpu = np.zeros((0, 3))
        samples = torch.rand(1, sample_num, 3).float().to(self.device) * 3 - 1.5
        samples.requires_grad = True
        encoding,indices  = self.transformer.encode_to_z(inputs)
        sos_tokens = torch.ones(samples.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")
        start_indices = indices[:, :0]
        full_indices = self.transformer.sample(start_indices, sos_tokens, steps=indices.shape[1])
        print(full_indices.shape)
        print(indices.shape)
        i = 0
        while len(samples_cpu) < num_points:
            print('iteration', i)
            for j in range(num_steps):
                full_sample = self.transformer.z_to_udf(samples, full_indices)
                print('refinement', j)
                df_pred = torch.clamp(full_sample, max=self.threshold)
                df_pred.sum().backward()
                gradient = samples.grad.clone().detach()
                samples = samples.clone().detach()
                df_pred = df_pred.clone().detach()
                inputs = inputs.clone().detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
                samples = samples.detach()
                samples.requires_grad = True
            print('finished refinement')
            if not i == 0:
                samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))
            samples = samples[df_pred < 0.03].unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * torch.randn(samples.shape).to(self.device)  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            print(samples_cpu.shape)

        duration = time.time() - start
        print(duration)
        return samples_cpu*0.5, duration
