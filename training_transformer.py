import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models.data import voxelized_data_shapenet as voxelized_data
from transformer import Transformer
from memory_profiler import profile

class TrainTransformer:
    def __init__(self,device="cuda"):
        self.model = Transformer().to(device = "cuda")
        #self.model = torch.nn.DataParallel(self.model, device_ids=[1, 2]).cuda()
        self.optim = self.configure_optimizers()
        self.train(2500)
    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer
    
    def train(self,epochs):
        train_dataset = self.load_data()
        self.model = self.load_checkpoint(240)    
        for epoch in range(epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, t in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    device = "cuda"
                    p= t.get('grid_coords').to(device)
                    df_gt = t.get('df') 
                    inputs = t.get('inputs').to(device)
                    logits, targets = self.model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
                if epoch % 30 ==  0:
                    torch.save(self.model.state_dict(), os.path.join("Transformer", f"Good_{epoch}.pt"))
        torch.save(self.model.state_dict(), os.path.join("Transformer", f"Good_{epoch}.pt"))
        
    def load_checkpoint(self,epoch):
        self.model.load_state_dict(torch.load(os.path.join("Transformer",f"Good_{epoch}.pt")))
        print("loaded.....funny")
        return self.model
    
    def load_data(self):
        input_res = 32
        num_points = 5000 
        data_dir = "/home/Tejaswini/PycharmProjects/NDF/Data/00_good"
        split_file = "/home/Tejaswini/PycharmProjects/NDF/Data/Int_cars.npz"
        batch_size = 8
        num_sample_points_training = 10000    
        sample_ratio = [0.01, 0.49, 0.5]
        sample_std_dev = [0.08, 0.02, 0.003]
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
        train_dataset = train_dataset.get_loader()
        return train_dataset

model = TrainTransformer()
net = torch.nn.DataParallel(model, device_ids=[0,1, 2]).cuda()
net.to(device)


